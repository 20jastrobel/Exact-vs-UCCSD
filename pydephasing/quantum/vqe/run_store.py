"""Run storage backends for benchmarking and reproducibility.

This module is intentionally lightweight:
  - stable RunStore interface (start_run/log_step/add_artifact/finish_run)
  - JsonRunStore: legacy JSON/JSONL on-disk format
  - SqliteRunStore: SQLite ledger (migrations + batched inserts)

Algorithm code can be wired to this later without changing core logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON encoding for hashing and storage."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(int(chunk_size))
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@runtime_checkable
class RunStore(Protocol):
    def start_run(self, run_config: dict) -> str: ...

    def log_step(self, run_id: str, step_idx: int, metrics: dict) -> None: ...

    def add_artifact(
        self,
        run_id: str,
        kind: str,
        path: str,
        sha256: str,
        bytes: int | None,
        extra: dict | None = None,
    ) -> None: ...

    def finish_run(self, run_id: str, status: str, summary_metrics: dict) -> None: ...


class MigrationError(RuntimeError):
    pass


def _default_migrations_dir() -> Path:
    return Path(__file__).resolve().parent / "migrations"


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations(
          version TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL
        )
        """
    )


def apply_sqlite_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> None:
    """Apply all *.sql migrations in lexicographic order (idempotent)."""
    mig_dir = Path(migrations_dir) if migrations_dir is not None else _default_migrations_dir()
    if not mig_dir.exists():
        raise MigrationError(f"Missing migrations dir: {mig_dir}")

    _ensure_schema_migrations(conn)
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}

    sql_paths = sorted(p for p in mig_dir.glob("*.sql") if p.is_file())
    for path in sql_paths:
        version = path.name
        if version in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (version, _utc_now_iso()),
            )
            conn.commit()
        except Exception as exc:  # pragma: no cover - catastrophic / corrupt migration
            conn.rollback()
            raise MigrationError(f"Failed applying migration {version}: {exc}") from exc


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_energy(metrics: Mapping[str, Any]) -> float | None:
    return _coerce_float(metrics.get("energy"))


def _extract_grad_norm(metrics: Mapping[str, Any]) -> float | None:
    # Prefer explicit grad norm; fall back to "max_grad" which is commonly logged.
    if "grad_norm" in metrics:
        return _coerce_float(metrics.get("grad_norm"))
    if "max_grad" in metrics:
        return _coerce_float(metrics.get("max_grad"))
    return None


def _extract_wall_time_s(metrics: Mapping[str, Any]) -> float | None:
    if "wall_time_s" in metrics:
        return _coerce_float(metrics.get("wall_time_s"))
    # Current JSONL logs use t_cum_s as the canonical runtime proxy.
    if "t_cum_s" in metrics:
        return _coerce_float(metrics.get("t_cum_s"))
    return None


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    run_id: str
    kind: str
    path: str
    sha256: str
    bytes: int | None
    created_at: str
    extra_json: str | None


class SqliteRunStore:
    """SQLite-backed run ledger.

    Notes:
      - uses a small migration system (plain SQL files)
      - batches step inserts for reasonable performance
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        batch_size: int = 200,
        migrations_dir: str | Path | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._apply_pragmas()
        apply_sqlite_migrations(self.conn, Path(migrations_dir) if migrations_dir is not None else None)

        self.batch_size = int(batch_size)
        self._pending_steps: list[tuple[str, int, str, float | None, float | None, float | None]] = []
        self._pending_artifacts: list[ArtifactRecord] = []

    def _apply_pragmas(self) -> None:
        # Safe-ish defaults: WAL for concurrent readers; NORMAL synchronous for speed; FK for consistency.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self.flush()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def flush(self) -> None:
        if self._pending_steps:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO steps(
                  run_id, step_idx, metrics_json, energy, grad_norm, wall_time_s
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                self._pending_steps,
            )
            self._pending_steps.clear()

        if self._pending_artifacts:
            self.conn.executemany(
                """
                INSERT INTO artifacts(
                  artifact_id, run_id, kind, path, sha256, bytes, created_at, extra_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        a.artifact_id,
                        a.run_id,
                        a.kind,
                        a.path,
                        a.sha256,
                        a.bytes,
                        a.created_at,
                        a.extra_json,
                    )
                    for a in self._pending_artifacts
                ],
            )
            self._pending_artifacts.clear()

        self.conn.commit()

    def start_run(self, run_config: dict) -> str:
        run_id = str(run_config.get("run_id") or uuid.uuid4().hex)
        created_at = _utc_now_iso()

        config_json = stable_json_dumps(run_config)
        config_hash = sha256_text(config_json)

        seed = _coerce_int(run_config.get("seed"))
        git_commit = run_config.get("git_commit")
        git_commit = str(git_commit) if git_commit is not None else None

        system_json = run_config.get("system")
        ansatz_json = run_config.get("ansatz")
        optimizer_json = run_config.get("optimizer")

        system_txt = stable_json_dumps(system_json) if system_json is not None else None
        ansatz_txt = stable_json_dumps(ansatz_json) if ansatz_json is not None else None
        opt_txt = stable_json_dumps(optimizer_json) if optimizer_json is not None else None

        self.conn.execute(
            """
            INSERT INTO runs(
              run_id, created_at, started_at, finished_at, status,
              git_commit, seed,
              config_json, config_hash,
              system_json, ansatz_json, optimizer_json,
              summary_json
            ) VALUES(?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                run_id,
                created_at,
                created_at,
                "running",
                git_commit,
                seed,
                config_json,
                config_hash,
                system_txt,
                ansatz_txt,
                opt_txt,
            ),
        )
        self.conn.commit()
        return run_id

    def log_step(self, run_id: str, step_idx: int, metrics: dict) -> None:
        metrics_json = stable_json_dumps(metrics)
        energy = _extract_energy(metrics)
        grad_norm = _extract_grad_norm(metrics)
        wall_time_s = _extract_wall_time_s(metrics)

        self._pending_steps.append(
            (str(run_id), int(step_idx), metrics_json, energy, grad_norm, wall_time_s)
        )
        if self.batch_size > 0 and len(self._pending_steps) >= self.batch_size:
            self.flush()

    def add_artifact(
        self,
        run_id: str,
        kind: str,
        path: str,
        sha256: str,
        bytes: int | None,
        extra: dict | None = None,
    ) -> None:
        b = bytes
        if b is None:
            try:
                b = int(os.stat(path).st_size)
            except Exception:
                b = None
        rec = ArtifactRecord(
            artifact_id=uuid.uuid4().hex,
            run_id=str(run_id),
            kind=str(kind),
            path=str(path),
            sha256=str(sha256),
            bytes=b,
            created_at=_utc_now_iso(),
            extra_json=stable_json_dumps(extra) if extra is not None else None,
        )
        self._pending_artifacts.append(rec)
        if self.batch_size > 0 and len(self._pending_artifacts) >= self.batch_size:
            self.flush()

    def finish_run(self, run_id: str, status: str, summary_metrics: dict) -> None:
        self.flush()
        finished_at = _utc_now_iso()
        summary_json = stable_json_dumps(summary_metrics)
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, finished_at = ?, summary_json = ?
            WHERE run_id = ?
            """,
            (str(status), finished_at, summary_json, str(run_id)),
        )
        self.conn.commit()


class JsonRunStore:
    """Legacy JSON/JSONL store compatible with the repo's current run artifacts.

    Layout per run:
      <base_dir>/<run_dir_name>/
        meta.json
        history.jsonl
        result.json

    This is intended for compatibility with existing scripts that scan `runs/`.
    """

    def __init__(self, base_dir: str | Path = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._run_dirs: dict[str, Path] = {}
        self._artifact_records: dict[str, list[dict]] = {}

    def start_run(self, run_config: dict) -> str:
        run_id = str(run_config.get("run_id") or uuid.uuid4().hex)
        # Allow callers to explicitly choose the output directory (useful for
        # scripts that already have a canonical log_dir layout).
        explicit_dir = run_config.get("run_dir") or run_config.get("log_dir")
        if explicit_dir:
            run_dir = Path(str(explicit_dir))
            if not run_dir.is_absolute():
                run_dir = (self.base_dir / run_dir).resolve()
        else:
            # Best-effort compatibility with existing directory naming.
            sites = run_config.get("sites", run_config.get("n_sites"))
            n_up = run_config.get("n_up")
            n_down = run_config.get("n_down")
            if sites is not None and n_up is not None and n_down is not None:
                run_dir = self.base_dir / f"{run_id}_L{int(sites)}_Nup{int(n_up)}_Ndown{int(n_down)}"
            else:
                run_dir = self.base_dir / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = dict(run_config)
        meta.setdefault("run_id", run_id)
        meta.setdefault("created_at", _utc_now_iso())
        meta.setdefault("status", "running")
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "history.jsonl").write_text("", encoding="utf-8")
        self._run_dirs[run_id] = run_dir
        self._artifact_records[run_id] = []
        return run_id

    def _run_dir(self, run_id: str) -> Path:
        if run_id not in self._run_dirs:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self._run_dirs[run_id]

    def log_step(self, run_id: str, step_idx: int, metrics: dict) -> None:
        run_dir = self._run_dir(run_id)
        row = {"iter": int(step_idx)}
        row.update(metrics)
        with open(run_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(stable_json_dumps(row) + "\n")

    def add_artifact(
        self,
        run_id: str,
        kind: str,
        path: str,
        sha256: str,
        bytes: int | None,
        extra: dict | None = None,
    ) -> None:
        rec = {
            "kind": str(kind),
            "path": str(path),
            "sha256": str(sha256),
            "bytes": None if bytes is None else int(bytes),
            "created_at": _utc_now_iso(),
            "extra": extra,
        }
        self._artifact_records.setdefault(str(run_id), []).append(rec)

    def finish_run(self, run_id: str, status: str, summary_metrics: dict) -> None:
        run_dir = self._run_dir(run_id)

        # Update meta.json status.
        meta_path = run_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"run_id": run_id}
        meta["status"] = str(status)
        meta["finished_at"] = _utc_now_iso()
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Write result.json.
        payload = dict(summary_metrics)
        payload.setdefault("run_id", run_id)
        payload.setdefault("status", str(status))
        payload.setdefault("artifacts", self._artifact_records.get(str(run_id), []))
        (run_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RunStoreLogger:
    """Adapter with the logger API expected by ADAPT runners.

    `pydephasing/quantum/vqe/adapt_vqe_meta.py:run_meta_adapt_vqe` expects a logger that
    provides:
      - start_iter()
      - end_iter() -> (dt, t_cum)
      - log_point(it=..., energy=..., max_grad=..., chosen_op=..., t_iter_s=..., t_cum_s=..., extra=...)

    This class forwards each logged row into the configured RunStore.
    """

    def __init__(self, store: RunStore, run_id: str):
        self.store = store
        self.run_id = str(run_id)
        self.t_cum = 0.0
        self._t0 = None

    def start_iter(self) -> None:
        import time

        self._t0 = time.perf_counter()

    def end_iter(self) -> tuple[float, float]:
        import time

        if self._t0 is None:
            dt = 0.0
        else:
            dt = float(time.perf_counter() - float(self._t0))
        self.t_cum += dt
        return float(dt), float(self.t_cum)

    def log_point(
        self,
        *,
        it,
        energy,
        max_grad=None,
        chosen_op=None,
        t_iter_s=None,
        t_cum_s=None,
        extra=None,
    ) -> None:
        metrics = {
            "energy": float(energy),
            "max_grad": None if max_grad is None else float(max_grad),
            "chosen_op": None if chosen_op is None else str(chosen_op),
            "t_iter_s": None if t_iter_s is None else float(t_iter_s),
            "t_cum_s": None if t_cum_s is None else float(t_cum_s),
        }
        if extra:
            metrics.update(dict(extra))
        self.store.log_step(self.run_id, int(it), metrics)


class MultiRunStore:
    """Fan-out RunStore that writes to multiple backends.

    This is useful during migration: default to SQLite while still emitting the legacy
    JSON/JSONL artifacts for compatibility with existing scripts.
    """

    def __init__(self, stores: Iterable[RunStore]):
        stores_list = list(stores)
        if not stores_list:
            raise ValueError("MultiRunStore requires at least one backend.")
        self._stores = stores_list

    def start_run(self, run_config: dict) -> str:
        # Ensure a stable run_id across all stores when possible.
        run_id = str(run_config.get("run_id") or uuid.uuid4().hex)
        run_config = dict(run_config)
        run_config["run_id"] = run_id

        ids = []
        for s in self._stores:
            ids.append(str(s.start_run(run_config)))
        # Prefer the explicit run_id; sanity-check that stores didn't diverge.
        for got in ids:
            if got != run_id:
                raise RuntimeError(f"MultiRunStore backend returned mismatched run_id: expected {run_id}, got {got}")
        return run_id

    def log_step(self, run_id: str, step_idx: int, metrics: dict) -> None:
        for s in self._stores:
            s.log_step(run_id, step_idx, metrics)

    def add_artifact(
        self,
        run_id: str,
        kind: str,
        path: str,
        sha256: str,
        bytes: int | None,
        extra: dict | None = None,
    ) -> None:
        for s in self._stores:
            s.add_artifact(run_id, kind, path, sha256, bytes, extra)

    def finish_run(self, run_id: str, status: str, summary_metrics: dict) -> None:
        for s in self._stores:
            s.finish_run(run_id, status, summary_metrics)

    def close(self) -> None:
        for s in self._stores:
            if hasattr(s, "close"):
                try:
                    s.close()  # type: ignore[attr-defined]
                except Exception:
                    # Don't mask the primary error path due to cleanup issues.
                    pass
