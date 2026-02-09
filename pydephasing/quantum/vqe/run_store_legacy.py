"""Legacy JSON run import/export utilities for the SQLite run ledger."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .run_store import (
    MigrationError,
    apply_sqlite_migrations,
    sha256_file,
    sha256_text,
    stable_json_dumps,
)


@dataclass(frozen=True)
class ImportStats:
    runs_found: int
    runs_imported: int
    runs_skipped: int
    runs_conflicts: int
    steps_written: int
    artifacts_written: int


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat()


def _relative_to_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def discover_legacy_run_dirs(runs_dir: str | Path) -> list[Path]:
    """Discover legacy run directories by reading meta.json and requiring history.jsonl.

    This does not use directory-name heuristics; it discovers by locating meta.json,
    parsing it, and verifying that the directory looks like a run (has history.jsonl).
    """
    root = Path(runs_dir)
    if not root.exists():
        return []
    out: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        run_dir = meta_path.parent
        hist_path = run_dir / "history.jsonl"
        if not hist_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        out.append(run_dir)
    # Stable ordering helps tests and makes repeated imports deterministic.
    out.sort(key=lambda p: str(p))
    return out


def _extract_promoted_step_fields(metrics: dict) -> tuple[float | None, float | None, float | None]:
    def _f(x: Any) -> float | None:
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    energy = _f(metrics.get("energy"))
    grad_norm = _f(metrics.get("grad_norm"))
    if grad_norm is None:
        grad_norm = _f(metrics.get("max_grad"))
    wall_time_s = _f(metrics.get("wall_time_s"))
    if wall_time_s is None:
        wall_time_s = _f(metrics.get("t_cum_s"))
    return energy, grad_norm, wall_time_s


def _iter_history_rows(path: Path) -> Iterable[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row
    except FileNotFoundError:
        return


def init_db(*, db_path: str | Path) -> None:
    """Create/open a SQLite DB and apply migrations."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    try:
        apply_sqlite_migrations(conn)
    finally:
        conn.close()


def import_legacy_runs(
    *,
    runs_dir: str | Path,
    db_path: str | Path,
    dry_run: bool = False,
) -> ImportStats:
    root = Path(runs_dir)
    run_dirs = discover_legacy_run_dirs(root)

    if dry_run:
        # Best-effort counting without touching the DB.
        steps = 0
        artifacts = 0
        for run_dir in run_dirs:
            hist = run_dir / "history.jsonl"
            steps += sum(1 for _ in _iter_history_rows(hist))
            artifacts += 1  # meta.json
            artifacts += 1  # history.jsonl
            if (run_dir / "result.json").exists():
                artifacts += 1
        return ImportStats(
            runs_found=len(run_dirs),
            runs_imported=0,
            runs_skipped=0,
            runs_conflicts=0,
            steps_written=steps,
            artifacts_written=artifacts,
        )

    dbp = Path(db_path)
    dbp.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(dbp))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

        apply_sqlite_migrations(conn)

        runs_imported = 0
        runs_skipped = 0
        runs_conflicts = 0
        steps_written = 0
        artifacts_written = 0

        for run_dir in run_dirs:
            meta_path = run_dir / "meta.json"
            hist_path = run_dir / "history.jsonl"
            result_path = run_dir / "result.json"

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                runs_skipped += 1
                continue
            if not isinstance(meta, dict):
                runs_skipped += 1
                continue

            config_json = stable_json_dumps(meta)
            config_hash = sha256_text(config_json)

            run_id_val = meta.get("run_id")
            run_id = str(run_id_val) if run_id_val else f"legacy_{run_dir.name}_{config_hash[:12]}"

            # Extract a few common fields when present.
            seed = meta.get("seed")
            try:
                seed_i = int(seed) if seed is not None else None
            except Exception:
                seed_i = None
            git_commit = meta.get("git_commit")
            git_commit_s = str(git_commit) if git_commit is not None else None

            system_json = {
                k: meta.get(k)
                for k in ("sites", "n_up", "n_down", "sz_target", "ham_params")
                if k in meta
            }
            ansatz_json = {k: meta.get(k) for k in ("pool", "cse", "uccsd") if k in meta}
            optimizer_json = {
                k: meta.get(k)
                for k in ("inner_optimizer", "max_depth", "inner_steps", "eps_grad", "eps_energy")
                if k in meta
            }

            created_at = meta.get("created_at") or _mtime_iso(meta_path)
            started_at = meta.get("started_at") or created_at
            finished_at = meta.get("finished_at")
            if finished_at is None and result_path.exists():
                finished_at = _mtime_iso(result_path)
            if finished_at is None:
                finished_at = _mtime_iso(hist_path)

            status = meta.get("status")
            summary: dict | None = None
            if result_path.exists():
                try:
                    summary = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception:
                    summary = None
                if status is None and isinstance(summary, dict):
                    status = summary.get("status")
            if status is None:
                status = "completed" if summary is not None else "imported"

            existing = conn.execute(
                "SELECT config_hash FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != str(config_hash):
                    runs_conflicts += 1
                    continue
                # Run already imported: still upsert steps/artifacts (idempotent),
                # and update status/summary in case they changed.
            else:
                conn.execute(
                    """
                    INSERT INTO runs(
                      run_id, created_at, started_at, finished_at, status,
                      git_commit, seed,
                      config_json, config_hash,
                      system_json, ansatz_json, optimizer_json,
                      summary_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        str(created_at),
                        str(started_at),
                        str(finished_at),
                        str(status),
                        git_commit_s,
                        seed_i,
                        config_json,
                        config_hash,
                        stable_json_dumps(system_json) if system_json else None,
                        stable_json_dumps(ansatz_json) if ansatz_json else None,
                        stable_json_dumps(optimizer_json) if optimizer_json else None,
                        stable_json_dumps(summary) if summary is not None else None,
                    ),
                )
                runs_imported += 1

            # Always refresh status/summary/finished_at from disk (cheap and keeps DB in sync).
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, summary_json = ? WHERE run_id = ?",
                (
                    str(status),
                    str(finished_at),
                    stable_json_dumps(summary) if summary is not None else None,
                    run_id,
                ),
            )

            # Steps: insert-or-replace.
            pending_steps: list[tuple[str, int, str, float | None, float | None, float | None]] = []
            for row in _iter_history_rows(hist_path):
                step_idx_val = row.get("iter")
                try:
                    step_idx = int(step_idx_val)
                except Exception:
                    # If missing, fall back to a sequential index. This preserves ordering even if
                    # legacy logs were malformed.
                    step_idx = len(pending_steps)
                metrics_json = stable_json_dumps(row)
                energy, grad_norm, wall_time_s = _extract_promoted_step_fields(row)
                pending_steps.append((run_id, step_idx, metrics_json, energy, grad_norm, wall_time_s))

            if pending_steps:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO steps(
                      run_id, step_idx, metrics_json, energy, grad_norm, wall_time_s
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    pending_steps,
                )
                steps_written += len(pending_steps)

            # Artifacts: deterministic artifact_id for idempotency.
            def _add_art(kind: str, path: Path) -> None:
                nonlocal artifacts_written
                try:
                    sha = sha256_file(path)
                    size = int(path.stat().st_size)
                    created = _mtime_iso(path)
                except FileNotFoundError:
                    return
                except Exception:
                    return
                artifact_id = sha256_text(f"{run_id}|{kind}|{sha}")
                rel = _relative_to_or_abs(path, root)
                extra = {"legacy_relpath": rel}
                conn.execute(
                    """
                    INSERT OR REPLACE INTO artifacts(
                      artifact_id, run_id, kind, path, sha256, bytes, created_at, extra_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        run_id,
                        str(kind),
                        str(rel),
                        str(sha),
                        size,
                        created,
                        stable_json_dumps(extra),
                    ),
                )
                artifacts_written += 1

            _add_art("legacy_meta_json", meta_path)
            _add_art("legacy_history_jsonl", hist_path)
            if result_path.exists():
                _add_art("legacy_result_json", result_path)

        conn.commit()
        return ImportStats(
            runs_found=len(run_dirs),
            runs_imported=runs_imported,
            runs_skipped=runs_skipped,
            runs_conflicts=runs_conflicts,
            steps_written=steps_written,
            artifacts_written=artifacts_written,
        )
    except MigrationError:
        raise
    finally:
        conn.close()


def export_legacy_run(*, run_id: str, out_dir: str | Path, db_path: str | Path) -> Path:
    """Export a single run from SQLite back to meta.json/history.jsonl/result.json."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / str(run_id)
    dest.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        apply_sqlite_migrations(conn)

        row = conn.execute(
            "SELECT config_json, status, summary_json FROM runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        config_json, status, summary_json = row
        meta = json.loads(config_json) if config_json else {}
        if not isinstance(meta, dict):
            meta = {"run_id": str(run_id)}
        meta.setdefault("run_id", str(run_id))
        meta.setdefault("status", str(status))
        (dest / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Steps -> history.jsonl
        lines: list[str] = []
        for step_idx, metrics_json in conn.execute(
            "SELECT step_idx, metrics_json FROM steps WHERE run_id = ? ORDER BY step_idx ASC",
            (str(run_id),),
        ):
            step_idx_i = int(step_idx)
            metrics = json.loads(metrics_json) if metrics_json else {}
            if not isinstance(metrics, dict):
                metrics = {}
            # Ensure iter is consistent with the DB primary key.
            metrics["iter"] = step_idx_i
            lines.append(json.dumps(metrics, sort_keys=True) + "\n")
        (dest / "history.jsonl").write_text("".join(lines), encoding="utf-8")

        # Summary -> result.json
        if summary_json:
            summary = json.loads(summary_json)
        else:
            summary = {}
        if not isinstance(summary, dict):
            summary = {"summary": summary}
        summary.setdefault("run_id", str(run_id))
        summary.setdefault("status", str(status))
        (dest / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        return dest
    finally:
        conn.close()
