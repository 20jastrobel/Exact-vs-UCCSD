"""Read-only query helpers for the SQLite run ledger.

These utilities intentionally do NOT expose arbitrary SQL execution. Queries are
parameterized and limited to the run-ledger schema defined in
`pydephasing/quantum/vqe/migrations/0001_init.sql`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote


def _json_loads(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open the run ledger in read-only mode (URI `mode=ro`)."""
    p = Path(db_path).expanduser().resolve()
    # Quote but keep path separators and drive separators intact.
    uri = f"file:{quote(p.as_posix(), safe='/:')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
    except Exception:
        # Not supported on all SQLite builds; best-effort only.
        pass
    return conn


def _get_dotted(obj: Any, key: str) -> Any:
    cur = obj
    for part in str(key).split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _matches_filters(rec: dict, filters: dict) -> bool:
    for raw_k, raw_v in (filters or {}).items():
        k = str(raw_k)
        # Convenience aliases.
        if k in {"sites", "n_up", "n_down"}:
            k = f"system.{k}"
        if k in {"pool"}:
            k = "ansatz.pool"

        got = _get_dotted(rec, k)
        want = raw_v

        # Normalize ints when possible.
        if isinstance(want, (int, float)) and got is not None:
            try:
                if int(got) != int(want):
                    return False
                continue
            except Exception:
                pass

        if got != want:
            return False
    return True


def list_runs(*, db_path: str | Path, filters: dict | None = None, limit: int = 50) -> list[dict]:
    """List runs with simple parameterized filters.

    Supported SQL-level filter keys:
      - run_id, status, seed, git_commit, config_hash

    Any other keys are applied post-hoc on the returned JSON-decoded fields using
    dotted paths (e.g. `system.sites`, `ansatz.pool`).
    """
    filters = dict(filters or {})
    limit_i = max(1, int(limit))
    fetch_limit = min(max(limit_i * 20, 200), 5000)

    where = []
    params: list[Any] = []

    if "run_id" in filters:
        where.append("run_id = ?")
        params.append(_to_str(filters.pop("run_id")))
    if "status" in filters:
        where.append("status = ?")
        params.append(_to_str(filters.pop("status")))
    if "seed" in filters:
        where.append("seed = ?")
        params.append(_to_int(filters.pop("seed")))
    if "git_commit" in filters:
        val = _to_str(filters.pop("git_commit"))
        if val is not None and val.endswith("*"):
            where.append("git_commit LIKE ?")
            params.append(val[:-1] + "%")
        else:
            where.append("git_commit = ?")
            params.append(val)
    if "config_hash" in filters:
        where.append("config_hash = ?")
        params.append(_to_str(filters.pop("config_hash")))

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
              run_id, created_at, started_at, finished_at, status,
              git_commit, seed, config_hash,
              system_json, ansatz_json, optimizer_json, summary_json
            FROM runs
            """
            + where_sql
            + " ORDER BY created_at DESC, run_id DESC LIMIT ?",
            tuple(params + [fetch_limit]),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict] = []
    for row in rows:
        rec = {
            "run_id": str(row["run_id"]),
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "status": row["status"],
            "git_commit": row["git_commit"],
            "seed": row["seed"],
            "config_hash": row["config_hash"],
            "system": _json_loads(row["system_json"]) or {},
            "ansatz": _json_loads(row["ansatz_json"]) or {},
            "optimizer": _json_loads(row["optimizer_json"]) or {},
            "summary": _json_loads(row["summary_json"]) or {},
        }
        if not _matches_filters(rec, filters):
            continue
        out.append(rec)
        if len(out) >= limit_i:
            break
    return out


def get_run(*, db_path: str | Path, run_id: str) -> dict | None:
    conn = connect_readonly(db_path)
    try:
        row = conn.execute(
            """
            SELECT
              run_id, created_at, started_at, finished_at, status,
              git_commit, seed, config_hash,
              config_json,
              system_json, ansatz_json, optimizer_json,
              summary_json
            FROM runs
            WHERE run_id = ?
            """,
            (str(run_id),),
        ).fetchone()
        if row is None:
            return None
        steps_n = conn.execute(
            "SELECT COUNT(*) AS n FROM steps WHERE run_id = ?", (str(run_id),)
        ).fetchone()["n"]
        arts_n = conn.execute(
            "SELECT COUNT(*) AS n FROM artifacts WHERE run_id = ?", (str(run_id),)
        ).fetchone()["n"]
    finally:
        conn.close()

    return {
        "run_id": str(row["run_id"]),
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "status": row["status"],
        "git_commit": row["git_commit"],
        "seed": row["seed"],
        "config_hash": row["config_hash"],
        "config": _json_loads(row["config_json"]) or {},
        "system": _json_loads(row["system_json"]) or {},
        "ansatz": _json_loads(row["ansatz_json"]) or {},
        "optimizer": _json_loads(row["optimizer_json"]) or {},
        "summary": _json_loads(row["summary_json"]) or {},
        "counts": {"steps": int(steps_n), "artifacts": int(arts_n)},
    }


def get_steps(
    *,
    db_path: str | Path,
    run_id: str,
    metric: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> dict:
    """Return steps for a run.

    - `start`/`end` are inclusive bounds on `step_idx`.
    - If `metric` is provided, returns a compact list of `{step_idx, value}`.
    - Otherwise returns `{step_idx, energy, grad_norm, wall_time_s, metrics}`.
    """
    where = ["run_id = ?"]
    params: list[Any] = [str(run_id)]
    if start is not None:
        where.append("step_idx >= ?")
        params.append(int(start))
    if end is not None:
        where.append("step_idx <= ?")
        params.append(int(end))
    where_sql = " WHERE " + " AND ".join(where)

    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT step_idx, metrics_json, energy, grad_norm, wall_time_s FROM steps"
            + where_sql
            + " ORDER BY step_idx ASC",
            tuple(params),
        ).fetchall()
    finally:
        conn.close()

    metric_s = None if metric is None else str(metric)
    steps: list[dict] = []
    for row in rows:
        step_idx = int(row["step_idx"])
        if metric_s is None:
            metrics = _json_loads(row["metrics_json"])
            if not isinstance(metrics, dict):
                metrics = {}
            steps.append(
                {
                    "step_idx": step_idx,
                    "energy": row["energy"],
                    "grad_norm": row["grad_norm"],
                    "wall_time_s": row["wall_time_s"],
                    "metrics": metrics,
                }
            )
            continue

        if metric_s == "energy":
            val = row["energy"]
        elif metric_s == "grad_norm":
            val = row["grad_norm"]
        elif metric_s == "wall_time_s":
            val = row["wall_time_s"]
        else:
            metrics = _json_loads(row["metrics_json"])
            if not isinstance(metrics, dict):
                metrics = {}
            val = metrics.get(metric_s)
        steps.append({"step_idx": step_idx, "value": val})

    return {"run_id": str(run_id), "metric": metric_s, "steps": steps}


def list_artifacts(*, db_path: str | Path, run_id: str, kind: str | None = None) -> list[dict]:
    where = ["run_id = ?"]
    params: list[Any] = [str(run_id)]
    if kind is not None:
        where.append("kind = ?")
        params.append(str(kind))
    where_sql = " WHERE " + " AND ".join(where)

    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            """
            SELECT artifact_id, kind, path, sha256, bytes, created_at, extra_json
            FROM artifacts
            """
            + where_sql
            + " ORDER BY created_at ASC, artifact_id ASC",
            tuple(params),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "artifact_id": str(row["artifact_id"]),
                "kind": row["kind"],
                "path": row["path"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "created_at": row["created_at"],
                "extra": _json_loads(row["extra_json"]),
            }
        )
    return out


def export_artifact_path(*, db_path: str | Path, artifact_id: str) -> dict | None:
    """Return artifact path + sha256 only (no file contents)."""
    conn = connect_readonly(db_path)
    try:
        row = conn.execute(
            "SELECT artifact_id, path, sha256 FROM artifacts WHERE artifact_id = ?",
            (str(artifact_id),),
        ).fetchone()
        if row is None:
            return None
        return {
            "artifact_id": str(row["artifact_id"]),
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
        }
    finally:
        conn.close()

