#!/usr/bin/env python3
"""MCP server exposing *read-only* access to the SQLite run ledger.

This server intentionally does NOT provide any tool for executing arbitrary SQL.
It only exposes a small, parameterized query surface over the run-ledger schema.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mcp.server import FastMCP

# Allow running from any cwd without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydephasing.quantum.vqe.run_ledger_ro import (
    export_artifact_path as _export_artifact_path,
    get_run as _get_run,
    get_steps as _get_steps,
    list_artifacts as _list_artifacts,
    list_runs as _list_runs,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        type=str,
        default=os.environ.get("RUNS_DB_PATH", "data/runs.db"),
        help="SQLite DB path (opened read-only). Env: RUNS_DB_PATH.",
    )
    ap.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport.",
    )
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    mcp = FastMCP(name="run-ledger-ro", host=str(args.host), port=int(args.port))

    @mcp.tool()
    def list_runs(filters: dict | None = None, limit: int = 50) -> list[dict]:
        """List runs from the ledger with simple filters (no arbitrary SQL)."""
        return _list_runs(db_path=db_path, filters=filters or {}, limit=int(limit))

    @mcp.tool()
    def get_run(run_id: str) -> dict | None:
        """Get a single run by run_id."""
        return _get_run(db_path=db_path, run_id=str(run_id))

    @mcp.tool()
    def get_steps(
        run_id: str,
        metric: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> dict:
        """Get per-step metrics for a run (optionally a single metric, optionally sliced)."""
        return _get_steps(
            db_path=db_path,
            run_id=str(run_id),
            metric=None if metric is None else str(metric),
            start=None if start is None else int(start),
            end=None if end is None else int(end),
        )

    @mcp.tool()
    def list_artifacts(run_id: str, kind: str | None = None) -> list[dict]:
        """List artifacts attached to a run."""
        return _list_artifacts(
            db_path=db_path, run_id=str(run_id), kind=None if kind is None else str(kind)
        )

    @mcp.tool()
    def export_artifact_path(artifact_id: str) -> dict | None:
        """Return artifact path + sha256 only (no file contents)."""
        return _export_artifact_path(db_path=db_path, artifact_id=str(artifact_id))

    mcp.run(transport=str(args.transport))


if __name__ == "__main__":
    main()
