#!/usr/bin/env python3
"""Run-ledger utilities (SQLite) for importing/exporting legacy JSON runs.

Examples:
  python scripts/run_store_db.py db init --db runs/run_ledger.sqlite3
  python scripts/run_store_db.py db import-legacy --runs-dir runs --db runs/run_ledger.sqlite3
  python scripts/run_store_db.py db export-legacy --run-id <ID> --out-dir /tmp/export --db runs/run_ledger.sqlite3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Allow running as `python scripts/...py` without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydephasing.quantum.vqe.run_store_legacy import (
    export_legacy_run,
    import_legacy_runs,
    init_db,
)

def _load_json(text: str | None):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _print_run_row(*, row) -> None:
    run_id, created_at, status, seed, git_commit, system_json, ansatz_json, optimizer_json = row
    sysj = _load_json(system_json)
    ansj = _load_json(ansatz_json)
    optj = _load_json(optimizer_json)
    if not isinstance(sysj, dict):
        sysj = {}
    if not isinstance(ansj, dict):
        # Older/experimental rows may store a scalar; keep a useful label.
        if isinstance(ansj, str):
            ansj = {"kind": ansj}
        else:
            ansj = {}
    if not isinstance(optj, dict):
        if isinstance(optj, str):
            optj = {"name": optj}
        else:
            optj = {}

    sites = sysj.get("sites")
    n_up = sysj.get("n_up")
    n_down = sysj.get("n_down")
    pool = ansj.get("pool") or ansj.get("kind") or ansj.get("ansatz") or ansj.get("name")
    opt_name = optj.get("name") or optj.get("inner_optimizer") or optj.get("optimizer")

    parts = [
        str(run_id),
        str(status),
        str(created_at),
        f"seed={seed}" if seed is not None else "seed=?",
    ]
    if sites is not None and n_up is not None and n_down is not None:
        parts.append(f"L={sites} (n_up={n_up}, n_down={n_down})")
    if pool is not None:
        parts.append(f"ansatz/pool={pool}")
    if opt_name is not None:
        parts.append(f"opt={opt_name}")
    if git_commit is not None:
        parts.append(f"git={git_commit}")
    print(" | ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    db = sub.add_parser("db", help="SQLite run ledger utilities")
    db_sub = db.add_subparsers(dest="db_cmd", required=True)

    p_init = db_sub.add_parser("init", help="Initialize the DB (apply migrations)")
    p_init.add_argument("--db", type=str, required=True)

    p_imp = db_sub.add_parser("import-legacy", help="Import legacy JSON run folders into SQLite")
    p_imp.add_argument("--runs-dir", type=str, required=True)
    p_imp.add_argument("--db", type=str, required=True)
    p_imp.add_argument("--dry-run", action="store_true", default=False)

    p_exp = db_sub.add_parser("export-legacy", help="Export a run back to legacy JSON files")
    p_exp.add_argument("--run-id", type=str, required=True)
    p_exp.add_argument("--out-dir", type=str, required=True)
    p_exp.add_argument("--db", type=str, required=True)

    p_list = db_sub.add_parser("list", help="List runs in the DB")
    p_list.add_argument("--db", type=str, required=True)
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--status", type=str, default=None)
    p_list.add_argument("--where", type=str, default=None, help="Optional raw SQL WHERE clause fragment.")

    p_show = db_sub.add_parser("show", help="Show a single run (config + summary + counts)")
    p_show.add_argument("--db", type=str, required=True)
    p_show.add_argument("--run-id", type=str, required=True)

    p_steps = db_sub.add_parser("steps", help="Dump steps for a run")
    p_steps.add_argument("--db", type=str, required=True)
    p_steps.add_argument("--run-id", type=str, required=True)
    p_steps.add_argument("--metric", type=str, default="energy")
    p_steps.add_argument("--csv", action="store_true", default=False)

    args = ap.parse_args()

    if args.cmd != "db":
        raise SystemExit("Only 'db' subcommands are supported.")

    if args.db_cmd == "init":
        init_db(db_path=args.db)
        print(f"Initialized DB: {args.db}")
        return

    if args.db_cmd == "import-legacy":
        stats = import_legacy_runs(runs_dir=args.runs_dir, db_path=args.db, dry_run=bool(args.dry_run))
        if args.dry_run:
            print(
                f"[dry-run] runs_found={stats.runs_found}, steps={stats.steps_written}, artifacts={stats.artifacts_written}"
            )
        else:
            print(
                "Imported legacy runs: "
                f"runs_found={stats.runs_found}, runs_imported={stats.runs_imported}, "
                f"runs_skipped={stats.runs_skipped}, runs_conflicts={stats.runs_conflicts}, "
                f"steps_written={stats.steps_written}, artifacts_written={stats.artifacts_written}"
            )
        return

    if args.db_cmd == "export-legacy":
        dest = export_legacy_run(run_id=args.run_id, out_dir=args.out_dir, db_path=args.db)
        print(f"Exported {args.run_id} to {dest}")
        return

    if args.db_cmd == "list":
        conn = sqlite3.connect(str(args.db))
        try:
            where = []
            params = []
            if args.status:
                where.append("status = ?")
                params.append(str(args.status))
            if args.where:
                where.append(f"({args.where})")
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""
            sql = (
                "SELECT run_id, created_at, status, seed, git_commit, system_json, ansatz_json, optimizer_json "
                "FROM runs"
                f"{where_sql} "
                "ORDER BY created_at DESC "
                "LIMIT ?"
            )
            params.append(int(args.limit))
            rows = conn.execute(sql, tuple(params)).fetchall()
            for row in rows:
                _print_run_row(row=row)
        finally:
            conn.close()
        return

    if args.db_cmd == "show":
        conn = sqlite3.connect(str(args.db))
        try:
            row = conn.execute(
                "SELECT run_id, created_at, started_at, finished_at, status, git_commit, seed, config_hash, config_json, system_json, ansatz_json, optimizer_json, summary_json "
                "FROM runs WHERE run_id = ?",
                (str(args.run_id),),
            ).fetchone()
            if row is None:
                raise SystemExit(f"Unknown run_id: {args.run_id}")
            (
                run_id,
                created_at,
                started_at,
                finished_at,
                status,
                git_commit,
                seed,
                config_hash,
                config_json,
                system_json,
                ansatz_json,
                optimizer_json,
                summary_json,
            ) = row
            steps_n = conn.execute("SELECT COUNT(*) FROM steps WHERE run_id = ?", (str(args.run_id),)).fetchone()[0]
            arts_n = conn.execute("SELECT COUNT(*) FROM artifacts WHERE run_id = ?", (str(args.run_id),)).fetchone()[0]
            print(f"run_id:      {run_id}")
            print(f"status:      {status}")
            print(f"created_at:  {created_at}")
            print(f"started_at:  {started_at}")
            print(f"finished_at: {finished_at}")
            print(f"seed:        {seed}")
            print(f"git_commit:  {git_commit}")
            print(f"config_hash: {config_hash}")
            print(f"steps:       {steps_n}")
            print(f"artifacts:   {arts_n}")
            if system_json:
                print("\nsystem_json:")
                print(json.dumps(_load_json(system_json), indent=2, sort_keys=True))
            if ansatz_json:
                print("\nansatz_json:")
                print(json.dumps(_load_json(ansatz_json), indent=2, sort_keys=True))
            if optimizer_json:
                print("\noptimizer_json:")
                print(json.dumps(_load_json(optimizer_json), indent=2, sort_keys=True))
            if summary_json:
                print("\nsummary_json:")
                print(json.dumps(_load_json(summary_json), indent=2, sort_keys=True))
            if config_json:
                print("\nconfig_json:")
                cfg = _load_json(config_json)
                print(json.dumps(cfg, indent=2, sort_keys=True))
        finally:
            conn.close()
        return

    if args.db_cmd == "steps":
        conn = sqlite3.connect(str(args.db))
        try:
            rows = conn.execute(
                "SELECT step_idx, metrics_json, energy, grad_norm, wall_time_s FROM steps WHERE run_id = ? ORDER BY step_idx ASC",
                (str(args.run_id),),
            ).fetchall()
            metric = str(args.metric)
            if args.csv:
                print(f"step_idx,{metric}")
            for step_idx, metrics_json, energy, grad_norm, wall_time_s in rows:
                step = int(step_idx)
                if metric == "energy":
                    val = energy
                elif metric == "grad_norm":
                    val = grad_norm
                elif metric == "wall_time_s":
                    val = wall_time_s
                else:
                    mj = _load_json(metrics_json) or {}
                    val = mj.get(metric)
                if args.csv:
                    print(f"{step},{'' if val is None else val}")
                else:
                    print(f"{step}: {val}")
        finally:
            conn.close()
        return

    raise SystemExit(f"Unknown db subcommand: {args.db_cmd}")


if __name__ == "__main__":
    main()
