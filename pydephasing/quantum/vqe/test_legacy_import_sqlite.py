from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pydephasing.quantum.vqe.run_store import sha256_file, sha256_text, stable_json_dumps
from pydephasing.quantum.vqe.run_store_legacy import export_legacy_run, import_legacy_runs


def _count(conn: sqlite3.Connection, sql: str, params=()) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def test_import_legacy_run_dir_and_idempotency(tmp_path: Path) -> None:
    runs_dir = tmp_path / "legacy_runs_root"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Use a nonstandard directory name to ensure discovery isn't name-heuristic based.
    run_dir = runs_dir / "some_weird_folder_name"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": "legacy_run_001",
        "sites": 2,
        "n_up": 1,
        "n_down": 1,
        "pool": "cse_density_ops",
        "inner_optimizer": "hybrid",
        "eps_grad": 1e-4,
        "eps_energy": 1e-3,
        "ham_params": {"t": 1.0, "u": 4.0, "dv": 0.5},
        "status": "completed",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hist_rows = [
        {"iter": 0, "energy": -0.5, "max_grad": None, "t_cum_s": 0.0},
        {"iter": 1, "energy": -0.7, "max_grad": 0.2, "t_cum_s": 1.0},
        {"iter": 2, "energy": -0.8, "max_grad": 0.05, "t_cum_s": 2.5},
    ]
    (run_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in hist_rows) + "\n",
        encoding="utf-8",
    )

    result = {"energy": -0.8, "theta": [0.0, 0.1], "operators": ["op1", "op2"], "status": "completed"}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    db_path = tmp_path / "ledger.sqlite3"

    stats1 = import_legacy_runs(runs_dir=runs_dir, db_path=db_path, dry_run=False)
    assert stats1.runs_found == 1
    assert stats1.runs_imported == 1
    assert stats1.runs_conflicts == 0
    assert stats1.steps_written == 3
    assert stats1.artifacts_written == 3  # meta/history/result

    # Re-import should not duplicate rows.
    stats2 = import_legacy_runs(runs_dir=runs_dir, db_path=db_path, dry_run=False)
    assert stats2.runs_found == 1
    assert stats2.runs_imported == 0  # already present
    assert stats2.runs_conflicts == 0

    conn = sqlite3.connect(str(db_path))
    try:
        assert _count(conn, "SELECT COUNT(*) FROM runs") == 1
        assert _count(conn, "SELECT COUNT(*) FROM steps") == 3
        assert _count(conn, "SELECT COUNT(*) FROM artifacts") == 3

        # Check config_hash matches stable dump of meta.json.
        config_hash = conn.execute("SELECT config_hash FROM runs WHERE run_id = ?", ("legacy_run_001",)).fetchone()[0]
        expected = sha256_text(stable_json_dumps(meta))
        assert str(config_hash) == expected

        # Check artifact kinds and sha256 match.
        arts = conn.execute(
            "SELECT kind, path, sha256 FROM artifacts WHERE run_id = ? ORDER BY kind ASC",
            ("legacy_run_001",),
        ).fetchall()
        kinds = [k for (k, _p, _s) in arts]
        assert kinds == ["legacy_history_jsonl", "legacy_meta_json", "legacy_result_json"]

        sha_meta = sha256_file(run_dir / "meta.json")
        sha_hist = sha256_file(run_dir / "history.jsonl")
        sha_res = sha256_file(run_dir / "result.json")
        sha_map = {k: s for (k, _p, s) in arts}
        assert sha_map["legacy_meta_json"] == sha_meta
        assert sha_map["legacy_history_jsonl"] == sha_hist
        assert sha_map["legacy_result_json"] == sha_res

        # Step promoted columns should be present.
        step1 = conn.execute(
            "SELECT energy, grad_norm, wall_time_s FROM steps WHERE run_id = ? AND step_idx = 1",
            ("legacy_run_001",),
        ).fetchone()
        assert step1 is not None
        assert pytest.approx(float(step1[0])) == -0.7
        # grad_norm promoted from max_grad
        assert pytest.approx(float(step1[1])) == 0.2
        # wall_time_s promoted from t_cum_s
        assert pytest.approx(float(step1[2])) == 1.0
    finally:
        conn.close()

    # Export should recreate files.
    export_dir = tmp_path / "exported"
    exported_run_dir = export_legacy_run(run_id="legacy_run_001", out_dir=export_dir, db_path=db_path)
    assert (exported_run_dir / "meta.json").exists()
    assert (exported_run_dir / "history.jsonl").exists()
    assert (exported_run_dir / "result.json").exists()

    exported_meta = json.loads((exported_run_dir / "meta.json").read_text(encoding="utf-8"))
    assert exported_meta["run_id"] == "legacy_run_001"
    assert exported_meta["sites"] == 2

    exported_hist = (exported_run_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(exported_hist) == 3
    # Ensure iter fields are consistent
    assert json.loads(exported_hist[0])["iter"] == 0
    assert json.loads(exported_hist[2])["iter"] == 2

