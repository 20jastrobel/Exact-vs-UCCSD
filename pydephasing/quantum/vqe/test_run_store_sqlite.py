from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydephasing.quantum.vqe.run_store import SqliteRunStore, sha256_file


def _tables(conn) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(r[0]) for r in rows}


def test_sqlite_run_store_migrates_and_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite3"
    store = SqliteRunStore(db_path, batch_size=1)

    try:
        tables = _tables(store.conn)
        assert "schema_migrations" in tables
        assert "runs" in tables
        assert "steps" in tables
        assert "artifacts" in tables

        run_config = {
            "seed": 7,
            "git_commit": "deadbeef",
            "system": {"sites": 2, "n_up": 1, "n_down": 1},
            "ansatz": {"kind": "uccsd", "reps": 2},
            "optimizer": {"name": "COBYLA", "maxiter": 50},
            "sites": 2,
            "n_up": 1,
            "n_down": 1,
        }
        run_id = store.start_run(run_config)

        row = store.conn.execute(
            "SELECT run_id, status, git_commit, seed, config_hash FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == run_id
        assert row[1] == "running"
        assert row[2] == "deadbeef"
        assert row[3] == 7
        assert isinstance(row[4], str) and len(row[4]) == 64

        store.log_step(
            run_id,
            0,
            {"energy": -1.23, "max_grad": 0.5, "t_cum_s": 1.5, "note": "hello"},
        )
        step = store.conn.execute(
            "SELECT step_idx, metrics_json, energy, grad_norm, wall_time_s FROM steps WHERE run_id = ? AND step_idx = ?",
            (run_id, 0),
        ).fetchone()
        assert step is not None
        assert int(step[0]) == 0
        metrics = json.loads(step[1])
        assert metrics["note"] == "hello"
        assert pytest.approx(float(step[2])) == -1.23
        # grad_norm is promoted from max_grad if no explicit grad_norm is present.
        assert pytest.approx(float(step[3])) == 0.5
        # wall_time_s is promoted from t_cum_s if present.
        assert pytest.approx(float(step[4])) == 1.5

        art_path = tmp_path / "artifact.txt"
        art_path.write_text("abc\n", encoding="utf-8")
        art_sha = sha256_file(art_path)
        store.add_artifact(
            run_id,
            kind="test_artifact",
            path=str(art_path),
            sha256=art_sha,
            bytes=None,
            extra={"k": 1},
        )
        store.flush()
        art = store.conn.execute(
            "SELECT kind, path, sha256, bytes, extra_json FROM artifacts WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert art is not None
        assert art[0] == "test_artifact"
        assert art[1] == str(art_path)
        assert art[2] == art_sha
        assert int(art[3]) == art_path.stat().st_size
        assert json.loads(art[4])["k"] == 1

        store.finish_run(run_id, status="completed", summary_metrics={"energy_final": -1.23})
        row2 = store.conn.execute(
            "SELECT status, finished_at, summary_json FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row2 is not None
        assert row2[0] == "completed"
        assert row2[1] is not None
        assert json.loads(row2[2])["energy_final"] == -1.23

        mig_count = store.conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        assert int(mig_count) >= 1
    finally:
        store.close()


def test_sqlite_run_store_idempotent_migrations(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite3"
    store1 = SqliteRunStore(db_path, batch_size=10)
    try:
        c1 = int(store1.conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0])
        assert c1 >= 1
    finally:
        store1.close()

    store2 = SqliteRunStore(db_path, batch_size=10)
    try:
        c2 = int(store2.conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0])
        assert c2 == c1
    finally:
        store2.close()

