import json
from pathlib import Path


def test_run_ledger_ro_queries(tmp_path: Path) -> None:
    from pydephasing.quantum.vqe.run_ledger_ro import (
        export_artifact_path,
        get_run,
        get_steps,
        list_artifacts,
        list_runs,
    )
    from pydephasing.quantum.vqe.run_store import SqliteRunStore, sha256_file

    db_path = tmp_path / "runs.db"
    store = SqliteRunStore(db_path)
    run_id = "ro_query_test"
    store.start_run(
        {
            "run_id": run_id,
            "seed": 123,
            "system": {"sites": 2, "n_up": 1, "n_down": 1},
            "ansatz": {"pool": "ham_terms"},
            "optimizer": {"name": "none"},
        }
    )
    store.log_step(run_id, 0, {"energy": -1.0, "max_grad": 0.1, "chosen_op": None})
    store.log_step(run_id, 1, {"energy": -1.1, "max_grad": 0.01, "chosen_op": "X0"})

    artifact_path = tmp_path / "dummy.txt"
    artifact_path.write_text("hello", encoding="utf-8")
    store.add_artifact(
        run_id,
        kind="dummy",
        path=str(artifact_path),
        sha256=sha256_file(artifact_path),
        bytes=int(artifact_path.stat().st_size),
        extra={"tag": "x"},
    )
    store.finish_run(run_id, status="completed", summary_metrics={"energy": -1.1})
    store.close()

    runs = list_runs(db_path=db_path, filters={}, limit=10)
    assert any(r["run_id"] == run_id for r in runs)

    filtered = list_runs(db_path=db_path, filters={"status": "completed", "system.sites": 2}, limit=10)
    assert any(r["run_id"] == run_id for r in filtered)

    run = get_run(db_path=db_path, run_id=run_id)
    assert run is not None
    assert run["run_id"] == run_id
    assert run["counts"]["steps"] == 2
    assert run["counts"]["artifacts"] == 1
    assert run["system"]["sites"] == 2

    steps_full = get_steps(db_path=db_path, run_id=run_id)
    assert steps_full["run_id"] == run_id
    assert steps_full["metric"] is None
    assert [s["step_idx"] for s in steps_full["steps"]] == [0, 1]
    assert steps_full["steps"][0]["energy"] == -1.0

    steps_energy = get_steps(db_path=db_path, run_id=run_id, metric="energy")
    assert [s["value"] for s in steps_energy["steps"]] == [-1.0, -1.1]

    arts = list_artifacts(db_path=db_path, run_id=run_id)
    assert len(arts) == 1
    assert arts[0]["kind"] == "dummy"
    assert arts[0]["extra"]["tag"] == "x"

    art_ref = export_artifact_path(db_path=db_path, artifact_id=str(arts[0]["artifact_id"]))
    assert art_ref is not None
    assert art_ref["sha256"] == arts[0]["sha256"]
    assert str(artifact_path) in art_ref["path"]

