from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from qiskit.primitives import StatevectorEstimator

from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.utils_particles import jw_reference_occupations_from_particles
from pydephasing.quantum.vqe.adapt_vqe_meta import build_reference_state, run_meta_adapt_vqe
from pydephasing.quantum.vqe.cost_model import CostCounters, CountingEstimator
from pydephasing.quantum.vqe.run_store import JsonRunStore, RunStoreLogger, SqliteRunStore


def _setup_hubbard_l2():
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-0.25, 0.25],
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    n_up, n_down = 1, 1
    reference = build_reference_state(
        qubit_op.num_qubits,
        jw_reference_occupations_from_particles(2, n_up, n_down),
    )
    return ferm_op, qubit_op, mapper, reference, n_up, n_down


def _load_json_history(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("iter", 0)))
    return rows


def _load_sqlite_steps(db_path: Path, run_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT step_idx, metrics_json FROM steps WHERE run_id = ? ORDER BY step_idx ASC",
            (str(run_id),),
        ).fetchall()
    finally:
        conn.close()
    out: list[dict] = []
    for step_idx, metrics_json in rows:
        m = json.loads(metrics_json)
        if not isinstance(m, dict):
            m = {}
        m["iter"] = int(step_idx)
        out.append(m)
    out.sort(key=lambda r: int(r.get("iter", 0)))
    return out


def test_json_vs_sqlite_store_parity_for_adapt() -> None:
    """Logging backend must not perturb numerics (energies / selected ops / counters)."""
    ferm_op, qubit_op, mapper, reference, n_up, n_down = _setup_hubbard_l2()

    # Run once with JsonRunStore.
    tmp = Path(__file__).resolve().parent
    # Use a temp-ish per-test directory under pytest's tmp_path isn't available here; avoid collisions by
    # scoping into a unique dir per process.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        json_dir = root / "json_run"
        json_store = JsonRunStore(base_dir=root)
        json_run_id = "parity_json"
        json_store.start_run(
            {
                "run_id": json_run_id,
                "run_dir": str(json_dir),
                "sites": 2,
                "n_up": n_up,
                "n_down": n_down,
                "system": {"sites": 2, "n_up": n_up, "n_down": n_down},
                "ansatz": {"pool": "ham_terms_plus_imag_partners"},
                "optimizer": {"name": "lbfgs"},
            }
        )
        cost_json = CostCounters()
        est_json = CountingEstimator(StatevectorEstimator(), cost_json)
        res_json = run_meta_adapt_vqe(
            qubit_op,
            reference,
            est_json,
            pool_mode="ham_terms_plus_imag_partners",
            ferm_op=ferm_op,
            mapper=mapper,
            n_sites=2,
            n_up=n_up,
            n_down=n_down,
            enforce_sector=True,
            max_depth=2,
            inner_steps=2,
            eps_grad=-1.0,
            eps_energy=-1.0,
            inner_optimizer="lbfgs",
            lbfgs_restarts=1,
            seed=7,
            cost_counters=cost_json,
            logger=RunStoreLogger(json_store, json_run_id),
            log_every=1,
            verbose=False,
        )
        json_store.finish_run(json_run_id, status="completed", summary_metrics={"energy": float(res_json.energy)})
        hist_json = _load_json_history(json_dir / "history.jsonl")

        # Run once with SqliteRunStore.
        db_path = root / "runs.db"
        sql_store = SqliteRunStore(db_path)
        sql_run_id = "parity_sql"
        sql_store.start_run(
            {
                "run_id": sql_run_id,
                "sites": 2,
                "n_up": n_up,
                "n_down": n_down,
                "system": {"sites": 2, "n_up": n_up, "n_down": n_down},
                "ansatz": {"pool": "ham_terms_plus_imag_partners"},
                "optimizer": {"name": "lbfgs"},
            }
        )
        cost_sql = CostCounters()
        est_sql = CountingEstimator(StatevectorEstimator(), cost_sql)
        res_sql = run_meta_adapt_vqe(
            qubit_op,
            reference,
            est_sql,
            pool_mode="ham_terms_plus_imag_partners",
            ferm_op=ferm_op,
            mapper=mapper,
            n_sites=2,
            n_up=n_up,
            n_down=n_down,
            enforce_sector=True,
            max_depth=2,
            inner_steps=2,
            eps_grad=-1.0,
            eps_energy=-1.0,
            inner_optimizer="lbfgs",
            lbfgs_restarts=1,
            seed=7,
            cost_counters=cost_sql,
            logger=RunStoreLogger(sql_store, sql_run_id),
            log_every=1,
            verbose=False,
        )
        sql_store.finish_run(sql_run_id, status="completed", summary_metrics={"energy": float(res_sql.energy)})
        sql_store.close()
        hist_sql = _load_sqlite_steps(db_path, sql_run_id)

    # Energies should match (logging backend is side-effect only).
    assert np.isfinite(float(res_json.energy))
    assert np.isfinite(float(res_sql.energy))
    assert abs(float(res_json.energy) - float(res_sql.energy)) < 1e-10

    assert [int(r["iter"]) for r in hist_json] == [int(r["iter"]) for r in hist_sql]

    # Compare the non-timing fields we expect to be deterministic.
    ignore = {"t_iter_s", "t_cum_s"}
    keys = {
        "energy",
        "max_grad",
        "chosen_op",
        "ansatz_len",
        "n_params",
        "pool_size",
        "stop_reason",
        # cost counters
        "n_energy_evals",
        "n_grad_evals",
        "n_estimator_calls",
        "n_circuits_executed",
        "n_pauli_terms_measured",
        "total_shots",
        # sector diagnostics
        "N_mean",
        "Sz_mean",
        "VarN",
        "VarSz",
        "abs_N_err",
        "abs_Sz_err",
    }

    for a, b in zip(hist_json, hist_sql):
        for k in keys:
            if k in ignore:
                continue
            va = a.get(k)
            vb = b.get(k)
            if va is None or vb is None:
                assert va == vb
                continue
            if isinstance(va, (int, float)) or isinstance(vb, (int, float)):
                assert abs(float(va) - float(vb)) < 1e-10
            else:
                assert va == vb

