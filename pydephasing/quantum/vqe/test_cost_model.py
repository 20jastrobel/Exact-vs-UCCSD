from __future__ import annotations

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


def test_counting_estimator_counts_pubs_and_terms() -> None:
    counters = CostCounters()
    est = CountingEstimator(StatevectorEstimator(), counters)

    # Simple 2-qubit pub with a 2-term observable (no identity).
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(2)
    op = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])
    job = est.run([(qc, op, [])])
    _ = job.result()

    assert counters.n_estimator_calls == 1
    assert counters.n_circuits_executed == 1
    assert counters.n_pauli_terms_measured == 2


def test_adapt_logs_cost_counters_and_increments_categories() -> None:
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

    counters = CostCounters()
    estimator = CountingEstimator(StatevectorEstimator(), counters)

    class _MemLogger:
        def __init__(self):
            self.rows: list[dict] = []

        def start_iter(self):
            return None

        def end_iter(self):
            return 0.0, 0.0

        def log_point(self, **kwargs):
            self.rows.append(kwargs)

    logger = _MemLogger()

    result = run_meta_adapt_vqe(
        qubit_op,
        reference,
        estimator,
        pool_mode="uccsd_excitations",
        mapper=mapper,
        n_sites=2,
        n_up=n_up,
        n_down=n_down,
        enforce_sector=True,
        max_depth=1,
        inner_steps=1,
        eps_grad=-1.0,
        eps_energy=-1.0,
        inner_optimizer="lbfgs",
        lbfgs_maxiter=1,
        lbfgs_restarts=1,
        cost_counters=counters,
        logger=logger,
        verbose=False,
    )

    assert np.isfinite(float(result.energy))
    # Categories should have nonzero work for a single ADAPT outer step.
    assert counters.n_energy_evals > 0
    assert counters.n_grad_evals > 0
    assert counters.n_circuits_executed > 0

    # Logger rows should carry the flattened cost keys.
    assert logger.rows
    extras = logger.rows[-1].get("extra") or {}
    for key in (
        "n_energy_evals",
        "n_grad_evals",
        "n_estimator_calls",
        "n_circuits_executed",
        "n_pauli_terms_measured",
    ):
        assert key in extras
