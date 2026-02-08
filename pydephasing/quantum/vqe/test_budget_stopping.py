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
    return qubit_op, mapper, reference, n_up, n_down


def test_adapt_stops_immediately_when_budget_too_small() -> None:
    qubit_op, mapper, reference, n_up, n_down = _setup_hubbard_l2()

    cost = CostCounters()
    estimator = CountingEstimator(StatevectorEstimator(), cost)

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
        max_depth=10,
        inner_steps=1,
        eps_grad=-1.0,
        eps_energy=-1.0,
        inner_optimizer="lbfgs",
        lbfgs_maxiter=1,
        lbfgs_restarts=1,
        max_circuits_executed=1,
        cost_counters=cost,
        verbose=False,
    )

    assert len(result.operators) == 0
    assert str(result.diagnostics.get("stop_reason", "")).startswith("budget:")
    assert int(cost.n_circuits_executed) == 1


def test_adapt_stops_in_outer_loop_under_tight_budget() -> None:
    qubit_op, mapper, reference, n_up, n_down = _setup_hubbard_l2()

    cost = CostCounters()
    estimator = CountingEstimator(StatevectorEstimator(), cost)

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
        max_depth=10,
        inner_steps=1,
        eps_grad=-1.0,
        eps_energy=-1.0,
        inner_optimizer="lbfgs",
        lbfgs_maxiter=1,
        lbfgs_restarts=1,
        max_circuits_executed=2,
        cost_counters=cost,
        verbose=False,
    )

    assert len(result.operators) <= 1
    assert str(result.diagnostics.get("stop_reason", "")).startswith("budget:")
    assert int(cost.n_circuits_executed) == 2
    assert np.isfinite(float(result.energy))

