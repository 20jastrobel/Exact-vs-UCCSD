from __future__ import annotations

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from pydephasing.quantum.ansatz import build_ansatz
from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)


def _smoke_test_uccsd_for_size(n_sites: int, num_particles: tuple[int, int]) -> None:
    ferm_op = build_fermionic_hubbard(
        n_sites=n_sites,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(n_sites, periodic=False),
        v=[-0.25, 0.25] if n_sites == 2 else None,
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)

    ansatz = build_ansatz(
        "uccsd",
        qubit_op.num_qubits,
        reps=1,
        mapper=mapper,
        n_sites=n_sites,
        num_particles=num_particles,
    )
    assert ansatz.num_qubits == qubit_op.num_qubits

    estimator = StatevectorEstimator()
    optimizer = COBYLA(maxiter=25)
    initial_point = np.zeros(ansatz.num_parameters, dtype=float)
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
    )
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    energy = float(np.real(result.eigenvalue))
    assert np.isfinite(energy)


def test_uccsd_smoke_small_sizes() -> None:
    # Keep the sector fixed and fast for a smoke test.
    for n_sites in (2, 3, 4):
        _smoke_test_uccsd_for_size(n_sites, (1, 1))

