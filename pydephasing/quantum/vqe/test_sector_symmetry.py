import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper

from pydephasing.quantum.hamiltonian.hubbard import build_fermionic_hubbard, default_1d_chain_edges, build_qubit_hamiltonian_from_fermionic
from pydephasing.quantum.symmetry import (
    commutes,
    exact_ground_energy_sector,
    map_symmetry_ops_to_qubits,
    computational_basis_eigs,
)
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_cse_density_pool_from_fermionic,
    build_reference_state,
    build_adapt_circuit_grouped,
    estimate_energy,
)


def test_sector_ops_commutation() -> None:
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-0.25, 0.25],
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    n_q, sz_q = map_symmetry_ops_to_qubits(mapper, 2)
    pool = build_cse_density_pool_from_fermionic(ferm_op, mapper, enforce_symmetry=True, n_sites=2)
    assert pool
    for spec in pool:
        op = SparsePauliOp.from_list(spec["paulis"])
        assert commutes(op, n_q)
        assert commutes(op, sz_q)


def test_sector_exact_energy_matches_manual() -> None:
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-0.25, 0.25],
    )
    qubit_op, _mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    exact_sector = exact_ground_energy_sector(qubit_op, 2, 2, 0.0)

    mat = qubit_op.to_matrix()
    indices = []
    for idx in range(2 ** 4):
        n_val, sz_val = computational_basis_eigs(2, idx)
        if n_val == 2 and abs(sz_val - 0.0) < 1e-12:
            indices.append(idx)
    sub = mat[np.ix_(indices, indices)]
    evals = np.linalg.eigvalsh(sub)
    manual = float(np.min(np.real(evals)))
    assert np.isclose(exact_sector, manual)


def test_variational_bound_in_sector() -> None:
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-0.25, 0.25],
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    exact_sector = exact_ground_energy_sector(qubit_op, 2, 2, 0.0)
    pool = build_cse_density_pool_from_fermionic(ferm_op, mapper, enforce_symmetry=True, n_sites=2)

    reference = build_reference_state(qubit_op.num_qubits, [0, 2])
    circuit, params = build_adapt_circuit_grouped(reference, pool[:1])
    theta = np.array([0.25], dtype=float)
    estimator = StatevectorEstimator()
    energy = estimate_energy(estimator, circuit, qubit_op, theta)

    assert energy >= exact_sector - 1e-9
