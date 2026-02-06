import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_nature.second_q.mappers import JordanWignerMapper

from pydephasing.quantum.hamiltonian.hubbard import build_fermionic_hubbard, default_1d_chain_edges, build_qubit_hamiltonian_from_fermionic
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_cse_density_pool_from_fermionic,
    build_reference_state,
    _compute_grouped_pool_gradients,
)


def test_cse_density_pool_size_dimer() -> None:
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-0.25, 0.25],
    )
    mapper = JordanWignerMapper()
    pool = build_cse_density_pool_from_fermionic(ferm_op, mapper)
    assert len(pool) >= 2


def test_cse_density_pool_gradient_nonzero() -> None:
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-0.25, 0.25],
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    pool = build_cse_density_pool_from_fermionic(ferm_op, mapper)

    reference_state = build_reference_state(qubit_op.num_qubits, [0, 2])
    estimator = StatevectorEstimator()

    grads = _compute_grouped_pool_gradients(
        estimator,
        reference_state,
        qubit_op,
        np.zeros((0,)),
        pool,
        probe_convention="half_angle",
    )
    assert np.max(np.abs(grads)) > 0.0


def test_cse_density_pool_includes_both_quadratures() -> None:
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-0.25, 0.25],
    )
    mapper = JordanWignerMapper()

    pool_im_only = build_cse_density_pool_from_fermionic(
        ferm_op,
        mapper,
        include_antihermitian_part=True,
        include_hermitian_part=False,
    )
    pool_both = build_cse_density_pool_from_fermionic(
        ferm_op,
        mapper,
        include_antihermitian_part=True,
        include_hermitian_part=True,
    )

    assert len(pool_im_only) > 0
    assert len(pool_both) >= len(pool_im_only)

    names = [spec.get("name", "") for spec in pool_both]
    assert any(name.startswith("gamma_im(") for name in names)
    assert any(name.startswith("gamma_re(") for name in names)
