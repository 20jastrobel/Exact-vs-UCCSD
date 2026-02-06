import numpy as np
from qiskit.primitives import StatevectorEstimator

from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.utils_particles import jw_reference_occupations_from_particles
from pydephasing.quantum.vqe.adapt_vqe_meta import build_reference_state, run_meta_adapt_vqe


def test_sector_diagnostics_present_and_finite() -> None:
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
    estimator = StatevectorEstimator()

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
        verbose=False,
    )

    assert result.diagnostics.get("outer")
    sector = result.diagnostics["outer"][0].get("sector")
    assert isinstance(sector, dict)
    for key in ("N_mean", "Sz_mean", "VarN", "VarSz", "abs_N_err", "abs_Sz_err"):
        assert key in sector
        assert np.isfinite(float(sector[key]))

