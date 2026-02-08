from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.state_quality import (
    energy_variance,
    hubbard_observables,
    infidelity,
    sector_probability,
)
from pydephasing.quantum.symmetry import (
    exact_ground_energy_sector,
    exact_ground_state_sector,
    sector_basis_indices,
)


def test_exact_sector_state_has_zero_variance_and_unit_sector_mass() -> None:
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=1.0,
        u=4.0,
        edges=default_1d_chain_edges(2, periodic=False),
        v=[-0.25, 0.25],
    )
    qubit_op, _mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)

    e0_energy_only = exact_ground_energy_sector(qubit_op, 2, 2, 0.0)
    e0, psi = exact_ground_state_sector(qubit_op, 2, 2, 0.0)
    assert np.isclose(e0, e0_energy_only, atol=1e-10)

    st = Statevector(psi)
    var_h = energy_variance(qubit_op=qubit_op, state=st)
    assert var_h < 1e-10

    idx = sector_basis_indices(2, 2, 0.0)
    p_sector = sector_probability(statevector=np.asarray(st.data), indices=idx)
    assert abs(p_sector - 1.0) < 1e-12

    # Fidelity to itself should be 1.
    infid = infidelity(psi_exact=np.asarray(st.data), psi_approx=np.asarray(st.data))
    assert infid < 1e-12

    obs = hubbard_observables(state=st, n_sites=2)
    assert isinstance(obs["double_occ"], float)
    assert isinstance(obs["szsz_nn"], float)
    dens = obs["densities"]
    assert isinstance(dens, list) and len(dens) == 2
    # n_i is between 0 and 2.
    assert all(0.0 <= float(x) <= 2.0 + 1e-9 for x in dens)

