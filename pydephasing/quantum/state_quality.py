"""State-quality metrics for benchmarking (statevector-only for now)."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from .observables_hubbard import (
    jw_site_density_ops,
    jw_total_double_occupancy_op,
    jw_sz_sz_nn_sum_op,
)


def infidelity(*, psi_exact: np.ndarray, psi_approx: np.ndarray) -> float:
    """1 - |<psi_exact|psi_approx>|^2."""
    a = np.asarray(psi_exact, dtype=complex)
    b = np.asarray(psi_approx, dtype=complex)
    if a.shape != b.shape:
        raise ValueError("psi_exact and psi_approx must have the same shape.")
    ov = np.vdot(a, b)
    fid = float(np.abs(ov) ** 2)
    # numerical noise guard
    fid = min(1.0, max(0.0, fid))
    return float(1.0 - fid)


def sector_probability(*, statevector: np.ndarray, indices: list[int]) -> float:
    """Probability mass in the provided basis indices."""
    psi = np.asarray(statevector, dtype=complex)
    if psi.ndim != 1:
        raise ValueError("statevector must be 1D.")
    if not indices:
        return 0.0
    idx = np.asarray(indices, dtype=int)
    probs = np.abs(psi[idx]) ** 2
    return float(np.sum(probs))


def energy_variance(*, qubit_op: SparsePauliOp, state: Statevector) -> float:
    """Var(H) = <H^2> - <H>^2 via ||H|psi>||^2 - <H>^2 (H assumed Hermitian)."""
    psi = np.asarray(state.data, dtype=complex)
    mat = qubit_op.to_matrix(sparse=True).tocsc()
    hpsi = mat.dot(psi)
    e = np.vdot(psi, hpsi)
    e2 = np.vdot(hpsi, hpsi)
    var = float(np.real(e2) - float(np.real(e)) ** 2)
    return float(max(0.0, var))


def expectation(*, state: Statevector, op: SparsePauliOp) -> float:
    return float(np.real(state.expectation_value(op)))


def hubbard_observables(*, state: Statevector, n_sites: int) -> dict:
    """Return a small set of Hubbard-relevant observable expectations."""
    L = int(n_sites)
    dens_ops = jw_site_density_ops(n_sites=L)
    dens = [expectation(state=state, op=op) for op in dens_ops]

    d_tot = expectation(state=state, op=jw_total_double_occupancy_op(n_sites=L))
    szsz_nn = expectation(state=state, op=jw_sz_sz_nn_sum_op(n_sites=L))

    return {
        "double_occ": float(d_tot),
        "densities": [float(x) for x in dens],
        "szsz_nn": float(szsz_nn),
    }


def density_mae(*, dens_a: list[float], dens_b: list[float]) -> float:
    if len(dens_a) != len(dens_b):
        raise ValueError("density vectors must have same length")
    if not dens_a:
        return 0.0
    a = np.asarray(dens_a, dtype=float)
    b = np.asarray(dens_b, dtype=float)
    return float(np.mean(np.abs(a - b)))

