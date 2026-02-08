"""Symmetry operators and sector utilities."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from scipy.sparse.linalg import eigsh


def fermionic_number_op(n_sites: int) -> FermionicOp:
    data = {}
    n_orb = 2 * n_sites
    for p in range(n_orb):
        data[f"+_{p} -_{p}"] = 1.0
    return FermionicOp(data, num_spin_orbitals=n_orb)


def fermionic_sz_op(n_sites: int) -> FermionicOp:
    data = {}
    n_orb = 2 * n_sites
    for i in range(n_sites):
        p_up = i
        p_dn = i + n_sites
        data[f"+_{p_up} -_{p_up}"] = 0.5
        data[f"+_{p_dn} -_{p_dn}"] = -0.5
    return FermionicOp(data, num_spin_orbitals=n_orb)


def map_symmetry_ops_to_qubits(
    mapper: JordanWignerMapper,
    n_sites: int,
) -> tuple[SparsePauliOp, SparsePauliOp]:
    n_op = fermionic_number_op(n_sites)
    sz_op = fermionic_sz_op(n_sites)
    n_q = mapper.map(n_op).simplify(atol=1e-12)
    sz_q = mapper.map(sz_op).simplify(atol=1e-12)
    return n_q, sz_q


def computational_basis_eigs(n_sites: int, bitstring_int: int) -> tuple[int, float]:
    n_orb = 2 * n_sites
    bits = [(bitstring_int >> idx) & 1 for idx in range(n_orb)]
    n_total = int(sum(bits))
    n_up = int(sum(bits[:n_sites]))
    n_dn = int(sum(bits[n_sites:]))
    sz = 0.5 * (n_up - n_dn)
    return n_total, sz


def sector_basis_indices(
    n_sites: int,
    n_target: int,
    sz_target: float,
) -> list[int]:
    """Return computational-basis indices that lie in the requested (N, Sz) sector.

    Indexing convention matches Qiskit's little-endian basis ordering, i.e. qubit
    index 0 is the least-significant bit.
    """
    dim = 2 ** (2 * int(n_sites))
    indices: list[int] = []
    for idx in range(dim):
        n_val, sz_val = computational_basis_eigs(int(n_sites), idx)
        if n_val == int(n_target) and abs(float(sz_val) - float(sz_target)) < 1e-12:
            indices.append(int(idx))
    return indices


def exact_ground_state_sector(
    qubit_op: SparsePauliOp,
    n_sites: int,
    n_target: int,
    sz_target: float,
) -> tuple[float, np.ndarray]:
    """Return (E0, |psi0>) for the exact ground state within a symmetry sector.

    The returned |psi0> is embedded in the full 2^(2*n_sites) Hilbert space and is
    normalized. It has support only on basis indices in the requested sector.
    """
    indices = sector_basis_indices(int(n_sites), int(n_target), float(sz_target))
    if not indices:
        raise ValueError("No basis states in target sector.")

    # Prefer sparse matrices to avoid repeated format conversions and warnings.
    mat_sparse = qubit_op.to_matrix(sparse=True).tocsc()
    sub_sparse = mat_sparse[indices, :][:, indices].tocsc()
    sub_dim = int(sub_sparse.shape[0])

    if sub_dim <= 256:
        sub = sub_sparse.toarray()
        evals, evecs = np.linalg.eigh(sub)
        idx0 = int(np.argmin(np.real(evals)))
        e0 = float(np.real(evals[idx0]))
        vec_sub = np.asarray(evecs[:, idx0], dtype=complex)
    else:
        evals, evecs = eigsh(sub_sparse, k=1, which="SA", return_eigenvectors=True)
        e0 = float(np.real(evals[0]))
        vec_sub = np.asarray(evecs[:, 0], dtype=complex)

    norm = float(np.linalg.norm(vec_sub))
    if norm <= 0.0:
        raise RuntimeError("Exact eigenvector has zero norm (unexpected).")
    vec_sub = vec_sub / norm

    full_dim = 2 ** (2 * int(n_sites))
    psi = np.zeros((full_dim,), dtype=complex)
    psi[np.asarray(indices, dtype=int)] = vec_sub
    psi_norm = float(np.linalg.norm(psi))
    if psi_norm <= 0.0:
        raise RuntimeError("Embedded exact eigenvector has zero norm (unexpected).")
    psi = psi / psi_norm
    return e0, psi


def exact_ground_energy_sector(
    qubit_op: SparsePauliOp,
    n_sites: int,
    n_target: int,
    sz_target: float,
) -> float:
    indices = sector_basis_indices(int(n_sites), int(n_target), float(sz_target))
    if not indices:
        raise ValueError("No basis states in target sector.")

    # Prefer sparse matrices to avoid repeated format conversions and warnings.
    mat_sparse = qubit_op.to_matrix(sparse=True).tocsc()
    sub_sparse = mat_sparse[indices, :][:, indices].tocsc()
    dim = int(sub_sparse.shape[0])

    if dim <= 256:
        sub = sub_sparse.toarray()
        evals = np.linalg.eigvalsh(sub)
        return float(np.min(np.real(evals)))

    val = eigsh(sub_sparse, k=1, which="SA", return_eigenvectors=False)[0]
    return float(np.real(val))


def commutes(op: SparsePauliOp, sym: SparsePauliOp, tol: float = 1e-12) -> bool:
    comm = (op @ sym - sym @ op).simplify(atol=tol)
    if len(comm.coeffs) == 0:
        return True
    return float(np.max(np.abs(comm.coeffs))) < tol
