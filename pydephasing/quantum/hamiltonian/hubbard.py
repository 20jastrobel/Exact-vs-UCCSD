"""Hubbard Hamiltonian builders."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

Edge = Union[Tuple[int, int], Tuple[int, int, complex]]


def _spin_orbital_index(site: int, spin: int, n_sites: int) -> int:
    if site < 0 or site >= n_sites:
        raise ValueError(f"site out of range: {site} (n_sites={n_sites})")
    if spin not in (0, 1):
        raise ValueError(f"spin must be 0(up) or 1(down), got {spin}")
    return site + spin * n_sites


def _add_term(data: Dict[str, complex], label: str, coeff: complex, *, atol: float = 0.0) -> None:
    if atol and abs(coeff) <= atol:
        return
    data[label] = data.get(label, 0.0) + coeff
    if atol and abs(data[label]) <= atol:
        data.pop(label, None)


def _normalize_onsite_potential(
    v: Optional[Union[Sequence[float], Sequence[Sequence[float]]]],
    n_sites: int,
) -> np.ndarray:
    if v is None:
        return np.zeros((n_sites, 2), dtype=float)

    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != n_sites:
            raise ValueError(f"v must have length n_sites={n_sites}, got {arr.shape[0]}")
        return np.column_stack([arr, arr])
    if arr.ndim == 2:
        if arr.shape != (n_sites, 2):
            raise ValueError(f"v must have shape (n_sites,2)={(n_sites,2)}, got {arr.shape}")
        return arr
    raise ValueError("v must be None, length-n_sites, or shape (n_sites,2)")


def _normalize_u(u: Union[float, Sequence[float]], n_sites: int) -> np.ndarray:
    if isinstance(u, (int, float, np.floating)):
        return np.full(n_sites, float(u), dtype=float)
    arr = np.asarray(u, dtype=float)
    if arr.shape != (n_sites,):
        raise ValueError(f"u must be scalar or length-n_sites={n_sites}, got shape {arr.shape}")
    return arr


def _default_1d_chain_edges(n_sites: int, *, periodic: bool = False) -> list[tuple[int, int]]:
    edges = [(i, i + 1) for i in range(n_sites - 1)]
    if periodic and n_sites > 2:
        edges.append((n_sites - 1, 0))
    return edges


def default_1d_chain_edges(n_sites: int, *, periodic: bool = False) -> list[tuple[int, int]]:
    return _default_1d_chain_edges(n_sites, periodic=periodic)


def build_fermionic_hubbard(
    n_sites: int,
    t: Union[float, complex] = 1.0,
    u: Union[float, Sequence[float]] = 4.0,
    *,
    edges: Optional[Iterable[Edge]] = None,
    v: Optional[Union[Sequence[float], Sequence[Sequence[float]]]] = None,
    atol: float = 0.0,
) -> FermionicOp:
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")

    v_mat = _normalize_onsite_potential(v, n_sites)
    u_vec = _normalize_u(u, n_sites)

    if edges is None:
        edges_list: list[Edge] = _default_1d_chain_edges(n_sites, periodic=False)
    else:
        edges_list = list(edges)

    data: Dict[str, complex] = {}

    for i in range(n_sites):
        for spin in (0, 1):
            p = _spin_orbital_index(i, spin, n_sites)
            _add_term(data, f"+_{p} -_{p}", complex(v_mat[i, spin]), atol=atol)

    for edge in edges_list:
        if len(edge) == 2:
            i, j = int(edge[0]), int(edge[1])
            tij = complex(t)
        elif len(edge) == 3:
            i, j = int(edge[0]), int(edge[1])
            tij = complex(edge[2])
        else:
            raise ValueError(f"edge must be (i,j) or (i,j,tij), got: {edge}")

        if i == j:
            raise ValueError(f"invalid edge with i==j: {(i, j)}")
        if not (0 <= i < n_sites and 0 <= j < n_sites):
            raise ValueError(f"edge indices out of range: {(i, j)} (n_sites={n_sites})")

        for spin in (0, 1):
            pi = _spin_orbital_index(i, spin, n_sites)
            pj = _spin_orbital_index(j, spin, n_sites)
            _add_term(data, f"+_{pi} -_{pj}", -tij, atol=atol)
            _add_term(data, f"+_{pj} -_{pi}", -np.conjugate(tij), atol=atol)

    for i in range(n_sites):
        p_up = _spin_orbital_index(i, 0, n_sites)
        p_dn = _spin_orbital_index(i, 1, n_sites)
        _add_term(
            data,
            f"+_{p_up} -_{p_up} +_{p_dn} -_{p_dn}",
            complex(u_vec[i]),
            atol=atol,
        )

    return FermionicOp(data, num_spin_orbitals=2 * n_sites)


def _legacy_dimer_fermionic(t: float, u: float, dv: float) -> FermionicOp:
    data = {
        "+_0 -_1": -t,
        "+_1 -_0": -t,
        "+_2 -_3": -t,
        "+_3 -_2": -t,
        "+_1 -_1": dv / 2,
        "+_3 -_3": dv / 2,
        "+_0 -_0": -dv / 2,
        "+_2 -_2": -dv / 2,
        "+_0 -_0 +_2 -_2": u,
        "+_1 -_1 +_3 -_3": u,
    }
    return FermionicOp(data, num_spin_orbitals=4)


def build_qubit_hamiltonian_from_fermionic(
    ferm_op: FermionicOp,
    *,
    simplify_atol: float = 1e-12,
) -> tuple[SparsePauliOp, JordanWignerMapper]:
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(ferm_op)
    if simplify_atol is not None:
        qubit_op = qubit_op.simplify(atol=simplify_atol)
    return qubit_op, mapper


def build_qubit_hamiltonian(t: float, u: float, dv: float) -> tuple[SparsePauliOp, JordanWignerMapper]:
    ferm_op = build_fermionic_hubbard(
        n_sites=2,
        t=t,
        u=u,
        edges=[(0, 1)],
        v=[-dv / 2, dv / 2],
    )
    return build_qubit_hamiltonian_from_fermionic(ferm_op)

