"""Model-relevant observables for the spinful Hubbard model under JW ordering.

Assumed qubit ordering for 2*L spin-orbitals:
  [0..L-1]          = spin-up orbitals
  [L..2*L-1]        = spin-down orbitals

All operators below are returned as SparsePauliOp on 2*L qubits.
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp


def _label_with_z(num_qubits: int, *z_qubits: int) -> str:
    chars = ["I"] * int(num_qubits)
    for q in z_qubits:
        qq = int(q)
        if qq < 0 or qq >= int(num_qubits):
            raise ValueError(f"qubit index out of range: {qq}")
        chars[qq] = "Z"
    return "".join(chars)


def jw_number_op(*, n_sites: int, orbital: int) -> SparsePauliOp:
    """n_p = (I - Z_p)/2 for JW occupation convention (|1> == occupied)."""
    num_qubits = 2 * int(n_sites)
    p = int(orbital)
    if p < 0 or p >= num_qubits:
        raise ValueError(f"orbital index out of range: {p}")

    ident = "I" * num_qubits
    z = _label_with_z(num_qubits, p)
    return SparsePauliOp.from_list([(ident, 0.5), (z, -0.5)]).simplify(atol=1e-12)


def jw_site_density_op(*, n_sites: int, site: int) -> SparsePauliOp:
    """n_i = n_{i,up} + n_{i,down}."""
    L = int(n_sites)
    i = int(site)
    if i < 0 or i >= L:
        raise ValueError(f"site index out of range: {i}")

    return (
        jw_number_op(n_sites=L, orbital=i) + jw_number_op(n_sites=L, orbital=i + L)
    ).simplify(atol=1e-12)


def jw_site_density_ops(*, n_sites: int) -> list[SparsePauliOp]:
    return [jw_site_density_op(n_sites=int(n_sites), site=i) for i in range(int(n_sites))]


def jw_double_occupancy_op(*, n_sites: int, site: int) -> SparsePauliOp:
    """d_i = n_{i,up} n_{i,down} = (I - Z_up - Z_dn + Z_up Z_dn)/4."""
    L = int(n_sites)
    i = int(site)
    if i < 0 or i >= L:
        raise ValueError(f"site index out of range: {i}")

    num_qubits = 2 * L
    up = i
    dn = i + L
    ident = "I" * num_qubits
    z_up = _label_with_z(num_qubits, up)
    z_dn = _label_with_z(num_qubits, dn)
    z_up_dn = _label_with_z(num_qubits, up, dn)

    return SparsePauliOp.from_list(
        [
            (ident, 0.25),
            (z_up, -0.25),
            (z_dn, -0.25),
            (z_up_dn, 0.25),
        ]
    ).simplify(atol=1e-12)


def jw_total_double_occupancy_op(*, n_sites: int) -> SparsePauliOp:
    L = int(n_sites)
    num_qubits = 2 * L
    out = SparsePauliOp.from_list([("I" * num_qubits, 0.0)])
    for i in range(L):
        out = out + jw_double_occupancy_op(n_sites=L, site=i)
    return out.simplify(atol=1e-12)


def jw_sz_op(*, n_sites: int, site: int) -> SparsePauliOp:
    """S^z_i = (n_{i,up} - n_{i,down})/2 = (Z_dn - Z_up)/4."""
    L = int(n_sites)
    i = int(site)
    if i < 0 or i >= L:
        raise ValueError(f"site index out of range: {i}")

    num_qubits = 2 * L
    up = i
    dn = i + L
    z_up = _label_with_z(num_qubits, up)
    z_dn = _label_with_z(num_qubits, dn)
    return SparsePauliOp.from_list([(z_dn, 0.25), (z_up, -0.25)]).simplify(atol=1e-12)


def jw_sz_sz_op(*, n_sites: int, i: int, j: int) -> SparsePauliOp:
    """S^z_i S^z_j expanded in the Z basis."""
    L = int(n_sites)
    i = int(i)
    j = int(j)
    if i < 0 or i >= L or j < 0 or j >= L:
        raise ValueError(f"site indices out of range: {(i, j)}")
    if i == j:
        # (Sz_i)^2 is also a valid observable; keep it consistent with expansion.
        # Sz_i = (Z_dn - Z_up)/4, so Sz_i^2 = (I - Z_up Z_dn)/16.
        num_qubits = 2 * L
        ident = "I" * num_qubits
        up = i
        dn = i + L
        z_up_dn = _label_with_z(num_qubits, up, dn)
        return SparsePauliOp.from_list([(ident, 1.0 / 16.0), (z_up_dn, -1.0 / 16.0)]).simplify(atol=1e-12)

    num_qubits = 2 * L
    up_i, dn_i = i, i + L
    up_j, dn_j = j, j + L
    # (Z_dn_i - Z_up_i)(Z_dn_j - Z_up_j) / 16
    return SparsePauliOp.from_list(
        [
            (_label_with_z(num_qubits, dn_i, dn_j), 1.0 / 16.0),
            (_label_with_z(num_qubits, dn_i, up_j), -1.0 / 16.0),
            (_label_with_z(num_qubits, up_i, dn_j), -1.0 / 16.0),
            (_label_with_z(num_qubits, up_i, up_j), 1.0 / 16.0),
        ]
    ).simplify(atol=1e-12)


def jw_sz_sz_nn_sum_op(*, n_sites: int) -> SparsePauliOp:
    """Sum_{i=0..L-2} S^z_i S^z_{i+1} for an open chain."""
    L = int(n_sites)
    num_qubits = 2 * L
    out = SparsePauliOp.from_list([("I" * num_qubits, 0.0)])
    for i in range(L - 1):
        out = out + jw_sz_sz_op(n_sites=L, i=i, j=i + 1)
    return out.simplify(atol=1e-12)

