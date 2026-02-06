"""Helpers for particle-number sectors and Jordan-Wigner reference occupations."""

from __future__ import annotations


def half_filling_num_particles(n_sites: int, *, sz_target: float = 0.0) -> tuple[int, int]:
    """
    Spinful Hubbard with 2*n_sites spin orbitals.
    Half filling means N_total = n_sites electrons.

    If sz_target=0 and n_sites is even: (N_up, N_down) = (n_sites/2, n_sites/2).
    If n_sites is odd and sz_target=0 is impossible, this will raise unless you set sz_target=±0.5.
    """
    n_sites = int(n_sites)
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")

    n_total = n_sites  # half-filling for spinful Hubbard
    delta = int(round(2.0 * float(sz_target)))  # since Sz = (N_up - N_down)/2

    # Solve:
    #   N_up + N_down = n_total
    #   N_up - N_down = delta
    # => N_up = (n_total + delta)/2, N_down = (n_total - delta)/2
    if (n_total + delta) % 2 != 0:
        raise ValueError(
            f"Incompatible (n_sites={n_sites}) with sz_target={sz_target}. "
            "Choose sz_target so that (n_total + 2*sz_target) is even."
        )

    n_up = (n_total + delta) // 2
    n_down = (n_total - delta) // 2
    if n_up < 0 or n_down < 0:
        raise ValueError("Invalid (N_up, N_down) computed; check sz_target.")
    return int(n_up), int(n_down)


def jw_reference_occupations_from_particles(n_sites: int, n_up: int, n_down: int) -> list[int]:
    """
    Occupied spin-orbital indices for Jordan-Wigner ordering used in this repo:
      [0..n_sites-1]          = spin-up orbitals
      [n_sites..2*n_sites-1]  = spin-down orbitals
    """
    n_sites = int(n_sites)
    n_up = int(n_up)
    n_down = int(n_down)
    if not (0 <= n_up <= n_sites and 0 <= n_down <= n_sites):
        raise ValueError("n_up and n_down must be between 0 and n_sites.")
    return list(range(n_up)) + list(range(n_sites, n_sites + n_down))

