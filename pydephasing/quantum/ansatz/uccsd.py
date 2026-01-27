"""UCCSD ansatz builder for spinful Hubbard models."""

from __future__ import annotations

from typing import Tuple, Union

from qiskit.circuit import QuantumCircuit

try:
    # Qiskit Nature newer API
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
except ImportError:  # pragma: no cover
    # Older qiskit-nature fallback
    from qiskit_nature.circuit.library import HartreeFock, UCCSD


def build_uccsd_ansatz(
    *,
    n_sites: int,
    num_particles: Union[int, Tuple[int, int]],
    reps: int = 1,
    qubit_mapper=None,
) -> QuantumCircuit:
    """
    Build a UCCSD ansatz for a spinful Hubbard model with n_sites spatial orbitals.

    Args:
        n_sites: number of spatial orbitals (Hubbard sites).
        num_particles:
            - either total electrons (int), assuming Sz=0 split,
            - or explicit (n_alpha, n_beta).
        qubit_mapper: same mapper used for Hamiltonian mapping (e.g., JordanWignerMapper()).
        reps: trotter repetitions (keep 1 for low depth; >1 increases depth quickly).

    Returns:
        QuantumCircuit: UCCSD ansatz circuit, including Hartree-Fock initial state.
    """
    if qubit_mapper is None:
        raise ValueError("qubit_mapper must be provided for UCCSD.")

    if isinstance(num_particles, int):
        n_total = int(num_particles)
        if n_total % 2 != 0:
            raise ValueError(
                "If num_particles is an int, it must be even (assumes Sz=0). "
                "Pass (n_alpha, n_beta) explicitly for odd totals."
            )
        num_particles = (n_total // 2, n_total // 2)

    num_spatial_orbitals = int(n_sites)
    num_particles = (int(num_particles[0]), int(num_particles[1]))

    try:
        initial_state = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=qubit_mapper,
        )
    except TypeError:
        initial_state = HartreeFock(num_spatial_orbitals, num_particles, qubit_mapper)

    try:
        ansatz = UCCSD(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=qubit_mapper,
            reps=int(reps),
            initial_state=initial_state,
        )
    except TypeError:
        ansatz = UCCSD(
            num_spatial_orbitals,
            num_particles,
            qubit_mapper,
            reps=int(reps),
            initial_state=initial_state,
        )

    return ansatz


# Backwards-compatible alias used by the ansatz dispatcher.
build_ansatz = build_uccsd_ansatz
