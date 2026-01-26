"""UCCSD ansatz builder for the Hubbard dimer."""

from __future__ import annotations

from typing import Optional


def build_ansatz(
    *,
    num_qubits: int,
    reps: int = 1,
    mapper,
    num_spatial_orbitals: int = 2,
    num_particles: tuple[int, int] = (1, 1),
    initial_state=None,
):
    try:
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
    except Exception as exc:
        raise ImportError("qiskit_nature is required for UCCSD") from exc

    if num_spatial_orbitals * 2 != num_qubits:
        raise ValueError(
            "UCCSD expects num_qubits == 2 * num_spatial_orbitals; "
            f"got num_qubits={num_qubits}, num_spatial_orbitals={num_spatial_orbitals}."
        )

    if initial_state is None:
        try:
            initial_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
        except TypeError:
            initial_state = HartreeFock(
                num_spatial_orbitals, num_particles, qubit_mapper=mapper
            )

    try:
        ansatz = UCCSD(
            num_spatial_orbitals,
            num_particles,
            mapper,
            initial_state=initial_state,
            reps=reps,
            preserve_spin=True,
        )
    except TypeError:
        ansatz = UCCSD(
            num_spatial_orbitals,
            num_particles,
            qubit_mapper=mapper,
            initial_state=initial_state,
            reps=reps,
            preserve_spin=True,
        )

    return ansatz


def build_uccsd_ansatz(
    *,
    num_qubits: int,
    reps: int = 1,
    mapper,
    num_spatial_orbitals: int = 2,
    num_particles: tuple[int, int] = (1, 1),
    initial_state=None,
):
    return build_ansatz(
        num_qubits=num_qubits,
        reps=reps,
        mapper=mapper,
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        initial_state=initial_state,
    )

