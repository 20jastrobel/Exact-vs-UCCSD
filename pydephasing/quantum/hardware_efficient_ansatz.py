"""Hardware-efficient ansatz builder for Hubbard dimer experiments."""

from __future__ import annotations

from typing import Iterable, List, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _entanglement_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []

    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]

    if entanglement == "circular":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        pairs.append((n_qubits - 1, 0))
        return pairs

    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    raise ValueError(f"Unknown entanglement pattern: {entanglement}")


def _apply_manual_hea(
    qc: QuantumCircuit,
    reps: int,
    entanglement: str,
    rotation: str,
    add_barriers: bool,
) -> None:
    n_qubits = qc.num_qubits
    if n_qubits == 0:
        return

    if rotation == "ry":
        params = ParameterVector("theta", reps * n_qubits)
        param_idx = 0
        for _ in range(reps):
            for q in range(n_qubits):
                qc.ry(params[param_idx], q)
                param_idx += 1
            for src, dst in _entanglement_pairs(n_qubits, entanglement):
                qc.cx(src, dst)
            if add_barriers:
                qc.barrier()
        return

    if rotation == "ryrz":
        params = ParameterVector("theta", reps * n_qubits * 2)
        param_idx = 0
        for _ in range(reps):
            for q in range(n_qubits):
                qc.ry(params[param_idx], q)
                param_idx += 1
                qc.rz(params[param_idx], q)
                param_idx += 1
            for src, dst in _entanglement_pairs(n_qubits, entanglement):
                qc.cx(src, dst)
            if add_barriers:
                qc.barrier()
        return

    raise ValueError(f"Unknown rotation: {rotation}")


def build_hardware_efficient_ansatz(
    n_qubits: int,
    reps: int = 1,
    entanglement: str = "linear",
    initial_occupations: Optional[list[int]] = None,
    rotation: str = "ry",
    add_barriers: bool = False,
) -> QuantumCircuit:
    """Build a low-depth hardware-efficient ansatz using Ry/Rz and CX only."""

    rotation = rotation.lower()
    if rotation not in {"ry", "ryrz"}:
        raise ValueError("rotation must be 'ry' or 'ryrz'")

    qc_init = QuantumCircuit(n_qubits)
    if initial_occupations is None:
        initial_occupations = [0, 2]

    for idx in initial_occupations:
        if idx < 0 or idx >= n_qubits:
            raise ValueError(f"Initial occupation index out of range: {idx}")
        qc_init.x(idx)

    try:
        from qiskit.circuit.library import RealAmplitudes, TwoLocal
    except Exception:  # pragma: no cover - fallback path
        RealAmplitudes = None
        TwoLocal = None

    if rotation == "ry" and RealAmplitudes is not None:
        hea = RealAmplitudes(
            n_qubits,
            entanglement=entanglement,
            reps=reps,
            insert_barriers=add_barriers,
        )
        return qc_init.compose(hea)

    if TwoLocal is not None:
        rotation_blocks = ["ry"] if rotation == "ry" else ["ry", "rz"]
        hea = TwoLocal(
            n_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks="cx",
            entanglement=entanglement,
            reps=reps,
            insert_barriers=add_barriers,
            skip_final_rotation_layer=False,
        )
        return qc_init.compose(hea)

    qc = qc_init.copy()
    _apply_manual_hea(qc, reps, entanglement, rotation, add_barriers)
    return qc
