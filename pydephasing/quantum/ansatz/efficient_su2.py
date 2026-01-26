"""EfficientSU2 ansatz for the Hubbard dimer."""

from qiskit.circuit.library import efficient_su2


def build_ansatz(*, num_qubits: int, reps: int = 2):
    return efficient_su2(
        num_qubits,
        su2_gates=["ry", "rz"],
        reps=reps,
        entanglement="full",
    )


def build_efficient_su2_ansatz(*, num_qubits: int, reps: int = 2):
    return build_ansatz(num_qubits=num_qubits, reps=reps)

