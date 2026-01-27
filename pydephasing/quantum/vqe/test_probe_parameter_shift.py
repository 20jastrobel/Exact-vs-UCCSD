import math

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector


def _energy(circuit: QuantumCircuit, op: SparsePauliOp) -> float:
    state = Statevector.from_instruction(circuit)
    return float(np.real(state.expectation_value(op)))


def _finite_diff(energy_fn, eps: float = 1e-6) -> float:
    return (energy_fn(eps) - energy_fn(-eps)) / (2 * eps)


def _random_base_circuit(n_qubits: int, seed: int = 7) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(float(rng.uniform(0, 2 * math.pi)), q)
        qc.rz(float(rng.uniform(0, 2 * math.pi)), q)
    qc.cx(0, 1)
    return qc


def test_probe_parameter_shift_half_angle() -> None:
    n_qubits = 2
    base = _random_base_circuit(n_qubits)
    op = SparsePauliOp.from_list([("ZZ", 0.7), ("XI", -0.2)])
    label = "XY"

    def energy_fn(phi: float) -> float:
        qc = QuantumCircuit(n_qubits)
        qc.compose(base, inplace=True)
        gate = PauliEvolutionGate(Pauli(label), time=phi / 2)
        qc.append(gate, [0, 1])
        return _energy(qc, op)

    shift = math.pi / 2
    ps_grad = 0.5 * (energy_fn(shift) - energy_fn(-shift))
    fd_grad = _finite_diff(energy_fn)
    assert np.isclose(ps_grad, fd_grad, atol=1e-5)


def test_probe_parameter_shift_full_angle() -> None:
    n_qubits = 2
    base = _random_base_circuit(n_qubits, seed=11)
    op = SparsePauliOp.from_list([("ZI", -0.5), ("XX", 0.3)])
    label = "YX"

    def energy_fn(phi: float) -> float:
        qc = QuantumCircuit(n_qubits)
        qc.compose(base, inplace=True)
        gate = PauliEvolutionGate(Pauli(label), time=phi)
        qc.append(gate, [0, 1])
        return _energy(qc, op)

    shift = math.pi / 4
    ps_grad = energy_fn(shift) - energy_fn(-shift)
    fd_grad = _finite_diff(energy_fn)
    assert np.isclose(ps_grad, fd_grad, atol=1e-5)
