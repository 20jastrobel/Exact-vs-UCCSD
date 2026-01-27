import math

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_adapt_circuit,
    parameter_shift_grad,
)


def _energy(circuit: QuantumCircuit, op: SparsePauliOp, theta: np.ndarray, params) -> float:
    bound = circuit.assign_parameters({p: v for p, v in zip(params, theta)}, inplace=False)
    state = Statevector.from_instruction(bound)
    return float(np.real(state.expectation_value(op)))


def _finite_diff(energy_fn, theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (energy_fn(plus) - energy_fn(minus)) / (2 * eps)
    return grad


def test_ansatz_parameter_shift_matches_finite_diff() -> None:
    rng = np.random.default_rng(5)
    num_qubits = 2
    reference = QuantumCircuit(num_qubits)
    reference.x(0)

    ops = ["XX", "YZ", "ZX"]
    circuit, params = build_adapt_circuit(reference, ops[:2])
    op = SparsePauliOp.from_list([("ZZ", 0.8), ("XI", -0.3)])

    theta = rng.uniform(-math.pi, math.pi, size=len(params))

    def energy_fn(values):
        return _energy(circuit, op, np.asarray(values, dtype=float), params)

    ps_grad = parameter_shift_grad(energy_fn, theta)
    fd_grad = _finite_diff(energy_fn, theta)

    assert np.allclose(ps_grad, fd_grad, atol=1e-5)
