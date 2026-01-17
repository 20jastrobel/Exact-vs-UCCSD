#!/usr/bin/env python3
import argparse
import time

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp


def build_fermionic_hubbard(t: float, u: float, dv: float) -> FermionicOp:
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


def build_qubit_hamiltonian(t: float, u: float, dv: float) -> SparsePauliOp:
    ferm_op = build_fermionic_hubbard(t, u, dv)
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(ferm_op)
    return qubit_op.simplify(atol=1e-12)


def exact_ground_energy(qubit_op: SparsePauliOp) -> float:
    mat = qubit_op.to_matrix()
    evals = np.linalg.eigvalsh(mat)
    return float(np.min(np.real(evals)))


def run_vqe(qubit_op: SparsePauliOp, maxiter: int, reps: int, seed: int) -> tuple[float, float]:
    estimator = StatevectorEstimator()
    ansatz = efficient_su2(
        qubit_op.num_qubits,
        su2_gates=["ry", "rz"],
        reps=reps,
        entanglement="linear",
    )
    optimizer = COBYLA(maxiter=maxiter)
    rng = np.random.default_rng(seed)
    initial_point = rng.random(ansatz.num_parameters) * 2 * np.pi

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
    )

    start = time.perf_counter()
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    elapsed = time.perf_counter() - start
    vqe_energy = float(np.real(result.eigenvalue))
    return vqe_energy, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local VQE vs exact ground-state energy for 2-site Hubbard (4 qubits)."
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.5)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    qubit_op = build_qubit_hamiltonian(args.t, args.u, args.dv)
    exact = exact_ground_energy(qubit_op)
    vqe_energy, elapsed = run_vqe(qubit_op, args.maxiter, args.reps, args.seed)

    diff = vqe_energy - exact

    print("Local VQE (no IBM runtime)")
    print(f"t={args.t}, U={args.u}, dv={args.dv}")
    print(f"Exact ground energy: {exact:.8f}")
    print(f"VQE energy:          {vqe_energy:.8f}")
    print(f"Abs diff:            {abs(diff):.8e}")
    print(f"Elapsed:             {elapsed:.2f}s")


if __name__ == "__main__":
    main()
