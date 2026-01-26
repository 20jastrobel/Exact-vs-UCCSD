#!/usr/bin/env python3
import argparse
import time

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from pydephasing.quantum.ansatz import build_clustered_ansatz
from pydephasing.quantum.hamiltonian.exact import exact_ground_energy
from pydephasing.quantum.hamiltonian.hubbard import build_qubit_hamiltonian

def run_vqe(qubit_op: SparsePauliOp, maxiter: int, reps: int, seed: int) -> tuple[float, float]:
    estimator = StatevectorEstimator()
    ansatz = build_clustered_ansatz(num_qubits=qubit_op.num_qubits, reps=reps)
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

    qubit_op, _mapper = build_qubit_hamiltonian(args.t, args.u, args.dv)
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
