"""VQE execution helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator

from .optimizers import build_optimizer


@dataclass
class LocalVQEResult:
    energy: float
    params: list[float]
    seconds: float


def run_vqe_with_estimator(
    qubit_op,
    ansatz,
    estimator,
    *,
    restarts: int,
    maxiter: int,
    optimizer_name: str,
    seed: int,
    initial_point: np.ndarray | None = None,
) -> LocalVQEResult:
    rng = np.random.default_rng(seed)

    best_energy: Optional[float] = None
    best_params: list[float] = []
    start = time.perf_counter()

    for restart_idx in range(restarts):
        optimizer = build_optimizer(optimizer_name, maxiter=maxiter)
        if initial_point is not None:
            point = np.asarray(initial_point, dtype=float)
        elif ansatz.__class__.__name__.lower() == "uccsd":
            # UCCSD is typically initialized at the HF reference point.
            point = np.zeros(ansatz.num_parameters, dtype=float)
        else:
            point = rng.random(ansatz.num_parameters) * 2 * np.pi
        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=point,
        )
        result = vqe.compute_minimum_eigenvalue(qubit_op)
        energy = float(np.real(result.eigenvalue))

        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_params = list(map(float, result.optimal_point))

        print(
            f"local-vqe restart {restart_idx + 1}/{restarts}: "
            f"energy={energy:.10f}"
        )

    if best_energy is None:
        raise RuntimeError("Local VQE failed to produce an energy.")

    elapsed = time.perf_counter() - start
    return LocalVQEResult(best_energy, best_params, elapsed)


def run_local_vqe(
    qubit_op,
    ansatz,
    *,
    restarts: int,
    maxiter: int,
    optimizer_name: str,
    seed: int,
    initial_point: np.ndarray | None = None,
) -> LocalVQEResult:
    estimator = StatevectorEstimator()
    return run_vqe_with_estimator(
        qubit_op,
        ansatz,
        estimator,
        restarts=restarts,
        maxiter=maxiter,
        optimizer_name=optimizer_name,
        seed=seed,
        initial_point=initial_point,
    )
