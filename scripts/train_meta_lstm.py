#!/usr/bin/env python3
"""Train a coordinate-wise LSTM meta-optimizer for Hubbard dimer instances."""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np

from qiskit.primitives import StatevectorEstimator

from pydephasing.quantum.hamiltonian.hubbard import build_qubit_hamiltonian
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_adapt_circuit,
    build_operator_pool_from_hamiltonian,
    build_reference_state,
    estimate_energy,
    parameter_shift_grad,
)
from pydephasing.quantum.vqe.meta_lstm import (
    CoordinateWiseLSTM,
    preprocess_gradients_torch,
    save_meta_lstm,
)


def _omega_schedule(t: int, T: int, schedule: str, alpha: float) -> np.ndarray:
    if schedule == "uniform":
        weights = np.ones(T + 1, dtype=float)
    elif schedule == "linear":
        weights = np.linspace(0.0, 1.0, T + 1, dtype=float)
    elif schedule == "exp":
        grid = np.linspace(0.0, 1.0, T + 1, dtype=float)
        weights = np.exp(alpha * grid)
    elif schedule == "final":
        weights = np.zeros(T + 1, dtype=float)
        weights[-1] = 1.0
    else:
        raise ValueError(f"Unknown omega schedule: {schedule}")
    total = float(np.sum(weights))
    if total == 0.0:
        return weights
    return weights / total


def _sample_uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train meta LSTM optimizer for Hubbard dimer.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--T-start", type=int, default=3)
    parser.add_argument("--T-max", type=int, default=10)
    parser.add_argument("--meta-hidden", type=int, default=20)
    parser.add_argument("--meta-r", type=float, default=10.0)
    parser.add_argument("--omega-schedule", type=str, default="final", choices=["uniform", "linear", "exp", "final"])
    parser.add_argument("--omega-alpha", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t-range", type=float, nargs=2, default=[0.5, 1.5])
    parser.add_argument("--u-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--dv-range", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--out", type=str, default="weights/meta_lstm_hubbard_dimer.pt")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Torch is required to train the meta-optimizer.") from exc

    rng = np.random.default_rng(args.seed)
    estimator = StatevectorEstimator()

    model = CoordinateWiseLSTM(hidden_size=args.meta_hidden, input_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8)

    for episode in range(args.episodes):
        t = _sample_uniform(rng, args.t_range)
        u = _sample_uniform(rng, args.u_range)
        dv = _sample_uniform(rng, args.dv_range)
        qubit_op, _ = build_qubit_hamiltonian(t, u, dv)

        pool = build_operator_pool_from_hamiltonian(qubit_op, mode="ham_terms_plus_imag_partners")
        ops = list(rng.choice(pool, size=args.depth, replace=True))

        reference = build_reference_state(qubit_op.num_qubits, [0, 2])
        circuit, params = build_adapt_circuit(reference, ops)

        def energy_fn(values):
            return estimate_energy(estimator, circuit, qubit_op, list(values))

        def grad_fn(values):
            return parameter_shift_grad(energy_fn, np.asarray(values, dtype=float))

        T = args.T_start + int((episode / max(1, args.episodes - 1)) * (args.T_max - args.T_start))
        T = max(args.T_start, min(args.T_max, T))
        omega = _omega_schedule(episode, T, args.omega_schedule, args.omega_alpha)

        theta = torch.zeros((args.depth,), dtype=torch.float32, requires_grad=True)
        theta = theta + 0.05 * torch.randn_like(theta)
        state = None
        loss = 0.0

        for step in range(T + 1):
            theta_np = theta.detach().cpu().numpy()
            energy = float(energy_fn(theta_np))
            grad = grad_fn(theta_np)

            grad_tensor = torch.tensor(grad, dtype=torch.float32)
            surrogate = energy + torch.dot(theta - theta.detach(), grad_tensor)
            loss = loss + float(omega[step]) * surrogate

            if step == T:
                break

            x = preprocess_gradients_torch(grad, r=args.meta_r)
            delta, state = model(x, state)
            theta = theta + delta

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"episode {episode + 1}/{args.episodes} - loss={float(loss):.6f}")

    config = {
        "hidden_size": args.meta_hidden,
        "input_size": 2,
        "r": args.meta_r,
        "omega_schedule": args.omega_schedule,
        "omega_alpha": args.omega_alpha,
        "depth": args.depth,
        "T_start": args.T_start,
        "T_max": args.T_max,
        "t_range": args.t_range,
        "u_range": args.u_range,
        "dv_range": args.dv_range,
    }
    save_meta_lstm(args.out, model, config)
    print(f"Saved meta-optimizer weights to {args.out}")


if __name__ == "__main__":
    main()
