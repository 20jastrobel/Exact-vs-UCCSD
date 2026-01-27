#!/usr/bin/env python3
"""Run Meta-ADAPT-VQE on the Hubbard dimer using a statevector estimator."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np
from qiskit.primitives import StatevectorEstimator

from pydephasing.quantum.hamiltonian.exact import exact_ground_energy
from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_reference_state,
    run_meta_adapt_vqe,
)
from pydephasing.quantum.vqe.meta_lstm_optimizer import CoordinateWiseLSTMOptimizer
from pydephasing.quantum.vqe.meta_lstm import load_meta_lstm, CoordinateWiseLSTM


def _parse_occ_list(value: str | None) -> list[int]:
    if value is None:
        return []
    text = value.strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _load_weights(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "W_x": data["W_x"],
            "W_h": data["W_h"],
            "b": data["b"],
            "W_out": data["W_out"],
            "b_out": data["b_out"],
        }


def _save_result(path: str, energy: float, params: Sequence[float], operators: Sequence[str]) -> None:
    payload = {
        "energy": float(energy),
        "theta": list(map(float, params)),
        "operators": list(operators),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-ADAPT-VQE for 2-site Hubbard (4 qubits) using StatevectorEstimator."
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.5)
    parser.add_argument("--sites", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--inner-steps", type=int, default=25)
    parser.add_argument("--eps-grad", type=float, default=1e-4)
    parser.add_argument("--eps-energy", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--occ", type=str, default="0,2")
    parser.add_argument(
        "--inner-optimizer",
        type=str,
        choices=["meta", "lbfgs", "hybrid"],
        default=None,
    )
    parser.add_argument(
        "--theta-bound",
        type=str,
        choices=["none", "pi"],
        default="none",
    )
    parser.add_argument("--meta-warmup-steps", type=int, default=15)
    parser.add_argument("--meta-step-scale", type=float, default=1.0)
    parser.add_argument("--meta-dtheta-clip", type=float, default=0.25)
    parser.add_argument("--meta-hidden", type=int, default=20)
    parser.add_argument("--meta-r", type=float, default=10.0)
    parser.add_argument("--theta-init-noise", type=float, default=0.0)
    parser.add_argument("--lbfgs-restarts", type=int, default=3)
    parser.add_argument("--allow-repeats", action="store_true")
    parser.add_argument(
        "--pool",
        type=str,
        default="ham_terms_plus_imag_partners",
        choices=["ham_terms", "ham_terms_plus_imag_partners", "cse_density_ops"],
    )
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ferm_op = build_fermionic_hubbard(
        n_sites=args.sites,
        t=args.t,
        u=args.u,
        edges=default_1d_chain_edges(args.sites, periodic=False),
        v=[-args.dv / 2, args.dv / 2] if args.sites == 2 else None,
    )
    qubit_op, _mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    exact = exact_ground_energy(qubit_op)

    occupations = _parse_occ_list(args.occ)
    reference_state = build_reference_state(qubit_op.num_qubits, occupations)

    warned_missing_weights = False
    meta_model = None
    if not args.weights:
        warned_missing_weights = True
        print(
            "Meta-optimizer weights not provided; using random init "
            "(expected poor performance)."
        )
    else:
        try:
            meta_model, meta_config = load_meta_lstm(args.weights, device="cpu")
            print(
                f"Loaded meta-optimizer weights from {args.weights} "
                f"(hidden={meta_config.get('hidden_size', 'n/a')}, "
                f"input={meta_config.get('input_size', 'n/a')})"
            )
        except FileNotFoundError:
            warned_missing_weights = True
            print(
                "Meta-optimizer weights not provided; using random init "
                "(expected poor performance)."
            )
        except Exception as exc:
            warned_missing_weights = True
            print(
                "Meta-optimizer weights not provided; using random init "
                f"(expected poor performance). ({exc})"
            )

    if args.inner_optimizer is None:
        args.inner_optimizer = "meta" if meta_model is not None else "lbfgs"

    if args.inner_optimizer in {"meta", "hybrid"} and meta_model is None:
        if not warned_missing_weights:
            print(
                "Meta-optimizer weights not provided; using random init "
                "(expected poor performance)."
            )
        try:
            meta_model = CoordinateWiseLSTM(hidden_size=args.meta_hidden, input_size=2)
        except Exception as exc:
            raise RuntimeError(
                "Torch is required for meta/hybrid inner optimization. "
                "Install torch or choose --inner-optimizer lbfgs."
            ) from exc

    weights = None
    if args.weights and args.weights.endswith(".npz"):
        weights = _load_weights(args.weights)
    lstm = CoordinateWiseLSTMOptimizer(seed=args.seed, weights=weights) if weights else CoordinateWiseLSTMOptimizer(seed=args.seed)

    estimator = StatevectorEstimator()
    result = run_meta_adapt_vqe(
        qubit_op,
        reference_state,
        estimator,
        max_depth=args.max_depth,
        inner_steps=args.inner_steps,
        eps_grad=args.eps_grad,
        eps_energy=args.eps_energy,
        lstm_optimizer=lstm,
        meta_model=meta_model,
        seed=args.seed,
        pool_mode=args.pool,
        ferm_op=ferm_op,
        mapper=_mapper,
        r=args.meta_r,
        inner_optimizer=args.inner_optimizer,
        theta_bound=args.theta_bound,
        meta_step_scale=args.meta_step_scale,
        meta_dtheta_clip=args.meta_dtheta_clip,
        meta_warmup_steps=args.meta_warmup_steps,
        lbfgs_maxiter=args.inner_steps,
        lbfgs_restarts=args.lbfgs_restarts,
        theta_init_noise=args.theta_init_noise,
        allow_repeats=args.allow_repeats,
        verbose=not args.quiet,
    )

    print("Meta-ADAPT-VQE (Statevector)")
    print(f"t={args.t}, U={args.u}, dv={args.dv}")
    print(f"Exact ground energy: {exact:.8f}")
    print(f"Meta-ADAPT energy:  {result.energy:.8f}")
    print(f"Abs diff:           {abs(result.energy - exact):.8e}")
    print(f"Depth:              {len(result.operators)}")
    if result.diagnostics.get("outer"):
        first = result.diagnostics["outer"][0]
        print(f"max|g| at n=0:      {first['max_grad']:.6e}")
        print(f"chosen operator:    {first['chosen_op']}")
        if args.pool == "cse_density_ops":
            pool_size = result.diagnostics.get("pool_size", [None])[0]
            print(f"pool size:          {pool_size}")
            comps = first.get("chosen_components") or []
            comp_preview = ", ".join(f"{lbl}:{wt:.3f}" for lbl, wt in comps[:6])
            if len(comps) > 6:
                comp_preview += ", ..."
            if comp_preview:
                print(f"components:         {comp_preview}")
        last = result.diagnostics["outer"][-1]
        if last.get("inner_optimizer") == "lbfgs":
            stats = last.get("inner_stats") or {}
            print(f"lbfgs nfev:         {stats.get('nfev')}")
            print(f"lbfgs njev:         {stats.get('njev')}")
            print(f"lbfgs restarts:     {stats.get('restarts')}")
        elif last.get("inner_optimizer") == "meta":
            stats = last.get("inner_stats") or {}
            print(f"meta avg||Δθ||:     {stats.get('avg_delta_norm')}")
        elif last.get("inner_optimizer") == "hybrid":
            stats = last.get("inner_stats") or {}
            meta_stats = stats.get("meta", {})
            lbfgs_stats = stats.get("lbfgs", {})
            print(f"meta avg||Δθ||:     {meta_stats.get('avg_delta_norm')}")
            print(f"lbfgs nfev:         {lbfgs_stats.get('nfev')}")
            print(f"lbfgs restarts:     {lbfgs_stats.get('restarts')}")

    if args.save:
        _save_result(args.save, result.energy, result.params, result.operators)
        print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
