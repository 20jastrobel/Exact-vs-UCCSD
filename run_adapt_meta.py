#!/usr/bin/env python3
"""Run Meta-ADAPT-VQE on the Hubbard dimer using a statevector estimator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
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
from pydephasing.quantum.vqe.cost_model import CostCounters, CountingEstimator
from pydephasing.quantum.symmetry import (
    exact_ground_energy_sector,
    map_symmetry_ops_to_qubits,
)
from pydephasing.quantum.vqe.run_store import (
    JsonRunStore,
    MultiRunStore,
    RunStoreLogger,
    SqliteRunStore,
    sha256_file,
)
from pydephasing.quantum.utils_particles import (
    half_filling_num_particles,
    jw_reference_occupations_from_particles,
)


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

def _git_info() -> dict[str, object] | None:
    """Best-effort git provenance (commit + dirty)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        is_dirty = bool(dirty)
    except Exception:
        is_dirty = None
    return {"git_commit": commit, "git_dirty": is_dirty}


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "history.jsonl"
        self.f = open(self.path, "a", buffering=1)
        self.t_cum = 0.0
        self._t0 = None

    def log_point(self, *, it, energy, max_grad=None, chosen_op=None,
                  t_iter_s=None, t_cum_s=None, extra=None):
        row = {
            "iter": int(it),
            "energy": float(energy),
            "max_grad": None if max_grad is None else float(max_grad),
            "chosen_op": None if chosen_op is None else str(chosen_op),
            "t_iter_s": None if t_iter_s is None else float(t_iter_s),
            "t_cum_s": None if t_cum_s is None else float(t_cum_s),
        }
        if extra:
            row.update(extra)
        self.f.write(json.dumps(row) + "\n")

    def start_iter(self):
        self._t0 = time.perf_counter()

    def end_iter(self):
        dt = time.perf_counter() - self._t0
        self.t_cum += dt
        return dt, self.t_cum

    def close(self):
        self.f.close()


def make_run_dir_and_meta(args) -> Path:
    run_id = getattr(args, "run_id", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = Path(getattr(args, "logdir", "runs"))
    run_dir = logdir / f"{run_id}_L{args.sites}_Nup{args.n_up}_Ndown{args.n_down}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "sites": args.sites,
        "n_up": args.n_up,
        "n_down": args.n_down,
        "sz_target": getattr(args, "sz_target", None),
        "pool": getattr(args, "pool", None),
        "cse": {
            "include_diagonal": getattr(args, "cse_include_diagonal", None),
            "include_antihermitian_part": getattr(args, "cse_include_antihermitian", None),
            "include_hermitian_part": getattr(args, "cse_include_hermitian", None),
        },
        "uccsd": {
            "reps": getattr(args, "uccsd_reps", None),
            "include_imaginary": getattr(args, "uccsd_include_imaginary", None),
            "generalized": getattr(args, "uccsd_generalized", None),
            "preserve_spin": getattr(args, "uccsd_preserve_spin", None),
        },
        "inner_optimizer": getattr(args, "inner_optimizer", None),
        "max_depth": getattr(args, "max_depth", None),
        "eps_grad": getattr(args, "eps_grad", None),
        "eps_energy": getattr(args, "eps_energy", None),
        "ham_params": {
            "t": getattr(args, "t", None),
            "u": getattr(args, "u", None),
            "dv": getattr(args, "dv", None),
        },
        "budget": getattr(args, "budget", None),
        "exact_energy": None,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[log] run_dir = {run_dir}")
    return run_dir


def update_meta_exact_energy(run_dir: Path, exact_energy: float):
    p = run_dir / "meta.json"
    meta = json.loads(p.read_text())
    meta["exact_energy"] = float(exact_energy)
    p.write_text(json.dumps(meta, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-ADAPT-VQE for 2-site Hubbard (4 qubits) using StatevectorEstimator."
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.5)
    parser.add_argument("--sites", type=int, default=2)
    parser.add_argument(
        "--sz-target",
        type=float,
        default=0.0,
        help="Target Sz sector used only for default half-filling (when --n-up/--n-down not set).",
    )
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--inner-steps", type=int, default=25)
    parser.add_argument("--eps-grad", type=float, default=1e-4)
    parser.add_argument("--eps-energy", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--occ", type=str, default="")
    parser.add_argument("--n-up", type=int, default=None)
    parser.add_argument("--n-down", type=int, default=None)
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
    parser.add_argument("--meta-alpha0", type=float, default=1.0)
    parser.add_argument("--meta-alpha-k", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--polish-steps", type=int, default=3)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--store",
        type=str,
        choices=["sqlite", "json"],
        default=os.environ.get("RUN_STORE", "sqlite"),
        help="Run storage backend (env: RUN_STORE).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.environ.get("RUNS_DB_PATH", "data/runs.db"),
        help="SQLite DB path when --store sqlite (env: RUNS_DB_PATH).",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--theta-init-noise", type=float, default=0.0)
    parser.add_argument("--lbfgs-restarts", type=int, default=3)
    parser.add_argument("--allow-repeats", action="store_true")
    parser.add_argument("--compute-var-h", dest="compute_var_h", action="store_true", default=False)
    parser.add_argument("--no-compute-var-h", dest="compute_var_h", action="store_false")
    parser.add_argument("--enforce-sector", action="store_true", default=True)
    parser.add_argument("--no-enforce-sector", dest="enforce_sector", action="store_false")
    parser.add_argument("--budget-k", type=float, default=2000.0)
    parser.add_argument("--no-budget", action="store_true")
    parser.add_argument("--max-pauli-terms", type=int, default=None)
    parser.add_argument("--max-circuits", type=int, default=None)
    parser.add_argument("--max-time-s", type=float, default=None)
    parser.add_argument(
        "--pool",
        type=str,
        default="ham_terms_plus_imag_partners",
        choices=["ham_terms", "ham_terms_plus_imag_partners", "cse_density_ops", "uccsd_excitations"],
    )
    parser.add_argument("--cse-include-diagonal", dest="cse_include_diagonal", action="store_true", default=True)
    parser.add_argument("--no-cse-include-diagonal", dest="cse_include_diagonal", action="store_false")
    parser.add_argument("--cse-include-antihermitian", dest="cse_include_antihermitian", action="store_true", default=True)
    parser.add_argument("--no-cse-include-antihermitian", dest="cse_include_antihermitian", action="store_false")
    parser.add_argument("--cse-include-hermitian", dest="cse_include_hermitian", action="store_true", default=True)
    parser.add_argument("--no-cse-include-hermitian", dest="cse_include_hermitian", action="store_false")
    parser.add_argument("--uccsd-reps", type=int, default=1)
    parser.add_argument("--uccsd-include-imaginary", dest="uccsd_include_imaginary", action="store_true", default=False)
    parser.add_argument("--no-uccsd-include-imaginary", dest="uccsd_include_imaginary", action="store_false")
    parser.add_argument("--uccsd-generalized", dest="uccsd_generalized", action="store_true", default=False)
    parser.add_argument("--no-uccsd-generalized", dest="uccsd_generalized", action="store_false")
    parser.add_argument("--uccsd-preserve-spin", dest="uccsd_preserve_spin", action="store_true", default=True)
    parser.add_argument("--no-uccsd-preserve-spin", dest="uccsd_preserve_spin", action="store_false")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.pool == "uccsd":
        args.pool = "uccsd_excitations"

    if args.no_budget:
        pauli_budget = None
        max_circuits_executed = None
        max_time_s = None
    else:
        pauli_budget = (
            int(args.max_pauli_terms)
            if args.max_pauli_terms is not None
            else int(float(args.budget_k) * (int(args.sites) ** 2))
        )
        max_circuits_executed = (
            int(args.max_circuits) if args.max_circuits is not None else None
        )
        max_time_s = float(args.max_time_s) if args.max_time_s is not None else None

    args.budget = {
        "enabled": bool(not args.no_budget),
        "budget_k": float(args.budget_k),
        "max_pauli_terms_measured": None if pauli_budget is None else int(pauli_budget),
        "max_circuits_executed": None
        if max_circuits_executed is None
        else int(max_circuits_executed),
        "max_time_s": None if max_time_s is None else float(max_time_s),
    }

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
    if occupations:
        n_up_occ = sum(1 for idx in occupations if idx < args.sites)
        n_down_occ = sum(1 for idx in occupations if idx >= args.sites)
        if args.n_up is None and args.n_down is None:
            args.n_up = n_up_occ
            args.n_down = n_down_occ
        elif args.n_up is None or args.n_down is None:
            raise ValueError("Both --n-up and --n-down must be set together.")
        elif n_up_occ != args.n_up or n_down_occ != args.n_down:
            raise ValueError("Provided --occ does not match --n-up/--n-down sector.")
    else:
        if args.n_up is None and args.n_down is None:
            try:
                args.n_up, args.n_down = half_filling_num_particles(args.sites, sz_target=args.sz_target)
            except ValueError as exc:
                raise ValueError(
                    f"{exc} For odd L half-filling, pass --sz-target 0.5 or -0.5, "
                    "or set --n-up/--n-down explicitly."
                ) from exc
        elif args.n_up is None or args.n_down is None:
            raise ValueError("Both --n-up and --n-down must be set together.")
        occupations = jw_reference_occupations_from_particles(args.sites, args.n_up, args.n_down)

    reference_state = build_reference_state(qubit_op.num_qubits, occupations)

    # Resolve run_id early so both JSON and SQLite stores share the same identifier.
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(args.run_id)

    exact_sector = exact_ground_energy_sector(
        qubit_op,
        args.sites,
        args.n_up + args.n_down,
        0.5 * (args.n_up - args.n_down),
    )

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
        if args.inner_optimizer == "meta":
            try:
                meta_model = CoordinateWiseLSTM(hidden_size=args.meta_hidden, input_size=2)
            except Exception as exc:
                raise RuntimeError(
                    "Torch is required for meta inner optimization. "
                    "Install torch or choose --inner-optimizer lbfgs/hybrid."
                ) from exc

    weights = None
    if args.weights and args.weights.endswith(".npz"):
        weights = _load_weights(args.weights)
    lstm = CoordinateWiseLSTMOptimizer(seed=args.seed, weights=weights) if weights else CoordinateWiseLSTMOptimizer(seed=args.seed)

    git_info = _git_info() or {}
    system_json = {
        "sites": int(args.sites),
        "n_up": int(args.n_up),
        "n_down": int(args.n_down),
        "n_total": int(args.n_up) + int(args.n_down),
        "sz_target": 0.5 * (int(args.n_up) - int(args.n_down)),
        "occupations": list(map(int, occupations)),
        "ham_params": {"t": float(args.t), "u": float(args.u), "dv": float(args.dv)},
    }
    ansatz_json = {
        "pool": str(args.pool),
        "enforce_sector": bool(args.enforce_sector),
        "cse": {
            "include_diagonal": bool(args.cse_include_diagonal),
            "include_antihermitian_part": bool(args.cse_include_antihermitian),
            "include_hermitian_part": bool(args.cse_include_hermitian),
        },
        "uccsd": {
            "reps": int(args.uccsd_reps),
            "include_imaginary": bool(args.uccsd_include_imaginary),
            "generalized": bool(args.uccsd_generalized),
            "preserve_spin": bool(args.uccsd_preserve_spin),
        },
    }
    optimizer_json = {
        "inner_optimizer": str(args.inner_optimizer),
        "max_depth": int(args.max_depth),
        "inner_steps": int(args.inner_steps),
        "warmup_steps": int(args.warmup_steps),
        "polish_steps": int(args.polish_steps),
        "eps_grad": float(args.eps_grad),
        "eps_energy": float(args.eps_energy),
        "theta_bound": str(args.theta_bound),
        "lbfgs_restarts": int(args.lbfgs_restarts),
        "meta": {
            "meta_r": float(args.meta_r),
            "meta_step_scale": float(args.meta_step_scale),
            "meta_dtheta_clip": float(args.meta_dtheta_clip),
            "meta_warmup_steps": int(args.meta_warmup_steps),
            "meta_hidden": int(args.meta_hidden),
            "meta_alpha0": float(args.meta_alpha0),
            "meta_alpha_k": float(args.meta_alpha_k),
        },
    }

    # RunStore config: keep legacy top-level keys for compatibility with scripts that
    # discover runs by reading meta.json, but also include structured JSON blobs for
    # DB queries (system/ansatz/optimizer).
    run_config = {
        "run_id": run_id,
        "sites": int(args.sites),
        "n_up": int(args.n_up),
        "n_down": int(args.n_down),
        "sz_target": float(args.sz_target),
        "pool": str(args.pool),
        "cse": dict(ansatz_json["cse"]),
        "uccsd": dict(ansatz_json["uccsd"]),
        "inner_optimizer": str(args.inner_optimizer),
        "max_depth": int(args.max_depth),
        "inner_steps": int(args.inner_steps),
        "eps_grad": float(args.eps_grad),
        "eps_energy": float(args.eps_energy),
        "ham_params": dict(system_json["ham_params"]),
        "budget": dict(args.budget),
        "exact_energy": float(exact_sector),
        "exact_energy_full": float(exact),
        "occ": list(map(int, occupations)),
        "seed": int(args.seed),
        "weights": args.weights,
        "store": str(args.store),
        "db_path": str(args.db),
        "logdir": str(args.logdir),
        "python": sys.version,
        "platform": platform.platform(),
        **git_info,
        "cli": vars(args),
        "system": system_json,
        "ansatz": ansatz_json,
        "optimizer": optimizer_json,
    }

    store_kind = str(args.store)
    if store_kind == "sqlite":
        store = MultiRunStore([SqliteRunStore(args.db), JsonRunStore(base_dir=args.logdir)])
    elif store_kind == "json":
        store = JsonRunStore(base_dir=args.logdir)
    else:
        raise ValueError(f"Unknown store kind: {store_kind}")

    # If the backend rewrites/chooses the run_id, use the returned value.
    run_id = store.start_run(run_config)
    logger = RunStoreLogger(store, run_id)

    cost = CostCounters()
    estimator = CountingEstimator(StatevectorEstimator(), cost)
    status = "completed"
    summary: dict[str, object] = {}
    result = None
    ops_out: list[str] = []
    try:
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
            n_sites=args.sites,
            n_up=args.n_up,
            n_down=args.n_down,
            enforce_sector=args.enforce_sector,
            cse_include_diagonal=args.cse_include_diagonal,
            cse_include_antihermitian_part=args.cse_include_antihermitian,
            cse_include_hermitian_part=args.cse_include_hermitian,
            uccsd_reps=args.uccsd_reps,
            uccsd_include_imaginary=args.uccsd_include_imaginary,
            uccsd_generalized=args.uccsd_generalized,
            uccsd_preserve_spin=args.uccsd_preserve_spin,
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
            warmup_steps=args.warmup_steps,
            polish_steps=args.polish_steps,
            meta_alpha0=args.meta_alpha0,
            meta_alpha_k=args.meta_alpha_k,
            max_pauli_terms_measured=pauli_budget,
            max_circuits_executed=max_circuits_executed,
            max_time_s=max_time_s,
            cost_counters=cost,
            compute_var_h=args.compute_var_h,
            logger=logger,
            log_every=args.log_every,
            verbose=not args.quiet,
        )

        for op in result.operators:
            if isinstance(op, dict) and "name" in op:
                ops_out.append(str(op["name"]))
            else:
                ops_out.append(str(op))

        print("Meta-ADAPT-VQE (Statevector)")
        print(f"t={args.t}, U={args.u}, dv={args.dv}")
        print(f"Exact ground energy (full):   {exact:.8f}")
        print(f"Exact ground energy (sector): {exact_sector:.8f}")
        print(f"Meta-ADAPT energy:            {result.energy:.8f}")
        print(f"Abs diff (sector):            {abs(result.energy - exact_sector):.8e}")
        print(f"Depth:              {len(result.operators)}")

        if result.diagnostics.get("outer"):
            first = result.diagnostics["outer"][0]
            print(f"max|g| at n=0:      {first['max_grad']:.6e}")
            print(f"chosen operator:    {first['chosen_op']}")
            if args.pool in {"cse_density_ops", "uccsd_excitations"}:
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
                warm_stats = stats.get("warm", {})
                polish_stats = stats.get("polish", {})
                print(f"meta avg||Δθ||:     {meta_stats.get('avg_delta_norm')}")
                print(f"warm nfev:          {warm_stats.get('nfev')}")
                print(f"polish nfev:        {polish_stats.get('nfev')}")

        if args.save:
            _save_result(args.save, result.energy, result.params, ops_out)
            sha = sha256_file(args.save)
            b = Path(args.save).stat().st_size
            store.add_artifact(run_id, kind="result_export_json", path=str(args.save), sha256=sha, bytes=int(b), extra=None)
            print(f"Saved: {args.save}")

        if args.enforce_sector:
            n_q, sz_q = map_symmetry_ops_to_qubits(_mapper, args.sites)
            from qiskit.quantum_info import Statevector
            from pydephasing.quantum.vqe.adapt_vqe_meta import build_adapt_circuit, build_adapt_circuit_grouped
            if args.pool in {"cse_density_ops", "uccsd_excitations"}:
                circuit, params = build_adapt_circuit_grouped(reference_state, result.operators)
            else:
                circuit, params = build_adapt_circuit(reference_state, result.operators)
            bound = circuit.assign_parameters({p: v for p, v in zip(params, result.params)}, inplace=False)
            state = Statevector.from_instruction(bound)
            n_val = float(np.real(state.expectation_value(n_q)))
            sz_val = float(np.real(state.expectation_value(sz_q)))
            if abs(n_val - (args.n_up + args.n_down)) > 1e-6 or abs(sz_val - 0.5 * (args.n_up - args.n_down)) > 1e-6:
                raise RuntimeError("Statevector left target sector.")
            if result.energy < exact_sector - 1e-9:
                raise RuntimeError("VQE energy below exact sector energy.")

        summary = {
            "energy": float(result.energy),
            "theta": list(map(float, result.params)),
            "operators": ops_out,
            "exact_full": float(exact),
            "exact_sector": float(exact_sector),
            "abs_delta_e_sector": float(abs(float(result.energy) - float(exact_sector))),
            "stop_reason": result.diagnostics.get("stop_reason"),
            "pool": str(args.pool),
            "cost": cost.snapshot(),
        }
    except Exception as exc:
        status = "error"
        summary = {"error": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        try:
            store.finish_run(run_id, status=status, summary_metrics=summary)

            # Register the legacy JSON artifacts in the DB so plotting/query tools
            # can back-reference the exact files used.
            run_dir = Path(args.logdir) / f"{run_id}_L{args.sites}_Nup{args.n_up}_Ndown{args.n_down}"
            if status == "error" and isinstance(summary, dict) and summary.get("traceback"):
                tb_path = run_dir / "traceback.txt"
                tb_path.write_text(str(summary["traceback"]) + "\n", encoding="utf-8")
                store.add_artifact(
                    run_id,
                    kind="traceback_txt",
                    path=str(tb_path),
                    sha256=sha256_file(tb_path),
                    bytes=int(tb_path.stat().st_size),
                    extra=None,
                )
            for kind, p in [
                ("legacy_meta_json", run_dir / "meta.json"),
                ("legacy_history_jsonl", run_dir / "history.jsonl"),
                ("legacy_result_json", run_dir / "result.json"),
            ]:
                if p.exists():
                    store.add_artifact(
                        run_id,
                        kind=kind,
                        path=str(p),
                        sha256=sha256_file(p),
                        bytes=int(p.stat().st_size),
                        extra=None,
                    )
        finally:
            if hasattr(store, "close"):
                store.close()


if __name__ == "__main__":
    main()
