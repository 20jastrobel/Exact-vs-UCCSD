#!/usr/bin/env python3
"""Compare ADAPT vs regular VQE ansaetze and plot delta-E by ansatz type per L."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

# Allow running as `python scripts/...py` without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from pydephasing.quantum.ansatz import build_ansatz
from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.symmetry import exact_ground_energy_sector
from pydephasing.quantum.utils_particles import half_filling_sector
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_adapt_circuit_grouped,
    build_cse_density_pool_from_fermionic,
    build_reference_state,
    run_meta_adapt_vqe,
)
from pydephasing.quantum.vqe.cost_model import CostCounters, CountingEstimator
from pydephasing.quantum.vqe.run_store import (
    JsonRunStore,
    MultiRunStore,
    RunStoreLogger,
    SqliteRunStore,
    sha256_file,
)


def sector_occ(n_sites: int, n_up: int, n_down: int) -> list[int]:
    return list(range(n_up)) + list(range(n_sites, n_sites + n_down))


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str, best_energy: float | None = None):
        self.reason = reason
        self.best_energy = best_energy
        super().__init__(f"Budget exceeded: {reason}")


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


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("iter", 0)))
    return rows


def _resolve_vqe_maxiter(
    *,
    budget_enabled: bool,
    n_params: int,
    maxiter_min: int,
    maxiter_per_param: int,
    maxiter_cap: int,
    maxiter_budget_cap: int,
) -> int:
    if budget_enabled:
        return int(maxiter_budget_cap)
    scaled = max(int(maxiter_min), int(maxiter_per_param) * int(n_params))
    return int(min(int(maxiter_cap), scaled))


def find_adapt_run_dir(
    runs_root: Path,
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    pool: str,
) -> Path:
    candidates: list[tuple[Path, dict]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        hist_path = run_dir / "history.jsonl"
        # Require result.json so we don't accidentally cache an interrupted run
        # (which may contain only partial history rows).
        result_path = run_dir / "result.json"
        if not meta_path.exists() or not hist_path.exists() or not result_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            sites = int(meta.get("sites", -1))
            n_up_meta = int(meta.get("n_up", -1))
            n_down_meta = int(meta.get("n_down", -1))
        except Exception:
            continue
        if sites != n_sites or n_up_meta != n_up or n_down_meta != n_down:
            continue
        if str(meta.get("pool")) != pool:
            continue
        candidates.append((run_dir, meta))
    if not candidates:
        raise FileNotFoundError(f"No cached {pool} ADAPT runs found for L={n_sites}")
    candidates.sort(
        key=lambda item: (item[1].get("exact_energy") is None, -item[0].stat().st_mtime)
    )
    return candidates[0][0]


def load_cached_adapt_energy(
    runs_root: Path,
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    pool: str,
) -> float:
    run_dir = find_adapt_run_dir(
        runs_root,
        n_sites=n_sites,
        n_up=n_up,
        n_down=n_down,
        pool=pool,
    )
    hist = _load_history(run_dir / "history.jsonl")
    if not hist:
        raise RuntimeError(f"No history in cached run {run_dir}")
    vals = [float(r["energy"]) for r in hist if r.get("energy") is not None]
    if not vals:
        raise RuntimeError(f"No energies in cached run {run_dir}")
    return float(min(vals))


def make_adapt_run_dir(
    runs_root: Path,
    *,
    run_id: str,
    n_sites: int,
    n_up: int,
    n_down: int,
    pool: str,
    inner_optimizer: str,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    t: float,
    u: float,
    dv: float,
    exact_energy: float,
    budget_k: float | None = None,
    max_pauli_terms_measured: int | None = None,
    max_circuits_executed: int | None = None,
    max_time_s: float | None = None,
) -> tuple[Path, dict]:
    run_dir = runs_root / f"{run_id}_L{n_sites}_Nup{n_up}_Ndown{n_down}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "sites": n_sites,
        "n_up": n_up,
        "n_down": n_down,
        "pool": pool,
        "inner_optimizer": inner_optimizer,
        "max_depth": max_depth,
        "eps_grad": eps_grad,
        "eps_energy": eps_energy,
        "budget": {
            "budget_k": None if budget_k is None else float(budget_k),
            "max_pauli_terms_measured": None
            if max_pauli_terms_measured is None
            else int(max_pauli_terms_measured),
            "max_circuits_executed": None
            if max_circuits_executed is None
            else int(max_circuits_executed),
            "max_time_s": None if max_time_s is None else float(max_time_s),
        },
        "ham_params": {"t": t, "u": u, "dv": dv},
        "exact_energy": exact_energy,
        "python": sys.version,
        "platform": platform.platform(),
    }
    return run_dir, meta


def load_cse_operator_specs(
    *,
    runs_root: Path,
    ferm_op,
    mapper,
    n_sites: int,
    n_up: int,
    n_down: int,
) -> list[dict]:
    run_dir = find_adapt_run_dir(
        runs_root,
        n_sites=n_sites,
        n_up=n_up,
        n_down=n_down,
        pool="cse_density_ops",
    )
    hist = _load_history(run_dir / "history.jsonl")
    chosen = [row.get("chosen_op") for row in hist if row.get("chosen_op")]
    if not chosen:
        raise RuntimeError(f"No chosen operators found in {run_dir}")

    pool_specs = build_cse_density_pool_from_fermionic(
        ferm_op,
        mapper,
        enforce_symmetry=True,
        n_sites=n_sites,
        include_diagonal=True,
        include_antihermitian_part=True,
        include_hermitian_part=True,
    )
    spec_map = {spec["name"]: spec for spec in pool_specs}

    ops: list[dict] = []
    missing: list[str] = []
    def _resolve(name: str) -> str | None:
        if name in spec_map:
            return name
        # Back-compat for older cached runs that used gamma(...) naming.
        if name.startswith("gamma(") and name.endswith(")"):
            label = name[len("gamma(") : -1]
            for prefix in ("gamma_im(", "gamma_re("):
                cand = f"{prefix}{label})"
                if cand in spec_map:
                    return cand
        return None

    for name in chosen:
        resolved = _resolve(str(name))
        if resolved is None:
            missing.append(str(name))
            continue
        ops.append(spec_map[resolved])
    if missing:
        raise RuntimeError(f"Missing CSE operators from pool: {', '.join(missing)}")
    return ops


def run_regular_vqe(
    *,
    n_sites: int,
    num_particles: tuple[int, int],
    qubit_op,
    mapper,
    ansatz_kind: str,
    exact_energy: float,
    log_dir: Path | None = None,
    reps: int = 2,
    seed: int = 7,
    max_pauli_terms_measured: int | None = None,
    max_circuits_executed: int | None = None,
    max_time_s: float | None = None,
    vqe_maxiter_min: int = 200,
    vqe_maxiter_per_param: int = 25,
    vqe_maxiter_cap: int = 5000,
    vqe_maxiter_budget_cap: int = 100000,
    store=None,
    run_id: str | None = None,
) -> float:
    use_store = store is not None
    store_run_id: str | None = None

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if log_dir is not None and not use_store:
        log_file = open(log_dir / "history.jsonl", "w", encoding="utf-8")

    cost = CostCounters()
    estimator = CountingEstimator(StatevectorEstimator(), cost)
    ansatz = build_ansatz(
        ansatz_kind,
        qubit_op.num_qubits,
        reps,
        mapper,
        n_sites=n_sites,
        num_particles=num_particles,
    )
    ansatz_num_params = int(ansatz.num_parameters)
    # UCCSD is commonly initialized at the Hartree-Fock reference point.
    initial_point = np.zeros(ansatz_num_params, dtype=float)

    budget_enabled = any(
        lim is not None
        for lim in (max_pauli_terms_measured, max_circuits_executed, max_time_s)
    )
    maxiter = _resolve_vqe_maxiter(
        budget_enabled=budget_enabled,
        n_params=int(ansatz.num_parameters),
        maxiter_min=int(vqe_maxiter_min),
        maxiter_per_param=int(vqe_maxiter_per_param),
        maxiter_cap=int(vqe_maxiter_cap),
        maxiter_budget_cap=int(vqe_maxiter_budget_cap),
    )
    optimizer = COBYLA(maxiter=int(maxiter))

    t_start = time.perf_counter()
    t_last = t_start
    best_energy: float | None = None
    best_params: list[float] | None = None
    stop_reason: str | None = None

    if use_store:
        if log_dir is None:
            raise ValueError("store logging requires log_dir to persist legacy artifacts.")
        # Ensure run_id is unique even if the same log_dir is reused across runs.
        run_id = run_id or f"vqe_{ansatz_kind}_{int(time.time())}_L{n_sites}_Nup{num_particles[0]}_Ndown{num_particles[1]}"
        run_config = {
            "run_id": str(run_id),
            "run_dir": str(log_dir.resolve()),
            "sites": int(n_sites),
            "n_up": int(num_particles[0]),
            "n_down": int(num_particles[1]),
            "reps": int(reps),
            "seed": int(seed),
            "exact_energy": float(exact_energy),
            "budget": {
                "max_pauli_terms_measured": None if max_pauli_terms_measured is None else int(max_pauli_terms_measured),
                "max_circuits_executed": None if max_circuits_executed is None else int(max_circuits_executed),
                "max_time_s": None if max_time_s is None else float(max_time_s),
            },
            "optimizer": {"name": "COBYLA", "maxiter": int(maxiter)},
            "system": {"sites": int(n_sites), "n_up": int(num_particles[0]), "n_down": int(num_particles[1])},
            "ansatz": {"kind": str(ansatz_kind), "reps": int(reps)},
        }
        store_run_id = str(store.start_run(run_config))

    def _budget_reason() -> str | None:
        if max_time_s is not None and (time.perf_counter() - t_start) >= float(max_time_s):
            return "max_time_s"
        if max_circuits_executed is not None and cost.n_circuits_executed >= int(max_circuits_executed):
            return "max_circuits_executed"
        if max_pauli_terms_measured is not None and cost.n_pauli_terms_measured >= int(max_pauli_terms_measured):
            return "max_pauli_terms_measured"
        return None

    def _callback(eval_count: int, params: np.ndarray, energy: float, _meta: dict) -> None:
        nonlocal t_last, best_energy, best_params, stop_reason
        e = float(energy)
        params_vec = np.asarray(params, dtype=float).reshape(-1)
        if best_energy is None or e < best_energy:
            best_energy = e
            # Qiskit Algorithms may currently pass only a 1-element params array
            # in the callback; only trust it if it matches the ansatz size.
            if int(params_vec.size) == ansatz_num_params:
                best_params = list(map(float, params_vec))

        now = time.perf_counter()
        t_iter_s = now - t_last
        t_cum_s = now - t_start
        t_last = now

        # COBYLA is gradient-free; eval_count is objective evaluations so far.
        cost.n_energy_evals = int(eval_count)
        cost.n_grad_evals = 0

        row = {
            "iter": int(eval_count),
            "energy": float(e),
            "delta_e": float(e - exact_energy),
            "t_iter_s": float(t_iter_s),
            "t_cum_s": float(t_cum_s),
            "ansatz": ansatz_kind,
            "n_params": int(ansatz_num_params),
            "callback_params_len": int(params_vec.size),
            **cost.snapshot(),
        }

        reason = _budget_reason()
        if reason is not None:
            stop_reason = f"budget:{reason}"
            row["stop_reason"] = stop_reason

        if use_store and store_run_id is not None:
            metrics = dict(row)
            metrics.pop("iter", None)
            store.log_step(store_run_id, int(eval_count), metrics)
        elif log_file is not None:
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()

        if reason is not None:
            raise BudgetExceeded(reason, best_energy=best_energy)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=_callback,
    )

    def _finalize(*, status: str, payload: dict) -> None:
        if log_dir is None:
            return
        if use_store and store_run_id is not None:
            store.finish_run(store_run_id, status=str(status), summary_metrics=payload)
            for kind, p in [
                ("legacy_meta_json", log_dir / "meta.json"),
                ("legacy_history_jsonl", log_dir / "history.jsonl"),
                ("legacy_result_json", log_dir / "result.json"),
            ]:
                if p.exists():
                    store.add_artifact(
                        store_run_id,
                        kind=kind,
                        path=str(p),
                        sha256=sha256_file(p),
                        bytes=int(p.stat().st_size),
                        extra=None,
                    )
        else:
            (log_dir / "result.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    energy_out: float = float("nan")
    opt_out: list[float] | None = None
    status_out = "completed"
    payload: dict[str, object]
    try:
        try:
            result = vqe.compute_minimum_eigenvalue(qubit_op)
        except BudgetExceeded as exc:
            stop_reason = stop_reason or f"budget:{exc.reason}"
            status_out = "stopped"
            energy_out = float(best_energy) if best_energy is not None else float("nan")
            opt_out = (
                best_params
                if best_params is not None and len(best_params) == ansatz_num_params
                else None
            )
        except Exception:
            # Some SciPy-backed optimizers (notably COBYLA) may surface Python
            # exceptions from callbacks as low-level errors (e.g. "capi_return is NULL").
            # If we already logged a budget stop and have a best-so-far energy, treat it
            # as an intentional abort.
            if (
                stop_reason is not None
                and stop_reason.startswith("budget:")
                and best_energy is not None
            ):
                status_out = "stopped"
                energy_out = float(best_energy)
                opt_out = (
                    best_params
                    if best_params is not None and len(best_params) == ansatz_num_params
                    else None
                )
            else:
                raise
        else:
            final_e = float(np.real(result.eigenvalue))
            # For non-aborted runs, trust the optimizer result for parameters.
            opt_out = (
                list(map(float, result.optimal_point))
                if result.optimal_point is not None
                else None
            )
            if opt_out is not None and len(opt_out) != ansatz_num_params:
                opt_out = None
            energy_out = float(final_e)

        payload = {
            "ansatz": str(ansatz_kind),
            "n_params": int(ansatz_num_params),
            "n_sites": int(n_sites),
            "num_particles": [int(num_particles[0]), int(num_particles[1])],
            "reps": int(reps),
            "energy": float(energy_out),
            "optimal_point": opt_out,
            "optimizer": {"name": "COBYLA", "maxiter": int(maxiter)},
            "seed": int(seed),
            "stop_reason": stop_reason,
            "budget": {
                "max_pauli_terms_measured": None
                if max_pauli_terms_measured is None
                else int(max_pauli_terms_measured),
                "max_circuits_executed": None
                if max_circuits_executed is None
                else int(max_circuits_executed),
                "max_time_s": None if max_time_s is None else float(max_time_s),
            },
        }
        _finalize(status=status_out, payload=payload)
    except Exception as exc:
        payload = {
            "ansatz": str(ansatz_kind),
            "n_params": int(ansatz_num_params),
            "n_sites": int(n_sites),
            "num_particles": [int(num_particles[0]), int(num_particles[1])],
            "reps": int(reps),
            "seed": int(seed),
            "optimizer": {"name": "COBYLA", "maxiter": int(maxiter)},
            "budget": {
                "max_pauli_terms_measured": None
                if max_pauli_terms_measured is None
                else int(max_pauli_terms_measured),
                "max_circuits_executed": None
                if max_circuits_executed is None
                else int(max_circuits_executed),
                "max_time_s": None if max_time_s is None else float(max_time_s),
            },
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _finalize(status="error", payload=payload)
        raise
    finally:
        if log_file is not None:
            log_file.close()

    # Return best-so-far energy from the log when available (more robust than trusting
    # any single termination point).
    if log_dir is not None:
        best_logged = _best_energy_from_log(log_dir)
        if best_logged is not None:
            return float(best_logged)
    return float(energy_out)


def run_vqe_cse_ops(
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    ferm_op,
    qubit_op,
    mapper,
    exact_energy: float,
    runs_root: Path,
    log_dir: Path | None = None,
    seed: int = 7,
    max_pauli_terms_measured: int | None = None,
    max_circuits_executed: int | None = None,
    max_time_s: float | None = None,
    vqe_maxiter_min: int = 200,
    vqe_maxiter_per_param: int = 25,
    vqe_maxiter_cap: int = 5000,
    vqe_maxiter_budget_cap: int = 100000,
    store=None,
    run_id: str | None = None,
) -> float:
    use_store = store is not None
    store_run_id: str | None = None

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if log_dir is not None and not use_store:
        log_file = open(log_dir / "history.jsonl", "w", encoding="utf-8")

    ops = load_cse_operator_specs(
        runs_root=runs_root,
        ferm_op=ferm_op,
        mapper=mapper,
        n_sites=n_sites,
        n_up=n_up,
        n_down=n_down,
    )
    reference = build_reference_state(
        qubit_op.num_qubits,
        sector_occ(n_sites, n_up, n_down),
    )
    ansatz, _params = build_adapt_circuit_grouped(reference, ops)

    cost = CostCounters()
    estimator = CountingEstimator(StatevectorEstimator(), cost)
    initial_point = np.zeros(ansatz.num_parameters, dtype=float)
    ansatz_num_params = int(ansatz.num_parameters)
    budget_enabled = any(
        lim is not None
        for lim in (max_pauli_terms_measured, max_circuits_executed, max_time_s)
    )
    maxiter = _resolve_vqe_maxiter(
        budget_enabled=budget_enabled,
        n_params=int(ansatz.num_parameters),
        maxiter_min=int(vqe_maxiter_min),
        maxiter_per_param=int(vqe_maxiter_per_param),
        maxiter_cap=int(vqe_maxiter_cap),
        maxiter_budget_cap=int(vqe_maxiter_budget_cap),
    )
    optimizer = COBYLA(maxiter=int(maxiter))

    t_start = time.perf_counter()
    t_last = t_start
    best_energy: float | None = None
    best_params: list[float] | None = None
    stop_reason: str | None = None

    if use_store:
        if log_dir is None:
            raise ValueError("store logging requires log_dir to persist legacy artifacts.")
        run_id = run_id or f"vqe_cse_ops_{int(time.time())}_L{n_sites}_Nup{n_up}_Ndown{n_down}"
        run_config = {
            "run_id": str(run_id),
            "run_dir": str(log_dir.resolve()),
            "sites": int(n_sites),
            "n_up": int(n_up),
            "n_down": int(n_down),
            "ansatz": {"kind": "vqe_cse_ops"},
            "seed": int(seed),
            "exact_energy": float(exact_energy),
            "budget": {
                "max_pauli_terms_measured": None if max_pauli_terms_measured is None else int(max_pauli_terms_measured),
                "max_circuits_executed": None if max_circuits_executed is None else int(max_circuits_executed),
                "max_time_s": None if max_time_s is None else float(max_time_s),
            },
            "optimizer": {"name": "COBYLA", "maxiter": int(maxiter)},
            "system": {"sites": int(n_sites), "n_up": int(n_up), "n_down": int(n_down)},
        }
        store_run_id = str(store.start_run(run_config))

    def _budget_reason() -> str | None:
        if max_time_s is not None and (time.perf_counter() - t_start) >= float(max_time_s):
            return "max_time_s"
        if max_circuits_executed is not None and cost.n_circuits_executed >= int(max_circuits_executed):
            return "max_circuits_executed"
        if max_pauli_terms_measured is not None and cost.n_pauli_terms_measured >= int(max_pauli_terms_measured):
            return "max_pauli_terms_measured"
        return None

    def _callback(eval_count: int, params: np.ndarray, energy: float, _meta: dict) -> None:
        nonlocal t_last, best_energy, best_params, stop_reason
        e = float(energy)
        params_vec = np.asarray(params, dtype=float).reshape(-1)
        if best_energy is None or e < best_energy:
            best_energy = e
            if int(params_vec.size) == ansatz_num_params:
                best_params = list(map(float, params_vec))
        now = time.perf_counter()
        t_iter_s = now - t_last
        t_cum_s = now - t_start
        t_last = now
        # COBYLA is gradient-free; eval_count is objective evaluations so far.
        cost.n_energy_evals = int(eval_count)
        cost.n_grad_evals = 0
        row = {
            "iter": int(eval_count),
            "energy": float(e),
            "delta_e": float(e - exact_energy),
            "t_iter_s": float(t_iter_s),
            "t_cum_s": float(t_cum_s),
            "ansatz": "vqe_cse_ops",
            "n_params": int(ansatz_num_params),
            "callback_params_len": int(params_vec.size),
            **cost.snapshot(),
        }
        reason = _budget_reason()
        if reason is not None:
            stop_reason = f"budget:{reason}"
            row["stop_reason"] = stop_reason
        if use_store and store_run_id is not None:
            metrics = dict(row)
            metrics.pop("iter", None)
            store.log_step(store_run_id, int(eval_count), metrics)
        elif log_file is not None:
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
        if reason is not None:
            raise BudgetExceeded(reason, best_energy=best_energy)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=_callback,
    )
    def _finalize(*, status: str, payload: dict) -> None:
        if log_dir is None:
            return
        if use_store and store_run_id is not None:
            store.finish_run(store_run_id, status=str(status), summary_metrics=payload)
            for kind, p in [
                ("legacy_meta_json", log_dir / "meta.json"),
                ("legacy_history_jsonl", log_dir / "history.jsonl"),
                ("legacy_result_json", log_dir / "result.json"),
            ]:
                if p.exists():
                    store.add_artifact(
                        store_run_id,
                        kind=kind,
                        path=str(p),
                        sha256=sha256_file(p),
                        bytes=int(p.stat().st_size),
                        extra=None,
                    )
        else:
            (log_dir / "result.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    energy_out: float = float("nan")
    opt_out: list[float] | None = None
    status_out = "completed"
    payload: dict[str, object]
    try:
        try:
            result = vqe.compute_minimum_eigenvalue(qubit_op)
        except BudgetExceeded as exc:
            stop_reason = stop_reason or f"budget:{exc.reason}"
            status_out = "stopped"
            energy_out = float(best_energy) if best_energy is not None else float("nan")
            opt_out = (
                best_params
                if best_params is not None and len(best_params) == ansatz_num_params
                else None
            )
        except Exception:
            if (
                stop_reason is not None
                and stop_reason.startswith("budget:")
                and best_energy is not None
            ):
                status_out = "stopped"
                energy_out = float(best_energy)
                opt_out = (
                    best_params
                    if best_params is not None and len(best_params) == ansatz_num_params
                    else None
                )
            else:
                raise
        else:
            final_e = float(np.real(result.eigenvalue))
            opt_out = (
                list(map(float, result.optimal_point))
                if result.optimal_point is not None
                else None
            )
            if opt_out is not None and len(opt_out) != ansatz_num_params:
                opt_out = None
            energy_out = float(final_e)

        payload = {
            "ansatz": "vqe_cse_ops",
            "n_params": int(ansatz_num_params),
            "n_sites": int(n_sites),
            "num_particles": [int(n_up), int(n_down)],
            "energy": float(energy_out),
            "optimal_point": opt_out,
            "optimizer": {"name": "COBYLA", "maxiter": int(maxiter)},
            "seed": int(seed),
            "stop_reason": stop_reason,
            "budget": {
                "max_pauli_terms_measured": None
                if max_pauli_terms_measured is None
                else int(max_pauli_terms_measured),
                "max_circuits_executed": None
                if max_circuits_executed is None
                else int(max_circuits_executed),
                "max_time_s": None if max_time_s is None else float(max_time_s),
            },
        }
        _finalize(status=status_out, payload=payload)
    except Exception as exc:
        payload = {
            "ansatz": "vqe_cse_ops",
            "n_params": int(ansatz_num_params),
            "n_sites": int(n_sites),
            "num_particles": [int(n_up), int(n_down)],
            "seed": int(seed),
            "optimizer": {"name": "COBYLA", "maxiter": int(maxiter)},
            "budget": {
                "max_pauli_terms_measured": None
                if max_pauli_terms_measured is None
                else int(max_pauli_terms_measured),
                "max_circuits_executed": None
                if max_circuits_executed is None
                else int(max_circuits_executed),
                "max_time_s": None if max_time_s is None else float(max_time_s),
            },
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _finalize(status="error", payload=payload)
        raise
    finally:
        if log_file is not None:
            log_file.close()

    if log_dir is not None:
        best_logged = _best_energy_from_log(log_dir)
        if best_logged is not None:
            return float(best_logged)
    return float(energy_out)

def run_adapt(
    *,
    n_sites: int,
    ferm_op,
    qubit_op,
    mapper,
    n_up: int,
    n_down: int,
    pool_mode: str,
    logger: RunLogger | None = None,
    max_depth: int = 6,
    inner_steps: int = 25,
    warmup_steps: int = 5,
    polish_steps: int = 3,
    max_pauli_terms_measured: int | None = None,
    max_circuits_executed: int | None = None,
    max_time_s: float | None = None,
    inner_steps_schedule: Callable[[int], int] | None = None,
) -> object:
    cost = CostCounters()
    estimator = CountingEstimator(StatevectorEstimator(), cost)
    reference = build_reference_state(qubit_op.num_qubits, sector_occ(n_sites, n_up, n_down))
    result = run_meta_adapt_vqe(
        qubit_op,
        reference,
        estimator,
        pool_mode=pool_mode,
        ferm_op=ferm_op,
        mapper=mapper,
        n_sites=n_sites,
        n_up=n_up,
        n_down=n_down,
        enforce_sector=True,
        inner_optimizer="hybrid",
        max_depth=max_depth,
        inner_steps=inner_steps,
        warmup_steps=warmup_steps,
        polish_steps=polish_steps,
        compute_var_h=True,
        max_pauli_terms_measured=max_pauli_terms_measured,
        max_circuits_executed=max_circuits_executed,
        max_time_s=max_time_s,
        inner_steps_schedule=inner_steps_schedule,
        cost_counters=cost,
        logger=logger,
        log_every=1,
        verbose=False,
    )
    return result


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return rows


def _rows_by_key(rows: list[dict]) -> dict[tuple[int, int, int, str], dict]:
    out: dict[tuple[int, int, int, str], dict] = {}
    for row in rows:
        try:
            sites = int(row.get("sites"))
            n_up = int(row.get("n_up"))
            n_down = int(row.get("n_down"))
            ansatz = str(row.get("ansatz"))
        except Exception:
            continue
        out[(sites, n_up, n_down, ansatz)] = row
    return out


def _best_energy_from_log(log_dir: Path) -> float | None:
    hist = _load_history(log_dir / "history.jsonl")
    if not hist:
        return None
    vals = [float(r["energy"]) for r in hist if r.get("energy") is not None]
    return float(min(vals)) if vals else None


def load_cached_exact_energy(
    runs_root: Path,
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
) -> float | None:
    """Load exact energy from cached ADAPT runs if available."""
    for pool in ("cse_density_ops", "uccsd_excitations"):
        try:
            run_dir = find_adapt_run_dir(
                runs_root,
                n_sites=n_sites,
                n_up=n_up,
                n_down=n_down,
                pool=pool,
            )
        except Exception:
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        exact = meta.get("exact_energy")
        if exact is None:
            continue
        return float(exact)
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", type=int, default=[2, 3, 4, 5])
    ap.add_argument(
        "--odd-policy",
        type=str,
        choices=["min_sz", "restrict", "dope_sz0"],
        default="min_sz",
        help=(
            "Half-filling sector policy for odd L. "
            "min_sz: (n_up,n_down)=((L+1)/2,(L-1)/2) (Sz=+1/2). "
            "restrict: error for odd L. "
            "dope_sz0: nearest Sz=0 sector (NOT half-filling): N=L-1."
        ),
    )
    ap.add_argument("--out-dir", type=str, default="runs/compare_vqe")
    ap.add_argument(
        "--store",
        type=str,
        choices=["sqlite", "json"],
        default=os.environ.get("RUN_STORE", "sqlite"),
        help="Run storage backend (env: RUN_STORE).",
    )
    ap.add_argument(
        "--db",
        type=str,
        default=os.environ.get("RUNS_DB_PATH", "data/runs.db"),
        help="SQLite DB path when --store sqlite (env: RUNS_DB_PATH).",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--allow-exact-compute",
        action="store_true",
        help="Allow computing exact energies when not cached.",
    )
    ap.add_argument("--budget-k", type=float, default=2000.0)
    ap.add_argument("--no-budget", action="store_true")
    ap.add_argument("--max-pauli-terms", type=int, default=None)
    ap.add_argument("--max-circuits", type=int, default=None)
    ap.add_argument("--max-time-s", type=float, default=None)
    ap.add_argument("--vqe-maxiter-min", type=int, default=200)
    ap.add_argument("--vqe-maxiter-per-param", type=int, default=25)
    ap.add_argument("--vqe-maxiter-cap", type=int, default=5000)
    ap.add_argument("--vqe-maxiter-budget-cap", type=int, default=100000)
    ap.add_argument("--adapt-maxdepth-cse-cap", type=int, default=60)
    ap.add_argument("--adapt-maxdepth-uccsd-cap", type=int, default=20)
    ap.add_argument("--adapt-inner-base", type=int, default=8)
    ap.add_argument("--adapt-inner-slope", type=int, default=2)
    ap.add_argument("--adapt-inner-max", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store_kind = str(args.store)
    if store_kind == "sqlite":
        # Migration mode: default to SQLite, but also emit legacy JSON artifacts so
        # the existing plotting/caching pipeline under runs/ keeps working.
        store = MultiRunStore([SqliteRunStore(args.db), JsonRunStore(base_dir=".")])
    elif store_kind == "json":
        store = JsonRunStore(base_dir=".")
    else:
        raise ValueError(f"Unknown store kind: {store_kind}")

    batch_run_id = f"compare_vqe_hist_{time.time_ns()}"
    batch_run_dir = (out_dir / batch_run_id).resolve()
    batch_run_dir.mkdir(parents=True, exist_ok=True)
    store.start_run(
        {
            "run_id": batch_run_id,
            "run_dir": str(batch_run_dir),
            "entrypoint": "scripts/compare_vqe_hist.py",
            "sites": list(map(int, args.sites)),
            "odd_policy": str(args.odd_policy),
            "budget": {
                "budget_k": None if args.no_budget else float(args.budget_k),
                "max_pauli_terms_measured": None if args.no_budget else args.max_pauli_terms,
                "max_circuits_executed": None if args.no_budget else args.max_circuits,
                "max_time_s": None if args.no_budget else args.max_time_s,
            },
            "store": str(args.store),
            "db_path": str(args.db),
            "cli": vars(args),
        }
    )

    t = 1.0
    u = 4.0
    dv = 0.5

    runs_root = Path("runs")

    sites = sorted(set(int(s) for s in args.sites))
    # Half-filling per size: N_total = L (spinful Hubbard convention in this repo).
    sectors = {n: half_filling_sector(n, odd_policy=args.odd_policy) for n in sites}

    def inner_steps_schedule(n_params: int) -> int:
        return int(
            min(
                int(args.adapt_inner_max),
                int(args.adapt_inner_base) + int(args.adapt_inner_slope) * int(n_params),
            )
        )

    def adapt_settings(n_sites: int, pool: str) -> dict:
        if pool == "cse_density_ops":
            max_depth = min(4 * int(n_sites), int(args.adapt_maxdepth_cse_cap))
        elif pool == "uccsd_excitations":
            max_depth = min(2 * int(n_sites), int(args.adapt_maxdepth_uccsd_cap))
        else:
            max_depth = 6
        return {
            "max_depth": int(max_depth),
            "inner_steps": int(args.adapt_inner_base),
            "warmup_steps": 5,
            "polish_steps": 3,
        }

    adapt_kinds = ["adapt_cse_hybrid", "adapt_uccsd_hybrid"]
    vqe_kinds = ["uccsd", "vqe_cse_ops"]
    allowed_ansatz = set(adapt_kinds + vqe_kinds)

    compare_path = out_dir / "compare_rows.json"
    rows = _load_rows(compare_path)
    row_map = _rows_by_key(rows)
    row_map = {k: v for k, v in row_map.items() if v.get("ansatz") in allowed_ansatz}

    try:
        for n_sites in sites:
            if n_sites not in sectors:
                raise ValueError(f"Missing sector mapping for L={n_sites}")
            n_up, n_down = sectors[n_sites]

            if args.no_budget:
                max_pauli_terms_measured = None
                max_circuits_executed = None
                max_time_s = None
            else:
                max_pauli_terms_measured = (
                    int(args.max_pauli_terms)
                    if args.max_pauli_terms is not None
                    else int(float(args.budget_k) * (int(n_sites) ** 2))
                )
                max_circuits_executed = (
                    int(args.max_circuits) if args.max_circuits is not None else None
                )
                max_time_s = float(args.max_time_s) if args.max_time_s is not None else None

            # Determine if anything is missing for this site.
            missing = []
            for kind in adapt_kinds + vqe_kinds:
                if args.force or (n_sites, n_up, n_down, kind) not in row_map:
                    missing.append(kind)
            if not missing:
                print(f"L={n_sites}: using cached comparison rows.")
                continue

            ferm_op = build_fermionic_hubbard(
                n_sites=n_sites,
                t=t,
                u=u,
                edges=default_1d_chain_edges(n_sites, periodic=False),
                v=[-dv / 2, dv / 2] if n_sites == 2 else None,
            )
            qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
            exact = load_cached_exact_energy(
                runs_root,
                n_sites=n_sites,
                n_up=n_up,
                n_down=n_down,
            )
            if exact is None:
                if not args.allow_exact_compute:
                    raise RuntimeError(
                        f"Exact energy not cached for L={n_sites}. "
                        "Provide it or rerun with --allow-exact-compute."
                    )
                exact = exact_ground_energy_sector(
                    qubit_op,
                    n_sites,
                    n_up + n_down,
                    0.5 * (n_up - n_down),
                )

            if "adapt_cse_hybrid" in missing:
                adapt_e = None
                if not args.force:
                    try:
                        adapt_e = load_cached_adapt_energy(
                            runs_root,
                            n_sites=n_sites,
                            n_up=n_up,
                            n_down=n_down,
                            pool="cse_density_ops",
                        )
                    except Exception:
                        adapt_e = None
                if adapt_e is None:
                    settings = adapt_settings(n_sites, "cse_density_ops")
                    run_id = f"adapt_cse_{time.time_ns()}"
                    run_dir, meta = make_adapt_run_dir(
                        runs_root,
                        run_id=run_id,
                        n_sites=n_sites,
                        n_up=n_up,
                        n_down=n_down,
                        pool="cse_density_ops",
                        inner_optimizer="hybrid",
                        max_depth=settings["max_depth"],
                        eps_grad=1e-4,
                        eps_energy=1e-3,
                        t=t,
                        u=u,
                        dv=dv,
                        exact_energy=exact,
                        budget_k=None if args.no_budget else float(args.budget_k),
                        max_pauli_terms_measured=max_pauli_terms_measured,
                        max_circuits_executed=max_circuits_executed,
                        max_time_s=max_time_s,
                    )
                    store_run_id = store.start_run(meta)
                    logger = RunStoreLogger(store, store_run_id)
                    status = "completed"
                    payload: dict[str, object] = {}
                    try:
                        adapt_result = run_adapt(
                            n_sites=n_sites,
                            ferm_op=ferm_op,
                            qubit_op=qubit_op,
                            mapper=mapper,
                            n_up=n_up,
                            n_down=n_down,
                            pool_mode="cse_density_ops",
                            logger=logger,
                            max_depth=settings["max_depth"],
                            inner_steps=settings["inner_steps"],
                            warmup_steps=settings["warmup_steps"],
                            polish_steps=settings["polish_steps"],
                            max_pauli_terms_measured=max_pauli_terms_measured,
                            max_circuits_executed=max_circuits_executed,
                            max_time_s=max_time_s,
                            inner_steps_schedule=inner_steps_schedule,
                        )
                        adapt_e_final = float(adapt_result.energy)
                        adapt_e_best = _best_energy_from_log(run_dir) or adapt_e_final
                        adapt_e = float(adapt_e_best)
                        # Persist for downstream reconstruction (fidelity/observables/etc).
                        ops_out = []
                        for op in adapt_result.operators:
                            if isinstance(op, dict) and "name" in op:
                                ops_out.append(str(op["name"]))
                            else:
                                ops_out.append(str(op))
                        payload = {
                            "energy": float(adapt_e_final),
                            "best_energy": float(adapt_e_best),
                            "theta": list(map(float, adapt_result.params)),
                            "operators": ops_out,
                            "pool": "cse_density_ops",
                            "stop_reason": adapt_result.diagnostics.get("stop_reason"),
                            "budget": {
                                "budget_k": None if args.no_budget else float(args.budget_k),
                                "max_pauli_terms_measured": max_pauli_terms_measured,
                                "max_circuits_executed": max_circuits_executed,
                                "max_time_s": max_time_s,
                            },
                            "exact_energy": float(exact),
                            "sites": int(n_sites),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                        }
                    except Exception as exc:
                        status = "error"
                        payload = {
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "pool": "cse_density_ops",
                            "sites": int(n_sites),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                        }
                        raise
                    finally:
                        store.finish_run(store_run_id, status=status, summary_metrics=payload)
                        # Register legacy on-disk artifacts for DB back-references.
                        for kind, p in [
                            ("legacy_meta_json", run_dir / "meta.json"),
                            ("legacy_history_jsonl", run_dir / "history.jsonl"),
                            ("legacy_result_json", run_dir / "result.json"),
                        ]:
                            if p.exists():
                                store.add_artifact(
                                    store_run_id,
                                    kind=kind,
                                    path=str(p),
                                    sha256=sha256_file(p),
                                    bytes=int(p.stat().st_size),
                                    extra=None,
                                )
                row_map[(n_sites, n_up, n_down, "adapt_cse_hybrid")] = {
                    "sites": n_sites,
                    "L": int(n_sites),
                    "n_up": int(n_up),
                    "n_down": int(n_down),
                    "N": int(n_up) + int(n_down),
                    "Sz": 0.5 * (int(n_up) - int(n_down)),
                    "filling": float((int(n_up) + int(n_down)) / float(n_sites)),
                    "ansatz": "adapt_cse_hybrid",
                    "energy": adapt_e,
                    "exact": exact,
                    "delta_e": adapt_e - exact,
                }

            if "adapt_uccsd_hybrid" in missing:
                adapt_uccsd_e = None
                if not args.force:
                    try:
                        adapt_uccsd_e = load_cached_adapt_energy(
                            runs_root,
                            n_sites=n_sites,
                            n_up=n_up,
                            n_down=n_down,
                            pool="uccsd_excitations",
                        )
                    except Exception:
                        adapt_uccsd_e = None
                if adapt_uccsd_e is None:
                    settings = adapt_settings(n_sites, "uccsd_excitations")
                    run_id = f"adapt_uccsd_{time.time_ns()}"
                    run_dir, meta = make_adapt_run_dir(
                        runs_root,
                        run_id=run_id,
                        n_sites=n_sites,
                        n_up=n_up,
                        n_down=n_down,
                        pool="uccsd_excitations",
                        inner_optimizer="hybrid",
                        max_depth=settings["max_depth"],
                        eps_grad=1e-4,
                        eps_energy=1e-3,
                        t=t,
                        u=u,
                        dv=dv,
                        exact_energy=exact,
                        budget_k=None if args.no_budget else float(args.budget_k),
                        max_pauli_terms_measured=max_pauli_terms_measured,
                        max_circuits_executed=max_circuits_executed,
                        max_time_s=max_time_s,
                    )
                    store_run_id = store.start_run(meta)
                    logger = RunStoreLogger(store, store_run_id)
                    status = "completed"
                    payload: dict[str, object] = {}
                    try:
                        adapt_uccsd_result = run_adapt(
                            n_sites=n_sites,
                            ferm_op=ferm_op,
                            qubit_op=qubit_op,
                            mapper=mapper,
                            n_up=n_up,
                            n_down=n_down,
                            pool_mode="uccsd_excitations",
                            logger=logger,
                            max_depth=settings["max_depth"],
                            inner_steps=settings["inner_steps"],
                            warmup_steps=settings["warmup_steps"],
                            polish_steps=settings["polish_steps"],
                            max_pauli_terms_measured=max_pauli_terms_measured,
                            max_circuits_executed=max_circuits_executed,
                            max_time_s=max_time_s,
                            inner_steps_schedule=inner_steps_schedule,
                        )
                        adapt_uccsd_final = float(adapt_uccsd_result.energy)
                        adapt_uccsd_best = _best_energy_from_log(run_dir) or adapt_uccsd_final
                        adapt_uccsd_e = float(adapt_uccsd_best)
                        ops_out = []
                        for op in adapt_uccsd_result.operators:
                            if isinstance(op, dict) and "name" in op:
                                ops_out.append(str(op["name"]))
                            else:
                                ops_out.append(str(op))
                        payload = {
                            "energy": float(adapt_uccsd_final),
                            "best_energy": float(adapt_uccsd_best),
                            "theta": list(map(float, adapt_uccsd_result.params)),
                            "operators": ops_out,
                            "pool": "uccsd_excitations",
                            "stop_reason": adapt_uccsd_result.diagnostics.get("stop_reason"),
                            "budget": {
                                "budget_k": None if args.no_budget else float(args.budget_k),
                                "max_pauli_terms_measured": max_pauli_terms_measured,
                                "max_circuits_executed": max_circuits_executed,
                                "max_time_s": max_time_s,
                            },
                            "exact_energy": float(exact),
                            "sites": int(n_sites),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                        }
                    except Exception as exc:
                        status = "error"
                        payload = {
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "pool": "uccsd_excitations",
                            "sites": int(n_sites),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                        }
                        raise
                    finally:
                        store.finish_run(store_run_id, status=status, summary_metrics=payload)
                        for kind, p in [
                            ("legacy_meta_json", run_dir / "meta.json"),
                            ("legacy_history_jsonl", run_dir / "history.jsonl"),
                            ("legacy_result_json", run_dir / "result.json"),
                        ]:
                            if p.exists():
                                store.add_artifact(
                                    store_run_id,
                                    kind=kind,
                                    path=str(p),
                                    sha256=sha256_file(p),
                                    bytes=int(p.stat().st_size),
                                    extra=None,
                                )
                row_map[(n_sites, n_up, n_down, "adapt_uccsd_hybrid")] = {
                    "sites": n_sites,
                    "L": int(n_sites),
                    "n_up": int(n_up),
                    "n_down": int(n_down),
                    "N": int(n_up) + int(n_down),
                    "Sz": 0.5 * (int(n_up) - int(n_down)),
                    "filling": float((int(n_up) + int(n_down)) / float(n_sites)),
                    "ansatz": "adapt_uccsd_hybrid",
                    "energy": adapt_uccsd_e,
                    "exact": exact,
                    "delta_e": adapt_uccsd_e - exact,
                }

            if "uccsd" in missing:
                log_dir = out_dir / f"logs_uccsd_L{n_sites}_Nup{n_up}_Ndown{n_down}"
                cached_e = None if args.force else _best_energy_from_log(log_dir)
                if cached_e is not None:
                    row_map[(n_sites, n_up, n_down, "uccsd")] = {
                        "sites": n_sites,
                        "L": int(n_sites),
                        "n_up": int(n_up),
                        "n_down": int(n_down),
                        "N": int(n_up) + int(n_down),
                        "Sz": 0.5 * (int(n_up) - int(n_down)),
                        "filling": float((int(n_up) + int(n_down)) / float(n_sites)),
                        "ansatz": "uccsd",
                        "energy": cached_e,
                        "exact": exact,
                        "delta_e": cached_e - exact,
                    }
                else:
                    try:
                        e = run_regular_vqe(
                            n_sites=n_sites,
                            num_particles=(n_up, n_down),
                            qubit_op=qubit_op,
                            mapper=mapper,
                            ansatz_kind="uccsd",
                            exact_energy=exact,
                            log_dir=log_dir,
                            max_pauli_terms_measured=max_pauli_terms_measured,
                            max_circuits_executed=max_circuits_executed,
                            max_time_s=max_time_s,
                            vqe_maxiter_min=int(args.vqe_maxiter_min),
                            vqe_maxiter_per_param=int(args.vqe_maxiter_per_param),
                            vqe_maxiter_cap=int(args.vqe_maxiter_cap),
                            vqe_maxiter_budget_cap=int(args.vqe_maxiter_budget_cap),
                            store=store,
                        )
                    except Exception as exc:
                        row_map[(n_sites, n_up, n_down, "uccsd")] = {
                            "sites": n_sites,
                            "L": int(n_sites),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                            "N": int(n_up) + int(n_down),
                            "Sz": 0.5 * (int(n_up) - int(n_down)),
                            "filling": float((int(n_up) + int(n_down)) / float(n_sites)),
                            "ansatz": "uccsd",
                            "energy": None,
                            "exact": exact,
                            "delta_e": None,
                            "error": str(exc),
                        }
                    else:
                        row_map[(n_sites, n_up, n_down, "uccsd")] = {
                            "sites": n_sites,
                            "L": int(n_sites),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                            "N": int(n_up) + int(n_down),
                            "Sz": 0.5 * (int(n_up) - int(n_down)),
                            "filling": float((int(n_up) + int(n_down)) / float(n_sites)),
                            "ansatz": "uccsd",
                            "energy": e,
                            "exact": exact,
                            "delta_e": e - exact,
                        }

            if "vqe_cse_ops" in missing:
                log_dir = out_dir / f"logs_vqe_cse_ops_L{n_sites}_Nup{n_up}_Ndown{n_down}"
                cached_e = None if args.force else _best_energy_from_log(log_dir)
                if cached_e is not None:
                    row_map[(n_sites, n_up, n_down, "vqe_cse_ops")] = {
                        "sites": n_sites,
                        "L": int(n_sites),
                        "n_up": int(n_up),
                        "n_down": int(n_down),
                        "N": int(n_up) + int(n_down),
                        "Sz": 0.5 * (int(n_up) - int(n_down)),
                        "filling": float((int(n_up) + int(n_down)) / float(n_sites)),
                        "ansatz": "vqe_cse_ops",
                        "energy": cached_e,
                        "exact": exact,
                        "delta_e": cached_e - exact,
                    }
                else:
                    try:
                        e = run_vqe_cse_ops(
                            n_sites=n_sites,
                            n_up=n_up,
                            n_down=n_down,
                            ferm_op=ferm_op,
                            qubit_op=qubit_op,
                            mapper=mapper,
                            exact_energy=exact,
                            runs_root=runs_root,
                            log_dir=log_dir,
                            max_pauli_terms_measured=max_pauli_terms_measured,
                            max_circuits_executed=max_circuits_executed,
                            max_time_s=max_time_s,
                            vqe_maxiter_min=int(args.vqe_maxiter_min),
                            vqe_maxiter_per_param=int(args.vqe_maxiter_per_param),
                            vqe_maxiter_cap=int(args.vqe_maxiter_cap),
                            vqe_maxiter_budget_cap=int(args.vqe_maxiter_budget_cap),
                            store=store,
                        )
                    except Exception as exc:
                        row_map[(n_sites, n_up, n_down, "vqe_cse_ops")] = {
                            "sites": n_sites,
                            "L": int(n_sites),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                            "N": int(n_up) + int(n_down),
                            "Sz": 0.5 * (int(n_up) - int(n_down)),
                            "filling": float((int(n_up) + int(n_down)) / float(n_sites)),
                            "ansatz": "vqe_cse_ops",
                            "energy": None,
                            "exact": exact,
                            "delta_e": None,
                            "error": str(exc),
                        }
                    else:
                        row_map[(n_sites, n_up, n_down, "vqe_cse_ops")] = {
                            "sites": n_sites,
                            "L": int(n_sites),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                            "N": int(n_up) + int(n_down),
                            "Sz": 0.5 * (int(n_up) - int(n_down)),
                            "filling": float((int(n_up) + int(n_down)) / float(n_sites)),
                            "ansatz": "vqe_cse_ops",
                            "energy": e,
                            "exact": exact,
                            "delta_e": e - exact,
                        }
        rows = list(row_map.values())
        compare_path.write_text(json.dumps(rows, indent=2))
        store.add_artifact(
            batch_run_id,
            kind="compare_rows_json",
            path=str(compare_path),
            sha256=sha256_file(compare_path),
            bytes=int(compare_path.stat().st_size),
            extra=None,
        )

        # Per-L bar charts.
        for n_sites in sites:
            n_up, n_down = sectors[int(n_sites)]
            subset = [
                r
                for r in rows
                if int(r.get("sites", -1)) == int(n_sites)
                and int(r.get("n_up", -999)) == int(n_up)
                and int(r.get("n_down", -999)) == int(n_down)
                and r.get("delta_e") is not None
            ]
            if not subset:
                continue
            n_total = int(n_up) + int(n_down)
            sz = 0.5 * (int(n_up) - int(n_down))
            fill = float(n_total) / float(n_sites)
            delta_pairs = [(str(r["ansatz"]), float(r["delta_e"])) for r in subset]
            delta_pairs.sort(key=lambda kv: kv[1])
            labels = [k for k, _v in delta_pairs]
            deltas = np.array([v for _k, v in delta_pairs])

            fig, ax = plt.subplots(figsize=(9, 4.6))
            ax.bar(labels, deltas, color="#264653")
            ax.set_ylabel("ΔE = E_ansatz - E_exact_sector")
            ax.set_title(
                f"Delta-E by ansatz (L={n_sites}, n_up={n_up}, n_down={n_down}, N={n_total}, Sz={sz:+.1f}, filling={fill:.3f})"
            )
            ax.grid(True, axis="y", alpha=0.3)
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.tick_params(axis="x", rotation=20)

            for idx, val in enumerate(deltas):
                ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

            fig.tight_layout()
            out_path = out_dir / f"delta_e_hist_L{n_sites}_Nup{n_up}_Ndown{n_down}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            store.add_artifact(
                batch_run_id,
                kind="compare_plot_png",
                path=str(out_path),
                sha256=sha256_file(out_path),
                bytes=int(out_path.stat().st_size),
                extra={"L": int(n_sites), "n_up": int(n_up), "n_down": int(n_down), "plot": "delta_e"},
            )
            print(f"Saved plot to {out_path}")

            # Absolute error bars.
            abs_pairs = [(str(r["ansatz"]), abs(float(r["delta_e"]))) for r in subset]
            abs_pairs.sort(key=lambda kv: kv[1])
            abs_labels = [k for k, _v in abs_pairs]
            abs_vals = np.array([v for _k, v in abs_pairs])

            fig, ax = plt.subplots(figsize=(9, 4.6))
            ax.bar(abs_labels, abs_vals, color="#2a9d8f")
            ax.set_ylabel("|ΔE| = |E_ansatz - E_exact_sector|")
            ax.set_title(
                f"Absolute error by ansatz (L={n_sites}, n_up={n_up}, n_down={n_down}, N={n_total}, Sz={sz:+.1f}, filling={fill:.3f})"
            )
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=20)

            for idx, val in enumerate(abs_vals):
                ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

            fig.tight_layout()
            out_path = out_dir / f"delta_e_abs_hist_L{n_sites}_Nup{n_up}_Ndown{n_down}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            store.add_artifact(
                batch_run_id,
                kind="compare_plot_png",
                path=str(out_path),
                sha256=sha256_file(out_path),
                bytes=int(out_path.stat().st_size),
                extra={"L": int(n_sites), "n_up": int(n_up), "n_down": int(n_down), "plot": "abs_delta_e"},
            )
            print(f"Saved plot to {out_path}")

            # Relative error bars.
            exact = float(subset[0]["exact"])
            rel_pairs = [
                (str(r["ansatz"]), abs(float(r["delta_e"])) / abs(exact)) for r in subset
            ]
            rel_pairs.sort(key=lambda kv: kv[1])
            rel_labels = [k for k, _v in rel_pairs]
            rel_vals = np.array([v for _k, v in rel_pairs])

            fig, ax = plt.subplots(figsize=(9, 4.6))
            ax.bar(rel_labels, rel_vals, color="#e9c46a")
            ax.set_ylabel("|ΔE| / |E_exact|")
            ax.set_title(
                f"Relative error by ansatz (L={n_sites}, n_up={n_up}, n_down={n_down}, N={n_total}, Sz={sz:+.1f}, filling={fill:.3f})"
            )
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=20)

            for idx, val in enumerate(rel_vals):
                ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

            fig.tight_layout()
            out_path = out_dir / f"delta_e_rel_hist_L{n_sites}_Nup{n_up}_Ndown{n_down}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            store.add_artifact(
                batch_run_id,
                kind="compare_plot_png",
                path=str(out_path),
                sha256=sha256_file(out_path),
                bytes=int(out_path.stat().st_size),
                extra={"L": int(n_sites), "n_up": int(n_up), "n_down": int(n_down), "plot": "rel_delta_e"},
            )
            print(f"Saved plot to {out_path}")

        print(f"Saved comparison rows to {compare_path}")
        store.finish_run(
            batch_run_id,
            status="completed",
            summary_metrics={
                "rows": int(len(rows)),
                "sites": list(map(int, sites)),
                "out_dir": str(out_dir),
            },
        )
    except Exception as exc:
        # Keep the batch run consistent in the DB even if a single sub-run fails.
        try:
            store.finish_run(
                batch_run_id,
                status="error",
                summary_metrics={"error": str(exc), "traceback": traceback.format_exc()},
            )
        except Exception:
            pass
        raise
    finally:
        # Ensure any pending SQLite writes are flushed.
        if hasattr(store, "close"):
            store.close()


if __name__ == "__main__":
    main()
