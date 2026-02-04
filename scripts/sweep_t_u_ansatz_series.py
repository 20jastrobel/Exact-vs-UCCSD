#!/usr/bin/env python3
"""Compute regular-ansatz sweeps and append them to the sweep JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from pydephasing.quantum.ansatz import build_ansatz
from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)


DEFAULT_ANSATZ = ["uccsd"]


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sweep json: {path}")
    return json.loads(path.read_text())


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _as_site_key(n_sites: int) -> str:
    return str(int(n_sites))


def _ensure_site_arrays(store: dict[str, list[float | None]], n_sites: int, n_points: int) -> list[float | None]:
    key = _as_site_key(n_sites)
    arr = store.get(key)
    if arr is None or len(arr) != n_points:
        arr = [None] * n_points
        store[key] = arr
    return arr


def run_regular_vqe(
    *,
    n_sites: int,
    num_particles: tuple[int, int],
    qubit_op,
    mapper,
    ansatz_kind: str,
    reps: int,
    seed: int,
    maxiter: int,
) -> float:
    estimator = StatevectorEstimator()
    ansatz = build_ansatz(
        ansatz_kind,
        qubit_op.num_qubits,
        reps,
        mapper,
        n_sites=n_sites,
        num_particles=num_particles,
    )
    rng = np.random.default_rng(seed)
    if ansatz_kind == "uccsd":
        initial_point = np.zeros(ansatz.num_parameters, dtype=float)
    else:
        initial_point = rng.random(ansatz.num_parameters) * 2 * np.pi
    optimizer = COBYLA(maxiter=maxiter)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
    )
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    return float(np.real(result.eigenvalue))


def _compute_point(
    *,
    n_sites: int,
    t: float,
    u: float,
    dv: float,
    n_up: int,
    n_down: int,
    ansatz_kinds: list[str],
    seed: int,
    reps: int,
    maxiter: int,
) -> dict[str, float | None]:
    ferm_op = build_fermionic_hubbard(
        n_sites=n_sites,
        t=t,
        u=u,
        edges=default_1d_chain_edges(n_sites, periodic=False),
        v=[-dv / 2, dv / 2] if n_sites == 2 else None,
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    results: dict[str, float | None] = {}
    for ansatz in ansatz_kinds:
        try:
            energy = run_regular_vqe(
                n_sites=n_sites,
                num_particles=(n_up, n_down),
                qubit_op=qubit_op,
                mapper=mapper,
                ansatz_kind=ansatz,
                reps=reps,
                seed=seed,
                maxiter=maxiter,
            )
        except Exception as exc:
            print(f"Failed {ansatz} at L={n_sites}, t={t:.3f}, U={u:.3f}: {exc}")
            energy = None
        results[ansatz] = energy
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-json", type=str, default="runs/sweeps/sweep_t_u_multi.json")
    ap.add_argument("--sites", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--ansatz", nargs="+", type=str, default=DEFAULT_ANSATZ)
    ap.add_argument("--n-up", type=int, default=None)
    ap.add_argument("--n-down", type=int, default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxiter", type=int, default=150)
    args = ap.parse_args()

    sweep_path = Path(args.sweep_json)
    payload = _load_payload(sweep_path)

    t_values = list(map(float, payload.get("t_values", [])))
    u_values = list(map(float, payload.get("u_values", [])))
    if not t_values or not u_values:
        raise ValueError("Sweep payload missing t_values or u_values.")

    u_fixed = float(payload.get("u_fixed", 0.0))
    t_fixed = float(payload.get("t_fixed", 0.0))
    dv = float(payload.get("dv", 0.0))
    n_up = int(payload.get("n_up") if args.n_up is None else args.n_up)
    n_down = int(payload.get("n_down") if args.n_down is None else args.n_down)

    ansatz_kinds = [a.strip() for a in args.ansatz if a.strip()]
    if not ansatz_kinds:
        raise ValueError("No ansatz specified.")

    sites = sorted(set(int(s) for s in args.sites))
    payload["sites"] = sorted(set(int(s) for s in payload.get("sites", [])) | set(sites))

    for ansatz in ansatz_kinds:
        payload.setdefault(f"{ansatz}_vs_t", {})
        payload.setdefault(f"{ansatz}_vs_u", {})

    _write_payload(sweep_path, payload)

    for n_sites in sites:
        t_series = {
            ansatz: _ensure_site_arrays(payload[f"{ansatz}_vs_t"], n_sites, len(t_values))
            for ansatz in ansatz_kinds
        }
        u_series = {
            ansatz: _ensure_site_arrays(payload[f"{ansatz}_vs_u"], n_sites, len(u_values))
            for ansatz in ansatz_kinds
        }

        for idx, t in enumerate(t_values):
            missing = [a for a in ansatz_kinds if t_series[a][idx] is None]
            if not missing:
                continue
            print(f"Computing L={n_sites} at t={t:.3f}, U={u_fixed:.3f} for {', '.join(missing)}")
            results = _compute_point(
                n_sites=n_sites,
                t=float(t),
                u=float(u_fixed),
                dv=float(dv),
                n_up=n_up,
                n_down=n_down,
                ansatz_kinds=missing,
                seed=int(args.seed),
                reps=int(args.reps),
                maxiter=int(args.maxiter),
            )
            for ansatz, energy in results.items():
                t_series[ansatz][idx] = None if energy is None else float(energy)
            _write_payload(sweep_path, payload)

        for idx, u in enumerate(u_values):
            missing = [a for a in ansatz_kinds if u_series[a][idx] is None]
            if not missing:
                continue
            print(f"Computing L={n_sites} at t={t_fixed:.3f}, U={u:.3f} for {', '.join(missing)}")
            results = _compute_point(
                n_sites=n_sites,
                t=float(t_fixed),
                u=float(u),
                dv=float(dv),
                n_up=n_up,
                n_down=n_down,
                ansatz_kinds=missing,
                seed=int(args.seed),
                reps=int(args.reps),
                maxiter=int(args.maxiter),
            )
            for ansatz, energy in results.items():
                u_series[ansatz][idx] = None if energy is None else float(energy)
            _write_payload(sweep_path, payload)

    print(f"Updated sweep data at {sweep_path}")


if __name__ == "__main__":
    main()
