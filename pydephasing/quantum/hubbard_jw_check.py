#!/usr/bin/env python3
"""Hubbard dimer JW mapping, local VQE warm start, and IBM eval-only modes."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.primitives import StatevectorEstimator

from pydephasing.quantum.hardware_efficient_ansatz import (
    build_hardware_efficient_ansatz,
)
from pydephasing.quantum.ibm_runtime_tools import (
    choose_backend,
    get_runtime_service,
    make_estimator,
    runtime_context,
)
from pydephasing.quantum.ibm_sim_tools import (
    build_aer_estimator_for_backend,
    load_fake_backend,
)


@dataclass
class LocalVQEResult:
    energy: float
    params: list[float]
    seconds: float


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.strip().lower()
    if val in {"1", "true", "yes", "y"}:
        return True
    if val in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_occ_list(value: str) -> list[int]:
    if value is None:
        return []
    text = value.strip()
    if not text:
        return []
    items = [item.strip() for item in text.split(",") if item.strip()]
    return [int(item) for item in items]


def _normalize_ansatz(name: str) -> str:
    if name in {"hea", "hardware", "hardware-efficient"}:
        return "hea"
    return name


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


def build_qubit_hamiltonian(t: float, u: float, dv: float) -> tuple[SparsePauliOp, JordanWignerMapper]:
    ferm_op = build_fermionic_hubbard(t, u, dv)
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(ferm_op).simplify(atol=1e-12)
    return qubit_op, mapper


def exact_ground_energy(qubit_op: SparsePauliOp) -> float:
    mat = qubit_op.to_matrix()
    evals = np.linalg.eigvalsh(mat)
    return float(np.min(np.real(evals)))


def build_ansatz(
    ansatz_name: str,
    num_qubits: int,
    reps: int,
    mapper: JordanWignerMapper,
    *,
    hea_rotation: str = "ry",
    hea_entanglement: str = "linear",
    hea_occ: Optional[list[int]] = None,
) -> QuantumCircuit:
    if ansatz_name == "clustered":
        return efficient_su2(
            num_qubits,
            su2_gates=["ry", "rz"],
            reps=reps,
            entanglement="linear",
        )

    if ansatz_name == "uccsd":
        num_spatial_orbitals = 2
        num_particles = (1, 1)
        initial_state = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
        )
        return UCCSD(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
            reps=reps,
            initial_state=initial_state,
        )

    if ansatz_name == "hea":
        return build_hardware_efficient_ansatz(
            n_qubits=num_qubits,
            reps=reps,
            entanglement=hea_entanglement,
            initial_occupations=hea_occ,
            rotation=hea_rotation,
        )

    raise ValueError(f"Unknown ansatz: {ansatz_name}")


def _build_optimizer(name: str, maxiter: int):
    if name == "L_BFGS_B":
        return L_BFGS_B(maxiter=maxiter)
    if name == "SLSQP":
        return SLSQP(maxiter=maxiter)
    if name == "COBYLA":
        return COBYLA(maxiter=maxiter)
    raise ValueError(f"Unknown optimizer: {name}")


def run_vqe_with_estimator(
    qubit_op: SparsePauliOp,
    ansatz: QuantumCircuit,
    estimator,
    *,
    restarts: int,
    maxiter: int,
    optimizer_name: str,
    seed: int,
) -> LocalVQEResult:
    rng = np.random.default_rng(seed)

    best_energy: Optional[float] = None
    best_params: list[float] = []
    start = time.perf_counter()

    for restart_idx in range(restarts):
        optimizer = _build_optimizer(optimizer_name, maxiter)
        initial_point = rng.random(ansatz.num_parameters) * 2 * np.pi
        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
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
    qubit_op: SparsePauliOp,
    ansatz: QuantumCircuit,
    *,
    restarts: int,
    maxiter: int,
    optimizer_name: str,
    seed: int,
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
    )


def save_params(path: str, params: Sequence[float], meta: dict) -> None:
    payload = {
        "theta": list(map(float, params)),
        **meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_params(path: str) -> list[float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "theta" in data:
        return list(map(float, data["theta"]))
    if "params" in data:
        return list(map(float, data["params"]))
    raise KeyError("No params found in JSON; expected 'theta' or 'params'.")


def compile_ansatz_and_op(
    ansatz: QuantumCircuit,
    qubit_op: SparsePauliOp,
    backend,
    *,
    opt_level: int = 1,
) -> tuple[QuantumCircuit, SparsePauliOp]:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
    isa_circ = pm.run(ansatz)
    isa_op = qubit_op.apply_layout(isa_circ.layout)
    return isa_circ, isa_op


def compile_ansatz_and_op_for_sim(
    ansatz: QuantumCircuit,
    qubit_op: SparsePauliOp,
    backend,
    *,
    opt_level: int = 1,
) -> tuple[QuantumCircuit, SparsePauliOp, object, Optional[list[str]]]:
    from qiskit import transpile

    num_qubits = ansatz.num_qubits
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is not None:
        try:
            coupling_map = coupling_map.reduce(list(range(num_qubits)))
        except Exception:
            pass

    basis_gates = None
    try:
        cfg = backend.configuration()
        basis_gates = list(cfg.basis_gates) if cfg is not None else None
    except Exception:
        basis_gates = getattr(backend, "basis_gates", None)

    isa_circ = transpile(
        ansatz,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=opt_level,
    )
    isa_op = qubit_op.apply_layout(isa_circ.layout)
    return isa_circ, isa_op, coupling_map, basis_gates


def ibm_estimate_energies(
    qubit_op: SparsePauliOp,
    ansatz: QuantumCircuit,
    theta_list: list[list[float]],
    backend,
    *,
    shots: int,
    resilience: int,
    max_exec: int,
    opt_level: int = 1,
):
    isa_circ, isa_op = compile_ansatz_and_op(
        ansatz,
        qubit_op,
        backend,
        opt_level=opt_level,
    )

    estimator = make_estimator(backend, shots, resilience, max_exec)

    job = estimator.run([(isa_circ, isa_op, theta_list)])
    result = job.result(timeout=max_exec + 30)
    pub = result[0]
    evs = list(pub.data.evs)
    return evs, job.job_id()


def ibm_optimize_on_hardware(
    qubit_op: SparsePauliOp,
    ansatz: QuantumCircuit,
    backend,
    service,
    *,
    rounds: int,
    k: int,
    sigma: float,
    shrink: float,
    seed: int,
    shots_opt: int,
    shots_final: int,
    resilience_opt: int,
    resilience_final: int,
    max_exec_opt: int,
    max_exec_final: int,
    opt_level: int = 1,
):
    if k < 1:
        raise ValueError("--ibm-opt-k must be >= 1")
    if rounds < 1:
        raise ValueError("--ibm-opt-rounds must be >= 1")

    isa_circ, isa_op = compile_ansatz_and_op(
        ansatz,
        qubit_op,
        backend,
        opt_level=opt_level,
    )

    num_params = isa_circ.num_parameters
    if num_params == 0:
        raise RuntimeError("Ansatz has no parameters to optimize.")

    rng = np.random.default_rng(seed)
    center = np.zeros(num_params, dtype=float)
    job_ids: list[str] = []
    best_energy = None

    with runtime_context(service, backend, prefer_batch=True, prefer_session=False) as mode_obj:
        for round_idx in range(rounds):
            candidates = [center]
            for _ in range(k - 1):
                candidates.append(center + sigma * rng.normal(size=num_params))

            theta_list = [list(map(float, cand)) for cand in candidates]
            estimator = make_estimator(mode_obj, shots_opt, resilience_opt, max_exec_opt)
            job = estimator.run([(isa_circ, isa_op, theta_list)])
            job_ids.append(job.job_id())
            result = job.result(timeout=max_exec_opt + 30)
            evs = np.asarray(result[0].data.evs, dtype=float)
            idx = int(np.argmin(evs))
            center = np.asarray(candidates[idx], dtype=float)
            best_energy = float(evs[idx])
            print(
                f"ibm-opt round {round_idx + 1}/{rounds}: "
                f"best_energy={best_energy:.10f} sigma={sigma:.6f}"
            )
            sigma *= shrink

        final_theta = [list(map(float, center))]
        estimator = make_estimator(mode_obj, shots_final, resilience_final, max_exec_final)
        job = estimator.run([(isa_circ, isa_op, final_theta)])
        job_ids.append(job.job_id())
        result = job.result(timeout=max_exec_final + 30)
        final_energy = float(np.asarray(result[0].data.evs, dtype=float)[0])

    return final_energy, best_energy, list(center), job_ids


def _run_self_test() -> None:
    qubit_op, mapper = build_qubit_hamiltonian(1.0, 2.0, 0.5)
    exact = exact_ground_energy(qubit_op)
    ansatz = build_ansatz("clustered", qubit_op.num_qubits, 1, mapper)
    result = run_local_vqe(
        qubit_op,
        ansatz,
        restarts=1,
        maxiter=5,
        optimizer_name="COBYLA",
        seed=0,
    )
    print("SELF-TEST")
    print(f"Exact: {exact:.10f}")
    print(f"VQE:   {result.energy:.10f}")
    print("SELF-TEST OK")


def _run_vqe_on_ibm_sim(
    qubit_op: SparsePauliOp,
    ansatz: QuantumCircuit,
    *,
    fake_backend_name: str,
    shots: int,
    seed_sim: int,
    noisy: bool,
    restarts: int,
    maxiter: int,
    optimizer_name: str,
    seed: int,
    exact: float,
) -> None:
    backend = load_fake_backend(fake_backend_name)
    isa_circ, isa_op, coupling_map, basis_gates = compile_ansatz_and_op_for_sim(
        ansatz,
        qubit_op,
        backend,
        opt_level=1,
    )
    estimator, _sim = build_aer_estimator_for_backend(
        backend,
        shots=shots,
        seed=seed_sim,
        noisy=noisy,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
    )

    print(
        f"SIM backend={fake_backend_name} noisy={noisy} shots={shots}"
    )

    result = run_vqe_with_estimator(
        isa_op,
        isa_circ,
        estimator,
        restarts=restarts,
        maxiter=maxiter,
        optimizer_name=optimizer_name,
        seed=seed,
    )

    delta = result.energy - exact
    print(f"VQE energy: {result.energy:.10f}")
    print(f"E_exact:    {exact:.10f}")
    print(f"Delta:      {delta:.10e}")

    default_tol = 0.5 if noisy else 0.05
    tol_env = os.environ.get("EXACT_TOL")
    tol = float(tol_env) if tol_env is not None else default_tol
    if abs(delta) > tol:
        msg = f"|delta|={abs(delta):.6f} exceeds tol={tol}"
        if tol_env is not None:
            raise RuntimeError(msg)
        print(f"Warning: {msg}")


def _build_theta_list(
    theta_best: Sequence[float],
    *,
    k: int,
    sigma: float,
    seed: int,
) -> list[list[float]]:
    if k < 1:
        raise ValueError("--ibm-k must be >= 1")
    rng = np.random.default_rng(seed)
    theta_best = np.asarray(theta_best, dtype=float)

    thetas = [theta_best]
    for _ in range(k - 1):
        jitter = sigma * rng.standard_normal(theta_best.shape)
        candidate = theta_best + jitter
        wrapped = (candidate + np.pi) % (2 * np.pi) - np.pi
        thetas.append(wrapped)

    return [list(map(float, t)) for t in thetas]


def _print_queue_warning(backend) -> None:
    try:
        status = backend.status()
    except Exception:
        return
    pending = getattr(status, "pending_jobs", None)
    if pending is not None and pending > 100:
        print(
            f"Warning: backend {backend.name} has pending_jobs={pending}. "
            "Consider switching backends."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hubbard dimer JW mapping with local VQE warm-start and IBM eval-only modes."
    )
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=2.0)
    parser.add_argument("--dv", type=float, default=0.5)

    parser.add_argument(
        "--ansatz",
        choices=[
            "clustered",
            "uccsd",
            "hea",
            "hardware",
            "hardware-efficient",
        ],
        default=None,
    )
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--hea-rotation", choices=["ry", "ryrz"], default="ry")
    parser.add_argument(
        "--hea-entanglement",
        choices=["linear", "full", "circular"],
        default="linear",
    )
    parser.add_argument("--hea-hf-occ", type=str, default="0,2")
    parser.add_argument("--self-test", action="store_true")

    parser.add_argument("--local-vqe", action="store_true")
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument(
        "--optimizer",
        choices=["L_BFGS_B", "SLSQP", "COBYLA"],
        default="L_BFGS_B",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-params", type=str, default=None)
    parser.add_argument("--load-params", type=str, default=None)

    parser.add_argument("--ibm-eval", action="store_true")
    parser.add_argument("--ibm-search", action="store_true")
    parser.add_argument("--ibm-opt", action="store_true")
    parser.add_argument("--ibm-sim", action="store_true")
    parser.add_argument("--fake-backend", type=str, default="FakeBrisbane")
    parser.add_argument(
        "--sim-noise",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--seed-sim", type=int, default=11)
    parser.add_argument("--ibm-opt-rounds", type=int, default=3)
    parser.add_argument("--ibm-opt-k", type=int, default=3)
    parser.add_argument("--ibm-opt-sigma", type=float, default=0.4)
    parser.add_argument("--ibm-opt-shrink", type=float, default=0.5)
    parser.add_argument("--shots-opt", type=int, default=64)
    parser.add_argument("--shots-final", type=int, default=512)
    parser.add_argument("--resilience-opt", type=int, default=0)
    parser.add_argument("--resilience-final", type=int, default=1)
    parser.add_argument("--max-exec-opt", type=int, default=120)
    parser.add_argument("--max-exec-final", type=int, default=240)
    parser.add_argument("--ibm-k", type=int, default=1)
    parser.add_argument("--ibm-sigma", type=float, default=0.05)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--resilience", type=int, default=0)
    parser.add_argument("--max-exec", type=int, default=120)

    parser.add_argument(
        "--no-backend-scan",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--force-hardware", action="store_true")

    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    if args.ibm_sim and (args.ibm_opt or args.ibm_eval or args.ibm_search):
        raise RuntimeError("ibm-sim cannot be combined with other IBM runtime modes.")
    if args.ibm_sim and args.local_vqe:
        raise RuntimeError("ibm-sim cannot be combined with local-vqe.")

    if args.ibm_opt and (args.ibm_eval or args.ibm_search):
        raise RuntimeError("ibm-opt cannot be combined with ibm-eval or ibm-search.")
    if args.ibm_opt and args.local_vqe:
        raise RuntimeError("ibm-opt cannot be combined with local-vqe.")

    ansatz_choice = args.ansatz
    if ansatz_choice is None:
        if args.ibm_opt:
            ansatz_choice = "hea"
        elif args.ibm_sim:
            ansatz_choice = "uccsd"
        else:
            ansatz_choice = "clustered"
    ansatz_choice = _normalize_ansatz(ansatz_choice)

    reps = args.reps
    if reps is None:
        if ansatz_choice == "clustered":
            reps = 2
        elif ansatz_choice == "uccsd":
            reps = 1
        else:
            reps = 1

    qubit_op, mapper = build_qubit_hamiltonian(args.t, args.u, args.dv)
    exact = exact_ground_energy(qubit_op)
    hea_occ = _parse_occ_list(args.hea_hf_occ)
    try:
        ansatz = build_ansatz(
            ansatz_choice,
            qubit_op.num_qubits,
            reps,
            mapper,
            hea_rotation=args.hea_rotation,
            hea_entanglement=args.hea_entanglement,
            hea_occ=hea_occ if hea_occ else None,
        )
    except Exception as exc:
        if args.ibm_sim and ansatz_choice == "uccsd":
            print(f"Warning: UCCSD failed; falling back to clustered ({exc}).")
            ansatz_choice = "clustered"
            if args.reps is None:
                reps = 2
            ansatz = build_ansatz(
                ansatz_choice,
                qubit_op.num_qubits,
                reps,
                mapper,
                hea_rotation=args.hea_rotation,
                hea_entanglement=args.hea_entanglement,
                hea_occ=hea_occ if hea_occ else None,
            )
        else:
            raise

    save_params_path = args.save_params
    if args.ibm_opt and save_params_path is None:
        save_params_path = "hea_ibm_params.json"

    if args.ibm_sim:
        _run_vqe_on_ibm_sim(
            qubit_op,
            ansatz,
            fake_backend_name=args.fake_backend,
            shots=args.shots,
            seed_sim=args.seed_sim,
            noisy=args.sim_noise,
            restarts=args.restarts,
            maxiter=args.maxiter,
            optimizer_name=args.optimizer,
            seed=args.seed,
            exact=exact,
        )
        return

    theta_best: Optional[list[float]] = None
    if args.load_params:
        theta_best = load_params(args.load_params)

    needs_warm_start = args.local_vqe or ((args.ibm_eval or args.ibm_search) and theta_best is None)
    if needs_warm_start:
        local_result = run_local_vqe(
            qubit_op,
            ansatz,
            restarts=args.restarts,
            maxiter=args.maxiter,
            optimizer_name=args.optimizer,
            seed=args.seed,
        )
        theta_best = local_result.params
        print(f"local-vqe best energy: {local_result.energy:.10f}")
        print(f"exact energy:          {exact:.10f}")
        print(f"delta:                 {local_result.energy - exact:.10e}")
        print(f"elapsed:               {local_result.seconds:.2f}s")

        if args.save_params:
            save_params(
                args.save_params,
                theta_best,
                {
                    "t": args.t,
                    "u": args.u,
                    "dv": args.dv,
                    "ansatz": ansatz_choice,
                    "reps": reps,
                },
            )

    if args.ibm_opt:
        backend_arg = args.backend.strip() if args.backend else None
        if not backend_arg and args.no_backend_scan:
            raise RuntimeError(
                "Set IBM_BACKEND or pass --backend to avoid API scans/timeouts."
            )

        if ansatz_choice != "hea":
            print(f"Warning: ibm-opt is intended for HEA; using {ansatz_choice}.")

        service = get_runtime_service()
        backend = choose_backend(
            service,
            backend_arg,
            no_scan=args.no_backend_scan,
            force_hardware=args.force_hardware,
        )

        active_instance = None
        try:
            active_instance = service.active_instance()
        except Exception:
            active_instance = None

        if active_instance:
            print(f"Active instance: {active_instance}")
        print(f"Backend: {backend.name}")
        _print_queue_warning(backend)

        final_energy, best_energy, theta_best, job_ids = ibm_optimize_on_hardware(
            qubit_op,
            ansatz,
            backend,
            service,
            rounds=args.ibm_opt_rounds,
            k=args.ibm_opt_k,
            sigma=args.ibm_opt_sigma,
            shrink=args.ibm_opt_shrink,
            seed=args.seed,
            shots_opt=args.shots_opt,
            shots_final=args.shots_final,
            resilience_opt=args.resilience_opt,
            resilience_final=args.resilience_final,
            max_exec_opt=args.max_exec_opt,
            max_exec_final=args.max_exec_final,
        )

        for idx, job_id in enumerate(job_ids, start=1):
            print(f"Job ID {idx}: {job_id}")

        print(f"E_ibm_opt:   {best_energy:.10f}")
        print(f"E_ibm_final: {final_energy:.10f}")
        print(f"E_exact:     {exact:.10f}")
        print(f"Delta:       {final_energy - exact:.10e}")

        if save_params_path:
            save_params(
                save_params_path,
                theta_best,
                {
                    "t": args.t,
                    "u": args.u,
                    "dv": args.dv,
                    "ansatz": ansatz_choice,
                    "reps": reps,
                    "backend": backend.name,
                    "E_final": final_energy,
                    "E_opt": best_energy,
                    "shots_final": args.shots_final,
                    "resilience_final": args.resilience_final,
                },
            )
        return

    if args.ibm_eval or args.ibm_search:
        backend_arg = args.backend.strip() if args.backend else None
        if not backend_arg and args.no_backend_scan:
            raise RuntimeError(
                "Set IBM_BACKEND or pass --backend to avoid API scans/timeouts."
            )

        if theta_best is None:
            raise RuntimeError("No parameters available; use --local-vqe or --load-params.")

        theta_list = _build_theta_list(
            theta_best,
            k=args.ibm_k if args.ibm_search else 1,
            sigma=args.ibm_sigma,
            seed=args.seed,
        )

        service = get_runtime_service()
        backend = choose_backend(
            service,
            backend_arg,
            no_scan=args.no_backend_scan,
            force_hardware=args.force_hardware,
        )

        active_instance = None
        try:
            active_instance = service.active_instance()
        except Exception:
            active_instance = None

        if active_instance:
            print(f"Active instance: {active_instance}")
        print(f"Backend: {backend.name}")
        _print_queue_warning(backend)

        evs, job_id = ibm_estimate_energies(
            qubit_op,
            ansatz,
            theta_list,
            backend,
            shots=args.shots,
            resilience=args.resilience,
            max_exec=args.max_exec,
        )

        if args.ibm_eval:
            energy = float(np.real(evs[0]))
            print(f"Job ID: {job_id}")
            print(f"E_ibm:  {energy:.10f}")
            print(f"E_exact:{exact:.10f}")
            print(f"Delta:  {energy - exact:.10e}")
        else:
            evs_np = np.asarray(evs, dtype=float)
            idx = int(np.argmin(evs_np))
            energy = float(evs_np[idx])
            print(f"Job ID: {job_id}")
            print(f"E_ibm_min: {energy:.10f}")
            print(f"Best index: {idx}")
            print(f"E_exact:   {exact:.10f}")
            print(f"Delta:     {energy - exact:.10e}")


if __name__ == "__main__":
    main()
