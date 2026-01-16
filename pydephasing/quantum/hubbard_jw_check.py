#!/usr/bin/env python3
"""Verify JW mapping for 2-site Hubbard model against reference image Hamiltonian.

Reference Hamiltonian transcription (from image):
H = - (t/2) ( X0 X1 + Y0 Y1 + X2 X3 + Y2 Y3 )
    + (Δv/4) ( Z0 + Z2 - Z1 - Z3 )
    + (U/4) ( Z0 Z2 + Z1 Z3 )
    - (U/4) ( Z0 + Z1 + Z2 + Z3 )
    + (U/2) I.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

REFERENCE_HAMILTONIAN = (
    "H = - (t/2) ( X0 X1 + Y0 Y1 + X2 X3 + Y2 Y3 )\n"
    "    + (Δv/4) ( Z0 + Z2 - Z1 - Z3 )\n"
    "    + (U/4) ( Z0 Z2 + Z1 Z3 )\n"
    "    - (U/4) ( Z0 + Z1 + Z2 + Z3 )\n"
    "    + (U/2) I."
)

DEFAULT_MAPPING_PARAMS = [
    (1.0, 2.0, 0.5),
    (1.0, 0.0, 0.0),
    (0.0, 2.0, 0.5),
]

DEFAULT_UP_QUBITS = [0, 1]
DEFAULT_DOWN_QUBITS = [2, 3]


def _simplify_sparse_pauli(op, atol: float = 1e-12):
    try:
        return op.simplify(atol=atol)
    except TypeError:
        return op.simplify()


def _sort_pauli_list(pauli_list):
    return sorted(pauli_list, key=lambda item: item[0])


def _sorted_terms_str(pauli_list):
    return [f"{label}: {coeff}" for label, coeff in _sort_pauli_list(pauli_list)]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run local noiseless VQE (default: enabled).",
    )
    parser.add_argument("--ibm", action="store_true")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=2.0)
    parser.add_argument("--dv", type=float, default=0.5)
    parser.add_argument(
        "--ansatz",
        type=str,
        choices=["clustered", "uccsd", "both"],
        default="both",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Ansatz repetitions (default: clustered=2, uccsd=1).",
    )
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["SLSQP", "L_BFGS_B", "COBYLA"],
        default="L_BFGS_B",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mapping-trials", type=int, default=0)
    parser.add_argument("--mapping-seed", type=int, default=11)
    return parser.parse_args()


def _parse_index_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _default_sector(up_qubits, down_qubits):
    return {
        "n_electrons": 2,
        "sz": 0.0,
        "up_qubits": up_qubits,
        "down_qubits": down_qubits,
    }


def _parse_exact_sector(default_sector):
    raw = os.environ.get("EXACT_SECTOR")
    if raw is None:
        return default_sector
    raw = raw.strip().lower()
    if not raw or raw in {"none", "full", "global", "all"}:
        return None

    sector: dict[str, float | int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip()
        if key in {"n", "ne", "n_electrons", "electrons"}:
            sector["n_electrons"] = int(value)
        elif key in {"sz", "s_z"}:
            sector["sz"] = float(value)
        else:
            print(f"Unknown EXACT_SECTOR key: {key}")

    return sector or default_sector


def _build_hubbard_qubit_op(t: float, u: float, dv: float, mode: str = "JW"):
    from pydephasing.quantum.hubbard_dimer import build_hubbard_dimer_qubit_polynomial
    from pydephasing.quantum.qiskit_bridge import pauli_polynomial_to_sparse_pauli_op

    poly = build_hubbard_dimer_qubit_polynomial(t, u, dv, mode=mode)
    return poly, pauli_polynomial_to_sparse_pauli_op(poly)


def _assert_ops_close(op_a, op_b, label_a, label_b, idx, t, u, dv, tol=1e-10):
    diff = _simplify_sparse_pauli(op_a - op_b, atol=1e-12)
    diff_norm = np.linalg.norm(diff.to_matrix())
    if diff_norm > tol:
        print(f"Mismatch on trial {idx} ({label_a} vs {label_b})")
        print(f"t={t}, u={u}, dv={dv}")
        print(f"{label_a} terms:")
        print("\n".join(_sorted_terms_str(op_a.to_list())))
        print(f"{label_b} terms:")
        print("\n".join(_sorted_terms_str(op_b.to_list())))
        print("Diff terms:")
        print("\n".join(_sorted_terms_str(diff.to_list())))
        raise AssertionError(f"Operator norm mismatch: {diff_norm}")

    evals_a = np.linalg.eigvalsh(op_a.to_matrix())
    evals_b = np.linalg.eigvalsh(op_b.to_matrix())
    if not np.allclose(np.sort(evals_a), np.sort(evals_b), atol=1e-10):
        print(f"Eigenvalue mismatch on trial {idx} ({label_a} vs {label_b})")
        print(f"t={t}, u={u}, dv={dv}")
        print(f"{label_a} terms:")
        print("\n".join(_sorted_terms_str(op_a.to_list())))
        print(f"{label_b} terms:")
        print("\n".join(_sorted_terms_str(op_b.to_list())))
        raise AssertionError("Eigenvalue mismatch")


def _run_mapping_equivalence(param_list) -> None:
    from pydephasing.quantum.hubbard_dimer import (
        build_hubbard_dimer_qiskit_nature_jw,
        build_hubbard_dimer_jw_polynomial,
        build_reference_hubbard_image_polynomial,
    )
    from pydephasing.quantum.qiskit_bridge import pauli_polynomial_to_sparse_pauli_op

    qn_available = True

    for idx, (t, u, dv) in enumerate(param_list):
        ref_poly = build_reference_hubbard_image_polynomial(t, u, dv)
        ref_op = pauli_polynomial_to_sparse_pauli_op(ref_poly)

        jw_poly = build_hubbard_dimer_jw_polynomial(t, u, dv)
        jw_op = pauli_polynomial_to_sparse_pauli_op(jw_poly)

        op_qn = None
        if qn_available:
            try:
                op_qn = build_hubbard_dimer_qiskit_nature_jw(t, u, dv)
            except ImportError:
                qn_available = False

        _assert_ops_close(jw_op, ref_op, "poly", "ref", idx, t, u, dv)
        if op_qn is not None:
            _assert_ops_close(op_qn, ref_op, "qiskit_nature", "ref", idx, t, u, dv)
            _assert_ops_close(op_qn, jw_op, "qiskit_nature", "poly", idx, t, u, dv)

    print("PASS")


def _run_mapping_regression() -> None:
    print("Mapping regression checks...")
    _run_mapping_equivalence(DEFAULT_MAPPING_PARAMS)


def _run_mapping_random(trials: int, seed: int) -> None:
    if trials <= 0:
        return
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(trials):
        t = float(rng.uniform(-2.0, 2.0))
        u = float(rng.uniform(-2.0, 2.0))
        dv = float(rng.uniform(-2.0, 2.0))
        params.append((t, u, dv))
    print("Mapping random checks...")
    _run_mapping_equivalence(params)


def _local_estimator():
    try:
        from qiskit.primitives import StatevectorEstimator

        return StatevectorEstimator()
    except Exception:
        try:
            from qiskit_aer.primitives import EstimatorV2

            return EstimatorV2(options={"backend_options": {"method": "statevector"}})
        except Exception:
            from qiskit.primitives import Estimator

            return Estimator()


def _build_optimizer(name, maxiter, tol):
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP

    name_norm = name.strip().upper()
    if name_norm == "SLSQP":
        return SLSQP(maxiter=maxiter, ftol=tol, tol=tol)
    if name_norm in {"L_BFGS_B", "L-BFGS-B", "LBFGSB"}:
        return L_BFGS_B(maxiter=maxiter, ftol=tol)
    if name_norm == "COBYLA":
        return COBYLA(maxiter=maxiter, tol=tol)
    raise ValueError(f"Unknown optimizer: {name}")


def _summarize_restart_energies(energies):
    if not energies:
        print("VQE restarts: no energies recorded.")
        return
    sorted_energies = sorted(energies)
    mean = float(np.mean(sorted_energies))
    print(f"VQE restarts energies (sorted): {sorted_energies}")
    print(
        "VQE restarts summary: "
        f"min={sorted_energies[0]}, max={sorted_energies[-1]}, mean={mean}"
    )


def _print_uccsd_details(ansatz):
    print(f"UCCSD parameters: {getattr(ansatz, 'num_parameters', 'n/a')}")
    excitation_list = getattr(ansatz, "excitation_list", None)
    if excitation_list is not None:
        print(f"UCCSD excitations: {excitation_list}")
    metadata = getattr(ansatz, "metadata", None) or {}
    if "hf_occupations" in metadata:
        print(f"UCCSD HF occupations: {metadata['hf_occupations']}")
    if "initial_point_source" in metadata:
        print(f"UCCSD initial point source: {metadata['initial_point_source']}")


def _estimate_energy(hamiltonian, ansatz, estimator, params):
    from pydephasing.quantum.quantum_eigensolver import _evaluate_energy

    return _evaluate_energy(estimator, ansatz, hamiltonian, params)


def _run_vqe_multistart(
    *,
    hamiltonian,
    ansatz,
    optimizer_name,
    maxiter,
    restarts,
    seed,
    estimator,
    transpiler=None,
):
    from pydephasing.quantum.quantum_eigensolver import vqe_ground_energy

    rng = np.random.default_rng(seed)
    energies = []
    best_energy = None
    best_params = None
    best_result = None
    n_params = getattr(ansatz, "num_parameters", 0) or 0
    restarts = max(1, restarts)

    for idx in range(restarts):
        point = rng.uniform(-0.2, 0.2, size=n_params) if n_params else None
        optimizer = _build_optimizer(optimizer_name, maxiter, tol=1e-10)
        energy, result = vqe_ground_energy(
            hamiltonian,
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=estimator,
            transpiler=transpiler,
            initial_point=point,
        )
        energies.append(energy)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_result = result
            best_params = getattr(result, "optimal_parameters", None)

    _summarize_restart_energies(energies)
    print(f"VQE best energy across restarts: {best_energy}")
    return best_energy, best_params, energies, best_result


def _default_reps(kind: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    return 2 if kind == "clustered" else 1


def _summarize_best_trace(trace):
    if not trace:
        print("Best energy trace: []")
        return
    head = trace[:5]
    tail = trace[-5:] if len(trace) > 5 else trace
    print(f"Best energy trace (first 5): {head}")
    print(f"Best energy trace (last 5): {tail}")


def _run_exact_invariants() -> None:
    from pydephasing.quantum.quantum_eigensolver import exact_ground_energy

    _, op = _build_hubbard_qubit_op(1.0, 0.0, 0.0, mode="JW")
    e_half = exact_ground_energy(op, sector={"n_electrons": 2}, n_qubits=4)
    if abs(e_half + 2.0) > 1e-12:
        raise AssertionError(
            "Exact sanity check failed (t=1,U=0,dv=0, N=2). "
            f"Expected -2, got {e_half}."
        )

    for u, dv in [(2.0, 0.5), (0.5, 1.0)]:
        _, op = _build_hubbard_qubit_op(0.0, u, dv, mode="JW")
        expected = min(0.0, u - abs(dv))
        e_half = exact_ground_energy(op, sector={"n_electrons": 2}, n_qubits=4)
        if abs(e_half - expected) > 1e-12:
            raise AssertionError(
                "Exact sanity check failed (t=0, N=2). "
                f"Expected {expected}, got {e_half}."
            )


def _run_local_vqe_case(
    *,
    hamiltonian,
    estimator,
    t,
    u,
    dv,
    kind,
    reps,
    restarts,
    maxiter,
    optimizer_name,
    seed,
    e_exact_full,
    allow_retry,
):
    from pydephasing.quantum.quantum_eigensolver import (
        _optimizer_status,
        ansatz_factory,
    )

    ansatz, kind_used = ansatz_factory(
        kind,
        t=t,
        U=u,
        dv=dv,
        reps=reps,
    )
    if kind == "uccsd" and kind_used != "uccsd":
        raise ImportError("UCCSD ansatz unavailable; install qiskit_nature.")
    if kind_used == "uccsd":
        _print_uccsd_details(ansatz)

    n_params = getattr(ansatz, "num_parameters", 0) or 0
    e_init = None
    if n_params:
        zero_point = np.zeros(n_params, dtype=float)
        e_init = _estimate_energy(hamiltonian, ansatz, estimator, zero_point)
    print(f"E_init({kind_used}): {e_init}")

    energy, optimal_params, energies, best_result = _run_vqe_multistart(
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        restarts=restarts,
        seed=seed,
        estimator=estimator,
    )
    print(f"E_vqe_best({kind_used}): {energy}")
    if best_result is not None:
        print(f"Best optimizer status: {_optimizer_status(best_result)}")

    delta = abs(energy - e_exact_full)
    if kind_used == "uccsd":
        target_tol = 1e-5
    else:
        target_tol = 1e-3

    if delta > target_tol and allow_retry:
        if kind_used == "clustered" and reps == 2:
            print("Clustered threshold not met at reps=2; retrying with reps=3.")
            return _run_local_vqe_case(
                hamiltonian=hamiltonian,
                estimator=estimator,
                t=t,
                u=u,
                dv=dv,
                kind=kind,
                reps=3,
                restarts=restarts,
                maxiter=maxiter,
                optimizer_name=optimizer_name,
                seed=seed,
                e_exact_full=e_exact_full,
                allow_retry=False,
            )
        if kind_used == "uccsd" and reps == 1:
            print("UCCSD threshold not met at reps=1; retrying with reps=2.")
            return _run_local_vqe_case(
                hamiltonian=hamiltonian,
                estimator=estimator,
                t=t,
                u=u,
                dv=dv,
                kind=kind,
                reps=2,
                restarts=restarts,
                maxiter=maxiter,
                optimizer_name=optimizer_name,
                seed=seed,
                e_exact_full=e_exact_full,
                allow_retry=False,
            )

    if delta > target_tol:
        if best_result is not None:
            trace = getattr(best_result, "energy_trace", None)
            _summarize_best_trace(trace)
        raise AssertionError(
            f"VQE({kind_used}) deviates from exact by {delta} (> {target_tol})."
        )

    if _env_bool("DEBUG_VQE_PARAMS", False):
        print(f"VQE({kind_used}) optimal params: {optimal_params}")

    return energy, delta


def _run_local_vqe(t, u, dv, args, *, ansatz_choice, require_thresholds):
    from pydephasing.quantum.quantum_eigensolver import exact_ground_energy

    _, hamiltonian = _build_hubbard_qubit_op(t, u, dv, mode="JW")
    estimator = _local_estimator()

    up_qubits = DEFAULT_UP_QUBITS
    down_qubits = DEFAULT_DOWN_QUBITS
    sector = _default_sector(up_qubits, down_qubits)
    e_exact_full = exact_ground_energy(hamiltonian, sector=None, n_qubits=4)
    e_exact_sector = exact_ground_energy(hamiltonian, sector=sector, n_qubits=4)

    print(f"E_exact_full: {e_exact_full}")
    print(f"E_exact_sector: {e_exact_sector} | {sector}")

    kinds = ["clustered", "uccsd"] if ansatz_choice == "both" else [ansatz_choice]

    for kind in kinds:
        reps = _default_reps(kind, args.reps)
        print(f"Running local VQE for {kind} (reps={reps})")
        _run_local_vqe_case(
            hamiltonian=hamiltonian,
            estimator=estimator,
            t=t,
            u=u,
            dv=dv,
            kind=kind,
            reps=reps,
            restarts=args.restarts,
            maxiter=args.maxiter,
            optimizer_name=args.optimizer,
            seed=args.seed,
            e_exact_full=e_exact_full,
            allow_retry=require_thresholds,
        )


def _run_self_test(args) -> None:
    print("Running self-tests...")
    try:
        from pydephasing.quantum.qiskit_bridge import (
            self_test_label_convention,
            self_test_roundtrip,
        )

        if not self_test_label_convention():
            raise AssertionError("qiskit_bridge label convention test failed")
        if not self_test_roundtrip():
            raise AssertionError("qiskit_bridge roundtrip test failed")
    except ImportError as exc:
        raise ImportError("self-tests require qiskit and pydephasing deps") from exc

    _run_mapping_regression()
    _run_exact_invariants()
    _run_local_vqe(1.0, 2.0, 0.5, args, ansatz_choice="both", require_thresholds=True)
    print("SELF-TEST PASS")


def _load_ibm_runtime_service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    channel = os.environ.get("IBM_RUNTIME_CHANNEL", "ibm_quantum_platform")
    token = os.environ.get("IBM_RUNTIME_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("IBM_RUNTIME_INSTANCE") or os.environ.get(
        "QISKIT_IBM_INSTANCE"
    )
    if not token or not instance:
        raise RuntimeError(
            "IBM credentials must be set via IBM_RUNTIME_TOKEN/QISKIT_IBM_TOKEN "
            "and IBM_RUNTIME_INSTANCE/QISKIT_IBM_INSTANCE."
        )
    print(f"Using IBM Runtime channel: {channel}")
    return QiskitRuntimeService(channel=channel, token=token, instance=instance)


def _select_ibm_backend(service):
    name = os.environ.get("IBM_BACKEND")
    if not name:
        raise RuntimeError("IBM_BACKEND must be set for IBM Runtime mode.")
    try:
        backend = service.backend(name)
    except Exception as exc:
        raise RuntimeError(f"IBM backend {name!r} is not available.") from exc
    print(f"Using backend: {backend.name}")
    return backend


def _run_vqe_local(args) -> None:
    print("Running local VQE (StatevectorEstimator).")
    print(f"Parameters: t={args.t}, u={args.u}, dv={args.dv}")
    _run_local_vqe(
        args.t,
        args.u,
        args.dv,
        args,
        ansatz_choice=args.ansatz,
        require_thresholds=True,
    )


def _run_vqe_on_ibm(args) -> None:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Estimator, EstimatorOptions, Session
    from qiskit_ibm_runtime.api.exceptions import RequestsApiError

    from pydephasing.quantum.quantum_eigensolver import (
        ansatz_factory,
        exact_ground_energy,
    )

    shots = int(os.environ.get("VQE_SHOTS", "1024"))
    maxiter = args.maxiter
    restarts = max(1, args.restarts)
    seed = args.seed
    optimizer_name = args.optimizer

    _, hamiltonian = _build_hubbard_qubit_op(args.t, args.u, args.dv, mode="JW")
    operator_source = "PauliPolynomial (JW) -> SparsePauliOp"

    print("Running VQE on IBM Quantum Platform.")
    print(f"Operator source: {operator_source}")
    print(f"Parameters: t={args.t}, u={args.u}, dv={args.dv}")

    service = _load_ibm_runtime_service()
    backend = _select_ibm_backend(service)
    options = EstimatorOptions(default_shots=shots)
    transpiler = generate_preset_pass_manager(optimization_level=1, backend=backend)
    up_qubits = _parse_index_list(os.environ.get("UP_QUBITS"), DEFAULT_UP_QUBITS)
    down_qubits = _parse_index_list(os.environ.get("DOWN_QUBITS"), DEFAULT_DOWN_QUBITS)
    default_sector = _default_sector(up_qubits, down_qubits)
    exact_sector = _parse_exact_sector(default_sector)
    if exact_sector and "sz" in exact_sector:
        exact_sector.setdefault("up_qubits", up_qubits)
        exact_sector.setdefault("down_qubits", down_qubits)

    e_exact_full = exact_ground_energy(hamiltonian, sector=None, n_qubits=4)
    e_exact_sector = (
        exact_ground_energy(hamiltonian, sector=exact_sector, n_qubits=4)
        if exact_sector
        else e_exact_full
    )

    print(f"Exact ground (full): {e_exact_full}")
    if exact_sector:
        print(f"Exact ground (sector): {e_exact_sector} | {exact_sector}")

    ansatz, kind_used = ansatz_factory(
        args.ansatz if args.ansatz != "both" else "clustered",
        t=args.t,
        U=args.u,
        dv=args.dv,
        reps=_default_reps(args.ansatz if args.ansatz != "both" else "clustered", args.reps),
    )
    if args.ansatz == "both":
        print("IBM mode uses a single ansatz; defaulting to clustered.")
    if kind_used == "uccsd":
        _print_uccsd_details(ansatz)

    try:
        with Session(backend) as session:
            estimator = Estimator(session, options=options)
            energy, optimal_params, energies, best_result = _run_vqe_multistart(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                optimizer_name=optimizer_name,
                maxiter=maxiter,
                restarts=restarts,
                seed=seed,
                estimator=estimator,
                transpiler=transpiler,
            )
    except RequestsApiError as exc:
        if "open plan" not in str(exc) and "1352" not in str(exc):
            raise
        print("Session mode not authorized; retrying in job mode.")
        estimator = Estimator(backend, options=options)
        energy, optimal_params, energies, best_result = _run_vqe_multistart(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            optimizer_name=optimizer_name,
            maxiter=maxiter,
            restarts=restarts,
            seed=seed,
            estimator=estimator,
            transpiler=transpiler,
        )

    print(f"VQE ground state energy: {energy}")
    if _env_bool("DEBUG_VQE_PARAMS", False):
        print(f"Optimal parameters: {optimal_params}")

    tol_raw = os.environ.get("EXACT_TOL")
    if tol_raw is None:
        tol_value = 5e-1
        print("Warning: shots/hardware noise -> using EXACT_TOL=0.5.")
    else:
        tol_value = float(tol_raw)
    delta = abs(energy - e_exact_sector)
    print(f"Exact comparison tolerance: {tol_value}")
    print(f"VQE vs exact (sector) deltaE: {delta}")
    if delta > tol_value:
        raise AssertionError(
            f"VQE energy differs from exact by {delta} (> {tol_value}). "
            "Increase EXACT_TOL to relax."
        )


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _run_self_test(args)
        return

    if args.mapping_trials > 0:
        _run_mapping_random(args.mapping_trials, args.mapping_seed)

    if args.local:
        _run_vqe_local(args)

    if args.ibm:
        _run_vqe_on_ibm(args)


if __name__ == "__main__":
    main()
