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
import json
import os
import re
import sys

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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using default {default}.")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using default {default}.")
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ibm", action="store_true")
    parser.add_argument("--vqe-maxiter", type=int, default=None)
    parser.add_argument("--vqe-restarts", type=int, default=None)
    parser.add_argument("--vqe-seed", type=int, default=None)
    parser.add_argument(
        "--vqe-optimizer",
        type=str,
        choices=["SLSQP", "L_BFGS_B", "COBYLA"],
        default=None,
    )
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


def _vqe_settings(args, *, for_ibm=False, for_test=False):
    if for_test:
        maxiter = (
            args.vqe_maxiter
            if args.vqe_maxiter is not None
            else _env_int("VQE_TEST_MAXITER", 500)
        )
        restarts = (
            args.vqe_restarts
            if args.vqe_restarts is not None
            else _env_int("VQE_TEST_RESTARTS", 3)
        )
        seed = (
            args.vqe_seed
            if args.vqe_seed is not None
            else _env_int("VQE_TEST_SEED", 0)
        )
        optimizer_name = (
            args.vqe_optimizer
            if args.vqe_optimizer is not None
            else os.environ.get("VQE_TEST_OPTIMIZER", "L_BFGS_B")
        )
        tol = _env_float("VQE_TEST_OPT_TOL", 1e-10)
        return maxiter, restarts, seed, optimizer_name, tol

    if for_ibm:
        maxiter = (
            args.vqe_maxiter
            if args.vqe_maxiter is not None
            else _env_int("VQE_MAXITER", 50)
        )
        restarts = (
            args.vqe_restarts
            if args.vqe_restarts is not None
            else _env_int("VQE_RESTARTS", 1)
        )
    else:
        maxiter = (
            args.vqe_maxiter
            if args.vqe_maxiter is not None
            else _env_int("VQE_MAXITER", 1000)
        )
        restarts = (
            args.vqe_restarts
            if args.vqe_restarts is not None
            else _env_int("VQE_RESTARTS", 20)
        )

    seed = args.vqe_seed if args.vqe_seed is not None else _env_int("VQE_SEED", 0)
    optimizer_name = (
        args.vqe_optimizer
        if args.vqe_optimizer is not None
        else os.environ.get("VQE_OPTIMIZER", "L_BFGS_B")
    )
    tol = _env_float("VQE_OPT_TOL", 1e-10)
    return maxiter, restarts, seed, optimizer_name, tol


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


def _run_mapping_random() -> None:
    trials = _env_int("MAPPING_RANDOM_TRIALS", 0)
    if trials <= 0:
        return
    seed = _env_int("MAPPING_RANDOM_SEED", 11)
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


def _get_ansatz_initial_point(ansatz):
    metadata = getattr(ansatz, "metadata", None) or {}
    initial_point = metadata.get("initial_point")
    source = metadata.get("initial_point_source")
    point_range = metadata.get("initial_point_range")
    return initial_point, source, point_range


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
    if "initial_point_range" in metadata:
        print(f"UCCSD multistart range: {metadata['initial_point_range']}")


def _estimate_energy(hamiltonian, ansatz, estimator, params):
    from pydephasing.quantum.quantum_eigensolver import _evaluate_energy

    return _evaluate_energy(estimator, ansatz, hamiltonian, params)


def _expressibility_probe(
    *,
    hamiltonian,
    ansatz,
    estimator,
    seed,
    samples,
    param_range,
):
    if samples <= 0:
        print("Clustered expressibility probe: skipped (samples <= 0).")
        return
    n_params = getattr(ansatz, "num_parameters", 0) or 0
    if n_params == 0:
        print("Clustered expressibility probe: skipped (no parameters).")
        return

    rng = np.random.default_rng(seed)
    energies = []
    for _ in range(samples):
        params = rng.uniform(-param_range, param_range, size=n_params)
        energies.append(_estimate_energy(hamiltonian, ansatz, estimator, params))

    energies = np.asarray(energies, dtype=float)
    print(
        "Clustered expressibility probe: "
        f"min={energies.min()}, median={np.median(energies)}, max={energies.max()}"
    )
    print(f"Clustered expressibility probe: best sample energy={energies.min()}")


def _run_vqe_multistart(
    *,
    hamiltonian,
    ansatz,
    optimizer_name,
    maxiter,
    restarts,
    seed,
    estimator,
    tol,
    transpiler=None,
    initial_point=None,
    initial_point_source=None,
    initial_point_range=None,
):
    from pydephasing.quantum.quantum_eigensolver import compute_ground_state

    rng = np.random.default_rng(seed)
    energies = []
    best_energy = None
    best_params = None
    n_params = getattr(ansatz, "num_parameters", 0) or 0
    restarts = max(1, restarts)

    point_range = 0.2 if initial_point_range is None else float(initial_point_range)
    jitter_range = min(0.05, 0.25 * point_range)

    for idx in range(restarts):
        point = None
        if idx == 0 and initial_point is not None and n_params:
            point = np.asarray(initial_point, dtype=float)
            if point.shape[0] != n_params:
                print("Initial point length mismatch; falling back to random.")
                point = None
            elif np.allclose(point, 0.0):
                point = rng.uniform(-jitter_range, jitter_range, size=n_params)
        if point is None and n_params:
            point = rng.uniform(-point_range, point_range, size=n_params)
        if idx == 0 and point is not None and initial_point_source:
            print(f"VQE restart 1 uses initial point ({initial_point_source}).")
        optimizer = _build_optimizer(optimizer_name, maxiter, tol)
        energy, optimal_params = compute_ground_state(
            hamiltonian,
            method="vqe",
            vqe_options={
                "ansatz": ansatz,
                "optimizer": optimizer,
                "estimator": estimator,
                "transpiler": transpiler,
                "initial_point": point,
            },
        )
        energies.append(energy)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_params = optimal_params

    _summarize_restart_energies(energies)
    print(f"VQE best energy across restarts: {best_energy}")
    return best_energy, best_params, energies


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


def _run_vqe_regression(t, u, dv, args) -> None:
    from pydephasing.quantum.quantum_eigensolver import (
        ansatz_factory,
        exact_ground_energy,
    )

    _, op = _build_hubbard_qubit_op(t, u, dv, mode="JW")
    up_qubits = DEFAULT_UP_QUBITS
    down_qubits = DEFAULT_DOWN_QUBITS
    sector = _default_sector(up_qubits, down_qubits)
    e_exact = exact_ground_energy(op, sector=sector, n_qubits=4)

    estimator = _local_estimator()
    maxiter, restarts, seed, optimizer_name, opt_tol = _vqe_settings(
        args, for_test=True
    )
    tol_clustered = _env_float("VQE_TEST_CLUSTERED_TOL", 1e-2)
    tol_uccsd = _env_float("VQE_TEST_UCCSD_TOL", 1e-6)
    tol_other = _env_float("VQE_TEST_TOL", 0.3)
    reps = _env_int("VQE_TEST_REPS", 2)

    for kind in ("clustered", "uccsd"):
        ansatz, kind_used = ansatz_factory(
            kind,
            t=t,
            U=u,
            dv=dv,
            reps=reps,
        )
        if kind == "uccsd" and kind_used != "uccsd":
            print("UCCSD unavailable; skipping UCCSD VQE regression.")
            continue
        if kind_used == "uccsd":
            _print_uccsd_details(ansatz)

        initial_point, source, point_range = _get_ansatz_initial_point(ansatz)

        energy, optimal_params, energies = _run_vqe_multistart(
            hamiltonian=op,
            ansatz=ansatz,
            optimizer_name=optimizer_name,
            maxiter=maxiter,
            restarts=restarts,
            seed=seed,
            estimator=estimator,
            tol=opt_tol,
            initial_point=initial_point,
            initial_point_source=source,
            initial_point_range=point_range,
        )
        delta = abs(energy - e_exact)
        print(f"VQE({kind_used}) energy: {energy} | deltaE={delta}")
        if kind_used == "uccsd":
            target_tol = tol_uccsd
        elif kind_used == "clustered":
            target_tol = tol_clustered
        else:
            target_tol = tol_other
        if delta > target_tol:
            raise AssertionError(
                f"VQE({kind_used}) deviates from exact by {delta} (> {target_tol})."
            )

        if _env_bool("DEBUG_VQE_PARAMS", False):
            print(f"VQE({kind_used}) optimal params: {optimal_params}")


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
    t, u, dv = DEFAULT_MAPPING_PARAMS[0]
    _run_vqe_regression(t, u, dv, args)
    print("SELF-TEST PASS")


def _backend_is_simulator(backend) -> bool:
    for attr in ("simulator", "is_simulator"):
        value = getattr(backend, attr, None)
        if isinstance(value, bool):
            return value
    if hasattr(backend, "configuration"):
        try:
            cfg = backend.configuration()
        except Exception:
            cfg = None
        if cfg is not None and hasattr(cfg, "simulator"):
            return bool(cfg.simulator)
    return False


def _looks_like_token(value: str) -> bool:
    lowered = value.lower()
    if "<" in value or "your-api-key" in lowered or "api-key" in lowered:
        return False
    return len(value) >= 20


def _extract_notebook_credentials(path: str) -> tuple[str | None, str | None]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None, None

    sources = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            sources.append("".join(src))
        else:
            sources.append(str(src))

    text = "\n".join(sources)

    token_patterns = [
        r'QISKIT_IBM_TOKEN"\]\s*=\s*"([^"]+)"',
        r"QISKIT_IBM_TOKEN'\]\s*=\s*'([^']+)'",
        r'token\s*=\s*"([^"]+)"',
        r"token\s*=\s*'([^']+)'",
    ]
    instance_patterns = [
        r'instance\s*=\s*"(crn:[^"]+)"',
        r"instance\s*=\s*'(crn:[^']+)'",
        r'instance\s*=.*?"(crn:[^"]+)"',
        r"instance\s*=.*?'(crn:[^']+)'",
    ]

    token = None
    for pattern in token_patterns:
        match = re.search(pattern, text)
        if match and _looks_like_token(match.group(1)):
            token = match.group(1)
            break

    instance = None
    for pattern in instance_patterns:
        match = re.search(pattern, text)
        if match:
            instance = match.group(1)
            break

    return token, instance


def _load_ibm_runtime_service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    account_name = os.environ.get("IBM_RUNTIME_ACCOUNT_NAME")
    channel = os.environ.get("IBM_RUNTIME_CHANNEL")
    token = os.environ.get("IBM_RUNTIME_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("IBM_RUNTIME_INSTANCE") or os.environ.get(
        "QISKIT_IBM_INSTANCE"
    )

    if not token or not instance:
        notebook_path = os.environ.get(
            "IBM_RUNTIME_NOTEBOOK_PATH",
            os.path.join(os.getcwd(), "Untitled-1.ipynb"),
        )
        nb_token, nb_instance = _extract_notebook_credentials(notebook_path)
        if not token and nb_token:
            token = nb_token
            print("Loaded IBM Runtime token from notebook.")
        if not instance and nb_instance:
            instance = nb_instance
            print("Loaded IBM Runtime instance from notebook.")

    if token:
        if not channel:
            channel = "ibm_quantum_platform"
        kwargs = {"channel": channel, "token": token}
        if instance:
            kwargs["instance"] = instance
        print(f"Using IBM Runtime channel: {channel}")
        return QiskitRuntimeService(**kwargs)

    if account_name:
        print(f"Using IBM Runtime account name: {account_name}")
        return QiskitRuntimeService(name=account_name)
    if channel:
        print(f"Using IBM Runtime channel: {channel}")
        return QiskitRuntimeService(channel=channel)

    try:
        accounts = QiskitRuntimeService.saved_accounts()
    except Exception:
        accounts = {}

    for name, info in accounts.items():
        if info.get("channel") == "ibm_quantum_platform":
            print(f"Using saved IBM Quantum Platform account: {name}")
            return QiskitRuntimeService(name=name)

    print("Using default IBM Runtime account.")
    return QiskitRuntimeService()


def _select_ibm_backend(service):
    preferred = os.environ.get("IBM_RUNTIME_BACKEND", "ibm_qasm_simulator")
    backend = None
    try:
        backend = service.backend(preferred)
    except Exception:
        backend = None

    if backend is None:
        simulators = service.backends(simulator=True, operational=True)
        if simulators:
            backend = simulators[0]
        else:
            devices = service.backends(simulator=False, operational=True)
            if not devices:
                raise RuntimeError("No operational IBM backends available.")
            backend = devices[0]

    print(f"Using backend: {backend.name}")
    return backend


def _run_vqe_local(args) -> None:
    from pydephasing.quantum.quantum_eigensolver import (
        ansatz_factory,
        exact_ground_energy,
    )

    t = _env_float("VQE_T", 1.0)
    u = _env_float("VQE_U", 2.0)
    dv = _env_float("VQE_DV", 0.5)
    reps = _env_int("VQE_REPS", 1)
    ansatz_kind = os.environ.get("VQE_ANSATZ", "clustered")
    maxiter, restarts, seed, optimizer_name, opt_tol = _vqe_settings(args)

    _, hamiltonian = _build_hubbard_qubit_op(t, u, dv, mode="JW")
    operator_source = "PauliPolynomial (JW) -> SparsePauliOp"

    print("Running local VQE (Aer/statevector).")
    print(f"Operator source: {operator_source}")
    print(f"Parameters: t={t}, u={u}, dv={dv}")

    estimator = _local_estimator()

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
        ansatz_kind,
        t=t,
        U=u,
        dv=dv,
        reps=reps,
    )
    if kind_used != ansatz_kind:
        print(f"Ansatz fallback: requested {ansatz_kind}, using {kind_used}.")

    if kind_used == "uccsd":
        _print_uccsd_details(ansatz)

    initial_point, source, point_range = _get_ansatz_initial_point(ansatz)

    if kind_used == "clustered":
        initial_energy = None
        n_params = getattr(ansatz, "num_parameters", 0) or 0
        if n_params:
            zero_point = np.zeros(n_params, dtype=float)
            initial_energy = _estimate_energy(
                hamiltonian, ansatz, estimator, zero_point
            )
            print(f"Clustered initial state energy: {initial_energy}")
        else:
            print("Clustered initial state energy: n/a (no parameters).")

        probe_samples = _env_int("EXPRESSIBILITY_SAMPLES", 200)
        probe_seed = _env_int("EXPRESSIBILITY_SEED", 0)
        probe_range = _env_float("EXPRESSIBILITY_RANGE", float(np.pi))
        _expressibility_probe(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            estimator=estimator,
            seed=probe_seed,
            samples=probe_samples,
            param_range=probe_range,
        )

    energy, optimal_params, energies = _run_vqe_multistart(
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        restarts=restarts,
        seed=seed,
        estimator=estimator,
        tol=opt_tol,
        initial_point=initial_point,
        initial_point_source=source,
        initial_point_range=point_range,
    )

    print(f"VQE ground state energy: {energy}")
    if _env_bool("DEBUG_VQE_PARAMS", False):
        print(f"Optimal parameters: {optimal_params}")
    if kind_used == "clustered" and initial_energy is not None:
        print(
            "Clustered energy summary: "
            f"initial={initial_energy}, vqe_best={energy}, exact={e_exact_sector}"
        )

    if os.environ.get("VQE_LOCAL_TOL") is not None:
        tol_value = _env_float("VQE_LOCAL_TOL", 0.3)
    elif kind_used == "uccsd":
        tol_value = _env_float("VQE_LOCAL_UCCSD_TOL", 1e-6)
    elif kind_used == "clustered":
        tol_value = _env_float("VQE_LOCAL_CLUSTERED_TOL", 1e-2)
    else:
        tol_value = _env_float("VQE_LOCAL_TOL", 0.3)
    delta = abs(energy - e_exact_sector)
    print(f"Exact comparison tolerance: {tol_value}")
    print(f"VQE vs exact (sector) deltaE: {delta}")
    if delta > tol_value:
        raise AssertionError(
            f"VQE energy differs from exact by {delta} (> {tol_value})."
        )


def _run_vqe_on_ibm(args) -> None:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Estimator, EstimatorOptions, Session
    from qiskit_ibm_runtime.api.exceptions import RequestsApiError

    from pydephasing.quantum.quantum_eigensolver import (
        ansatz_factory,
        exact_ground_energy,
    )

    t = _env_float("VQE_T", 1.0)
    u = _env_float("VQE_U", 2.0)
    dv = _env_float("VQE_DV", 0.5)
    shots = _env_int("VQE_SHOTS", 1024)
    reps = _env_int("VQE_REPS", 1)
    ansatz_kind = os.environ.get("VQE_ANSATZ", "clustered")
    maxiter, restarts, seed, optimizer_name, opt_tol = _vqe_settings(args, for_ibm=True)

    _, hamiltonian = _build_hubbard_qubit_op(t, u, dv, mode="JW")
    operator_source = "PauliPolynomial (JW) -> SparsePauliOp"

    print("Running VQE on IBM Quantum Platform.")
    print(f"Operator source: {operator_source}")
    print(f"Parameters: t={t}, u={u}, dv={dv}")

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
        ansatz_kind,
        t=t,
        U=u,
        dv=dv,
        reps=reps,
    )
    if kind_used != ansatz_kind:
        print(f"Ansatz fallback: requested {ansatz_kind}, using {kind_used}.")
    if kind_used == "uccsd":
        _print_uccsd_details(ansatz)

    initial_point, source, point_range = _get_ansatz_initial_point(ansatz)

    try:
        with Session(backend) as session:
            estimator = Estimator(session, options=options)
            energy, optimal_params, energies = _run_vqe_multistart(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                optimizer_name=optimizer_name,
                maxiter=maxiter,
                restarts=restarts,
                seed=seed,
                estimator=estimator,
                tol=opt_tol,
                transpiler=transpiler,
                initial_point=initial_point,
                initial_point_source=source,
                initial_point_range=point_range,
            )
    except RequestsApiError as exc:
        if "open plan" not in str(exc) and "1352" not in str(exc):
            raise
        print("Session mode not authorized; retrying in job mode.")
        estimator = Estimator(backend, options=options)
        energy, optimal_params, energies = _run_vqe_multistart(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            optimizer_name=optimizer_name,
            maxiter=maxiter,
            restarts=restarts,
            seed=seed,
            estimator=estimator,
            tol=opt_tol,
            transpiler=transpiler,
            initial_point=initial_point,
            initial_point_source=source,
            initial_point_range=point_range,
        )

    print(f"VQE ground state energy: {energy}")
    if _env_bool("DEBUG_VQE_PARAMS", False):
        print(f"Optimal parameters: {optimal_params}")

    tol_raw = os.environ.get("EXACT_TOL")
    if tol_raw is None:
        if _backend_is_simulator(backend):
            tol_value = 5e-2 if shots >= 4096 else 1e-1
        else:
            tol_value = 5e-1
    else:
        tol_value = _env_float("EXACT_TOL", 1e-3)
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
    self_test_only = args.self_test
    use_ibm = args.ibm or _env_bool("USE_IBM_RUNTIME", False)

    _run_self_test(args)
    if self_test_only:
        return

    _run_mapping_random()

    if use_ibm:
        _run_vqe_on_ibm(args)
    else:
        _run_vqe_local(args)


if __name__ == "__main__":
    main()
