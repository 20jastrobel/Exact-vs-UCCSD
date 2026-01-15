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

import os
import numpy as np

REFERENCE_HAMILTONIAN = (
    "H = - (t/2) ( X0 X1 + Y0 Y1 + X2 X3 + Y2 Y3 )\n"
    "    + (Δv/4) ( Z0 + Z2 - Z1 - Z3 )\n"
    "    + (U/4) ( Z0 Z2 + Z1 Z3 )\n"
    "    - (U/4) ( Z0 + Z1 + Z2 + Z3 )\n"
    "    + (U/2) I."
)


def _simplify_sparse_pauli(op, atol: float = 1e-12):
    try:
        return op.simplify(atol=atol)
    except TypeError:
        return op.simplify()


def _sort_pauli_list(pauli_list):
    return sorted(pauli_list, key=lambda item: item[0])


def _sorted_terms_str(pauli_list):
    return [f"{label}: {coeff}" for label, coeff in _sort_pauli_list(pauli_list)]


def _build_reference_sparse_pauli(t: float, u: float, dv: float):
    from qiskit.quantum_info import SparsePauliOp

    ref = SparsePauliOp.from_list(
        [
            ("IIXX", -t / 2.0),
            ("IIYY", -t / 2.0),
            ("XXII", -t / 2.0),
            ("YYII", -t / 2.0),
            ("IIIZ", +dv / 4.0),
            ("IZII", +dv / 4.0),
            ("IIZI", -dv / 4.0),
            ("ZIII", -dv / 4.0),
            ("IZIZ", +u / 4.0),
            ("ZIZI", +u / 4.0),
            ("IIIZ", -u / 4.0),
            ("IIZI", -u / 4.0),
            ("IZII", -u / 4.0),
            ("ZIII", -u / 4.0),
            ("IIII", +u / 2.0),
        ]
    )
    return _simplify_sparse_pauli(ref)


def _run_qiskit_trials(trials: int = 8, seed: int = 11) -> None:
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import FermionicOp

    def build_fermionic_op(t: float, u: float, dv: float):
        terms = {
            "+_0 -_1": -t,
            "+_1 -_0": -t,
            "+_2 -_3": -t,
            "+_3 -_2": -t,
            "+_1 -_1": dv / 2.0,
            "+_3 -_3": dv / 2.0,
            "+_0 -_0": -dv / 2.0,
            "+_2 -_2": -dv / 2.0,
            "+_0 -_0 +_2 -_2": u,
            "+_1 -_1 +_3 -_3": u,
        }
        try:
            op = FermionicOp(terms, num_spin_orbitals=4)
        except TypeError:
            op = FermionicOp(terms, register_length=4)
        try:
            op = op.simplify()
        except Exception:
            pass
        return op

    mapper = JordanWignerMapper()
    rng = np.random.default_rng(seed)

    for idx in range(trials):
        t = float(rng.uniform(-2.0, 2.0))
        u = float(rng.uniform(-2.0, 2.0))
        dv = float(rng.uniform(-2.0, 2.0))

        fermionic_op = build_fermionic_op(t, u, dv)
        jw_op = mapper.map(fermionic_op)
        if hasattr(jw_op, "primitive"):
            jw_op = jw_op.primitive
        jw_op = _simplify_sparse_pauli(jw_op)

        ref_op = _build_reference_sparse_pauli(t, u, dv)
        diff = _simplify_sparse_pauli(jw_op - ref_op, atol=1e-12)

        diff_norm = np.linalg.norm(diff.to_matrix())
        if diff_norm > 1e-10:
            print(f"Mismatch on trial {idx}")
            print(f"t={t}, u={u}, dv={dv}")
            print("Mapped terms:")
            print("\n".join(_sorted_terms_str(jw_op.to_list())))
            print("Reference terms:")
            print("\n".join(_sorted_terms_str(ref_op.to_list())))
            print("Diff terms:")
            print("\n".join(_sorted_terms_str(diff.to_list())))
            raise AssertionError(f"Operator norm mismatch: {diff_norm}")

        evals_jw = np.linalg.eigvalsh(jw_op.to_matrix())
        evals_ref = np.linalg.eigvalsh(ref_op.to_matrix())
        if not np.allclose(np.sort(evals_jw), np.sort(evals_ref), atol=1e-10):
            raise AssertionError("Eigenvalue mismatch")

    print("PASS")


def _build_jw_qubit_op(t: float, u: float, dv: float):
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import FermionicOp

    terms = {
        "+_0 -_1": -t,
        "+_1 -_0": -t,
        "+_2 -_3": -t,
        "+_3 -_2": -t,
        "+_1 -_1": dv / 2.0,
        "+_3 -_3": dv / 2.0,
        "+_0 -_0": -dv / 2.0,
        "+_2 -_2": -dv / 2.0,
        "+_0 -_0 +_2 -_2": u,
        "+_1 -_1 +_3 -_3": u,
    }
    try:
        fermionic_op = FermionicOp(terms, num_spin_orbitals=4)
    except TypeError:
        fermionic_op = FermionicOp(terms, register_length=4)

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)
    if hasattr(qubit_op, "primitive"):
        qubit_op = qubit_op.primitive
    return _simplify_sparse_pauli(qubit_op)


def _openfermion_term_label(term) -> str:
    if not term:
        return "I"
    return " ".join(f"{pauli}{index}" for index, pauli in term)


def _sorted_openfermion_terms(op) -> list[str]:
    items = []
    for term, coeff in op.terms.items():
        items.append((_openfermion_term_label(term), coeff))
    items.sort(key=lambda item: item[0])
    return [f"{label}: {coeff}" for label, coeff in items]


def _run_openfermion_trials(trials: int = 8, seed: int = 11) -> None:
    from openfermion import (
        FermionOperator,
        QubitOperator,
        get_sparse_operator,
        jordan_wigner,
        normal_ordered,
    )

    def build_fermionic_op(t: float, u: float, dv: float):
        op = FermionOperator()
        op += FermionOperator("0^ 1", -t)
        op += FermionOperator("1^ 0", -t)
        op += FermionOperator("2^ 3", -t)
        op += FermionOperator("3^ 2", -t)
        op += FermionOperator("1^ 1", dv / 2.0)
        op += FermionOperator("3^ 3", dv / 2.0)
        op += FermionOperator("0^ 0", -dv / 2.0)
        op += FermionOperator("2^ 2", -dv / 2.0)
        op += FermionOperator("0^ 0 2^ 2", u)
        op += FermionOperator("1^ 1 3^ 3", u)
        return normal_ordered(op)

    def build_reference_op(t: float, u: float, dv: float):
        op = QubitOperator()
        op += QubitOperator("X0 X1", -t / 2.0)
        op += QubitOperator("Y0 Y1", -t / 2.0)
        op += QubitOperator("X2 X3", -t / 2.0)
        op += QubitOperator("Y2 Y3", -t / 2.0)
        op += QubitOperator("Z0", +dv / 4.0)
        op += QubitOperator("Z2", +dv / 4.0)
        op += QubitOperator("Z1", -dv / 4.0)
        op += QubitOperator("Z3", -dv / 4.0)
        op += QubitOperator("Z0 Z2", +u / 4.0)
        op += QubitOperator("Z1 Z3", +u / 4.0)
        op += QubitOperator("Z0", -u / 4.0)
        op += QubitOperator("Z1", -u / 4.0)
        op += QubitOperator("Z2", -u / 4.0)
        op += QubitOperator("Z3", -u / 4.0)
        op += QubitOperator("", +u / 2.0)
        return op

    rng = np.random.default_rng(seed)

    for idx in range(trials):
        t = float(rng.uniform(-2.0, 2.0))
        u = float(rng.uniform(-2.0, 2.0))
        dv = float(rng.uniform(-2.0, 2.0))

        fermionic_op = build_fermionic_op(t, u, dv)
        jw_op = jordan_wigner(fermionic_op)
        ref_op = build_reference_op(t, u, dv)
        diff = jw_op - ref_op

        diff_mat = get_sparse_operator(diff, n_qubits=4).toarray()
        diff_norm = np.linalg.norm(diff_mat)
        if diff_norm > 1e-10:
            print(f"Mismatch on trial {idx}")
            print(f"t={t}, u={u}, dv={dv}")
            print("Mapped terms:")
            print("\n".join(_sorted_openfermion_terms(jw_op)))
            print("Reference terms:")
            print("\n".join(_sorted_openfermion_terms(ref_op)))
            print("Diff terms:")
            print("\n".join(_sorted_openfermion_terms(diff)))
            raise AssertionError(f"Operator norm mismatch: {diff_norm}")

        evals_jw = np.linalg.eigvalsh(get_sparse_operator(jw_op, n_qubits=4).toarray())
        evals_ref = np.linalg.eigvalsh(get_sparse_operator(ref_op, n_qubits=4).toarray())
        if not np.allclose(np.sort(evals_jw), np.sort(evals_ref), atol=1e-10):
            raise AssertionError("Eigenvalue mismatch")

    print("PASS")


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


def _load_ibm_runtime_service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    account_name = os.environ.get("IBM_RUNTIME_ACCOUNT_NAME")
    channel = os.environ.get("IBM_RUNTIME_CHANNEL")

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


def _run_vqe_on_ibm() -> None:
    from qiskit.circuit.library import efficient_su2
    from qiskit_algorithms.minimum_eigensolvers import VQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_ibm_runtime import Estimator, EstimatorOptions, Session

    t = _env_float("VQE_T", 1.0)
    u = _env_float("VQE_U", 2.0)
    dv = _env_float("VQE_DV", 0.5)
    maxiter = _env_int("VQE_MAXITER", 50)
    shots = _env_int("VQE_SHOTS", 1024)
    reps = _env_int("VQE_REPS", 1)

    try:
        hamiltonian = _build_jw_qubit_op(t, u, dv)
        operator_source = "JW-mapped FermionicOp"
    except ImportError:
        hamiltonian = _build_reference_sparse_pauli(t, u, dv)
        operator_source = "reference SparsePauliOp"

    print("Running VQE on IBM Quantum Platform.")
    print(f"Operator source: {operator_source}")
    print(f"Parameters: t={t}, u={u}, dv={dv}")

    ansatz = efficient_su2(num_qubits=4, reps=reps, entanglement="full")
    optimizer = COBYLA(maxiter=maxiter)

    service = _load_ibm_runtime_service()
    backend = _select_ibm_backend(service)
    options = EstimatorOptions(default_shots=shots)

    with Session(service=service, backend=backend) as session:
        estimator = Estimator(session=session, options=options)
        vqe = VQE(estimator, ansatz, optimizer=optimizer)
        result = vqe.compute_minimum_eigenvalue(hamiltonian)

    energy = float(np.real(result.eigenvalue))
    print(f"VQE ground state energy: {energy}")
    print(f"Optimal parameters: {result.optimal_parameters}")


def main() -> None:
    try:
        _run_qiskit_trials()
    except ImportError as exc:
        print("Qiskit Nature path unavailable; falling back to OpenFermion.")
        print(f"Reason: {exc}")
        try:
            _run_openfermion_trials()
        except ImportError as exc2:
            print("Mapping checks skipped; qiskit_nature and openfermion missing.")
            print(f"Reason: {exc2}")

    _run_vqe_on_ibm()


if __name__ == "__main__":
    main()
