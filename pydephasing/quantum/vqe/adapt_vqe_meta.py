"""Meta-ADAPT-VQE implementation using a coordinate-wise LSTM optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

from .meta_lstm_optimizer import CoordinateWiseLSTMOptimizer, LSTMState
from .meta_lstm import preprocess_gradients_torch


@dataclass
class MetaAdaptVQEResult:
    energy: float
    params: list[float]
    operators: list[str]
    diagnostics: dict


def build_operator_pool_from_hamiltonian(
    qubit_op: SparsePauliOp,
    *,
    mode: str = "ham_terms_plus_imag_partners",
) -> list[str]:
    """Return a Pauli-string operator pool derived from the Hamiltonian."""
    identity = "I" * qubit_op.num_qubits
    seen: set[str] = set()
    base_pool: list[str] = []
    for label in qubit_op.paulis.to_labels():
        if label == identity:
            continue
        if label not in seen:
            seen.add(label)
            base_pool.append(label)

    if mode == "ham_terms":
        return base_pool
    if mode != "ham_terms_plus_imag_partners":
        raise ValueError(f"Unknown pool mode: {mode}")

    pool = list(base_pool)
    pool_set = set(pool)
    for label in base_pool:
        for partner in _imaginary_partners(label):
            if partner != identity and partner not in pool_set:
                pool_set.add(partner)
                pool.append(partner)
    # Add single-qubit X/Y terms to enrich the pool and avoid trivial stagnation.
    for q in range(qubit_op.num_qubits):
        for pauli in ("X", "Y"):
            label = ["I"] * qubit_op.num_qubits
            label[q] = pauli
            term = "".join(label)
            if term not in pool_set:
                pool_set.add(term)
                pool.append(term)
    return pool


def build_cse_density_pool_from_fermionic(
    ferm_op: FermionicOp,
    mapper: JordanWignerMapper,
    *,
    include_diagonal: bool = False,
    dedup_tol: float = 1e-12,
) -> list[dict]:
    """Return CSE-inspired density operator pool from a fermionic Hamiltonian."""
    if ferm_op is None:
        raise ValueError("ferm_op must be provided for cse_density_ops pool.")

    pool: list[dict] = []
    seen: set[tuple] = set()
    data = getattr(ferm_op, "_data", {})

    for label in data.keys():
        terms = label.split()
        if len(terms) == 2 and terms[0].startswith("+_") and terms[1].startswith("-_"):
            p = int(terms[0][2:])
            q = int(terms[1][2:])
            if p == q and not include_diagonal:
                continue
            gamma = FermionicOp({label: 1.0}, num_spin_orbitals=ferm_op.num_spin_orbitals)
        elif include_diagonal and len(terms) in (4,):
            gamma = FermionicOp({label: 1.0}, num_spin_orbitals=ferm_op.num_spin_orbitals)
        else:
            continue

        k_op = gamma - gamma.adjoint()
        if len(getattr(k_op, "_data", {})) == 0:
            continue

        qubit_k = mapper.map(k_op)
        if qubit_k is None:
            continue
        qubit_k = qubit_k.simplify(atol=dedup_tol)
        a_op = (-1j) * qubit_k

        labels = a_op.paulis.to_labels()
        coeffs = a_op.coeffs
        paulis: list[tuple[str, float]] = []
        for lbl, coeff in zip(labels, coeffs):
            if lbl == "I" * a_op.num_qubits:
                continue
            if abs(coeff.imag) > dedup_tol:
                continue
            weight = float(coeff.real)
            if abs(weight) <= dedup_tol:
                continue
            paulis.append((lbl, weight))

        if not paulis:
            continue

        paulis.sort(key=lambda item: item[0])
        key = tuple((lbl, round(w, 12)) for lbl, w in paulis)
        if key in seen:
            continue
        seen.add(key)
        pool.append(
            {
                "name": f"gamma({label})",
                "paulis": paulis,
            }
        )

    return pool


def build_reference_state(num_qubits: int, occupations: Sequence[int]) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for idx in occupations:
        if idx < 0 or idx >= num_qubits:
            raise ValueError(f"Occupation index out of range: {idx}")
        qc.x(idx)
    return qc


def build_adapt_circuit(
    reference_state: QuantumCircuit,
    operators: Sequence[str],
) -> tuple[QuantumCircuit, ParameterVector]:
    """Build the ADAPT circuit with parameter placeholders."""
    num_qubits = reference_state.num_qubits
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(reference_state, inplace=True)

    params = ParameterVector("theta", len(operators))
    for idx, label in enumerate(operators):
        pauli = Pauli(label)
        gate = PauliEvolutionGate(pauli, time=params[idx] / 2)
        circuit.append(gate, list(range(num_qubits)))

    return circuit, params


def build_adapt_circuit_grouped(
    reference_state: QuantumCircuit,
    operator_specs: Sequence[dict],
) -> tuple[QuantumCircuit, ParameterVector]:
    """Build ADAPT circuit from grouped generator specs."""
    num_qubits = reference_state.num_qubits
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(reference_state, inplace=True)

    params = ParameterVector("theta", len(operator_specs))
    for idx, spec in enumerate(operator_specs):
        for label, weight in spec["paulis"]:
            pauli = Pauli(label)
            gate = PauliEvolutionGate(pauli, time=params[idx] * weight / 2)
            circuit.append(gate, list(range(num_qubits)))

    return circuit, params


def estimate_energy(
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    qubit_op: SparsePauliOp,
    params: Sequence[float],
) -> float:
    job = estimator.run([(circuit, qubit_op, list(params))])
    result = job.result()
    ev = result[0].data.evs
    return float(np.real(ev))


def parameter_shift_grad(
    energy_fn,
    theta: np.ndarray,
    *,
    shift: float = math.pi / 2,
) -> np.ndarray:
    n = theta.size
    grad = np.zeros(n, dtype=float)
    for idx in range(n):
        plus = theta.copy()
        minus = theta.copy()
        plus[idx] += shift
        minus[idx] -= shift
        grad[idx] = 0.5 * (energy_fn(plus) - energy_fn(minus))
    return grad


def _append_probe_gate(
    circuit: QuantumCircuit,
    label: str,
    *,
    angle: float,
) -> QuantumCircuit:
    num_qubits = circuit.num_qubits
    probe = QuantumCircuit(num_qubits)
    probe.compose(circuit, inplace=True)
    gate = PauliEvolutionGate(Pauli(label), time=angle / 2)
    probe.append(gate, list(range(num_qubits)))
    return probe


def _imaginary_partners(label: str) -> list[str]:
    partners: list[str] = []
    chars = list(label)
    for idx, char in enumerate(chars):
        if char == "X":
            partner = chars.copy()
            partner[idx] = "Y"
            partners.append("".join(partner))
        elif char == "Y":
            partner = chars.copy()
            partner[idx] = "X"
            partners.append("".join(partner))
    return partners


def _wrap_angles(theta: np.ndarray) -> np.ndarray:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def _probe_shift_and_coeff(convention: str) -> tuple[float, float]:
    if convention == "half_angle":
        return math.pi / 2, 0.5
    if convention == "full_angle":
        return math.pi / 4, 1.0
    raise ValueError(f"Unknown probe convention: {convention}")


def _compute_grouped_pool_gradients(
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    qubit_op: SparsePauliOp,
    theta: np.ndarray,
    pool_specs: Sequence[dict],
    *,
    probe_convention: str,
) -> list[float]:
    shift, coeff = _probe_shift_and_coeff(probe_convention)
    label_set: set[str] = set()
    for spec in pool_specs:
        for label, _weight in spec["paulis"]:
            label_set.add(label)

    g_cache: dict[str, float] = {}
    for label in sorted(label_set):
        probe_plus = _append_probe_gate(circuit, label, angle=shift)
        probe_minus = _append_probe_gate(circuit, label, angle=-shift)
        e_plus = estimate_energy(estimator, probe_plus, qubit_op, theta)
        e_minus = estimate_energy(estimator, probe_minus, qubit_op, theta)
        g_cache[label] = coeff * (e_plus - e_minus)

    gradients: list[float] = []
    for spec in pool_specs:
        g_val = 0.0
        for label, weight in spec["paulis"]:
            g_val += weight * g_cache[label]
        gradients.append(g_val)
    return gradients


def _maybe_import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Torch is required for meta/hybrid inner optimization. "
            "Install torch or choose --inner-optimizer lbfgs."
        ) from exc
    return torch


def _lbfgs_optimize(
    theta0: np.ndarray,
    energy_fn,
    grad_fn,
    *,
    theta_bound: str = "none",
    maxiter: int | None = None,
    restarts: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    bounds = None
    if theta_bound == "pi":
        bounds = [(-math.pi, math.pi) for _ in range(theta0.size)]
    elif theta_bound != "none":
        raise ValueError(f"Unknown theta bound: {theta_bound}")

    if restarts < 1:
        restarts = 1

    best_x = None
    best_fun = None
    total_nfev = 0
    total_njev = 0
    best_stats = {}

    for idx in range(restarts):
        if idx == 0:
            x0 = theta0
        else:
            if rng is None:
                rng = np.random.default_rng()
            x0 = rng.uniform(-math.pi, math.pi, size=theta0.size)

        result = minimize(
            fun=lambda x: energy_fn(x),
            x0=x0,
            jac=lambda x: grad_fn(x),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter} if maxiter is not None else None,
        )

        total_nfev += int(result.nfev)
        total_njev += int(result.njev) if result.njev is not None else 0

        if best_fun is None or result.fun < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x, dtype=float)
            best_stats = {
                "status": int(result.status),
                "success": bool(result.success),
                "message": str(result.message),
                "grad_norm": float(np.linalg.norm(result.jac)) if result.jac is not None else None,
            }

    stats = {
        "nfev": total_nfev,
        "njev": total_njev,
        "restarts": restarts,
        **best_stats,
    }
    return best_x if best_x is not None else theta0, stats


def _meta_optimize(
    theta0: np.ndarray,
    energy_fn,
    grad_fn,
    *,
    steps: int,
    model,
    r: float,
    step_scale: float,
    dtheta_clip: float | None,
    verbose: bool,
) -> tuple[np.ndarray, dict]:
    torch = _maybe_import_torch()
    device = next(model.parameters()).device
    model.eval()

    theta = theta0.copy()
    n_params = theta.size
    if n_params == 0:
        return theta, {"avg_delta_norm": 0.0}

    h = torch.zeros((n_params, model.hidden_size), device=device)
    c = torch.zeros((n_params, model.hidden_size), device=device)

    delta_norms: list[float] = []
    max_delta_vals: list[float] = []
    energies: list[float] = []

    for inner_idx in range(steps):
        energy = energy_fn(theta)
        grad = grad_fn(theta)

        x = preprocess_gradients_torch(grad, r=r, device=device)
        with torch.no_grad():
            delta, (h, c) = model(x, (h, c))
        delta = delta.cpu().numpy().astype(float)

        if dtheta_clip is not None:
            delta = np.clip(delta, -dtheta_clip, dtheta_clip)
        delta = step_scale * delta

        theta = theta + delta

        grad_norm = float(np.linalg.norm(grad))
        delta_norm = float(np.linalg.norm(delta))
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        energies.append(energy)
        delta_norms.append(delta_norm)
        max_delta_vals.append(max_delta)

        if verbose:
            print(
                f"  inner {inner_idx + 1}/{steps}: "
                f"E={energy:.10f}, ||g||={grad_norm:.3e}, "
                f"||Δθ||={delta_norm:.3e}, max|Δθ|={max_delta:.3e}"
            )

    stats = {
        "avg_delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0,
        "avg_max_delta": float(np.mean(max_delta_vals)) if max_delta_vals else 0.0,
        "final_energy": energies[-1] if energies else energy_fn(theta),
    }
    return theta, stats


def _hybrid_optimize(
    theta0: np.ndarray,
    energy_fn,
    grad_fn,
    *,
    model,
    r: float,
    step_scale: float,
    dtheta_clip: float | None,
    warmup_steps: int,
    theta_bound: str,
    lbfgs_maxiter: int | None,
    lbfgs_restarts: int,
    verbose: bool,
) -> tuple[np.ndarray, dict]:
    theta_meta, meta_stats = _meta_optimize(
        theta0,
        energy_fn,
        grad_fn,
        steps=warmup_steps,
        model=model,
        r=r,
        step_scale=step_scale,
        dtheta_clip=dtheta_clip,
        verbose=verbose,
    )
    theta_lbfgs, lbfgs_stats = _lbfgs_optimize(
        theta_meta,
        energy_fn,
        grad_fn,
        theta_bound=theta_bound,
        maxiter=lbfgs_maxiter,
        restarts=lbfgs_restarts,
    )
    stats = {
        "meta": meta_stats,
        "lbfgs": lbfgs_stats,
    }
    return theta_lbfgs, stats


def run_meta_adapt_vqe(
    qubit_op: SparsePauliOp,
    reference_state: QuantumCircuit,
    estimator: BaseEstimatorV2,
    *,
    pool: Sequence[str] | None = None,
    pool_mode: str = "ham_terms_plus_imag_partners",
    ferm_op: FermionicOp | None = None,
    mapper: JordanWignerMapper | None = None,
    cse_include_diagonal: bool = False,
    max_depth: int = 20,
    inner_steps: int = 30,
    eps_grad: float = 1e-4,
    eps_energy: float = 1e-3,
    lstm_optimizer: CoordinateWiseLSTMOptimizer | None = None,
    meta_model=None,
    seed: int = 11,
    r: float = 10.0,
    reuse_lstm_state: bool = True,
    wrap_angles: bool = True,
    probe_convention: str = "half_angle",
    inner_optimizer: str = "lbfgs",
    theta_bound: str = "none",
    meta_step_scale: float = 1.0,
    meta_dtheta_clip: float | None = 0.25,
    meta_warmup_steps: int = 15,
    lbfgs_maxiter: int | None = None,
    lbfgs_restarts: int = 1,
    theta_init_noise: float = 0.0,
    allow_repeats: bool = False,
    verbose: bool = True,
) -> MetaAdaptVQEResult:
    pool_specs: list[dict] | None = None
    pool_labels: list[str] | None = None

    if pool_mode == "cse_density_ops":
        if pool is not None:
            pool_specs = list(pool)  # assume list of dict specs
        else:
            if ferm_op is None or mapper is None:
                raise ValueError("ferm_op and mapper are required for cse_density_ops pool.")
            pool_specs = build_cse_density_pool_from_fermionic(
                ferm_op,
                mapper,
                include_diagonal=cse_include_diagonal,
            )
        if not pool_specs:
            raise ValueError("Operator pool is empty.")
    else:
        if pool is None:
            pool = build_operator_pool_from_hamiltonian(qubit_op, mode=pool_mode)
        pool_labels = list(pool)
        if not pool_labels:
            raise ValueError("Operator pool is empty.")

    if lstm_optimizer is None:
        lstm_optimizer = CoordinateWiseLSTMOptimizer(seed=seed)

    rng = np.random.default_rng(seed)
    ops: list = []
    theta = np.zeros((0,), dtype=float)
    energy_current = estimate_energy(estimator, reference_state, qubit_op, [])
    prev_energy: float | None = None

    lstm_state = lstm_optimizer.init_state(0)

    diagnostics: dict[str, list] = {"outer": []}
    if pool_mode == "cse_density_ops":
        diagnostics["pool_size"] = [len(pool_specs)]
    else:
        diagnostics["pool_size"] = [len(pool_labels)]

    shift, coeff = _probe_shift_and_coeff(probe_convention)

    for outer_idx in range(max_depth):
        if pool_mode == "cse_density_ops":
            circuit, _params = build_adapt_circuit_grouped(reference_state, ops)
        else:
            circuit, _params = build_adapt_circuit(reference_state, ops)

        def energy_fn(values: Iterable[float]) -> float:
            return estimate_energy(estimator, circuit, qubit_op, list(values))

        if pool_mode == "cse_density_ops":
            pool_gradients = _compute_grouped_pool_gradients(
                estimator,
                circuit,
                qubit_op,
                theta,
                pool_specs,
                probe_convention=probe_convention,
            )
            max_idx = int(np.argmax(np.abs(pool_gradients)))
            max_abs_grad = float(abs(pool_gradients[max_idx]))
        else:
            pool_gradients = []
            max_abs_grad = -1.0
            max_idx = 0
            for idx, label in enumerate(pool_labels):
                probe_plus = _append_probe_gate(circuit, label, angle=shift)
                probe_minus = _append_probe_gate(circuit, label, angle=-shift)
                e_plus = estimate_energy(estimator, probe_plus, qubit_op, theta)
                e_minus = estimate_energy(estimator, probe_minus, qubit_op, theta)
                grad_val = coeff * (e_plus - e_minus)
                pool_gradients.append(grad_val)
                if abs(grad_val) > max_abs_grad:
                    max_abs_grad = abs(grad_val)
                    max_idx = idx

        if prev_energy is not None and abs(energy_current - prev_energy) < eps_energy:
            if verbose:
                print(
                    f"ADAPT stop: |ΔE|={abs(energy_current - prev_energy):.3e} < {eps_energy}"
                )
            break
        if max_abs_grad < eps_grad:
            if verbose:
                print(f"ADAPT stop: max|g|={max_abs_grad:.3e} < {eps_grad}")
            break

        if pool_mode == "cse_density_ops":
            chosen_spec = pool_specs[max_idx]
            ops.append(chosen_spec)
            chosen_op = chosen_spec["name"]
            if not allow_repeats:
                pool_specs.pop(max_idx)
        else:
            chosen_op = pool_labels[max_idx]
            ops.append(chosen_op)
            if not allow_repeats:
                pool_labels.pop(max_idx)
        if verbose:
            print(
                f"ADAPT iter {len(ops)}: op={chosen_op}, max|g|={max_abs_grad:.6e}"
            )
            if pool_mode == "cse_density_ops":
                comp_preview = ", ".join(
                    f"{lbl}:{wt:.3f}" for lbl, wt in chosen_spec["paulis"][:6]
                )
                if len(chosen_spec["paulis"]) > 6:
                    comp_preview += ", ..."
                print(f"  components: {comp_preview}")

        theta = np.concatenate([theta, [0.0]])
        if theta_init_noise > 0.0:
            theta[-1] = rng.normal(scale=theta_init_noise)
        if reuse_lstm_state:
            h = np.vstack([lstm_state.h, np.zeros((1, lstm_optimizer.hidden_size))])
            c = np.vstack([lstm_state.c, np.zeros((1, lstm_optimizer.hidden_size))])
            lstm_state = LSTMState(h=h, c=c)
        else:
            lstm_state = lstm_optimizer.init_state(len(theta))

        if pool_mode == "cse_density_ops":
            circuit, _params = build_adapt_circuit_grouped(reference_state, ops)
        else:
            circuit, _params = build_adapt_circuit(reference_state, ops)

        def energy_fn(values: Iterable[float]) -> float:
            return estimate_energy(estimator, circuit, qubit_op, list(values))

        def grad_fn(values: Iterable[float]) -> np.ndarray:
            return parameter_shift_grad(energy_fn, np.asarray(values, dtype=float))

        inner_stats: dict | None = None
        if inner_optimizer == "meta":
            if meta_model is None:
                raise RuntimeError("Meta optimizer requested without a loaded model.")
            theta, inner_stats = _meta_optimize(
                theta,
                energy_fn,
                grad_fn,
                steps=inner_steps,
                model=meta_model,
                r=r,
                step_scale=meta_step_scale,
                dtheta_clip=meta_dtheta_clip,
                verbose=verbose,
            )
        elif inner_optimizer == "lbfgs":
            theta, inner_stats = _lbfgs_optimize(
                theta,
                energy_fn,
                grad_fn,
                theta_bound=theta_bound,
                maxiter=lbfgs_maxiter,
                restarts=lbfgs_restarts,
                rng=rng,
            )
        elif inner_optimizer == "hybrid":
            if meta_model is None:
                raise RuntimeError("Hybrid optimizer requested without a loaded model.")
            theta, inner_stats = _hybrid_optimize(
                theta,
                energy_fn,
                grad_fn,
                model=meta_model,
                r=r,
                step_scale=meta_step_scale,
                dtheta_clip=meta_dtheta_clip,
                warmup_steps=meta_warmup_steps,
                theta_bound=theta_bound,
                lbfgs_maxiter=lbfgs_maxiter,
                lbfgs_restarts=lbfgs_restarts,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown inner optimizer: {inner_optimizer}")

        if wrap_angles:
            theta = _wrap_angles(theta)

        prev_energy = energy_current
        energy_current = energy_fn(theta)
        diagnostics["outer"].append(
            {
                "outer_iter": len(ops),
                "chosen_op": chosen_op,
                "chosen_components": chosen_spec["paulis"] if pool_mode == "cse_density_ops" else None,
                "max_grad": max_abs_grad,
                "energy": energy_current,
                "pool_gradients": pool_gradients,
                "inner_stats": inner_stats,
                "inner_optimizer": inner_optimizer,
            }
        )

    return MetaAdaptVQEResult(
        energy=energy_current,
        params=list(map(float, theta)),
        operators=ops,
        diagnostics=diagnostics,
    )
