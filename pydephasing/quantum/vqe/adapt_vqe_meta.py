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
from ..symmetry import commutes, map_symmetry_ops_to_qubits

GROUPED_POOL_MODES = {"cse_density_ops", "uccsd_excitations"}


@dataclass
class MetaAdaptVQEResult:
    energy: float
    params: list[float]
    operators: list
    diagnostics: dict


def _canonicalize_to_hermitian(op: SparsePauliOp, *, atol: float) -> SparsePauliOp:
    """Return a numerically Hermitian SparsePauliOp from a possibly non-Hermitian input.

    If the operator is closer to anti-Hermitian than Hermitian, rotate by -i so
    the returned operator is Hermitian. This is used to avoid silently dropping
    Pauli terms with nontrivial imaginary components when building grouped pools.
    """
    op = op.simplify(atol=atol)
    herm = 0.5 * (op + op.adjoint())
    anti = 0.5 * (op - op.adjoint())
    herm_norm = float(np.sum(np.abs(herm.coeffs))) if len(herm.coeffs) else 0.0
    anti_norm = float(np.sum(np.abs(anti.coeffs))) if len(anti.coeffs) else 0.0
    if anti_norm > herm_norm:
        op = (-1j) * anti
    else:
        op = herm
    return op.simplify(atol=atol)


def _grouped_spec_from_sparse_pauli_op(
    op: SparsePauliOp,
    *,
    name: str,
    dedup_tol: float,
    seen: set[tuple] | None = None,
) -> dict | None:
    """Convert an operator to the grouped {name, paulis} spec used by grouped ADAPT pools.

    The conversion:
    - removes identity components,
    - coerces numerically-real coefficients to real weights,
    - rejects truly complex coefficients after Hermitian canonicalization,
    - drops near-zero weights and deduplicates based on a rounded signature.
    """
    if op is None:
        return None

    op = _canonicalize_to_hermitian(op, atol=dedup_tol)
    identity = "I" * op.num_qubits

    paulis: list[tuple[str, float]] = []
    for lbl, coeff in zip(op.paulis.to_labels(), op.coeffs):
        if lbl == identity:
            continue
        if abs(coeff.imag) > dedup_tol:
            raise ValueError(
                f"Non-real Pauli coefficient after Hermitian canonicalization in {name}: {coeff}."
            )
        weight = float(coeff.real)
        if abs(weight) <= dedup_tol:
            continue
        paulis.append((lbl, weight))

    if not paulis:
        return None

    paulis.sort(key=lambda item: item[0])
    key = tuple((lbl, round(w, 12)) for lbl, w in paulis)
    if seen is not None:
        if key in seen:
            return None
        seen.add(key)
    return {"name": name, "paulis": paulis}


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
    include_antihermitian_part: bool = True,
    include_hermitian_part: bool = True,
    dedup_tol: float = 1e-12,
    enforce_symmetry: bool = False,
    n_sites: int | None = None,
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
            if enforce_symmetry and n_sites is not None:
                if not _fermionic_term_in_sector(label, n_sites):
                    continue
            gamma = FermionicOp({label: 1.0}, num_spin_orbitals=ferm_op.num_spin_orbitals)
        elif include_diagonal and len(terms) in (4,):
            if enforce_symmetry and n_sites is not None:
                if not _fermionic_term_in_sector(label, n_sites):
                    continue
            gamma = FermionicOp({label: 1.0}, num_spin_orbitals=ferm_op.num_spin_orbitals)
        else:
            continue

        if include_antihermitian_part:
            # Imag/current-like quadrature: A = i (gamma - gamma^\dagger) (Hermitian).
            k_op = gamma - gamma.adjoint()
            if len(getattr(k_op, "_data", {})) != 0:
                qubit_k = mapper.map(k_op)
                if qubit_k is not None:
                    op_im = (-1j) * qubit_k
                    spec = _grouped_spec_from_sparse_pauli_op(
                        op_im,
                        name=f"gamma_im({label})",
                        dedup_tol=dedup_tol,
                        seen=seen,
                    )
                    if spec is not None:
                        pool.append(spec)

        if include_hermitian_part:
            # Real/hopping-like quadrature: B = gamma + gamma^\dagger (Hermitian).
            h_op = gamma + gamma.adjoint()
            if len(getattr(h_op, "_data", {})) != 0:
                qubit_h = mapper.map(h_op)
                if qubit_h is not None:
                    spec = _grouped_spec_from_sparse_pauli_op(
                        qubit_h,
                        name=f"gamma_re({label})",
                        dedup_tol=dedup_tol,
                        seen=seen,
                    )
                    if spec is not None:
                        pool.append(spec)

    return pool


def build_uccsd_excitation_pool(
    *,
    n_sites: int,
    num_particles: tuple[int, int],
    mapper: JordanWignerMapper,
    reps: int = 1,
    include_imaginary: bool = False,
    preserve_spin: bool = True,
    generalized: bool = False,
    dedup_tol: float = 1e-12,
) -> list[dict]:
    """Return grouped Pauli operators from UCCSD excitation generators."""
    if mapper is None:
        raise ValueError("mapper must be provided for UCCSD excitation pool.")

    try:
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
    except ImportError:  # pragma: no cover
        from qiskit_nature.circuit.library import HartreeFock, UCCSD

    try:
        initial_state = HartreeFock(
            num_spatial_orbitals=int(n_sites),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            qubit_mapper=mapper,
        )
    except TypeError:
        initial_state = HartreeFock(int(n_sites), (int(num_particles[0]), int(num_particles[1])), mapper)

    try:
        ansatz = UCCSD(
            num_spatial_orbitals=int(n_sites),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            qubit_mapper=mapper,
            reps=int(reps),
            initial_state=initial_state,
            generalized=generalized,
            preserve_spin=preserve_spin,
            include_imaginary=include_imaginary,
        )
    except TypeError:
        ansatz = UCCSD(
            int(n_sites),
            (int(num_particles[0]), int(num_particles[1])),
            mapper,
            reps=int(reps),
            initial_state=initial_state,
        )

    operators = list(getattr(ansatz, "operators", []))
    excitation_list = list(getattr(ansatz, "excitation_list", []))

    pool: list[dict] = []
    seen: set[tuple] = set()
    for idx, op in enumerate(operators):
        exc = excitation_list[idx] if idx < len(excitation_list) else None
        name = f"uccsd_exc({exc})" if exc is not None else f"uccsd_exc_{idx}"
        spec = _grouped_spec_from_sparse_pauli_op(
            op,
            name=name,
            dedup_tol=dedup_tol,
            seen=seen,
        )
        if spec is not None:
            pool.append(spec)

    return pool


def _fermionic_term_in_sector(label: str, n_sites: int) -> bool:
    terms = label.split()
    delta_n = 0
    delta_sz = 0.0
    for term in terms:
        if term.startswith("+_"):
            idx = int(term[2:])
            delta_n += 1
            delta_sz += 0.5 if idx < n_sites else -0.5
        elif term.startswith("-_"):
            idx = int(term[2:])
            delta_n -= 1
            delta_sz -= 0.5 if idx < n_sites else -0.5
    return delta_n == 0 and abs(delta_sz) < 1e-12


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


def estimate_expectation(
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    op: SparsePauliOp,
    params: Sequence[float],
) -> float:
    job = estimator.run([(circuit, op, list(params))])
    result = job.result()
    ev = result[0].data.evs
    return float(np.real(ev))


def compute_sector_diagnostics(
    *,
    estimator: BaseEstimatorV2,
    circuit: QuantumCircuit,
    params: Sequence[float],
    mapper: JordanWignerMapper,
    n_sites: int,
    n_target: int,
    sz_target: float,
    simplify_atol: float = 1e-12,
) -> dict:
    """Compute N/Sz moments and simple sector leakage diagnostics for a circuit state."""
    n_q, sz_q = map_symmetry_ops_to_qubits(mapper, int(n_sites))

    n_mean = estimate_expectation(estimator, circuit, n_q, params)
    sz_mean = estimate_expectation(estimator, circuit, sz_q, params)

    n2_q = (n_q @ n_q).simplify(atol=simplify_atol)
    sz2_q = (sz_q @ sz_q).simplify(atol=simplify_atol)

    n2_mean = estimate_expectation(estimator, circuit, n2_q, params)
    sz2_mean = estimate_expectation(estimator, circuit, sz2_q, params)

    var_n = float(max(0.0, n2_mean - n_mean * n_mean))
    var_sz = float(max(0.0, sz2_mean - sz_mean * sz_mean))

    return {
        "N_mean": float(n_mean),
        "Sz_mean": float(sz_mean),
        "VarN": var_n,
        "VarSz": var_sz,
        "abs_N_err": float(abs(n_mean - float(n_target))),
        "abs_Sz_err": float(abs(sz_mean - float(sz_target))),
    }


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
    alpha0: float = 1.0,
    alpha_k: float = 0.0,
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
        alpha = alpha0 * (1.0 + alpha_k * (inner_idx / max(1, steps - 1)))
        delta = alpha * step_scale * delta

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
    cse_include_diagonal: bool = True,
    cse_include_antihermitian_part: bool = True,
    cse_include_hermitian_part: bool = True,
    uccsd_reps: int = 1,
    uccsd_include_imaginary: bool = False,
    uccsd_generalized: bool = False,
    uccsd_preserve_spin: bool = True,
    n_sites: int | None = None,
    n_up: int | None = None,
    n_down: int | None = None,
    enforce_sector: bool = False,
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
    warmup_steps: int = 5,
    polish_steps: int = 3,
    meta_alpha0: float = 1.0,
    meta_alpha_k: float = 0.0,
    logger=None,
    log_every: int = 1,
    verbose: bool = True,
) -> MetaAdaptVQEResult:
    pool_specs: list[dict] | None = None
    pool_labels: list[str] | None = None
    n_q = None
    sz_q = None
    if enforce_sector:
        if mapper is None or n_sites is None:
            raise ValueError("mapper and n_sites are required when enforce_sector is True.")
        n_q, sz_q = map_symmetry_ops_to_qubits(mapper, n_sites)

    if pool_mode in GROUPED_POOL_MODES:
        if pool is not None:
            pool_specs = list(pool)  # assume list of dict specs
        else:
            if pool_mode == "cse_density_ops":
                if ferm_op is None or mapper is None:
                    raise ValueError("ferm_op and mapper are required for cse_density_ops pool.")
                pool_specs = build_cse_density_pool_from_fermionic(
                    ferm_op,
                    mapper,
                    include_diagonal=cse_include_diagonal,
                    include_antihermitian_part=cse_include_antihermitian_part,
                    include_hermitian_part=cse_include_hermitian_part,
                    enforce_symmetry=enforce_sector,
                    n_sites=n_sites,
                )
            elif pool_mode == "uccsd_excitations":
                if mapper is None or n_sites is None or n_up is None or n_down is None:
                    raise ValueError("mapper, n_sites, n_up, n_down are required for uccsd_excitations pool.")
                pool_specs = build_uccsd_excitation_pool(
                    n_sites=n_sites,
                    num_particles=(int(n_up), int(n_down)),
                    mapper=mapper,
                    reps=uccsd_reps,
                    include_imaginary=uccsd_include_imaginary,
                    preserve_spin=uccsd_preserve_spin,
                    generalized=uccsd_generalized,
                )
            else:
                raise ValueError(f"Unknown pool mode: {pool_mode}")
        if not pool_specs:
            raise ValueError("Operator pool is empty.")
        if enforce_sector and n_q is not None and sz_q is not None:
            filtered = []
            for spec in pool_specs:
                op = SparsePauliOp.from_list(spec["paulis"])
                if commutes(op, n_q) and commutes(op, sz_q):
                    filtered.append(spec)
            pool_specs = filtered
            if not pool_specs:
                raise ValueError("Operator pool empty after symmetry filtering.")
    else:
        if pool is None:
            pool = build_operator_pool_from_hamiltonian(qubit_op, mode=pool_mode)
        pool_labels = list(pool)
        if not pool_labels:
            raise ValueError("Operator pool is empty.")
        if enforce_sector and n_q is not None and sz_q is not None:
            filtered = []
            for label in pool_labels:
                op = SparsePauliOp.from_list([(label, 1.0)])
                if commutes(op, n_q) and commutes(op, sz_q):
                    filtered.append(label)
            pool_labels = filtered
            if not pool_labels:
                raise ValueError("Operator pool empty after symmetry filtering.")

    if lstm_optimizer is None:
        lstm_optimizer = CoordinateWiseLSTMOptimizer(seed=seed)

    rng = np.random.default_rng(seed)
    ops: list = []
    theta = np.zeros((0,), dtype=float)
    energy_current = estimate_energy(estimator, reference_state, qubit_op, [])
    prev_energy: float | None = None

    lstm_state = lstm_optimizer.init_state(0)

    diagnostics: dict[str, list] = {"outer": []}
    if pool_mode in GROUPED_POOL_MODES:
        diagnostics["pool_size"] = [len(pool_specs)]
    else:
        diagnostics["pool_size"] = [len(pool_labels)]

    shift, coeff = _probe_shift_and_coeff(probe_convention)

    if logger is not None:
        sector_extra = None
        if mapper is not None and n_sites is not None and n_up is not None and n_down is not None:
            sector_extra = compute_sector_diagnostics(
                estimator=estimator,
                circuit=reference_state,
                params=[],
                mapper=mapper,
                n_sites=int(n_sites),
                n_target=int(n_up) + int(n_down),
                sz_target=0.5 * (int(n_up) - int(n_down)),
            )
        logger.log_point(
            it=0,
            energy=energy_current,
            max_grad=None,
            chosen_op=None,
            t_iter_s=0.0,
            t_cum_s=0.0,
            extra={
                "ansatz_len": 0,
                "n_params": 0,
                "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                **(sector_extra or {}),
            },
        )

    for outer_idx in range(max_depth):
        if logger is not None:
            logger.start_iter()
        if pool_mode in GROUPED_POOL_MODES and not pool_specs:
            if verbose:
                print("ADAPT stop: operator pool exhausted.")
            break
        if pool_mode not in GROUPED_POOL_MODES and not pool_labels:
            if verbose:
                print("ADAPT stop: operator pool exhausted.")
            break
        if pool_mode in GROUPED_POOL_MODES:
            circuit, _params = build_adapt_circuit_grouped(reference_state, ops)
        else:
            circuit, _params = build_adapt_circuit(reference_state, ops)

        def energy_fn(values: Iterable[float]) -> float:
            return estimate_energy(estimator, circuit, qubit_op, list(values))

        if pool_mode in GROUPED_POOL_MODES:
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
            if logger is not None:
                dt, tc = logger.end_iter()
                if (outer_idx + 1) % max(1, log_every) == 0:
                    logger.log_point(
                        it=outer_idx + 1,
                        energy=energy_current,
                        max_grad=max_abs_grad,
                        chosen_op=None,
                        t_iter_s=dt,
                        t_cum_s=tc,
                        extra={
                            "ansatz_len": len(ops),
                            "n_params": len(theta),
                            "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                        },
                    )
            break
        if max_abs_grad < eps_grad:
            if verbose:
                print(f"ADAPT stop: max|g|={max_abs_grad:.3e} < {eps_grad}")
            if logger is not None:
                dt, tc = logger.end_iter()
                if (outer_idx + 1) % max(1, log_every) == 0:
                    logger.log_point(
                        it=outer_idx + 1,
                        energy=energy_current,
                        max_grad=max_abs_grad,
                        chosen_op=None,
                        t_iter_s=dt,
                        t_cum_s=tc,
                        extra={
                            "ansatz_len": len(ops),
                            "n_params": len(theta),
                            "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                        },
                    )
            break

        if pool_mode in GROUPED_POOL_MODES:
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
            if pool_mode in GROUPED_POOL_MODES:
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

        if pool_mode in GROUPED_POOL_MODES:
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
                alpha0=meta_alpha0,
                alpha_k=meta_alpha_k,
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
            warm_steps = max(0, min(inner_steps, warmup_steps))
            meta_steps = max(0, inner_steps - warm_steps)

            theta_warm, warm_stats = _lbfgs_optimize(
                theta,
                energy_fn,
                grad_fn,
                theta_bound=theta_bound,
                maxiter=warm_steps if warm_steps > 0 else None,
                restarts=1,
                rng=rng,
            )
            if verbose and warm_steps > 0:
                e_warm = energy_fn(theta_warm)
                g_warm = grad_fn(theta_warm)
                print(
                    f"  warm-start: E={e_warm:.10f}, ||g||={np.linalg.norm(g_warm):.3e}, "
                    f"max|g|={np.max(np.abs(g_warm)):.3e}"
                )

            theta_meta = theta_warm.copy()
            delta_norms = []
            if meta_steps > 0:
                if meta_model is not None:
                    torch = _maybe_import_torch()
                    device = next(meta_model.parameters()).device
                    meta_model.eval()
                    n_params = theta_meta.size
                    h = torch.zeros((n_params, meta_model.hidden_size), device=device)
                    c = torch.zeros((n_params, meta_model.hidden_size), device=device)

                    for step_idx in range(meta_steps):
                        grad = grad_fn(theta_meta)
                        x = preprocess_gradients_torch(grad, r=r, device=device)
                        with torch.no_grad():
                            delta, (h, c) = meta_model(x, (h, c))
                        delta = delta.cpu().numpy().astype(float)
                        if meta_dtheta_clip is not None:
                            delta = np.clip(delta, -meta_dtheta_clip, meta_dtheta_clip)
                        alpha = meta_alpha0 * (1.0 + meta_alpha_k * (step_idx / max(1, meta_steps - 1)))
                        delta = alpha * delta
                        theta_meta = theta_meta + delta
                        delta_norms.append(float(np.linalg.norm(delta)))
                else:
                    for step_idx in range(meta_steps):
                        grad = grad_fn(theta_meta)
                        delta = -grad
                        if meta_dtheta_clip is not None:
                            delta = np.clip(delta, -meta_dtheta_clip, meta_dtheta_clip)
                        alpha = meta_alpha0 * (1.0 + meta_alpha_k * (step_idx / max(1, meta_steps - 1)))
                        delta = alpha * delta
                        theta_meta = theta_meta + delta
                        delta_norms.append(float(np.linalg.norm(delta)))

            meta_stats = {"avg_delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0}

            theta_polish, polish_stats = _lbfgs_optimize(
                theta_meta,
                energy_fn,
                grad_fn,
                theta_bound=theta_bound,
                maxiter=polish_steps if polish_steps > 0 else None,
                restarts=1,
                rng=rng,
            )
            if verbose and polish_steps > 0:
                e_polish = energy_fn(theta_polish)
                g_polish = grad_fn(theta_polish)
                print(
                    f"  polish: E={e_polish:.10f}, ||g||={np.linalg.norm(g_polish):.3e}, "
                    f"max|g|={np.max(np.abs(g_polish)):.3e}"
                )
            theta = theta_polish
            inner_stats = {
                "warm": warm_stats,
                "meta": meta_stats,
                "polish": polish_stats,
            }
        else:
            raise ValueError(f"Unknown inner optimizer: {inner_optimizer}")

        if wrap_angles:
            theta = _wrap_angles(theta)

        prev_energy = energy_current
        energy_current = energy_fn(theta)

        sector_diag = None
        if mapper is not None and n_sites is not None and n_up is not None and n_down is not None:
            sector_diag = compute_sector_diagnostics(
                estimator=estimator,
                circuit=circuit,
                params=theta,
                mapper=mapper,
                n_sites=int(n_sites),
                n_target=int(n_up) + int(n_down),
                sz_target=0.5 * (int(n_up) - int(n_down)),
            )

        if logger is not None:
            dt, tc = logger.end_iter()
            if (outer_idx + 1) % max(1, log_every) == 0:
                logger.log_point(
                    it=outer_idx + 1,
                    energy=energy_current,
                    max_grad=max_abs_grad,
                    chosen_op=chosen_op,
                    t_iter_s=dt,
                    t_cum_s=tc,
                    extra={
                        "ansatz_len": len(ops),
                        "n_params": len(theta),
                        "pool_size": len(pool_specs) if pool_specs is not None else len(pool_labels),
                        **(sector_diag or {}),
                    },
                )
        diagnostics["outer"].append(
            {
                "outer_iter": len(ops),
                "chosen_op": chosen_op,
                "chosen_components": chosen_spec["paulis"] if pool_mode in GROUPED_POOL_MODES else None,
                "max_grad": max_abs_grad,
                "energy": energy_current,
                "pool_gradients": pool_gradients,
                "inner_stats": inner_stats,
                "inner_optimizer": inner_optimizer,
                "sector": sector_diag,
            }
        )

    return MetaAdaptVQEResult(
        energy=energy_current,
        params=list(map(float, theta)),
        operators=ops,
        diagnostics=diagnostics,
    )
