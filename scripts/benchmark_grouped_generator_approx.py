#!/usr/bin/env python3
"""Microbenchmark grouped-generator implementation error (sum-form vs product-form).

For a grouped generator spec with G = sum_j c_j P_j, compare:
  U_exact(theta) = exp(-i (theta/2) G)
  U_prod(theta)  = Π_j exp(-i (theta/2) c_j P_j)   (current implementation)

and optionally a symmetric 2nd-order Trotter product:
  U_sym(theta)   = Π_j exp(-i (theta/4) c_j P_j) Π_j^rev exp(-i (theta/4) c_j P_j)

Metrics reported per selected operator:
  - distance proxy: 1 - |<psi| U_exact^† U_approx |psi>|^2
  - sector leakage: p_leak = 1 - sum_{basis in sector} |amp|^2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Allow running as `python scripts/...py` without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper
from scipy.sparse.linalg import expm_multiply

from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.observables_hubbard import jw_number_op, jw_sz_op
from pydephasing.quantum.symmetry import sector_basis_indices
from pydephasing.quantum.utils_particles import jw_reference_occupations_from_particles
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_cse_density_pool_from_fermionic,
    build_reference_state,
    build_uccsd_excitation_pool,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_history_jsonl(path: Path) -> list[dict]:
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


def _resolve_cse_spec_name(name: str, spec_map: dict[str, dict]) -> str | None:
    if name in spec_map:
        return name
    if name.startswith("gamma(") and name.endswith(")"):
        label = name[len("gamma(") : -1]
        for prefix in ("gamma_im(", "gamma_re("):
            cand = f"{prefix}{label})"
            if cand in spec_map:
                return cand
    return None


def _sector_probability(*, psi: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    idx = np.asarray(indices, dtype=int)
    return float(np.sum(np.abs(psi[idx]) ** 2))


def _apply_pauli_rotation(*, label: str, alpha: float, psi: np.ndarray, cache: dict[str, object], num_qubits: int) -> np.ndarray:
    """Apply exp(-i alpha P) to |psi>, where P is a Pauli string (SparsePauliOp label)."""
    if abs(alpha) < 1e-15:
        return psi
    mat = cache.get(label)
    if mat is None:
        mat = SparsePauliOp.from_list([(label, 1.0)]).to_matrix(sparse=True).tocsc()
        cache[label] = mat
    ppsi = mat.dot(psi)
    return (math.cos(alpha) * psi) - (1j * math.sin(alpha) * ppsi)


def _evolve_exact(*, G: SparsePauliOp, theta: float, psi0: np.ndarray) -> np.ndarray:
    mat = G.to_matrix(sparse=True).tocsc()
    A = (-1j * float(theta) / 2.0) * mat
    return np.asarray(expm_multiply(A, psi0), dtype=complex)


def _evolve_product(*, paulis: list[tuple[str, float]], theta: float, psi0: np.ndarray, order: str, seed: int, num_qubits: int) -> np.ndarray:
    terms = list(paulis)
    rng = np.random.default_rng(int(seed))
    if order == "random":
        rng.shuffle(terms)

    cache: dict[str, object] = {}
    psi = np.asarray(psi0, dtype=complex)
    for label, weight in terms:
        alpha = float(theta) * float(weight) / 2.0
        psi = _apply_pauli_rotation(label=label, alpha=alpha, psi=psi, cache=cache, num_qubits=num_qubits)
    return psi


def _evolve_symmetric(*, paulis: list[tuple[str, float]], theta: float, psi0: np.ndarray, order: str, seed: int, num_qubits: int) -> np.ndarray:
    terms = list(paulis)
    rng = np.random.default_rng(int(seed))
    if order == "random":
        rng.shuffle(terms)

    cache: dict[str, object] = {}
    psi = np.asarray(psi0, dtype=complex)
    # half-step forward
    for label, weight in terms:
        alpha = float(theta) * float(weight) / 4.0
        psi = _apply_pauli_rotation(label=label, alpha=alpha, psi=psi, cache=cache, num_qubits=num_qubits)
    # half-step reverse
    for label, weight in reversed(terms):
        alpha = float(theta) * float(weight) / 4.0
        psi = _apply_pauli_rotation(label=label, alpha=alpha, psi=psi, cache=cache, num_qubits=num_qubits)
    return psi


def _expect_number_and_sz(*, psi: np.ndarray, n_sites: int) -> tuple[float, float]:
    # N = sum_p n_p, Sz = sum_i Sz_i
    num_qubits = 2 * int(n_sites)
    st = Statevector(psi)
    n_mean = 0.0
    sz_mean = 0.0
    for p in range(num_qubits):
        n_mean += float(np.real(st.expectation_value(jw_number_op(n_sites=n_sites, orbital=p))))
    for i in range(int(n_sites)):
        sz_mean += float(np.real(st.expectation_value(jw_sz_op(n_sites=n_sites, site=i))))
    return float(n_mean), float(sz_mean)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None, help="ADAPT run directory containing meta.json/history.jsonl.")
    ap.add_argument("--runs-root", type=str, default="runs")
    ap.add_argument("--sites", type=int, default=None)
    ap.add_argument("--n-up", type=int, default=None)
    ap.add_argument("--n-down", type=int, default=None)
    ap.add_argument("--pool", type=str, choices=["cse_density_ops", "uccsd_excitations"], default=None)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--state", choices=["hf", "random_sector"], default="hf")
    ap.add_argument("--order", choices=["sorted", "random"], default="sorted", help="Ordering for product/symmetric evolutions.")
    ap.add_argument("--approx", choices=["prod", "symmetric"], default="prod")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--plot-out", type=str, default=None)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        if args.sites is None or args.n_up is None or args.n_down is None or args.pool is None:
            raise ValueError("Pass --run-dir or provide --sites/--n-up/--n-down/--pool.")
        # Discover latest run matching tuple.
        run_dir = None
        latest = -1.0
        for d in runs_root.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            hist_path = d / "history.jsonl"
            if not meta_path.exists() or not hist_path.exists():
                continue
            try:
                meta = _load_json(meta_path)
            except Exception:
                continue
            if int(meta.get("sites", -1)) != int(args.sites):
                continue
            if int(meta.get("n_up", -1)) != int(args.n_up) or int(meta.get("n_down", -1)) != int(args.n_down):
                continue
            if str(meta.get("pool")) != str(args.pool):
                continue
            mtime = meta_path.stat().st_mtime
            if mtime > latest:
                latest = mtime
                run_dir = d
        if run_dir is None:
            raise FileNotFoundError("No matching ADAPT run found.")

    meta = _load_json(run_dir / "meta.json")
    n_sites = int(meta["sites"])
    n_up = int(meta["n_up"])
    n_down = int(meta["n_down"])
    pool = str(meta["pool"])
    ham_params = meta.get("ham_params") or {}
    t = float(ham_params.get("t", 1.0))
    u = float(ham_params.get("u", 4.0))
    dv = float(ham_params.get("dv", 0.5))

    ferm_op = build_fermionic_hubbard(
        n_sites=n_sites,
        t=t,
        u=u,
        edges=default_1d_chain_edges(n_sites, periodic=False),
        v=[-dv / 2, dv / 2] if n_sites == 2 else None,
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)

    # Selected operators (names) from result.json preferred, fallback to history chosen_op.
    chosen: list[str] = []
    res_path = run_dir / "result.json"
    if res_path.exists():
        res = _load_json(res_path)
        chosen = [str(x) for x in (res.get("operators") or [])]
    if not chosen:
        hist = _load_history_jsonl(run_dir / "history.jsonl")
        chosen = [str(r["chosen_op"]) for r in hist if r.get("chosen_op")]
    if not chosen:
        raise RuntimeError(f"No chosen operators found in {run_dir}")

    # Build spec map for chosen operators.
    if pool == "cse_density_ops":
        cse_flags = meta.get("cse") or {}
        pool_specs = build_cse_density_pool_from_fermionic(
            ferm_op,
            mapper,
            enforce_symmetry=True,
            n_sites=n_sites,
            include_diagonal=bool(cse_flags.get("include_diagonal", True)),
            include_antihermitian_part=bool(cse_flags.get("include_antihermitian_part", True)),
            include_hermitian_part=bool(cse_flags.get("include_hermitian_part", True)),
        )
        spec_map = {spec["name"]: spec for spec in pool_specs}
        specs = []
        missing = []
        for name in chosen:
            resolved = _resolve_cse_spec_name(name, spec_map)
            if resolved is None:
                missing.append(name)
                continue
            specs.append(spec_map[resolved])
        if missing:
            raise RuntimeError(f"Missing chosen ops from CSE pool: {missing[:5]}")
    elif pool == "uccsd_excitations":
        uccsd_flags = meta.get("uccsd") or {}
        pool_specs = build_uccsd_excitation_pool(
            n_sites=n_sites,
            num_particles=(n_up, n_down),
            mapper=mapper,
            reps=int(uccsd_flags.get("reps", 1)),
            include_imaginary=bool(uccsd_flags.get("include_imaginary", False)),
            preserve_spin=bool(uccsd_flags.get("preserve_spin", True)),
            generalized=bool(uccsd_flags.get("generalized", False)),
        )
        spec_map = {spec["name"]: spec for spec in pool_specs}
        specs = []
        missing = []
        for name in chosen:
            if name not in spec_map:
                missing.append(name)
                continue
            specs.append(spec_map[name])
        if missing:
            raise RuntimeError(f"Missing chosen ops from UCCSD pool: {missing[:5]}")
    else:
        raise ValueError(f"Unsupported pool={pool!r}")

    # Build initial state |psi>.
    num_qubits = int(qubit_op.num_qubits)
    n_target = n_up + n_down
    sz_target = 0.5 * (n_up - n_down)
    sector_idx = sector_basis_indices(n_sites, n_target, sz_target)

    if args.state == "hf":
        occ = jw_reference_occupations_from_particles(n_sites, n_up, n_down)
        ref = build_reference_state(num_qubits, occ)
        psi0 = np.asarray(Statevector.from_instruction(ref).data, dtype=complex)
    else:
        rng = np.random.default_rng(int(args.seed))
        psi0 = np.zeros((2 ** num_qubits,), dtype=complex)
        # random vector supported only on the target sector
        z = rng.normal(size=len(sector_idx)) + 1j * rng.normal(size=len(sector_idx))
        z = z / np.linalg.norm(z)
        psi0[np.asarray(sector_idx, dtype=int)] = z

    results: list[dict] = []
    for spec in specs:
        name = str(spec["name"])
        paulis = list(spec["paulis"])
        G = SparsePauliOp.from_list(paulis).simplify(atol=1e-12)

        psi_exact = _evolve_exact(G=G, theta=float(args.theta), psi0=psi0)
        if args.approx == "prod":
            psi_apx = _evolve_product(
                paulis=paulis,
                theta=float(args.theta),
                psi0=psi0,
                order=args.order,
                seed=int(args.seed),
                num_qubits=num_qubits,
            )
        else:
            psi_apx = _evolve_symmetric(
                paulis=paulis,
                theta=float(args.theta),
                psi0=psi0,
                order=args.order,
                seed=int(args.seed),
                num_qubits=num_qubits,
            )

        ov = np.vdot(psi_exact, psi_apx)
        dist = float(1.0 - min(1.0, max(0.0, float(np.abs(ov) ** 2))))

        p_leak_exact = float(max(0.0, 1.0 - _sector_probability(psi=psi_exact, indices=sector_idx)))
        p_leak_apx = float(max(0.0, 1.0 - _sector_probability(psi=psi_apx, indices=sector_idx)))
        n_mean_apx, sz_mean_apx = _expect_number_and_sz(psi=psi_apx, n_sites=n_sites)

        results.append(
            {
                "name": name,
                "n_terms": int(len(paulis)),
                "theta": float(args.theta),
                "approx": str(args.approx),
                "order": str(args.order),
                "dist_proxy": float(dist),
                "p_leak_exact": float(p_leak_exact),
                "p_leak_approx": float(p_leak_apx),
                "N_mean_approx": float(n_mean_apx),
                "Sz_mean_approx": float(sz_mean_apx),
                "abs_N_err_approx": float(abs(n_mean_apx - float(n_target))),
                "abs_Sz_err_approx": float(abs(sz_mean_apx - float(sz_target))),
            }
        )

    out_path = Path(args.out) if args.out else (run_dir / f"grouped_generator_approx_theta{args.theta:.3f}_{args.approx}_{args.order}.json")
    out_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "sites": n_sites,
                "n_up": n_up,
                "n_down": n_down,
                "pool": pool,
                "theta": float(args.theta),
                "approx": str(args.approx),
                "order": str(args.order),
                "state": str(args.state),
                "seed": int(args.seed),
                "operators": results,
            },
            indent=2,
        )
    )
    print(f"Wrote: {out_path}")

    # Simple plot: dist proxy and leakage bars.
    names = [r["name"] for r in results]
    dist_vals = np.asarray([r["dist_proxy"] for r in results], dtype=float)
    leak_vals = np.asarray([r["p_leak_approx"] for r in results], dtype=float)
    x = np.arange(len(names))

    fig, ax1 = plt.subplots(figsize=(12.0, 4.8))
    ax1.bar(x - 0.2, dist_vals, width=0.4, label="dist_proxy", color="#264653")
    ax1.bar(x + 0.2, leak_vals, width=0.4, label="p_leak_approx", color="#e76f51")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i+1}" for i in range(len(names))])
    ax1.set_xlabel("operator index in selected sequence")
    ax1.set_ylabel("value")
    ax1.set_title(f"Grouped-Generator Approx Error (L={n_sites}, pool={pool}, theta={args.theta}, approx={args.approx}, order={args.order})")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=9, ncol=2)
    fig.tight_layout()

    plot_path = Path(args.plot_out) if args.plot_out else out_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()

