#!/usr/bin/env python3
"""Compute state-quality metrics (statevector-only) for cached compare runs.

Adds correctness signals beyond energy:
  - energy variance Var(H)
  - infidelity to exact sector ground state
  - physical observables (double occupancy, site densities, SzSz nearest-neighbor sum)
  - sector leakage p_leak (probability mass outside the target sector)

Also summarizes ADAPT sector-diagnostic trajectories (max errors, VarN/VarSz vs depth).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/...py` without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper

from pydephasing.quantum.ansatz import build_ansatz
from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.symmetry import exact_ground_state_sector, map_symmetry_ops_to_qubits, sector_basis_indices
from pydephasing.quantum.state_quality import (
    density_mae,
    energy_variance,
    hubbard_observables,
    infidelity,
    sector_probability,
)
from pydephasing.quantum.utils_particles import jw_reference_occupations_from_particles
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_adapt_circuit_grouped,
    build_cse_density_pool_from_fermionic,
    build_reference_state,
    build_uccsd_excitation_pool,
)


ANSATZ_ORDER = ["adapt_cse_hybrid", "adapt_uccsd_hybrid", "vqe_cse_ops", "uccsd"]
ANSATZ_LABELS = {
    "adapt_cse_hybrid": "ADAPT CSE (hybrid)",
    "adapt_uccsd_hybrid": "ADAPT UCCSD (hybrid)",
    "vqe_cse_ops": "VQE (CSE ops)",
    "uccsd": "UCCSD",
}
ANSATZ_COLORS = {
    "adapt_cse_hybrid": "#264653",
    "adapt_uccsd_hybrid": "#2a9d8f",
    "vqe_cse_ops": "#e9c46a",
    "uccsd": "#e76f51",
}


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")
    return rows


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
        if not meta_path.exists() or not hist_path.exists():
            continue
        try:
            meta = _load_json(meta_path)
            sites = int(meta.get("sites", -1))
            n_up_meta = int(meta.get("n_up", -1))
            n_down_meta = int(meta.get("n_down", -1))
        except Exception:
            continue
        if sites != int(n_sites) or n_up_meta != int(n_up) or n_down_meta != int(n_down):
            continue
        if str(meta.get("pool")) != str(pool):
            continue
        candidates.append((run_dir, meta))
    if not candidates:
        raise FileNotFoundError(f"No cached {pool} ADAPT runs found for L={n_sites}, (n_up,n_down)=({n_up},{n_down})")
    candidates.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return candidates[0][0]


def _resolve_cse_spec_name(name: str, spec_map: dict[str, dict]) -> str | None:
    """Back-compat for older cached runs that used gamma(...) naming."""
    if name in spec_map:
        return name
    if name.startswith("gamma(") and name.endswith(")"):
        label = name[len("gamma(") : -1]
        for prefix in ("gamma_im(", "gamma_re("):
            cand = f"{prefix}{label})"
            if cand in spec_map:
                return cand
    return None


def build_exact_state(
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    t: float,
    u: float,
    dv: float,
):
    ferm_op = build_fermionic_hubbard(
        n_sites=n_sites,
        t=t,
        u=u,
        edges=default_1d_chain_edges(n_sites, periodic=False),
        v=[-dv / 2, dv / 2] if n_sites == 2 else None,
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    n_target = int(n_up) + int(n_down)
    sz_target = 0.5 * (int(n_up) - int(n_down))
    e0, psi0 = exact_ground_state_sector(qubit_op, n_sites, n_target, sz_target)
    indices = sector_basis_indices(n_sites, n_target, sz_target)
    return ferm_op, qubit_op, mapper, float(e0), Statevector(psi0), indices


def _bound_state_from_grouped_ops(*, reference_state, specs: list[dict], theta: list[float]) -> Statevector:
    qc, params = build_adapt_circuit_grouped(reference_state, specs)
    bind = {p: v for p, v in zip(list(params), list(theta))}
    bound = qc.assign_parameters(bind, inplace=False)
    return Statevector.from_instruction(bound)


def _bound_state_from_uccsd(*, qubit_op, mapper, n_sites: int, n_up: int, n_down: int, reps: int, theta: list[float]) -> Statevector:
    ansatz = build_ansatz(
        "uccsd",
        qubit_op.num_qubits,
        reps=int(reps),
        mapper=mapper,
        n_sites=int(n_sites),
        num_particles=(int(n_up), int(n_down)),
    )
    params = list(ansatz.parameters)
    if len(params) != len(theta):
        raise ValueError(f"UCCSD parameter length mismatch: circuit has {len(params)} params, theta has {len(theta)}.")
    bound = ansatz.assign_parameters({p: v for p, v in zip(params, theta)}, inplace=False)
    return Statevector.from_instruction(bound)


def _extract_final_sector_metrics(*, state: Statevector, mapper: JordanWignerMapper, n_sites: int, n_target: int, sz_target: float) -> dict:
    n_q, sz_q = map_symmetry_ops_to_qubits(mapper, int(n_sites))
    n_mean = float(np.real(state.expectation_value(n_q)))
    sz_mean = float(np.real(state.expectation_value(sz_q)))
    n2 = float(np.real(state.expectation_value((n_q @ n_q).simplify(atol=1e-12))))
    sz2 = float(np.real(state.expectation_value((sz_q @ sz_q).simplify(atol=1e-12))))
    var_n = float(max(0.0, n2 - n_mean * n_mean))
    var_sz = float(max(0.0, sz2 - sz_mean * sz_mean))
    return {
        "N_mean_final": float(n_mean),
        "Sz_mean_final": float(sz_mean),
        "VarN_final": float(var_n),
        "VarSz_final": float(var_sz),
        "abs_N_err_final": float(abs(n_mean - float(n_target))),
        "abs_Sz_err_final": float(abs(sz_mean - float(sz_target))),
    }


def _plot_grouped_bars(
    *,
    rows: list[dict],
    sites: list[int],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    table: dict[int, dict[str, float]] = {s: {} for s in sites}
    for row in rows:
        s = int(row.get("sites", -1))
        if s not in table:
            continue
        ansatz = str(row.get("ansatz"))
        if ansatz not in ANSATZ_ORDER:
            continue
        val = row.get(metric)
        if val is None:
            continue
        table[s][ansatz] = float(val)

    data = []
    for ansatz in ANSATZ_ORDER:
        data.append([table[s].get(ansatz, np.nan) for s in sites])

    x = np.arange(len(sites))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    for idx, ansatz in enumerate(ANSATZ_ORDER):
        offset = (idx - (len(ANSATZ_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            data[idx],
            width=width,
            label=ANSATZ_LABELS.get(ansatz, ansatz),
            color=ANSATZ_COLORS.get(ansatz, None),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"L={s}" for s in sites])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=str, default="runs/compare_vqe/compare_rows.json")
    ap.add_argument("--compare-dir", type=str, default="runs/compare_vqe")
    ap.add_argument("--runs-root", type=str, default="runs")
    ap.add_argument("--sites", nargs="*", type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--u", type=float, default=4.0)
    ap.add_argument("--dv", type=float, default=0.5)
    ap.add_argument("--write-back", action="store_true", default=True, help="Write metrics back into compare_rows.json (default: True).")
    ap.add_argument("--no-write-back", dest="write_back", action="store_false")
    ap.add_argument("--out-dir", type=str, default="runs/compare_vqe")
    args = ap.parse_args()

    compare_path = Path(args.compare)
    rows = load_rows(compare_path)
    rows_out = [dict(r) for r in rows]  # preserve any rows we don't touch
    runs_root = Path(args.runs_root)
    compare_dir = Path(args.compare_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = sorted(set(int(s) for s in args.sites))

    leakage_traj_map: dict[tuple[int, str], dict] = {}

    for L in sites:
        # Infer (n_up,n_down) from compare_rows under half-filling (N=L).
        sectors: set[tuple[int, int]] = set()
        for row in rows_out:
            if int(row.get("sites", -1)) != int(L):
                continue
            if row.get("n_up") is None or row.get("n_down") is None:
                continue
            n_up = int(row["n_up"])
            n_down = int(row["n_down"])
            n_total = int(row.get("N", n_up + n_down))
            if n_total != int(L):
                continue
            sectors.add((n_up, n_down))
        if not sectors:
            print(f"[skip] L={L}: no half-filling sector rows found.")
            continue
        if len(sectors) != 1:
            raise ValueError(f"L={L}: multiple half-filling sectors found in compare rows: {sorted(sectors)}")
        n_up, n_down = next(iter(sectors))

        ferm_op, qubit_op, mapper, exact_e, exact_state, sector_idx = build_exact_state(
            n_sites=L,
            n_up=n_up,
            n_down=n_down,
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
        )

        exact_obs = hubbard_observables(state=exact_state, n_sites=L)

        for row in rows_out:
            if int(row.get("sites", -1)) != int(L):
                continue
            if int(row.get("n_up", -999)) != int(n_up) or int(row.get("n_down", -999)) != int(n_down):
                continue
            ansatz = str(row.get("ansatz"))
            if ansatz not in ANSATZ_ORDER:
                continue

            state = None
            err = None

            try:
                if ansatz == "adapt_cse_hybrid":
                    run_dir = find_adapt_run_dir(
                        runs_root,
                        n_sites=L,
                        n_up=n_up,
                        n_down=n_down,
                        pool="cse_density_ops",
                    )
                    res_path = run_dir / "result.json"
                    if not res_path.exists():
                        raise FileNotFoundError(f"Missing {res_path}")
                    result = _load_json(res_path)
                    theta = list(result.get("theta") or [])
                    chosen = list(result.get("operators") or [])

                    meta = _load_json(run_dir / "meta.json")
                    cse_flags = meta.get("cse") or {}
                    pool_specs = build_cse_density_pool_from_fermionic(
                        ferm_op,
                        mapper,
                        enforce_symmetry=True,
                        n_sites=L,
                        include_diagonal=bool(cse_flags.get("include_diagonal", True)),
                        include_antihermitian_part=bool(cse_flags.get("include_antihermitian_part", True)),
                        include_hermitian_part=bool(cse_flags.get("include_hermitian_part", True)),
                    )
                    spec_map = {spec["name"]: spec for spec in pool_specs}
                    specs: list[dict] = []
                    missing: list[str] = []
                    for name in chosen:
                        resolved = _resolve_cse_spec_name(str(name), spec_map)
                        if resolved is None:
                            missing.append(str(name))
                            continue
                        specs.append(spec_map[resolved])
                    if missing:
                        raise RuntimeError(f"Missing CSE operators from pool: {missing[:5]}")

                    ref = build_reference_state(
                        qubit_op.num_qubits,
                        jw_reference_occupations_from_particles(L, n_up, n_down),
                    )
                    state = _bound_state_from_grouped_ops(reference_state=ref, specs=specs, theta=theta)

                    # Trajectory summary for leakage.
                    hist = _load_history_jsonl(run_dir / "history.jsonl")
                    abs_n = [float(r.get("abs_N_err", np.nan)) for r in hist if r.get("abs_N_err") is not None]
                    abs_sz = [float(r.get("abs_Sz_err", np.nan)) for r in hist if r.get("abs_Sz_err") is not None]
                    var_n = [float(r.get("VarN", np.nan)) for r in hist if r.get("VarN") is not None]
                    var_sz = [float(r.get("VarSz", np.nan)) for r in hist if r.get("VarSz") is not None]
                    var_h_traj = [float(r.get("VarH", np.nan)) for r in hist if r.get("VarH") is not None]
                    max_abs_n = float(np.nanmax(abs_n)) if abs_n else None
                    max_abs_sz = float(np.nanmax(abs_sz)) if abs_sz else None
                    max_var_n = float(np.nanmax(var_n)) if var_n else None
                    max_var_sz = float(np.nanmax(var_sz)) if var_sz else None
                    max_var_h = float(np.nanmax(var_h_traj)) if var_h_traj else None
                    if abs_n or abs_sz or var_n or var_sz:
                        leakage_traj_map[(int(L), ansatz)] = {
                            "sites": int(L),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                            "ansatz": ansatz,
                            "depth": [int(r.get("ansatz_len", r.get("iter", 0))) for r in hist],
                            "abs_N_err": abs_n,
                            "abs_Sz_err": abs_sz,
                            "VarN": var_n,
                            "VarSz": var_sz,
                            "VarH": var_h_traj,
                            "max_abs_N_err": max_abs_n,
                            "max_abs_Sz_err": max_abs_sz,
                            "max_VarN": max_var_n,
                            "max_VarSz": max_var_sz,
                            "max_VarH": max_var_h,
                        }

                elif ansatz == "adapt_uccsd_hybrid":
                    run_dir = find_adapt_run_dir(
                        runs_root,
                        n_sites=L,
                        n_up=n_up,
                        n_down=n_down,
                        pool="uccsd_excitations",
                    )
                    res_path = run_dir / "result.json"
                    if not res_path.exists():
                        raise FileNotFoundError(f"Missing {res_path}")
                    result = _load_json(res_path)
                    theta = list(result.get("theta") or [])
                    chosen = list(result.get("operators") or [])

                    meta = _load_json(run_dir / "meta.json")
                    uccsd_flags = meta.get("uccsd") or {}
                    pool_specs = build_uccsd_excitation_pool(
                        n_sites=L,
                        num_particles=(n_up, n_down),
                        mapper=mapper,
                        reps=int(uccsd_flags.get("reps", 1)),
                        include_imaginary=bool(uccsd_flags.get("include_imaginary", False)),
                        preserve_spin=bool(uccsd_flags.get("preserve_spin", True)),
                        generalized=bool(uccsd_flags.get("generalized", False)),
                    )
                    spec_map = {spec["name"]: spec for spec in pool_specs}
                    specs: list[dict] = []
                    missing = []
                    for name in chosen:
                        if str(name) not in spec_map:
                            missing.append(str(name))
                            continue
                        specs.append(spec_map[str(name)])
                    if missing:
                        raise RuntimeError(f"Missing UCCSD operators from pool: {missing[:5]}")

                    ref = build_reference_state(
                        qubit_op.num_qubits,
                        jw_reference_occupations_from_particles(L, n_up, n_down),
                    )
                    state = _bound_state_from_grouped_ops(reference_state=ref, specs=specs, theta=theta)

                    hist = _load_history_jsonl(run_dir / "history.jsonl")
                    abs_n = [float(r.get("abs_N_err", np.nan)) for r in hist if r.get("abs_N_err") is not None]
                    abs_sz = [float(r.get("abs_Sz_err", np.nan)) for r in hist if r.get("abs_Sz_err") is not None]
                    var_n = [float(r.get("VarN", np.nan)) for r in hist if r.get("VarN") is not None]
                    var_sz = [float(r.get("VarSz", np.nan)) for r in hist if r.get("VarSz") is not None]
                    var_h_traj = [float(r.get("VarH", np.nan)) for r in hist if r.get("VarH") is not None]
                    max_abs_n = float(np.nanmax(abs_n)) if abs_n else None
                    max_abs_sz = float(np.nanmax(abs_sz)) if abs_sz else None
                    max_var_n = float(np.nanmax(var_n)) if var_n else None
                    max_var_sz = float(np.nanmax(var_sz)) if var_sz else None
                    max_var_h = float(np.nanmax(var_h_traj)) if var_h_traj else None
                    if abs_n or abs_sz or var_n or var_sz:
                        leakage_traj_map[(int(L), ansatz)] = {
                            "sites": int(L),
                            "n_up": int(n_up),
                            "n_down": int(n_down),
                            "ansatz": ansatz,
                            "depth": [int(r.get("ansatz_len", r.get("iter", 0))) for r in hist],
                            "abs_N_err": abs_n,
                            "abs_Sz_err": abs_sz,
                            "VarN": var_n,
                            "VarSz": var_sz,
                            "VarH": var_h_traj,
                            "max_abs_N_err": max_abs_n,
                            "max_abs_Sz_err": max_abs_sz,
                            "max_VarN": max_var_n,
                            "max_VarSz": max_var_sz,
                            "max_VarH": max_var_h,
                        }

                elif ansatz == "vqe_cse_ops":
                    log_dir = compare_dir / f"logs_vqe_cse_ops_L{L}_Nup{n_up}_Ndown{n_down}"
                    res_path = log_dir / "result.json"
                    if not res_path.exists():
                        raise FileNotFoundError(f"Missing {res_path}")
                    result = _load_json(res_path)
                    theta = list(result.get("optimal_point") or [])

                    # Use the same operator sequence as the cached ADAPT-CSE run.
                    run_dir = find_adapt_run_dir(
                        runs_root,
                        n_sites=L,
                        n_up=n_up,
                        n_down=n_down,
                        pool="cse_density_ops",
                    )
                    adapt_res_path = run_dir / "result.json"
                    if not adapt_res_path.exists():
                        raise FileNotFoundError(f"Missing {adapt_res_path} (needed to reconstruct VQE(CSE ops))")
                    adapt_res = _load_json(adapt_res_path)
                    chosen = list(adapt_res.get("operators") or [])

                    meta = _load_json(run_dir / "meta.json")
                    cse_flags = meta.get("cse") or {}
                    pool_specs = build_cse_density_pool_from_fermionic(
                        ferm_op,
                        mapper,
                        enforce_symmetry=True,
                        n_sites=L,
                        include_diagonal=bool(cse_flags.get("include_diagonal", True)),
                        include_antihermitian_part=bool(cse_flags.get("include_antihermitian_part", True)),
                        include_hermitian_part=bool(cse_flags.get("include_hermitian_part", True)),
                    )
                    spec_map = {spec["name"]: spec for spec in pool_specs}
                    specs: list[dict] = []
                    missing = []
                    for name in chosen:
                        resolved = _resolve_cse_spec_name(str(name), spec_map)
                        if resolved is None:
                            missing.append(str(name))
                            continue
                        specs.append(spec_map[resolved])
                    if missing:
                        raise RuntimeError(f"Missing CSE operators from pool: {missing[:5]}")

                    ref = build_reference_state(
                        qubit_op.num_qubits,
                        jw_reference_occupations_from_particles(L, n_up, n_down),
                    )
                    state = _bound_state_from_grouped_ops(reference_state=ref, specs=specs, theta=theta)

                elif ansatz == "uccsd":
                    log_dir = compare_dir / f"logs_uccsd_L{L}_Nup{n_up}_Ndown{n_down}"
                    res_path = log_dir / "result.json"
                    if not res_path.exists():
                        raise FileNotFoundError(f"Missing {res_path}")
                    result = _load_json(res_path)
                    theta = list(result.get("optimal_point") or [])
                    reps = int(result.get("reps", 2))
                    state = _bound_state_from_uccsd(
                        qubit_op=qubit_op,
                        mapper=mapper,
                        n_sites=L,
                        n_up=n_up,
                        n_down=n_down,
                        reps=reps,
                        theta=theta,
                    )
                else:
                    raise ValueError(f"Unknown ansatz {ansatz}")

            except Exception as exc:
                err = str(exc)

            if state is None:
                row["state_quality_error"] = err
                continue
            row.pop("state_quality_error", None)

            # State-quality metrics.
            var_h = energy_variance(qubit_op=qubit_op, state=state)
            infid = infidelity(psi_exact=np.asarray(exact_state.data), psi_approx=np.asarray(state.data))
            p_sector = sector_probability(statevector=np.asarray(state.data), indices=sector_idx)
            p_leak = float(max(0.0, 1.0 - p_sector))

            obs = hubbard_observables(state=state, n_sites=L)
            d_err = float(obs["double_occ"] - float(exact_obs["double_occ"]))
            dens_mae = density_mae(dens_a=obs["densities"], dens_b=exact_obs["densities"])
            szsz_err = float(obs["szsz_nn"] - float(exact_obs["szsz_nn"]))

            n_target = int(n_up) + int(n_down)
            sz_target = 0.5 * (int(n_up) - int(n_down))
            sector_metrics = _extract_final_sector_metrics(
                state=state,
                mapper=mapper,
                n_sites=L,
                n_target=n_target,
                sz_target=sz_target,
            )

            row.update(
                {
                    "var_h": float(var_h),
                    "infidelity": float(infid),
                    "p_leak": float(p_leak),
                    "double_occ": float(obs["double_occ"]),
                    "double_occ_exact": float(exact_obs["double_occ"]),
                    "double_occ_err": float(d_err),
                    "density_mae": float(dens_mae),
                    "szsz_nn": float(obs["szsz_nn"]),
                    "szsz_nn_exact": float(exact_obs["szsz_nn"]),
                    "szsz_nn_err": float(szsz_err),
                    **sector_metrics,
                    "exact_energy_check": float(exact_e),
                }
            )
            # Add ADAPT trajectory maxima if available.
            if ansatz.startswith("adapt_"):
                entry = leakage_traj_map.get((int(L), ansatz))
                if entry is not None:
                    for k in ("max_abs_N_err", "max_abs_Sz_err", "max_VarN", "max_VarSz", "max_VarH"):
                        if entry.get(k) is not None:
                            row[k] = float(entry[k])

    if args.write_back:
        compare_path.write_text(json.dumps(rows_out, indent=2))
        print(f"Wrote updated compare rows with state-quality metrics: {compare_path}")

    # Write leakage trajectory file for downstream analysis.
    leakage_traj = list(leakage_traj_map.values())
    leakage_path = out_dir / "leakage_traj.json"
    leakage_path.write_text(json.dumps(leakage_traj, indent=2))
    print(f"Wrote leakage trajectories: {leakage_path}")

    # Simple summary markdown table.
    summary_path = out_dir / "state_quality_summary.md"
    lines = ["# State-Quality Summary (Statevector)", ""]
    lines.append("Columns: `var_h`, `infidelity`, `p_leak`, `double_occ_err`, `density_mae`, `szsz_nn_err`.")
    lines.append("")
    lines.append("| L | Ansatz | var_h | infidelity | p_leak | double_occ_err | density_mae | szsz_nn_err |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for L in sites:
        for ansatz in ANSATZ_ORDER:
            row = next(
                (
                    r
                    for r in rows_out
                    if int(r.get("sites", -1)) == int(L)
                    and str(r.get("ansatz")) == ansatz
                    and r.get("var_h") is not None
                ),
                None,
            )
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(L)),
                        ANSATZ_LABELS.get(ansatz, ansatz),
                        f"{float(row['var_h']):.3e}",
                        f"{float(row['infidelity']):.3e}",
                        f"{float(row['p_leak']):.3e}",
                        f"{float(row['double_occ_err']):+.3e}",
                        f"{float(row['density_mae']):.3e}",
                        f"{float(row['szsz_nn_err']):+.3e}",
                    ]
                )
                + " |"
            )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {summary_path}")

    # Leakage summary table (ADAPT only).
    leak_summary_path = out_dir / "leakage_summary.md"
    def _fmt(x) -> str:
        if x is None:
            return "nan"
        try:
            return f"{float(x):.3e}"
        except Exception:
            return "nan"

    lines = ["# Leakage Summary (ADAPT)", ""]
    lines.append("| L | Ansatz | max_abs_N_err | max_abs_Sz_err | max_VarN | max_VarSz | max_VarH |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for entry in sorted(leakage_traj, key=lambda e: (int(e.get("sites", 0)), str(e.get("ansatz")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(entry["sites"])),
                    ANSATZ_LABELS.get(str(entry["ansatz"]), str(entry["ansatz"])),
                    _fmt(entry.get("max_abs_N_err")),
                    _fmt(entry.get("max_abs_Sz_err")),
                    _fmt(entry.get("max_VarN")),
                    _fmt(entry.get("max_VarSz")),
                    _fmt(entry.get("max_VarH")),
                ]
            )
            + " |"
        )
    leak_summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {leak_summary_path}")

    # Plots: grouped bars for a few metrics.
    _plot_grouped_bars(
        rows=rows_out,
        sites=sites,
        metric="var_h",
        ylabel="Var(H)",
        title="Energy Variance (final state)",
        out_path=out_dir / "state_quality_var_h.png",
    )
    _plot_grouped_bars(
        rows=rows_out,
        sites=sites,
        metric="infidelity",
        ylabel="1 - |<psi_exact|psi>|^2",
        title="Infidelity to Exact Sector Ground State (final state)",
        out_path=out_dir / "state_quality_infidelity.png",
    )
    _plot_grouped_bars(
        rows=rows_out,
        sites=sites,
        metric="p_leak",
        ylabel="1 - p_sector",
        title="Sector Leakage (final state probability mass outside sector)",
        out_path=out_dir / "state_quality_p_leak.png",
    )
    _plot_grouped_bars(
        rows=rows_out,
        sites=sites,
        metric="max_abs_N_err",
        ylabel="max_t |<N>-N_target|",
        title="ADAPT Max |N Error| Over Run (from history.jsonl)",
        out_path=out_dir / "leakage_max_abs_N_err.png",
    )
    _plot_grouped_bars(
        rows=rows_out,
        sites=sites,
        metric="max_VarN",
        ylabel="max_t Var(N)",
        title="ADAPT Max Var(N) Over Run (from history.jsonl)",
        out_path=out_dir / "leakage_max_varN.png",
    )
    _plot_grouped_bars(
        rows=rows_out,
        sites=sites,
        metric="max_VarH",
        ylabel="max_t Var(H)",
        title="ADAPT Max Var(H) Over Run (from history.jsonl)",
        out_path=out_dir / "state_quality_varH_over_depth.png",
    )
    print(f"Wrote plots under: {out_dir}")

    # Per-L leakage vs depth for ADAPT runs (if present).
    by_L: dict[int, list[dict]] = {}
    for entry in leakage_traj:
        by_L.setdefault(int(entry["sites"]), []).append(entry)
    for L, entries in by_L.items():
        if not entries:
            continue
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True)
        for entry in entries:
            ansatz = str(entry.get("ansatz"))
            depth = np.asarray(entry.get("depth") or [], dtype=float)
            var_n = np.asarray(entry.get("VarN") or [], dtype=float)
            var_sz = np.asarray(entry.get("VarSz") or [], dtype=float)
            var_h = np.asarray(entry.get("VarH") or [], dtype=float)
            ax1.plot(depth[: len(var_n)], var_n, label=f"{ANSATZ_LABELS.get(ansatz, ansatz)}", linewidth=2.0)
            ax2.plot(depth[: len(var_sz)], var_sz, label=f"{ANSATZ_LABELS.get(ansatz, ansatz)}", linewidth=2.0)
            if var_h.size:
                ax3.plot(depth[: len(var_h)], var_h, label=f"{ANSATZ_LABELS.get(ansatz, ansatz)}", linewidth=2.0)
        ax1.set_ylabel("VarN")
        ax2.set_ylabel("VarSz")
        ax3.set_ylabel("VarH")
        ax3.set_xlabel("ADAPT outer depth")
        ax1.grid(True, alpha=0.25)
        ax2.grid(True, alpha=0.25)
        ax3.grid(True, alpha=0.25)
        ax1.legend(fontsize=9, ncol=2)
        fig.suptitle(f"ADAPT Leakage vs Depth (L={L})")
        fig.tight_layout()
        out_path = out_dir / f"leakage_vs_depth_L{L}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
