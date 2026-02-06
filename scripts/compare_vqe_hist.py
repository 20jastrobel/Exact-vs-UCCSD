#!/usr/bin/env python3
"""Compare ADAPT vs regular VQE ansaetze and plot delta-E by ansatz type per L."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

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
from pydephasing.quantum.utils_particles import half_filling_num_particles
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_adapt_circuit_grouped,
    build_cse_density_pool_from_fermionic,
    build_reference_state,
    run_meta_adapt_vqe,
)


def sector_occ(n_sites: int, n_up: int, n_down: int) -> list[int]:
    return list(range(n_up)) + list(range(n_sites, n_sites + n_down))


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
    return float(hist[-1]["energy"])


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
) -> Path:
    run_dir = runs_root / f"{run_id}_L{n_sites}_Nup{n_up}_Ndown{n_down}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "sites": n_sites,
        "n_up": n_up,
        "n_down": n_down,
        "pool": pool,
        "inner_optimizer": inner_optimizer,
        "max_depth": max_depth,
        "eps_grad": eps_grad,
        "eps_energy": eps_energy,
        "ham_params": {"t": t, "u": u, "dv": dv},
        "exact_energy": exact_energy,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return run_dir


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
    )
    spec_map = {spec["name"]: spec for spec in pool_specs}

    ops: list[dict] = []
    missing: list[str] = []
    for name in chosen:
        spec = spec_map.get(name)
        if spec is None:
            missing.append(str(name))
        else:
            ops.append(spec)
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
) -> float:
    log_path = None
    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "history.jsonl"
        log_file = open(log_path, "w", encoding="utf-8")

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
    # UCCSD is commonly initialized at the Hartree-Fock reference point.
    initial_point = np.zeros(ansatz.num_parameters, dtype=float)
    optimizer = COBYLA(maxiter=150)

    t_start = time.perf_counter()
    t_last = t_start

    def _callback(eval_count: int, params: np.ndarray, energy: float, _meta: dict) -> None:
        nonlocal t_last
        if log_file is None:
            return
        now = time.perf_counter()
        t_iter_s = now - t_last
        t_cum_s = now - t_start
        t_last = now
        row = {
            "iter": int(eval_count),
            "energy": float(energy),
            "delta_e": float(energy - exact_energy),
            "t_iter_s": float(t_iter_s),
            "t_cum_s": float(t_cum_s),
            "ansatz": ansatz_kind,
            "n_params": int(len(params)),
        }
        log_file.write(json.dumps(row) + "\n")

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=_callback,
    )
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    if log_file is not None:
        log_file.close()
    return float(np.real(result.eigenvalue))


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
    maxiter: int = 150,
) -> float:
    log_path = None
    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "history.jsonl"
        log_file = open(log_path, "w", encoding="utf-8")

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

    estimator = StatevectorEstimator()
    initial_point = np.zeros(ansatz.num_parameters, dtype=float)
    optimizer = COBYLA(maxiter=maxiter)

    t_start = time.perf_counter()
    t_last = t_start

    def _callback(eval_count: int, params: np.ndarray, energy: float, _meta: dict) -> None:
        nonlocal t_last
        if log_file is None:
            return
        now = time.perf_counter()
        t_iter_s = now - t_last
        t_cum_s = now - t_start
        t_last = now
        row = {
            "iter": int(eval_count),
            "energy": float(energy),
            "delta_e": float(energy - exact_energy),
            "t_iter_s": float(t_iter_s),
            "t_cum_s": float(t_cum_s),
            "ansatz": "vqe_cse_ops",
            "n_params": int(len(params)),
        }
        log_file.write(json.dumps(row) + "\n")

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=_callback,
    )
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    if log_file is not None:
        log_file.close()
    return float(np.real(result.eigenvalue))

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
) -> float:
    estimator = StatevectorEstimator()
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
        logger=logger,
        log_every=1,
        verbose=False,
    )
    return float(result.energy)


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


def _rows_by_key(rows: list[dict]) -> dict[tuple[int, str], dict]:
    out: dict[tuple[int, str], dict] = {}
    for row in rows:
        try:
            sites = int(row.get("sites"))
            ansatz = str(row.get("ansatz"))
        except Exception:
            continue
        out[(sites, ansatz)] = row
    return out


def _last_energy_from_log(log_dir: Path) -> float | None:
    hist = _load_history(log_dir / "history.jsonl")
    if not hist:
        return None
    return float(hist[-1]["energy"])


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


def uccsd_adapt_settings(_n_sites: int) -> dict:
    """ADAPT-UCCSD settings (kept consistent across sizes)."""
    return {"max_depth": 6, "inner_steps": 25, "warmup_steps": 5, "polish_steps": 3}


def vqe_cse_settings(n_sites: int) -> dict:
    """Optimizer budget for CSE-ops VQE runs."""
    return {"maxiter": 150}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--allow-exact-compute",
        action="store_true",
        help="Allow computing exact energies when not cached.",
    )
    args = ap.parse_args()

    out_dir = Path("runs/compare_vqe")
    out_dir.mkdir(parents=True, exist_ok=True)

    t = 1.0
    u = 4.0
    dv = 0.5

    runs_root = Path("runs")

    # Half-filling per size: N_total = L. For odd L, Sz=0 is impossible, so pick Sz=+0.5.
    sectors = {
        n: half_filling_num_particles(n, sz_target=0.0 if n % 2 == 0 else 0.5)
        for n in (2, 3, 4, 5, 6)
    }

    adapt_kinds = ["adapt_cse_hybrid", "adapt_uccsd_hybrid"]
    vqe_kinds = ["uccsd", "vqe_cse_ops"]
    allowed_ansatz = set(adapt_kinds + vqe_kinds)

    compare_path = out_dir / "compare_rows.json"
    rows = _load_rows(compare_path)
    row_map = _rows_by_key(rows)
    row_map = {k: v for k, v in row_map.items() if v.get("ansatz") in allowed_ansatz}

    sites = sorted(set(int(s) for s in args.sites))

    for n_sites in sites:
        if n_sites not in sectors:
            raise ValueError(f"Missing sector mapping for L={n_sites}")
        n_up, n_down = sectors[n_sites]
        # Determine if anything is missing for this site.
        missing = []
        for kind in adapt_kinds + vqe_kinds:
            if args.force or (n_sites, kind) not in row_map:
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
            try:
                adapt_e = load_cached_adapt_energy(
                    runs_root,
                    n_sites=n_sites,
                    n_up=n_up,
                    n_down=n_down,
                    pool="cse_density_ops",
                )
            except Exception as exc:
                print(f"L={n_sites}: CSE cache missing, recomputing ({exc})")
                adapt_e = run_adapt(
                    n_sites=n_sites,
                    ferm_op=ferm_op,
                    qubit_op=qubit_op,
                    mapper=mapper,
                    n_up=n_up,
                    n_down=n_down,
                    pool_mode="cse_density_ops",
                )
            row_map[(n_sites, "adapt_cse_hybrid")] = {
                "sites": n_sites,
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
                settings = uccsd_adapt_settings(n_sites)
                run_id = f"adapt_uccsd_{int(time.time())}"
                run_dir = make_adapt_run_dir(
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
                )
                logger = RunLogger(run_dir)
                try:
                    adapt_uccsd_e = run_adapt(
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
                    )
                finally:
                    logger.close()
            row_map[(n_sites, "adapt_uccsd_hybrid")] = {
                "sites": n_sites,
                "ansatz": "adapt_uccsd_hybrid",
                "energy": adapt_uccsd_e,
                "exact": exact,
                "delta_e": adapt_uccsd_e - exact,
            }

        if "uccsd" in missing:
            log_dir = out_dir / f"logs_uccsd_L{n_sites}"
            cached_e = None if args.force else _last_energy_from_log(log_dir)
            if cached_e is not None:
                row_map[(n_sites, "uccsd")] = {
                    "sites": n_sites,
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
                    )
                except Exception as exc:
                    row_map[(n_sites, "uccsd")] = {
                        "sites": n_sites,
                        "ansatz": "uccsd",
                        "energy": None,
                        "exact": exact,
                        "delta_e": None,
                        "error": str(exc),
                    }
                else:
                    row_map[(n_sites, "uccsd")] = {
                        "sites": n_sites,
                        "ansatz": "uccsd",
                        "energy": e,
                        "exact": exact,
                        "delta_e": e - exact,
                    }

        if "vqe_cse_ops" in missing:
            log_dir = out_dir / f"logs_vqe_cse_ops_L{n_sites}"
            cached_e = None if args.force else _last_energy_from_log(log_dir)
            if cached_e is not None:
                row_map[(n_sites, "vqe_cse_ops")] = {
                    "sites": n_sites,
                    "ansatz": "vqe_cse_ops",
                    "energy": cached_e,
                    "exact": exact,
                    "delta_e": cached_e - exact,
                }
            else:
                try:
                    cse_settings = vqe_cse_settings(n_sites)
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
                        maxiter=cse_settings["maxiter"],
                    )
                except Exception as exc:
                    row_map[(n_sites, "vqe_cse_ops")] = {
                        "sites": n_sites,
                        "ansatz": "vqe_cse_ops",
                        "energy": None,
                        "exact": exact,
                        "delta_e": None,
                        "error": str(exc),
                    }
                else:
                    row_map[(n_sites, "vqe_cse_ops")] = {
                        "sites": n_sites,
                        "ansatz": "vqe_cse_ops",
                        "energy": e,
                        "exact": exact,
                        "delta_e": e - exact,
                    }

    rows = list(row_map.values())
    compare_path.write_text(json.dumps(rows, indent=2))

    # Per-L bar charts.
    for n_sites in sites:
        subset = [r for r in rows if r["sites"] == n_sites and r.get("delta_e") is not None]
        if not subset:
            continue
        delta_pairs = [(str(r["ansatz"]), float(r["delta_e"])) for r in subset]
        delta_pairs.sort(key=lambda kv: kv[1])
        labels = [k for k, _v in delta_pairs]
        deltas = np.array([v for _k, v in delta_pairs])

        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(labels, deltas, color="#264653")
        ax.set_ylabel("ΔE = E_ansatz - E_exact_sector")
        ax.set_title(f"Delta-E by ansatz type (L={n_sites}, n_up=1, n_down=1)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.tick_params(axis="x", rotation=20)

        for idx, val in enumerate(deltas):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        out_path = out_dir / f"delta_e_hist_L{n_sites}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot to {out_path}")

        # Absolute error bars.
        abs_pairs = [(str(r["ansatz"]), abs(float(r["delta_e"]))) for r in subset]
        abs_pairs.sort(key=lambda kv: kv[1])
        abs_labels = [k for k, _v in abs_pairs]
        abs_vals = np.array([v for _k, v in abs_pairs])

        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(abs_labels, abs_vals, color="#2a9d8f")
        ax.set_ylabel("|ΔE| = |E_ansatz - E_exact_sector|")
        ax.set_title(f"Absolute error by ansatz type (L={n_sites}, n_up=1, n_down=1)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

        for idx, val in enumerate(abs_vals):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        out_path = out_dir / f"delta_e_abs_hist_L{n_sites}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
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
        ax.set_title(f"Relative error by ansatz type (L={n_sites}, n_up=1, n_down=1)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

        for idx, val in enumerate(rel_vals):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        out_path = out_dir / f"delta_e_rel_hist_L{n_sites}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot to {out_path}")

    print(f"Saved comparison rows to {compare_path}")


if __name__ == "__main__":
    main()
