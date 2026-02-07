#!/usr/bin/env python3
"""
Benchmark circuit-structure metrics (depth, CX count, etc.) for the ansatz circuits
used in runs/compare_vqe/compare_rows.json.

This is intentionally hardware-agnostic: we transpile to a fixed basis gate set on an
all-to-all "virtual" backend (no coupling map) for a consistent apples-to-apples count.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Allow running as `python scripts/...py` without installing the repo as a package.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from qiskit import transpile
from qiskit_nature.second_q.mappers import JordanWignerMapper

from pydephasing.quantum.ansatz import build_ansatz
from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.utils_particles import half_filling_num_particles
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_adapt_circuit_grouped,
    build_cse_density_pool_from_fermionic,
    build_reference_state,
    build_uccsd_excitation_pool,
)


def sector_occ(n_sites: int, n_up: int, n_down: int) -> list[int]:
    return list(range(int(n_up))) + list(range(int(n_sites), int(n_sites) + int(n_down)))


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
        if sites != int(n_sites) or n_up_meta != int(n_up) or n_down_meta != int(n_down):
            continue
        if str(meta.get("pool")) != str(pool):
            continue
        candidates.append((run_dir, meta))
    if not candidates:
        raise FileNotFoundError(f"No cached {pool} ADAPT runs found for L={n_sites}, (n_up,n_down)=({n_up},{n_down})")
    candidates.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return candidates[0][0]


def chosen_operator_names_from_run_dir(run_dir: Path) -> list[str]:
    hist = _load_history(run_dir / "history.jsonl")
    chosen = [row.get("chosen_op") for row in hist if row.get("chosen_op")]
    return [str(x) for x in chosen]


def infer_sector_from_cache(
    runs_root: Path,
    *,
    n_sites: int,
    pool: str,
) -> tuple[int, int] | None:
    """Infer (n_up, n_down) from the newest cached run matching (L, pool)."""
    candidates: list[tuple[float, int, int]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        try:
            sites_meta = int(meta.get("sites", -1))
        except Exception:
            continue
        if sites_meta != int(n_sites):
            continue
        if str(meta.get("pool")) != str(pool):
            continue
        if meta.get("n_up") is None or meta.get("n_down") is None:
            continue
        try:
            n_up = int(meta["n_up"])
            n_down = int(meta["n_down"])
        except Exception:
            continue
        candidates.append((run_dir.stat().st_mtime, n_up, n_down))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _mtime, n_up, n_down = candidates[0]
    return int(n_up), int(n_down)


def circuit_metrics(qc) -> dict:
    ops = qc.count_ops()
    return {
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "n_params": int(qc.num_parameters),
        "ops": {str(k): int(v) for k, v in ops.items()},
        "cx": int(ops.get("cx", 0)),
    }


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument("--n-up", type=int, default=None, help="Override and use fixed n_up for all L.")
    ap.add_argument("--n-down", type=int, default=None, help="Override and use fixed n_down for all L.")
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--u", type=float, default=4.0)
    ap.add_argument("--dv", type=float, default=0.5)
    ap.add_argument("--runs-root", type=str, default="runs")
    ap.add_argument("--out", type=str, default="runs/compare_vqe/circuit_metrics.json")
    ap.add_argument("--plot-out", type=str, default="runs/compare_vqe/circuit_metrics_depth.png")
    ap.add_argument("--basis", type=str, default="rz,sx,x,cx")
    ap.add_argument("--opt", type=int, default=1)
    ap.add_argument("--seed-transpiler", type=int, default=7)
    args = ap.parse_args()

    sites = sorted(set(int(s) for s in args.sites))
    basis_gates = [g.strip() for g in str(args.basis).split(",") if g.strip()]
    runs_root = Path(args.runs_root)

    if (args.n_up is None) != (args.n_down is None):
        raise ValueError("Pass both --n-up and --n-down, or neither.")

    # Prefer matching whatever was cached for comparison runs; otherwise fall back to half-filling.
    sectors: dict[int, tuple[int, int]] = {}
    for n in sites:
        if args.n_up is not None and args.n_down is not None:
            sectors[n] = (int(args.n_up), int(args.n_down))
            continue
        sec = infer_sector_from_cache(runs_root, n_sites=n, pool="cse_density_ops")
        if sec is None:
            sec = infer_sector_from_cache(runs_root, n_sites=n, pool="uccsd_excitations")
        if sec is None:
            # Half-filling per size: N_total = L. For odd L, Sz=0 is impossible, so pick Sz=+0.5.
            sec = half_filling_num_particles(n, sz_target=0.0 if n % 2 == 0 else 0.5)
        sectors[n] = (int(sec[0]), int(sec[1]))

    rows: list[dict] = []
    t0 = time.perf_counter()

    for n_sites in sites:
        n_up, n_down = sectors[n_sites]
        mapper = JordanWignerMapper()

        ferm_op = build_fermionic_hubbard(
            n_sites=n_sites,
            t=args.t,
            u=args.u,
            edges=default_1d_chain_edges(n_sites, periodic=False),
            v=[-args.dv / 2, args.dv / 2] if n_sites == 2 else None,
        )
        qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
        num_qubits = int(qubit_op.num_qubits)

        ref = build_reference_state(num_qubits, sector_occ(n_sites, n_up, n_down))

        # ADAPT CSE: build circuit from cached chosen operators.
        cse_run = find_adapt_run_dir(
            runs_root,
            n_sites=n_sites,
            n_up=n_up,
            n_down=n_down,
            pool="cse_density_ops",
        )
        cse_chosen = chosen_operator_names_from_run_dir(cse_run)
        cse_pool = build_cse_density_pool_from_fermionic(
            ferm_op,
            mapper,
            enforce_symmetry=True,
            n_sites=n_sites,
            include_diagonal=True,
            include_antihermitian_part=True,
            include_hermitian_part=True,
        )
        cse_map = {spec["name"]: spec for spec in cse_pool}
        cse_specs: list[dict] = []
        missing: list[str] = []
        for name in cse_chosen:
            resolved = _resolve_cse_spec_name(name, cse_map)
            if resolved is None:
                missing.append(name)
                continue
            cse_specs.append(cse_map[resolved])
        if missing:
            raise RuntimeError(f"CSE chosen operators missing from pool for L={n_sites}: {missing[:5]}")
        cse_qc, _ = build_adapt_circuit_grouped(ref, cse_specs)

        # ADAPT UCCSD: build circuit from cached chosen operators.
        uccsd_run = find_adapt_run_dir(
            runs_root,
            n_sites=n_sites,
            n_up=n_up,
            n_down=n_down,
            pool="uccsd_excitations",
        )
        uccsd_chosen = chosen_operator_names_from_run_dir(uccsd_run)
        uccsd_pool = build_uccsd_excitation_pool(
            n_sites=n_sites,
            num_particles=(n_up, n_down),
            mapper=mapper,
            reps=1,
            include_imaginary=False,
            preserve_spin=True,
            generalized=False,
        )
        uccsd_map = {spec["name"]: spec for spec in uccsd_pool}
        uccsd_specs = [uccsd_map[name] for name in uccsd_chosen if name in uccsd_map]
        if len(uccsd_specs) != len(uccsd_chosen):
            missing = [name for name in uccsd_chosen if name not in uccsd_map]
            raise RuntimeError(f"UCCSD chosen operators missing from pool for L={n_sites}: {missing[:5]}")
        adapt_uccsd_qc, _ = build_adapt_circuit_grouped(ref, uccsd_specs)

        # Regular UCCSD (the one used in scripts/compare_vqe_hist.py): reps=2.
        uccsd_qc = build_ansatz(
            "uccsd",
            num_qubits,
            2,
            mapper,
            n_sites=n_sites,
            num_particles=(n_up, n_down),
        )

        circuits = [
            ("adapt_cse_hybrid", cse_qc, {"n_ops": len(cse_specs)}),
            ("vqe_cse_ops", cse_qc, {"n_ops": len(cse_specs)}),
            ("adapt_uccsd_hybrid", adapt_uccsd_qc, {"n_ops": len(uccsd_specs)}),
            ("uccsd", uccsd_qc, {"reps": 2}),
        ]

        for kind, qc, extra in circuits:
            raw = circuit_metrics(qc)
            tp = transpile(
                qc,
                basis_gates=basis_gates,
                optimization_level=int(args.opt),
                seed_transpiler=int(args.seed_transpiler),
            )
            transp = circuit_metrics(tp)
            row = {
                "sites": int(n_sites),
                "n_up": int(n_up),
                "n_down": int(n_down),
                "ansatz": str(kind),
                "n_qubits": int(num_qubits),
                "raw": raw,
                "transpiled": transp,
            }
            row.update(extra)
            rows.append(row)

    payload = {
        "meta": {
            "sites": sites,
            "ham_params": {"t": float(args.t), "u": float(args.u), "dv": float(args.dv)},
            "basis_gates": basis_gates,
            "transpile_opt_level": int(args.opt),
            "seed_transpiler": int(args.seed_transpiler),
            "runtime_s": float(time.perf_counter() - t0),
        },
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")

    # Optional plot: transpiled depth by L and ansatz.
    try:
        import matplotlib.pyplot as plt

        order = ["adapt_cse_hybrid", "adapt_uccsd_hybrid", "vqe_cse_ops", "uccsd"]
        labels = {
            "adapt_cse_hybrid": "ADAPT CSE",
            "adapt_uccsd_hybrid": "ADAPT UCCSD",
            "vqe_cse_ops": "VQE (CSE ops)",
            "uccsd": "UCCSD (reps=2)",
        }
        table = {s: {} for s in sites}
        for r in rows:
            table[int(r["sites"])][str(r["ansatz"])] = int(r["transpiled"]["depth"])

        x = np.arange(len(sites))
        width = 0.18
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for idx, kind in enumerate(order):
            offset = (idx - (len(order) - 1) / 2) * width
            ys = [table[s].get(kind, np.nan) for s in sites]
            ax.bar(x + offset, ys, width=width, label=labels.get(kind, kind))

        ax.set_xticks(x)
        ax.set_xticklabels([f"L={s}" for s in sites])
        ax.set_ylabel("Transpiled depth")
        ax.set_title("Circuit depth by L and ansatz (transpiled)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        plot_path = Path(args.plot_out)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"[warn] plot failed: {exc}")


if __name__ == "__main__":
    main()
