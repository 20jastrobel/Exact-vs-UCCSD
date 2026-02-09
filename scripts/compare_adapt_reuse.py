#!/usr/bin/env python3
"""
Run ADAPT (CSE + UCCSD pools) with and without operator reuse, and plot results.

This script is intentionally narrow: it answers "does allow_repeats change ADAPT
performance?" for specific L values in a fixed (n_up, n_down) sector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.primitives import StatevectorEstimator

from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.symmetry import exact_ground_energy_sector
from pydephasing.quantum.vqe.adapt_vqe_meta import (
    build_reference_state,
    run_meta_adapt_vqe,
)


def sector_occ(n_sites: int, n_up: int, n_down: int) -> list[int]:
    # Repo convention: [0..L-1] up, [L..2L-1] down.
    return list(range(int(n_up))) + list(range(int(n_sites), int(n_sites) + int(n_down)))


def run_adapt_once(
    *,
    n_sites: int,
    n_up: int,
    n_down: int,
    pool_mode: str,
    allow_repeats: bool,
    max_depth: int,
    inner_steps: int,
    warmup_steps: int,
    polish_steps: int,
    t: float,
    u: float,
    dv: float,
) -> dict:
    ferm_op = build_fermionic_hubbard(
        n_sites=int(n_sites),
        t=float(t),
        u=float(u),
        edges=default_1d_chain_edges(int(n_sites), periodic=False),
        v=[-float(dv) / 2, float(dv) / 2] if int(n_sites) == 2 else None,
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    exact = exact_ground_energy_sector(
        qubit_op,
        int(n_sites),
        int(n_up) + int(n_down),
        0.5 * (int(n_up) - int(n_down)),
    )
    reference = build_reference_state(
        int(qubit_op.num_qubits),
        sector_occ(int(n_sites), int(n_up), int(n_down)),
    )

    estimator = StatevectorEstimator()
    result = run_meta_adapt_vqe(
        qubit_op,
        reference,
        estimator,
        pool_mode=str(pool_mode),
        ferm_op=ferm_op,
        mapper=mapper,
        n_sites=int(n_sites),
        n_up=int(n_up),
        n_down=int(n_down),
        enforce_sector=True,
        inner_optimizer="hybrid",
        max_depth=int(max_depth),
        inner_steps=int(inner_steps),
        warmup_steps=int(warmup_steps),
        polish_steps=int(polish_steps),
        allow_repeats=bool(allow_repeats),
        verbose=False,
    )

    energy = float(result.energy)
    return {
        "sites": int(n_sites),
        "n_up": int(n_up),
        "n_down": int(n_down),
        "pool_mode": str(pool_mode),
        "allow_repeats": bool(allow_repeats),
        "energy": float(energy),
        "exact": float(exact),
        "delta_e": float(energy - float(exact)),
        "ansatz_len": int(len(result.operators)),
        "n_params": int(len(result.params)),
        "stop_reason": result.diagnostics.get("stop_reason"),
    }


def plot_bars(*, rows: list[dict], sites: list[int], out_path: Path) -> None:
    # Method order in the bar chart.
    order = [
        "adapt_cse_noreuse",
        "adapt_cse_reuse",
        "adapt_uccsd_noreuse",
        "adapt_uccsd_reuse",
    ]
    labels = {
        "adapt_cse_noreuse": "ADAPT CSE (no reuse)",
        "adapt_cse_reuse": "ADAPT CSE (reuse)",
        "adapt_uccsd_noreuse": "ADAPT UCCSD (no reuse)",
        "adapt_uccsd_reuse": "ADAPT UCCSD (reuse)",
    }

    table: dict[int, dict[str, float]] = {int(L): {} for L in sites}
    for row in rows:
        L = int(row["sites"])
        if L not in table:
            continue
        ansatz = str(row["ansatz"])
        table[L][ansatz] = abs(float(row["delta_e"]))

    data = [[table[int(L)].get(a, np.nan) for L in sites] for a in order]

    x = np.arange(len(sites))
    width = 0.20
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for idx, a in enumerate(order):
        offset = (idx - (len(order) - 1) / 2) * width
        bars = ax.bar(x + offset, data[idx], width=width, label=labels.get(a, a))
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"L={L}" for L in sites])
    ax.set_ylabel("|ΔE|")
    ax.set_title("ADAPT with/without operator reuse (absolute energy error)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", type=int, default=[2, 3])
    ap.add_argument("--n-up", type=int, default=1)
    ap.add_argument("--n-down", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--inner-steps", type=int, default=25)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--polish-steps", type=int, default=3)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--u", type=float, default=4.0)
    ap.add_argument("--dv", type=float, default=0.5)
    ap.add_argument("--out-dir", type=str, default="runs/compare_adapt_reuse")
    args = ap.parse_args()

    sites = sorted(set(int(s) for s in args.sites))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for L in sites:
        for pool_mode, pool_name in [
            ("cse_density_ops", "cse"),
            ("uccsd_excitations", "uccsd"),
        ]:
            for allow_repeats in (False, True):
                ansatz = f"adapt_{pool_name}_{'reuse' if allow_repeats else 'noreuse'}"
                print(f"Running L={L} {ansatz} ...")
                row = run_adapt_once(
                    n_sites=int(L),
                    n_up=int(args.n_up),
                    n_down=int(args.n_down),
                    pool_mode=pool_mode,
                    allow_repeats=bool(allow_repeats),
                    max_depth=int(args.max_depth),
                    inner_steps=int(args.inner_steps),
                    warmup_steps=int(args.warmup_steps),
                    polish_steps=int(args.polish_steps),
                    t=float(args.t),
                    u=float(args.u),
                    dv=float(args.dv),
                )
                row["ansatz"] = ansatz
                rows.append(row)

    (out_dir / "compare_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "sites": sites,
                "n_up": int(args.n_up),
                "n_down": int(args.n_down),
                "max_depth": int(args.max_depth),
                "inner_steps": int(args.inner_steps),
                "warmup_steps": int(args.warmup_steps),
                "polish_steps": int(args.polish_steps),
                "ham_params": {"t": float(args.t), "u": float(args.u), "dv": float(args.dv)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_png = out_dir / f"compare_bar_abs_L{'_'.join(str(s) for s in sites)}.png"
    plot_bars(rows=rows, sites=sites, out_path=out_png)
    print(f"Wrote {out_dir / 'compare_rows.json'}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

