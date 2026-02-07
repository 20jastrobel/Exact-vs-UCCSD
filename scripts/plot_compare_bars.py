#!/usr/bin/env python3
"""Grouped bar chart of comparison results by L and ansatz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANSATZ_ORDER = ["adapt_cse_hybrid", "adapt_uccsd_hybrid", "vqe_cse_ops", "uccsd"]
ANSATZ_LABELS = {
    "adapt_cse_hybrid": "ADAPT CSE (hybrid)",
    "adapt_uccsd_hybrid": "ADAPT UCCSD (hybrid)",
    "vqe_cse_ops": "VQE (CSE ops)",
    "uccsd": "UCCSD",
}


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=str, default="runs/compare_vqe/compare_rows.json")
    ap.add_argument("--sites", nargs="*", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--metric", choices=["delta", "abs", "rel"], default="abs")
    ap.add_argument("--out", type=str, default="runs/compare_vqe/compare_bar_L2_L5.png")
    args = ap.parse_args()

    rows = load_rows(Path(args.compare))
    sites = sorted(set(int(s) for s in args.sites))

    # Build table: site -> ansatz -> value
    table: dict[int, dict[str, float]] = {s: {} for s in sites}
    for row in rows:
        s = int(row.get("sites"))
        if s not in table:
            continue
        ansatz = str(row.get("ansatz"))
        if ansatz not in ANSATZ_ORDER:
            continue
        delta = row.get("delta_e")
        if delta is None:
            continue
        val = float(delta)
        if args.metric in ("abs", "rel"):
            val = abs(val)
        if args.metric == "rel":
            exact = row.get("exact")
            if exact is None:
                continue
            denom = abs(float(exact))
            if denom == 0.0:
                continue
            val = val / denom
        table[s][ansatz] = val

    # Ensure ordering and fill missing with NaN
    data = []
    for ansatz in ANSATZ_ORDER:
        series = [table[s].get(ansatz, np.nan) for s in sites]
        data.append(series)

    x = np.arange(len(sites))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for idx, ansatz in enumerate(ANSATZ_ORDER):
        offset = (idx - (len(ANSATZ_ORDER) - 1) / 2) * width
        bars = ax.bar(x + offset, data[idx], width=width, label=ANSATZ_LABELS.get(ansatz, ansatz))
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"L={s}" for s in sites])
    if args.metric == "abs":
        ylabel = "|ΔE|"
    elif args.metric == "rel":
        ylabel = "|ΔE| / |E_exact|"
    else:
        ylabel = "ΔE"
    ax.set_ylabel(ylabel)
    ax.set_title("Comparison by L and ansatz")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved bar chart to {out_path}")


if __name__ == "__main__":
    main()
