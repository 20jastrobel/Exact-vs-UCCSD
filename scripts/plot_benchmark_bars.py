#!/usr/bin/env python3
"""Plot runtime/cost counters and circuit metrics as grouped bar charts.

This is meant to answer: "Are runs actually comparable under the same hardware-ish
budget constraints?"

Inputs:
  - runs/compare_vqe/compare_rows.json: sector + ansatz mapping
  - runs/compare_vqe/logs_*/history.jsonl + result.json: VQE baselines
  - runs/*/meta.json + history.jsonl: ADAPT runs (matched by pool + sector)
  - optional circuit metrics JSON produced by scripts/benchmark_circuit_metrics.py
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


ANSATZ_ORDER = ["adapt_cse_hybrid", "adapt_uccsd_hybrid", "vqe_cse_ops", "uccsd"]
ANSATZ_LABELS = {
    "adapt_cse_hybrid": "ADAPT CSE (hybrid)",
    "adapt_uccsd_hybrid": "ADAPT UCCSD (hybrid)",
    "vqe_cse_ops": "VQE (CSE ops)",
    "uccsd": "VQE (UCCSD)",
}
ANSATZ_COLORS = {
    "adapt_cse_hybrid": "#264653",
    "adapt_uccsd_hybrid": "#2a9d8f",
    "vqe_cse_ops": "#e9c46a",
    "uccsd": "#e76f51",
}


def _load_json(path: Path):
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
    rows.sort(key=lambda r: (int(r.get("iter", 0)), float(r.get("t_cum_s", 0.0))))
    return rows


def _infer_unique_sector_from_compare_rows(*, rows: list[dict], sites: int) -> tuple[int, int]:
    sectors: set[tuple[int, int]] = set()
    for row in rows:
        if int(row.get("sites", -1)) != int(sites):
            continue
        if row.get("n_up") is None or row.get("n_down") is None:
            continue
        n_up = int(row["n_up"])
        n_down = int(row["n_down"])
        n_total = int(row.get("N", n_up + n_down))
        if n_total != int(sites):
            continue  # ignore non-half-filling rows if present
        sectors.add((n_up, n_down))
    if not sectors:
        raise ValueError(f"No sector found in compare rows for L={sites}.")
    if len(sectors) != 1:
        raise ValueError(f"Multiple sectors found in compare rows for L={sites}: {sorted(sectors)}")
    return next(iter(sectors))


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
        raise FileNotFoundError(
            f"No cached {pool} ADAPT runs found for L={n_sites}, (n_up,n_down)=({n_up},{n_down})"
        )
    candidates.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return candidates[0][0]


def _grouped_bar(
    *,
    ax,
    sites: list[int],
    values_by_site: dict[int, dict[str, float]],
    ylabel: str,
    title: str,
    yscale: str = "linear",
) -> None:
    x = np.arange(len(sites))
    width = 0.18
    for idx, ansatz in enumerate(ANSATZ_ORDER):
        offset = (idx - (len(ANSATZ_ORDER) - 1) / 2) * width
        ys = [values_by_site[s].get(ansatz, np.nan) for s in sites]
        ax.bar(
            x + offset,
            ys,
            width=width,
            label=ANSATZ_LABELS.get(ansatz, ansatz),
            color=ANSATZ_COLORS.get(ansatz, None),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={s}" for s in sites])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.grid(True, axis="y", alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=str, default="runs/compare_vqe/compare_rows.json")
    ap.add_argument("--compare-dir", type=str, default="runs/compare_vqe")
    ap.add_argument("--runs-root", type=str, default="runs")
    ap.add_argument("--sites", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument(
        "--circuit-metrics",
        type=str,
        default="runs/compare_vqe/circuit_metrics_L2_L3_L4.json",
        help="JSON output from scripts/benchmark_circuit_metrics.py",
    )
    ap.add_argument("--out-dir", type=str, default="runs/compare_vqe")
    args = ap.parse_args()

    sites = sorted(set(int(s) for s in args.sites))
    compare_rows = _load_json(Path(args.compare))
    compare_dir = Path(args.compare_dir)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sector per L inferred from compare rows.
    sectors: dict[int, tuple[int, int]] = {}
    for L in sites:
        sectors[L] = _infer_unique_sector_from_compare_rows(rows=compare_rows, sites=L)

    # Load transpiled circuit metrics if available.
    circ_metrics_path = Path(args.circuit_metrics)
    circ_table_depth: dict[int, dict[str, float]] = {L: {} for L in sites}
    circ_table_cx: dict[int, dict[str, float]] = {L: {} for L in sites}
    circ_table_params: dict[int, dict[str, float]] = {L: {} for L in sites}
    if circ_metrics_path.exists():
        payload = _load_json(circ_metrics_path)
        for row in payload.get("rows", []):
            L = int(row.get("sites", -1))
            if L not in circ_table_depth:
                continue
            ansatz = str(row.get("ansatz"))
            if ansatz not in ANSATZ_ORDER:
                continue
            transp = row.get("transpiled") or {}
            circ_table_depth[L][ansatz] = float(transp.get("depth", np.nan))
            circ_table_cx[L][ansatz] = float(transp.get("cx", np.nan))
            circ_table_params[L][ansatz] = float(transp.get("n_params", np.nan))

    # Load final cost/runtime rows.
    cost_tables: dict[str, dict[int, dict[str, float]]] = {
        "t_cum_s": {L: {} for L in sites},
        "n_estimator_calls": {L: {} for L in sites},
        "n_circuits_executed": {L: {} for L in sites},
        "n_pauli_terms_measured": {L: {} for L in sites},
    }
    frac_tables: dict[str, dict[int, dict[str, float]]] = {
        "pauli_frac": {L: {} for L in sites},
        "time_frac": {L: {} for L in sites},
    }

    for L in sites:
        n_up, n_down = sectors[L]

        def _load_vqe_final(ansatz: str) -> tuple[dict, dict]:
            log_dir = compare_dir / f"logs_{ansatz}_L{L}_Nup{n_up}_Ndown{n_down}"
            hist = _load_history_jsonl(log_dir / "history.jsonl")
            final = hist[-1] if hist else {}
            result = _load_json(log_dir / "result.json") if (log_dir / "result.json").exists() else {}
            return final, result

        def _load_adapt_final(pool: str) -> tuple[dict, dict]:
            run_dir = find_adapt_run_dir(
                runs_root,
                n_sites=L,
                n_up=n_up,
                n_down=n_down,
                pool=pool,
            )
            hist = _load_history_jsonl(run_dir / "history.jsonl")
            final = hist[-1] if hist else {}
            meta = _load_json(run_dir / "meta.json") if (run_dir / "meta.json").exists() else {}
            return final, meta

        for ansatz in ANSATZ_ORDER:
            if ansatz == "adapt_cse_hybrid":
                final, meta = _load_adapt_final("cse_density_ops")
                budget = meta.get("budget") or {}
            elif ansatz == "adapt_uccsd_hybrid":
                final, meta = _load_adapt_final("uccsd_excitations")
                budget = meta.get("budget") or {}
            elif ansatz == "vqe_cse_ops":
                final, result = _load_vqe_final("vqe_cse_ops")
                budget = result.get("budget") or {}
            elif ansatz == "uccsd":
                final, result = _load_vqe_final("uccsd")
                budget = result.get("budget") or {}
            else:
                continue

            # Core counters
            for key in cost_tables.keys():
                if final.get(key) is not None:
                    cost_tables[key][L][ansatz] = float(final.get(key))

            # Fractions relative to configured budgets (if present)
            max_pauli = budget.get("max_pauli_terms_measured")
            if max_pauli is not None and final.get("n_pauli_terms_measured") is not None:
                try:
                    frac_tables["pauli_frac"][L][ansatz] = float(final["n_pauli_terms_measured"]) / float(max_pauli)
                except Exception:
                    pass
            max_time = budget.get("max_time_s")
            if max_time is not None and final.get("t_cum_s") is not None:
                try:
                    frac_tables["time_frac"][L][ansatz] = float(final["t_cum_s"]) / float(max_time)
                except Exception:
                    pass

    # Plot: circuit metrics (transpiled)
    if any(circ_table_depth[L] for L in sites):
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
        _grouped_bar(
            ax=axes[0],
            sites=sites,
            values_by_site=circ_table_depth,
            ylabel="Transpiled depth",
            title="Circuit depth (transpiled)",
        )
        _grouped_bar(
            ax=axes[1],
            sites=sites,
            values_by_site=circ_table_cx,
            ylabel="CX count (transpiled)",
            title="CX count (transpiled)",
        )
        _grouped_bar(
            ax=axes[2],
            sites=sites,
            values_by_site=circ_table_params,
            ylabel="# parameters",
            title="# parameters",
        )
        axes[0].legend(ncol=2, fontsize=8)
        fig.tight_layout()
        out_path = out_dir / "bench_circuit_metrics_L2_L3_L4.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Plot: cost metrics (final)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.2))
    _grouped_bar(
        ax=axes[0, 0],
        sites=sites,
        values_by_site=cost_tables["t_cum_s"],
        ylabel="t_cum_s (final)",
        title="Runtime proxy (t_cum_s at stop)",
    )
    _grouped_bar(
        ax=axes[0, 1],
        sites=sites,
        values_by_site=cost_tables["n_estimator_calls"],
        ylabel="n_estimator_calls (final)",
        title="Estimator calls (final)",
    )
    _grouped_bar(
        ax=axes[1, 0],
        sites=sites,
        values_by_site=cost_tables["n_circuits_executed"],
        ylabel="n_circuits_executed (final)",
        title="Circuits executed (final)",
    )
    _grouped_bar(
        ax=axes[1, 1],
        sites=sites,
        values_by_site=cost_tables["n_pauli_terms_measured"],
        ylabel="n_pauli_terms_measured (final)",
        title="Pauli terms measured (final)",
    )
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "bench_cost_metrics_L2_L3_L4.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Plot: budget fractions
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6))
    _grouped_bar(
        ax=axes[0],
        sites=sites,
        values_by_site=frac_tables["pauli_frac"],
        ylabel="(pauli_terms used) / (pauli budget)",
        title="Pauli budget fraction (final)",
    )
    _grouped_bar(
        ax=axes[1],
        sites=sites,
        values_by_site=frac_tables["time_frac"],
        ylabel="(t_cum_s) / (time budget)",
        title="Time budget fraction (final)",
    )
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "bench_budget_fractions_L2_L3_L4.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

