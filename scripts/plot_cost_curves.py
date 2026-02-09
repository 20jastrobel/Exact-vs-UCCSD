#!/usr/bin/env python3
"""Plot cost-normalized error curves from cached run logs.

This reads:
  - regular VQE logs under runs/compare_vqe/logs_*/history.jsonl
  - ADAPT runs under runs/*/history.jsonl (matched by meta.json)

and plots best-achieved |ΔE| versus a cost proxy like n_circuits_executed.
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
    "uccsd": "UCCSD",
}
ANSATZ_COLORS = {
    "adapt_cse_hybrid": "#264653",
    "adapt_uccsd_hybrid": "#2a9d8f",
    "vqe_cse_ops": "#e9c46a",
    "uccsd": "#e76f51",
}


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
    rows.sort(key=lambda r: (float(r.get("t_cum_s", 0.0)), int(r.get("iter", 0))))
    return rows


def _load_json(path: Path):
    return json.loads(path.read_text())


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
        # Require result.json so we don't accidentally plot an interrupted run
        # (which may contain only partial history rows).
        result_path = run_dir / "result.json"
        if not meta_path.exists() or not hist_path.exists() or not result_path.exists():
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


def _best_so_far_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    best = np.minimum.accumulate(y)
    return x, best


def _extract_series_from_rows(
    *,
    hist: list[dict],
    exact_energy: float,
    x_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in hist:
        if row.get(x_key) is None:
            continue
        energy = row.get("energy")
        delta = row.get("delta_e")
        if delta is None:
            if energy is None:
                continue
            delta = float(energy) - float(exact_energy)
        xs.append(float(row[x_key]))
        ys.append(abs(float(delta)))
    if not xs:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=str, default="runs/compare_vqe/compare_rows.json")
    ap.add_argument("--compare-dir", type=str, default="runs/compare_vqe", help="Root dir that contains logs_* folders.")
    ap.add_argument("--runs-root", type=str, default="runs", help="Root dir that contains ADAPT runs with meta.json/history.jsonl.")
    ap.add_argument("--sites", nargs="*", type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument("--x", choices=["n_circuits_executed", "n_pauli_terms_measured", "n_estimator_calls", "t_cum_s"], default="n_circuits_executed")
    ap.add_argument("--logx", action="store_true", default=True)
    ap.add_argument("--no-logx", dest="logx", action="store_false")
    ap.add_argument("--logy", action="store_true", default=False)
    ap.add_argument("--out-dir", type=str, default="runs/compare_vqe")
    args = ap.parse_args()

    compare_rows = json.loads(Path(args.compare).read_text())
    if not isinstance(compare_rows, list):
        raise ValueError(f"Expected list in {args.compare}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = Path(args.compare_dir)
    runs_root = Path(args.runs_root)

    for n_sites in sorted(set(int(s) for s in args.sites)):
        n_up, n_down = _infer_unique_sector_from_compare_rows(rows=compare_rows, sites=n_sites)
        exact_vals = [row.get("exact") for row in compare_rows if int(row.get("sites", -1)) == n_sites and row.get("exact") is not None]
        if not exact_vals:
            raise ValueError(f"Missing exact energy in compare rows for L={n_sites}")
        exact = float(exact_vals[0])

        fig, ax = plt.subplots(figsize=(9.6, 5.0))

        for ansatz in ANSATZ_ORDER:
            if ansatz.startswith("adapt_"):
                pool = "cse_density_ops" if ansatz == "adapt_cse_hybrid" else "uccsd_excitations"
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
                hist = _load_history_jsonl(run_dir / "history.jsonl")
                x, y = _extract_series_from_rows(hist=hist, exact_energy=exact, x_key=args.x)
            else:
                log_dir = compare_dir / f"logs_{ansatz}_L{n_sites}_Nup{n_up}_Ndown{n_down}"
                hist = _load_history_jsonl(log_dir / "history.jsonl")
                x, y = _extract_series_from_rows(hist=hist, exact_energy=exact, x_key=args.x)

            if x.size == 0:
                continue
            x, y_best = _best_so_far_curve(x, y)
            ax.plot(
                x,
                y_best,
                label=ANSATZ_LABELS.get(ansatz, ansatz),
                color=ANSATZ_COLORS.get(ansatz, None),
                linewidth=2.0,
            )

        if args.logx:
            ax.set_xscale("log")
        if args.logy:
            ax.set_yscale("log")

        ax.set_xlabel(args.x)
        ax.set_ylabel("|ΔE|")
        sz = 0.5 * (n_up - n_down)
        ax.set_title(f"Best-so-far |ΔE| vs cost (L={n_sites}, n_up={n_up}, n_down={n_down}, Sz={sz:+.1f})")
        ax.grid(True, which="both", axis="both", alpha=0.25)
        ax.legend(fontsize=9, ncol=2)

        fig.tight_layout()
        out_path = out_dir / f"cost_curve_abs_delta_e_vs_{args.x}_L{n_sites}_Nup{n_up}_Ndown{n_down}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
