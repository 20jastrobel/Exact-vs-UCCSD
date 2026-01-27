#!/usr/bin/env python3
"""3D block plots with x=ansatz, z=ΔE, and y=iterations or cumulative time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ANSATZ_ORDER = ["adapt_cse_hybrid", "clustered", "efficient_su2", "hea", "uccsd"]
ANSATZ_LABELS = {
    "adapt_cse_hybrid": "ADAPT",
    "clustered": "Clustered",
    "efficient_su2": "EfficientSU2",
    "hea": "HEA",
    "uccsd": "UCCSD",
}
ANSATZ_COLORS = {
    "adapt_cse_hybrid": "#264653",
    "clustered": "#2a9d8f",
    "efficient_su2": "#e9c46a",
    "hea": "#f4a261",
    "uccsd": "#e76f51",
}


@dataclass
class AdaptSeries:
    sites: int
    exact: float
    iters: np.ndarray
    t_iter: np.ndarray
    t_cum: np.ndarray
    delta_e: np.ndarray


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    meta = load_json(run_dir / "meta.json")
    hist: list[dict] = []
    with open(run_dir / "history.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hist.append(json.loads(line))
    hist.sort(key=lambda r: int(r["iter"]))
    return meta, hist


def discover_latest_run_for_site(runs_root: Path, site: int) -> Path:
    latest_path: Path | None = None
    latest_mtime = -np.inf
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        hist_path = run_dir / "history.jsonl"
        if not meta_path.exists() or not hist_path.exists():
            continue
        try:
            meta = load_json(meta_path)
            sites = int(meta.get("sites"))
        except Exception:
            continue
        if sites != site:
            continue
        mtime = meta_path.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = run_dir
    if latest_path is None:
        raise FileNotFoundError(f"No cached run found for L={site} under {runs_root}")
    return latest_path


def load_compare_rows(path: Path) -> list[dict]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")
    return rows


def regular_delta_e_by_ansatz(rows: list[dict], sites: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if int(row.get("sites")) != sites:
            continue
        ansatz = str(row.get("ansatz"))
        delta = row.get("delta_e")
        if delta is None:
            continue
        out[ansatz] = float(delta)
    return out


def build_adapt_series(run_dir: Path) -> AdaptSeries:
    meta, hist = load_run(run_dir)
    exact = float(meta["exact_energy"])
    sites = int(meta["sites"])
    iters = np.array([r["iter"] for r in hist], dtype=float)
    t_iter = np.array([r.get("t_iter_s", 0.0) for r in hist], dtype=float)
    t_cum = np.array([r.get("t_cum_s", 0.0) for r in hist], dtype=float)
    energies = np.array([r["energy"] for r in hist], dtype=float)
    delta_e = energies - exact
    return AdaptSeries(sites=sites, exact=exact, iters=iters, t_iter=t_iter, t_cum=t_cum, delta_e=delta_e)


def _bar3d_delta_e(
    *,
    xpos: np.ndarray,
    ypos: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    delta_e: np.ndarray,
    colors: list[str],
    title: str,
    ylabel: str,
    ytick_values: np.ndarray | None,
    ytick_labels: list[str] | None,
    out_path: Path,
) -> None:
    # Allow negative ΔE bars by anchoring at the minimum of (0, ΔE).
    zpos = np.minimum(0.0, delta_e)
    dz = np.abs(delta_e)

    fig = plt.figure(figsize=(12, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        xpos,
        ypos,
        zpos,
        dx,
        dy,
        dz,
        shade=True,
        color=colors,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.96,
    )

    ax.set_title(title)
    ax.set_xlabel("ansatz")
    ax.set_ylabel(ylabel)
    ax.set_zlabel("ΔE = E_ansatz - E_exact_sector")
    ax.view_init(elev=26, azim=-56)

    # Discrete ansatz ticks.
    ax.set_xticks(np.arange(len(ANSATZ_ORDER)))
    ax.set_xticklabels([ANSATZ_LABELS[a] for a in ANSATZ_ORDER], rotation=15, ha="right")

    if ytick_values is not None and ytick_labels is not None:
        ax.set_yticks(ytick_values)
        ax.set_yticklabels(ytick_labels)

    # Z limits with a small margin.
    z_lo = float(np.min(zpos))
    z_hi = float(np.max(zpos + dz))
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0
    span = z_hi - z_lo
    ax.set_zlim(z_lo - 0.05 * span, z_hi + 0.05 * span)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved 3D ansatz ΔE plot to {out_path}")


def _ansatz_x_block(ansatz: str, width: float) -> tuple[float, float]:
    idx = ANSATZ_ORDER.index(ansatz)
    x0 = idx - 0.5 * width
    return x0, width


def _regular_y_anchor(y_max: float, width: float) -> tuple[float, float]:
    # We do not have cached per-iteration/time traces for regular VQE.
    # Anchor them near the end of the ADAPT trajectory for comparison.
    return max(0.0, y_max - width), width


def build_iter_plot(
    *,
    sites: int,
    regular_delta: dict[str, float],
    adapt: AdaptSeries,
    out_path: Path,
) -> None:
    xw = 0.8
    yw = 1.0
    iter_max = float(np.max(adapt.iters))

    xpos: list[float] = []
    ypos: list[float] = []
    dx: list[float] = []
    dy: list[float] = []
    dz: list[float] = []
    colors: list[str] = []

    # ADAPT trajectory blocks.
    x0, dx0 = _ansatz_x_block("adapt_cse_hybrid", xw)
    for n, de in zip(adapt.iters, adapt.delta_e):
        xpos.append(x0)
        ypos.append(float(n) - 0.5 * yw)
        dx.append(dx0)
        dy.append(yw)
        dz.append(float(de))
        colors.append(ANSATZ_COLORS["adapt_cse_hybrid"])

    # Regular ansatz final values as single blocks.
    for ansatz in ANSATZ_ORDER:
        if ansatz == "adapt_cse_hybrid":
            continue
        if ansatz not in regular_delta:
            continue
        x0, dx0 = _ansatz_x_block(ansatz, xw)
        y0, dy0 = _regular_y_anchor(iter_max + 1.0, yw)
        xpos.append(x0)
        ypos.append(y0)
        dx.append(dx0)
        dy.append(dy0)
        dz.append(float(regular_delta[ansatz]))
        colors.append(ANSATZ_COLORS[ansatz])

    ypos_arr = np.asarray(ypos, dtype=float)
    dy_arr = np.asarray(dy, dtype=float)
    y_hi = float(np.max(ypos_arr + dy_arr))
    y_ticks = np.arange(0.0, y_hi + 0.5, 1.0)
    y_labels = [str(int(v)) for v in y_ticks]

    _bar3d_delta_e(
        xpos=np.asarray(xpos, dtype=float),
        ypos=ypos_arr,
        dx=np.asarray(dx, dtype=float),
        dy=dy_arr,
        delta_e=np.asarray(dz, dtype=float),
        colors=colors,
        title=f"L={sites}: ΔE by ansatz with iteration axis",
        ylabel="iteration n",
        ytick_values=y_ticks,
        ytick_labels=y_labels,
        out_path=out_path,
    )


def build_time_plot(
    *,
    sites: int,
    regular_delta: dict[str, float],
    adapt: AdaptSeries,
    out_path: Path,
) -> None:
    xw = 0.8
    t_max = float(np.max(adapt.t_cum))
    regular_width = max(1.0, 0.03 * t_max)

    xpos: list[float] = []
    ypos: list[float] = []
    dx: list[float] = []
    dy: list[float] = []
    dz: list[float] = []
    colors: list[str] = []

    # ADAPT blocks span their iteration wall-time increments when available.
    x0, dx0 = _ansatz_x_block("adapt_cse_hybrid", xw)
    prev_t = 0.0
    for t_cum, t_iter, de in zip(adapt.t_cum, adapt.t_iter, adapt.delta_e):
        t_cum_f = float(t_cum)
        t_iter_f = float(t_iter)
        if t_iter_f <= 0.0:
            t_start = prev_t
            t_width = max(1e-9, t_cum_f - prev_t)
        else:
            t_start = max(0.0, t_cum_f - t_iter_f)
            t_width = t_iter_f
        prev_t = t_cum_f

        xpos.append(x0)
        ypos.append(t_start)
        dx.append(dx0)
        dy.append(t_width)
        dz.append(float(de))
        colors.append(ANSATZ_COLORS["adapt_cse_hybrid"])

    # Regular ansatz final values as single blocks near the end.
    for ansatz in ANSATZ_ORDER:
        if ansatz == "adapt_cse_hybrid":
            continue
        if ansatz not in regular_delta:
            continue
        x0, dx0 = _ansatz_x_block(ansatz, xw)
        y0, dy0 = _regular_y_anchor(t_max + regular_width, regular_width)
        xpos.append(x0)
        ypos.append(y0)
        dx.append(dx0)
        dy.append(dy0)
        dz.append(float(regular_delta[ansatz]))
        colors.append(ANSATZ_COLORS[ansatz])

    ypos_arr = np.asarray(ypos, dtype=float)
    dy_arr = np.asarray(dy, dtype=float)
    y_hi = float(np.max(ypos_arr + dy_arr))
    tick_count = 6
    y_ticks = np.linspace(0.0, y_hi, tick_count)
    y_labels = [f"{v:.0f}" for v in y_ticks]

    _bar3d_delta_e(
        xpos=np.asarray(xpos, dtype=float),
        ypos=ypos_arr,
        dx=np.asarray(dx, dtype=float),
        dy=dy_arr,
        delta_e=np.asarray(dz, dtype=float),
        colors=colors,
        title=f"L={sites}: ΔE by ansatz with cumulative time axis",
        ylabel="cumulative wall time t^(n) [s]",
        ytick_values=y_ticks,
        ytick_labels=y_labels,
        out_path=out_path,
    )


def main(sites: Iterable[int] = (2, 3, 4)) -> None:
    runs_root = Path("runs")
    compare_rows_path = runs_root / "compare_vqe" / "compare_rows.json"
    rows = load_compare_rows(compare_rows_path)

    out_dir = runs_root / "compare_vqe"

    for site in sites:
        site_i = int(site)
        run_dir = discover_latest_run_for_site(runs_root, site_i)
        adapt = build_adapt_series(run_dir)
        regular_delta = regular_delta_e_by_ansatz(rows, site_i)

        build_iter_plot(
            sites=site_i,
            regular_delta=regular_delta,
            adapt=adapt,
            out_path=out_dir / f"delta_e_hist3d_ansatz_iter_L{site_i}.png",
        )
        build_time_plot(
            sites=site_i,
            regular_delta=regular_delta,
            adapt=adapt,
            out_path=out_dir / f"delta_e_hist3d_ansatz_time_L{site_i}.png",
        )


if __name__ == "__main__":
    main()
