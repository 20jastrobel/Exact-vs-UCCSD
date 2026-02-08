#!/usr/bin/env python3
"""3D block plots with x=ansatz, z=ΔE, and y=iterations or cumulative time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ANSATZ_ORDER = ["adapt_cse_hybrid", "uccsd"]
ANSATZ_LABELS = {
    "adapt_cse_hybrid": "ADAPT CSE",
    "uccsd": "UCCSD",
}
ANSATZ_COLORS = {
    "adapt_cse_hybrid": "#264653",
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


@dataclass
class RegularSeries:
    ansatz: str
    iters: np.ndarray
    t_iter: np.ndarray
    t_cum: np.ndarray
    delta_e: np.ndarray


@dataclass
class BarSpec:
    ansatz: str
    x0: float
    y0: float
    z0: float
    dx: float
    dy: float
    dz: float


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


def discover_latest_run_for_site(
    runs_root: Path,
    site: int,
    *,
    pool: str = "cse_density_ops",
    n_up: int | None = None,
    n_down: int | None = None,
) -> Path:
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
            n_up_meta = int(meta.get("n_up", -1))
            n_down_meta = int(meta.get("n_down", -1))
        except Exception:
            continue
        if sites != site:
            continue
        if pool and str(meta.get("pool")) != pool:
            continue
        # Require sector information and optionally filter to a specific sector.
        if n_up_meta < 0 or n_down_meta < 0:
            continue
        if n_up is not None and int(n_up_meta) != int(n_up):
            continue
        if n_down is not None and int(n_down_meta) != int(n_down):
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


def regular_delta_e_by_ansatz(rows: list[dict], sites: int, *, n_up: int, n_down: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if int(row.get("sites")) != sites:
            continue
        if row.get("n_up") is None or row.get("n_down") is None:
            continue
        if int(row.get("n_up")) != int(n_up) or int(row.get("n_down")) != int(n_down):
            continue
        ansatz = str(row.get("ansatz"))
        delta = row.get("delta_e")
        if delta is None:
            continue
        out[ansatz] = float(delta)
    return out


def sector_for_site_from_compare_rows(*, rows: list[dict], sites: int, require_half_filling: bool = True) -> tuple[int, int]:
    """Infer a unique (n_up,n_down) sector for a site from compare_rows.json."""
    sectors: set[tuple[int, int]] = set()
    for row in rows:
        if int(row.get("sites", -1)) != int(sites):
            continue
        if row.get("n_up") is None or row.get("n_down") is None:
            continue
        n_up = int(row.get("n_up"))
        n_down = int(row.get("n_down"))
        if require_half_filling:
            n_total = int(row.get("N", n_up + n_down))
            if n_total != int(sites):
                continue
        sectors.add((n_up, n_down))
    if not sectors:
        raise ValueError(f"No sector info found in compare rows for L={sites}.")
    if len(sectors) != 1:
        raise ValueError(f"Multiple sectors found in compare rows for L={sites}: {sorted(sectors)}")
    return next(iter(sectors))


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


def load_regular_series(log_dir: Path, ansatz: str) -> RegularSeries | None:
    hist_path = log_dir / "history.jsonl"
    if not hist_path.exists():
        return None
    hist: list[dict] = []
    with open(hist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hist.append(json.loads(line))
    if not hist:
        return None
    hist.sort(key=lambda r: int(r.get("iter", 0)))
    iters = np.array([r.get("iter", 0) for r in hist], dtype=float)
    t_iter = np.array([r.get("t_iter_s", 0.0) for r in hist], dtype=float)
    t_cum = np.array([r.get("t_cum_s", 0.0) for r in hist], dtype=float)
    delta_e = np.array([r.get("delta_e", 0.0) for r in hist], dtype=float)
    return RegularSeries(ansatz=ansatz, iters=iters, t_iter=t_iter, t_cum=t_cum, delta_e=delta_e)


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


def _interactive_bar3d(
    *,
    bars: list[BarSpec],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("plotly is required for interactive plots") from exc

    fig = go.Figure()

    for ansatz in ANSATZ_ORDER:
        ansatz_bars = [b for b in bars if b.ansatz == ansatz]
        if not ansatz_bars:
            continue
        vertices: list[tuple[float, float, float]] = []
        i_idx: list[int] = []
        j_idx: list[int] = []
        k_idx: list[int] = []

        def add_cuboid(x0: float, y0: float, z0: float, dx: float, dy: float, dz: float) -> None:
            v = [
                (x0, y0, z0),
                (x0 + dx, y0, z0),
                (x0 + dx, y0 + dy, z0),
                (x0, y0 + dy, z0),
                (x0, y0, z0 + dz),
                (x0 + dx, y0, z0 + dz),
                (x0 + dx, y0 + dy, z0 + dz),
                (x0, y0 + dy, z0 + dz),
            ]
            base = len(vertices)
            vertices.extend(v)
            faces = [
                (0, 1, 2), (0, 2, 3),
                (4, 5, 6), (4, 6, 7),
                (0, 1, 5), (0, 5, 4),
                (1, 2, 6), (1, 6, 5),
                (2, 3, 7), (2, 7, 6),
                (3, 0, 4), (3, 4, 7),
            ]
            for a, b, c in faces:
                i_idx.append(base + a)
                j_idx.append(base + b)
                k_idx.append(base + c)

        for bar in ansatz_bars:
            add_cuboid(bar.x0, bar.y0, bar.z0, bar.dx, bar.dy, bar.dz)

        xs, ys, zs = zip(*vertices) if vertices else ([], [], [])
        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                color=ANSATZ_COLORS[ansatz],
                opacity=0.95,
                name=ANSATZ_LABELS[ansatz],
                flatshading=True,
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title="ansatz",
                tickmode="array",
                tickvals=list(range(len(ANSATZ_ORDER))),
                ticktext=[ANSATZ_LABELS[a] for a in ANSATZ_ORDER],
            ),
            yaxis=dict(title=ylabel),
            zaxis=dict(title="ΔE = E_ansatz - E_exact_sector"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved interactive 3D ansatz ΔE plot to {out_path}")


def _ansatz_x_block(ansatz: str, width: float) -> tuple[float, float]:
    idx = ANSATZ_ORDER.index(ansatz)
    x0 = idx - 0.5 * width
    return x0, width


def _regular_y_anchor(y_max: float, width: float) -> tuple[float, float]:
    # We do not have cached per-iteration/time traces for regular VQE.
    # Anchor them near the end of the ADAPT trajectory for comparison.
    return max(0.0, y_max - width), width


def _bars_for_iter(
    *,
    regular_delta: dict[str, float],
    regular_series: dict[str, RegularSeries],
    adapt: AdaptSeries,
) -> list[BarSpec]:
    bars: list[BarSpec] = []
    xw = 0.8
    yw = 1.0
    iter_max = float(np.max(adapt.iters))

    x0, dx0 = _ansatz_x_block("adapt_cse_hybrid", xw)
    for n, de in zip(adapt.iters, adapt.delta_e):
        z0 = min(0.0, float(de))
        bars.append(BarSpec("adapt_cse_hybrid", x0, float(n) - 0.5 * yw, z0, dx0, yw, abs(float(de))))

    for ansatz in ANSATZ_ORDER:
        if ansatz == "adapt_cse_hybrid":
            continue
        series = regular_series.get(ansatz)
        if series is not None:
            x0, dx0 = _ansatz_x_block(ansatz, xw)
            for n, de in zip(series.iters, series.delta_e):
                z0 = min(0.0, float(de))
                bars.append(BarSpec(ansatz, x0, float(n) - 0.5 * yw, z0, dx0, yw, abs(float(de))))
        elif ansatz in regular_delta:
            x0, dx0 = _ansatz_x_block(ansatz, xw)
            y0, dy0 = _regular_y_anchor(iter_max + 1.0, yw)
            de = float(regular_delta[ansatz])
            z0 = min(0.0, de)
            bars.append(BarSpec(ansatz, x0, y0, z0, dx0, dy0, abs(de)))

    return bars


def _bars_for_time(
    *,
    regular_delta: dict[str, float],
    regular_series: dict[str, RegularSeries],
    adapt: AdaptSeries,
) -> list[BarSpec]:
    bars: list[BarSpec] = []
    xw = 0.8
    t_max = float(np.max(adapt.t_cum))
    regular_width = max(1.0, 0.03 * t_max)

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
        z0 = min(0.0, float(de))
        bars.append(BarSpec("adapt_cse_hybrid", x0, t_start, z0, dx0, t_width, abs(float(de))))

    for ansatz in ANSATZ_ORDER:
        if ansatz == "adapt_cse_hybrid":
            continue
        series = regular_series.get(ansatz)
        if series is not None:
            x0, dx0 = _ansatz_x_block(ansatz, xw)
            prev_t = 0.0
            for t_cum, t_iter, de in zip(series.t_cum, series.t_iter, series.delta_e):
                t_cum_f = float(t_cum)
                t_iter_f = float(t_iter)
                if t_iter_f <= 0.0:
                    t_start = prev_t
                    t_width = max(1e-9, t_cum_f - prev_t)
                else:
                    t_start = max(0.0, t_cum_f - t_iter_f)
                    t_width = t_iter_f
                prev_t = t_cum_f
                z0 = min(0.0, float(de))
                bars.append(BarSpec(ansatz, x0, t_start, z0, dx0, t_width, abs(float(de))))
        elif ansatz in regular_delta:
            x0, dx0 = _ansatz_x_block(ansatz, xw)
            y0, dy0 = _regular_y_anchor(t_max + regular_width, regular_width)
            de = float(regular_delta[ansatz])
            z0 = min(0.0, de)
            bars.append(BarSpec(ansatz, x0, y0, z0, dx0, dy0, abs(de)))

    return bars


def _bars_for_count(
    *,
    regular_delta: dict[str, float],
    regular_series: dict[str, RegularSeries],
    adapt: AdaptSeries,
    count_factor: int,
) -> list[BarSpec]:
    bars: list[BarSpec] = []
    xw = 0.8
    dy = 1.0

    x0, dx0 = _ansatz_x_block("adapt_cse_hybrid", xw)
    for n, de in zip(adapt.iters, adapt.delta_e):
        count = float(n) * float(count_factor)
        z0 = min(0.0, float(de))
        bars.append(BarSpec("adapt_cse_hybrid", x0, max(0.0, count - 0.5), z0, dx0, dy, abs(float(de))))

    for ansatz in ANSATZ_ORDER:
        if ansatz == "adapt_cse_hybrid":
            continue
        series = regular_series.get(ansatz)
        if series is not None:
            x0, dx0 = _ansatz_x_block(ansatz, xw)
            for n, de in zip(series.iters, series.delta_e):
                count = float(n) * float(count_factor)
                z0 = min(0.0, float(de))
                bars.append(BarSpec(ansatz, x0, max(0.0, count - 0.5), z0, dx0, dy, abs(float(de))))
        elif ansatz in regular_delta:
            x0, dx0 = _ansatz_x_block(ansatz, xw)
            de = float(regular_delta[ansatz])
            z0 = min(0.0, de)
            bars.append(BarSpec(ansatz, x0, 0.0, z0, dx0, dy, abs(de)))

    return bars


def build_iter_plot(
    *,
    sites: int,
    regular_delta: dict[str, float],
    regular_series: dict[str, RegularSeries],
    adapt: AdaptSeries,
    out_path: Path,
) -> None:
    xw = 0.8
    yw = 1.0
    bars = _bars_for_iter(
        regular_delta=regular_delta,
        regular_series=regular_series,
        adapt=adapt,
    )

    ypos_arr = np.asarray([b.y0 for b in bars], dtype=float)
    dy_arr = np.asarray([b.dy for b in bars], dtype=float)
    y_hi = float(np.max(ypos_arr + dy_arr)) if len(bars) else 1.0
    y_ticks = np.arange(0.0, y_hi + 0.5, 1.0)
    y_labels = [str(int(v)) for v in y_ticks]

    _bar3d_delta_e(
        xpos=np.asarray([b.x0 for b in bars], dtype=float),
        ypos=ypos_arr,
        dx=np.asarray([b.dx for b in bars], dtype=float),
        dy=dy_arr,
        delta_e=np.asarray([b.z0 + b.dz for b in bars], dtype=float),
        colors=[ANSATZ_COLORS[b.ansatz] for b in bars],
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
    regular_series: dict[str, RegularSeries],
    adapt: AdaptSeries,
    out_path: Path,
) -> None:
    bars = _bars_for_time(
        regular_delta=regular_delta,
        regular_series=regular_series,
        adapt=adapt,
    )

    ypos_arr = np.asarray([b.y0 for b in bars], dtype=float)
    dy_arr = np.asarray([b.dy for b in bars], dtype=float)
    y_hi = float(np.max(ypos_arr + dy_arr)) if len(bars) else 1.0
    tick_count = 6
    y_ticks = np.linspace(0.0, y_hi, tick_count)
    y_labels = [f"{v:.0f}" for v in y_ticks]

    _bar3d_delta_e(
        xpos=np.asarray([b.x0 for b in bars], dtype=float),
        ypos=ypos_arr,
        dx=np.asarray([b.dx for b in bars], dtype=float),
        dy=dy_arr,
        delta_e=np.asarray([b.z0 + b.dz for b in bars], dtype=float),
        colors=[ANSATZ_COLORS[b.ansatz] for b in bars],
        title=f"L={sites}: ΔE by ansatz with cumulative time axis",
        ylabel="cumulative wall time t^(n) [s]",
        ytick_values=y_ticks,
        ytick_labels=y_labels,
        out_path=out_path,
    )


def _measurement_counts_for_site(site: int) -> tuple[int, int]:
    """Return (num_terms, num_groups) for the qubit Hamiltonian at site."""
    from pydephasing.quantum.hamiltonian.hubbard import (
        build_fermionic_hubbard,
        build_qubit_hamiltonian_from_fermionic,
        default_1d_chain_edges,
    )

    t = 1.0
    u = 4.0
    dv = 0.5
    ferm_op = build_fermionic_hubbard(
        n_sites=site,
        t=t,
        u=u,
        edges=default_1d_chain_edges(site, periodic=False),
        v=[-dv / 2, dv / 2] if site == 2 else None,
    )
    qubit_op, _mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    num_terms = len(qubit_op)
    num_groups = len(qubit_op.group_commuting())
    return int(num_terms), int(num_groups)


def build_count_plot(
    *,
    sites: int,
    regular_delta: dict[str, float],
    regular_series: dict[str, RegularSeries],
    adapt: AdaptSeries,
    out_path: Path,
    count_label: str,
    count_factor: int,
) -> None:
    bars = _bars_for_count(
        regular_delta=regular_delta,
        regular_series=regular_series,
        adapt=adapt,
        count_factor=count_factor,
    )

    ypos_arr = np.asarray([b.y0 for b in bars], dtype=float)
    dy_arr = np.asarray([b.dy for b in bars], dtype=float)
    y_hi = float(np.max(ypos_arr + dy_arr)) if len(bars) else 1.0
    tick_count = 6
    y_ticks = np.linspace(0.0, y_hi, tick_count)
    y_labels = [f"{v:.0f}" for v in y_ticks]

    _bar3d_delta_e(
        xpos=np.asarray([b.x0 for b in bars], dtype=float),
        ypos=ypos_arr,
        dx=np.asarray([b.dx for b in bars], dtype=float),
        dy=dy_arr,
        delta_e=np.asarray([b.z0 + b.dz for b in bars], dtype=float),
        colors=[ANSATZ_COLORS[b.ansatz] for b in bars],
        title=f"L={sites}: ΔE by ansatz with {count_label} axis",
        ylabel=count_label,
        ytick_values=y_ticks,
        ytick_labels=y_labels,
        out_path=out_path,
    )


def main(
    sites: Iterable[int] = (2, 3, 4),
    *,
    mode: str = "all",
    interactive: bool = False,
    compare_dir: str = "runs/compare_vqe",
) -> None:
    runs_root = Path("runs")
    out_dir = Path(compare_dir)
    compare_rows_path = out_dir / "compare_rows.json"
    rows = load_compare_rows(compare_rows_path)

    for site in sites:
        site_i = int(site)
        n_up, n_down = sector_for_site_from_compare_rows(rows=rows, sites=site_i, require_half_filling=True)
        run_dir = discover_latest_run_for_site(
            runs_root,
            site_i,
            pool="cse_density_ops",
            n_up=n_up,
            n_down=n_down,
        )
        adapt = build_adapt_series(run_dir)
        regular_delta = regular_delta_e_by_ansatz(rows, site_i, n_up=n_up, n_down=n_down)
        regular_series: dict[str, RegularSeries] = {}
        for ansatz in ANSATZ_ORDER:
            if ansatz == "adapt_cse_hybrid":
                continue
            log_dir = out_dir / f"logs_{ansatz}_L{site_i}_Nup{n_up}_Ndown{n_down}"
            series = load_regular_series(log_dir, ansatz)
            if series is not None:
                regular_series[ansatz] = series

        def emit_interactive(bars: list[BarSpec], title: str, ylabel: str, name: str) -> None:
            out_path = out_dir / f"{name}_L{site_i}_Nup{n_up}_Ndown{n_down}_interactive.html"
            _interactive_bar3d(
                bars=bars,
                title=title,
                ylabel=ylabel,
                out_path=out_path,
            )

        if mode in {"all", "iter"}:
            if interactive:
                bars = _bars_for_iter(
                    regular_delta=regular_delta,
                    regular_series=regular_series,
                    adapt=adapt,
                )
                emit_interactive(bars, f"L={site_i} (Nup={n_up},Ndown={n_down}): ΔE by ansatz with iteration axis", "iteration n",
                                 "delta_e_hist3d_ansatz_iter")
            else:
                build_iter_plot(
                    sites=site_i,
                    regular_delta=regular_delta,
                    regular_series=regular_series,
                    adapt=adapt,
                    out_path=out_dir / f"delta_e_hist3d_ansatz_iter_L{site_i}_Nup{n_up}_Ndown{n_down}.png",
                )
        if mode in {"all", "time"}:
            if interactive:
                bars = _bars_for_time(
                    regular_delta=regular_delta,
                    regular_series=regular_series,
                    adapt=adapt,
                )
                emit_interactive(
                    bars,
                    f"L={site_i} (Nup={n_up},Ndown={n_down}): ΔE by ansatz with cumulative time axis",
                    "cumulative wall time t^(n) [s]",
                    "delta_e_hist3d_ansatz_time",
                )
            else:
                build_time_plot(
                    sites=site_i,
                    regular_delta=regular_delta,
                    regular_series=regular_series,
                    adapt=adapt,
                    out_path=out_dir / f"delta_e_hist3d_ansatz_time_L{site_i}_Nup{n_up}_Ndown{n_down}.png",
                )
        if mode in {"all", "exec", "eval"}:
            num_terms, num_groups = _measurement_counts_for_site(site_i)
            if mode in {"all", "exec"}:
                if interactive:
                    bars = _bars_for_count(
                        regular_delta=regular_delta,
                        regular_series=regular_series,
                        adapt=adapt,
                        count_factor=num_groups,
                    )
                    emit_interactive(
                        bars,
                        f"L={site_i} (Nup={n_up},Ndown={n_down}): ΔE by ansatz with estimated circuit executions axis",
                        "estimated circuit executions",
                        "delta_e_hist3d_ansatz_exec",
                    )
                else:
                    build_count_plot(
                        sites=site_i,
                        regular_delta=regular_delta,
                        regular_series=regular_series,
                        adapt=adapt,
                        out_path=out_dir / f"delta_e_hist3d_ansatz_exec_L{site_i}_Nup{n_up}_Ndown{n_down}.png",
                        count_label="estimated circuit executions",
                        count_factor=num_groups,
                    )
            if mode in {"all", "eval"}:
                if interactive:
                    bars = _bars_for_count(
                        regular_delta=regular_delta,
                        regular_series=regular_series,
                        adapt=adapt,
                        count_factor=num_terms,
                    )
                    emit_interactive(
                        bars,
                        f"L={site_i} (Nup={n_up},Ndown={n_down}): ΔE by ansatz with estimated energy expectation values axis",
                        "estimated energy expectation values",
                        "delta_e_hist3d_ansatz_eval",
                    )
                else:
                    build_count_plot(
                        sites=site_i,
                        regular_delta=regular_delta,
                        regular_series=regular_series,
                        adapt=adapt,
                        out_path=out_dir / f"delta_e_hist3d_ansatz_eval_L{site_i}_Nup{n_up}_Ndown{n_down}.png",
                        count_label="estimated energy expectation values",
                        count_factor=num_terms,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="*", type=int, default=[2, 3, 4])
    parser.add_argument("--mode", choices=["all", "iter", "time", "exec", "eval"], default="all")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--compare-dir", type=str, default="runs/compare_vqe")
    args = parser.parse_args()
    main(args.sites, mode=args.mode, interactive=args.interactive, compare_dir=args.compare_dir)
