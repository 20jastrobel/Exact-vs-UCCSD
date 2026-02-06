#!/usr/bin/env python3
"""Sweep t and U across multiple L values and generate interactive 3D plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorEstimator

from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.symmetry import exact_ground_energy_sector
from pydephasing.quantum.utils_particles import (
    half_filling_num_particles,
    jw_reference_occupations_from_particles,
)
from pydephasing.quantum.vqe.adapt_vqe_meta import build_reference_state, run_meta_adapt_vqe


def run_case(*, n_sites: int, t: float, u: float, dv: float, n_up: int, n_down: int) -> tuple[float, float]:
    ferm_op = build_fermionic_hubbard(
        n_sites=n_sites,
        t=t,
        u=u,
        edges=default_1d_chain_edges(n_sites, periodic=False),
        v=[-dv / 2, dv / 2] if n_sites == 2 else None,
    )
    qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
    reference = build_reference_state(
        qubit_op.num_qubits,
        jw_reference_occupations_from_particles(n_sites, n_up, n_down),
    )
    estimator = StatevectorEstimator()

    exact = exact_ground_energy_sector(qubit_op, n_sites, n_up + n_down, 0.5 * (n_up - n_down))
    result = run_meta_adapt_vqe(
        qubit_op,
        reference,
        estimator,
        pool_mode="cse_density_ops",
        ferm_op=ferm_op,
        mapper=mapper,
        n_sites=n_sites,
        n_up=n_up,
        n_down=n_down,
        enforce_sector=True,
        inner_optimizer="hybrid",
        max_depth=6,
        inner_steps=25,
        warmup_steps=5,
        polish_steps=3,
        verbose=False,
    )
    return exact, float(result.energy)


def _as_site_key(n_sites: int) -> str:
    return str(int(n_sites))


def _ensure_site_arrays(store: dict[str, list[float | None]], n_sites: int, n_points: int) -> list[float | None]:
    key = _as_site_key(n_sites)
    arr = store.get(key)
    if arr is None or len(arr) != n_points:
        arr = [None] * n_points
        store[key] = arr
    return arr


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _seed_from_legacy_l2(path: Path, payload: dict[str, Any]) -> None:
    legacy = _load_json(path)
    if not legacy:
        return
    try:
        t_values = list(map(float, payload["t_values"]))
        u_values = list(map(float, payload["u_values"]))
        if t_values != list(map(float, legacy.get("t_values", []))):
            return
        if u_values != list(map(float, legacy.get("u_values", []))):
            return
        for field in ("exact_vs_t", "adapt_vs_t", "exact_vs_u", "adapt_vs_u"):
            legacy_vals = legacy.get(field)
            if not isinstance(legacy_vals, list) or len(legacy_vals) != len(
                t_values if "vs_t" in field else u_values
            ):
                return
        l2_key = _as_site_key(2)
        payload["exact_vs_t"][l2_key] = list(map(float, legacy["exact_vs_t"]))
        payload["adapt_vs_t"][l2_key] = list(map(float, legacy["adapt_vs_t"]))
        payload["exact_vs_u"][l2_key] = list(map(float, legacy["exact_vs_u"]))
        payload["adapt_vs_u"][l2_key] = list(map(float, legacy["adapt_vs_u"]))
        print(f"Seeded L=2 data from {path}")
    except Exception:
        # Best-effort seeding; ignore mismatches.
        return


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _compute_missing(
    *,
    payload: dict[str, Any],
    sweep_path: Path,
    n_sites: int,
    t_values: list[float],
    u_values: list[float],
    t_fixed: float,
    u_fixed: float,
    dv: float,
    n_up: int,
    n_down: int,
) -> None:
    exact_t = _ensure_site_arrays(payload["exact_vs_t"], n_sites, len(t_values))
    adapt_t = _ensure_site_arrays(payload["adapt_vs_t"], n_sites, len(t_values))
    exact_u = _ensure_site_arrays(payload["exact_vs_u"], n_sites, len(u_values))
    adapt_u = _ensure_site_arrays(payload["adapt_vs_u"], n_sites, len(u_values))

    for idx, t in enumerate(t_values):
        if exact_t[idx] is not None and adapt_t[idx] is not None:
            continue
        print(f"Computing L={n_sites} at t={t:.3f}, U={u_fixed:.3f}")
        exact, adapt = run_case(
            n_sites=n_sites,
            t=float(t),
            u=float(u_fixed),
            dv=float(dv),
            n_up=n_up,
            n_down=n_down,
        )
        exact_t[idx] = float(exact)
        adapt_t[idx] = float(adapt)
        _write_payload(sweep_path, payload)

    for idx, u in enumerate(u_values):
        if exact_u[idx] is not None and adapt_u[idx] is not None:
            continue
        print(f"Computing L={n_sites} at t={t_fixed:.3f}, U={u:.3f}")
        exact, adapt = run_case(
            n_sites=n_sites,
            t=float(t_fixed),
            u=float(u),
            dv=float(dv),
            n_up=n_up,
            n_down=n_down,
        )
        exact_u[idx] = float(exact)
        adapt_u[idx] = float(adapt)
        _write_payload(sweep_path, payload)


def _mesh_fill_trace(
    *,
    x_values: list[float],
    site: int,
    z_values: list[float],
    base_z: float,
    color: str,
    name: str,
):
    import plotly.graph_objects as go

    if len(x_values) < 2:
        return None

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []

    for seg in range(len(x_values) - 1):
        x0 = float(x_values[seg])
        x1 = float(x_values[seg + 1])
        z0 = float(z_values[seg])
        z1 = float(z_values[seg + 1])

        base_index = len(xs)
        xs.extend([x0, x1, x0, x1])
        ys.extend([site, site, site, site])
        zs.extend([z0, z1, base_z, base_z])

        # Two triangles forming a vertical quad under the line segment.
        i_idx.extend([base_index + 0, base_index + 1])
        j_idx.extend([base_index + 2, base_index + 2])
        k_idx.extend([base_index + 1, base_index + 3])

    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color=color,
        opacity=0.22,
        name=name,
        hoverinfo="skip",
        showlegend=False,
        flatshading=True,
    )


def _filter_series(
    x_values: list[float], values: list[float | None]
) -> tuple[list[float], list[float]] | tuple[None, None]:
    pairs = [(float(x), float(v)) for x, v in zip(x_values, values) if v is not None]
    if len(pairs) < 2:
        return None, None
    xs, zs = zip(*pairs)
    return list(xs), list(zs)


def _plot_interactive(
    *,
    payload: dict[str, Any],
    out_dir: Path,
    t_values: list[float],
    u_values: list[float],
    sites: list[int],
) -> tuple[Path, Path]:
    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("plotly is required for the interactive plots") from exc

    palette = qualitative.Plotly
    colors = {site: palette[i % len(palette)] for i, site in enumerate(sites)}

    def build_fig(
        *,
        x_values: list[float],
        series_key: str,
        x_title: str,
        title: str,
        out_name: str,
    ) -> Path:
        fig = go.Figure()

        adapt_store = payload[series_key]
        exact_store = payload["exact_" + series_key.split("adapt_")[-1]]

        all_adapt: list[float] = []
        for site in sites:
            key = _as_site_key(site)
            series = adapt_store.get(key, [])
            if not isinstance(series, list):
                continue
            all_adapt.extend([float(v) for v in series if v is not None])
        base_z = (min(all_adapt) - 0.25) if all_adapt else -1.0

        for site in sites:
            key = _as_site_key(site)
            adapt_series = adapt_store.get(key, [])
            exact_series = exact_store.get(key, [])
            if not isinstance(adapt_series, list) or not isinstance(exact_series, list):
                continue

            adapt_x, adapt_vals = _filter_series(x_values, adapt_series)
            exact_x, exact_vals = _filter_series(x_values, exact_series)
            if not adapt_vals and not exact_vals:
                continue

            color = colors[site]

            if adapt_vals:
                fill = _mesh_fill_trace(
                    x_values=adapt_x,
                    site=site,
                    z_values=adapt_vals,
                    base_z=base_z,
                    color=color,
                    name=f"L={site} fill",
                )
                if fill is not None:
                    fig.add_trace(fill)

                fig.add_trace(
                    go.Scatter3d(
                        x=adapt_x,
                        y=[site] * len(adapt_x),
                        z=adapt_vals,
                        mode="lines+markers",
                        line=dict(color=color, width=6),
                        marker=dict(size=4),
                        name=f"L={site} ADAPT",
                        hovertemplate=(
                            f"L={site} ADAPT<br>{x_title}=%{{x:.3f}}"
                            "<br>E=%{z:.6f}<extra></extra>"
                        ),
                    )
                )

            if exact_vals:
                fig.add_trace(
                    go.Scatter3d(
                        x=exact_x,
                        y=[site] * len(exact_x),
                        z=exact_vals,
                        mode="lines+markers",
                        line=dict(color=color, width=3, dash="dash"),
                        marker=dict(size=3, symbol="circle-open"),
                        name=f"L={site} Exact",
                        hovertemplate=(
                            f"L={site} Exact<br>{x_title}=%{{x:.3f}}"
                            "<br>E=%{z:.6f}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title=title,
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title=x_title,
                yaxis_title="sites L",
                zaxis_title="Energy E",
                yaxis=dict(tickmode="array", tickvals=sites, ticktext=[str(s) for s in sites]),
            ),
            legend=dict(orientation="h"),
        )

        out_path = out_dir / out_name
        fig.write_html(out_path, include_plotlyjs="cdn")
        return out_path

    t_out = build_fig(
        x_values=t_values,
        series_key="adapt_vs_t",
        x_title="t",
        title=f"E vs t vs sites (U={payload['u_fixed']:g}, dv={payload['dv']:g})",
        out_name="E_vs_t_vs_sites_interactive.html",
    )
    u_out = build_fig(
        x_values=u_values,
        series_key="adapt_vs_u",
        x_title="U",
        title=f"E vs U vs sites (t={payload['t_fixed']:g}, dv={payload['dv']:g})",
        out_name="E_vs_U_vs_sites_interactive.html",
    )
    return t_out, u_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument("--n-up", type=int, default=None)
    ap.add_argument("--n-down", type=int, default=None)
    ap.add_argument(
        "--odd-sz-target",
        type=float,
        default=0.5,
        help="Half-filling Sz target for odd L (default +0.5). Ignored if --n-up/--n-down are set.",
    )
    ap.add_argument("--dv", type=float, default=0.5)
    ap.add_argument("--t-fixed", type=float, default=1.0)
    ap.add_argument("--u-fixed", type=float, default=4.0)
    ap.add_argument("--t-min", type=float, default=0.5)
    ap.add_argument("--t-max", type=float, default=1.5)
    ap.add_argument("--u-min", type=float, default=2.0)
    ap.add_argument("--u-max", type=float, default=6.0)
    ap.add_argument("--n-points", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default="runs/sweeps")
    ap.add_argument("--sweep-json", type=str, default="sweep_t_u_multi.json")
    args = ap.parse_args()

    if (args.n_up is None) != (args.n_down is None):
        raise ValueError("Both --n-up and --n-down must be set together.")
    sector_mode = "fixed" if args.n_up is not None else "half_filling"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_sites = {int(s) for s in args.sites}
    t_values = list(map(float, np.linspace(args.t_min, args.t_max, args.n_points)))
    u_values = list(map(float, np.linspace(args.u_min, args.u_max, args.n_points)))

    sweep_path = out_dir / args.sweep_json
    payload = _load_json(sweep_path) or {}
    existing_sites = (
        {int(s) for s in payload.get("sites", [])}
        if isinstance(payload.get("sites", []), list)
        else set()
    )
    sites = sorted(requested_sites | existing_sites)
    sectors = {}
    for n_sites in sites:
        if sector_mode == "fixed":
            n_up, n_down = int(args.n_up), int(args.n_down)
        else:
            sz = 0.0 if int(n_sites) % 2 == 0 else float(args.odd_sz_target)
            n_up, n_down = half_filling_num_particles(int(n_sites), sz_target=sz)
        sectors[str(int(n_sites))] = [int(n_up), int(n_down)]
    payload.update(
        {
            "sites": sites,
            "t_values": t_values,
            "u_values": u_values,
            "u_fixed": float(args.u_fixed),
            "t_fixed": float(args.t_fixed),
            "dv": float(args.dv),
            "sector_mode": sector_mode,
            "odd_sz_target": float(args.odd_sz_target),
            "n_up": None if args.n_up is None else int(args.n_up),
            "n_down": None if args.n_down is None else int(args.n_down),
            "sectors": sectors,
            "exact_vs_t": payload.get("exact_vs_t", {}) if isinstance(payload.get("exact_vs_t", {}), dict) else {},
            "adapt_vs_t": payload.get("adapt_vs_t", {}) if isinstance(payload.get("adapt_vs_t", {}), dict) else {},
            "exact_vs_u": payload.get("exact_vs_u", {}) if isinstance(payload.get("exact_vs_u", {}), dict) else {},
            "adapt_vs_u": payload.get("adapt_vs_u", {}) if isinstance(payload.get("adapt_vs_u", {}), dict) else {},
        }
    )

    # Seed L=2 from the legacy single-site sweep, if compatible.
    _seed_from_legacy_l2(out_dir / "sweep_t_u_L2.json", payload)
    _write_payload(sweep_path, payload)

    for n_sites in sorted(requested_sites):
        if sector_mode == "fixed":
            n_up, n_down = int(args.n_up), int(args.n_down)
        else:
            sz = 0.0 if int(n_sites) % 2 == 0 else float(args.odd_sz_target)
            n_up, n_down = half_filling_num_particles(int(n_sites), sz_target=sz)
        _compute_missing(
            payload=payload,
            sweep_path=sweep_path,
            n_sites=n_sites,
            t_values=t_values,
            u_values=u_values,
            t_fixed=float(args.t_fixed),
            u_fixed=float(args.u_fixed),
            dv=float(args.dv),
            n_up=int(n_up),
            n_down=int(n_down),
        )

    t_out, u_out = _plot_interactive(
        payload=payload,
        out_dir=out_dir,
        t_values=t_values,
        u_values=u_values,
        sites=sites,
    )
    print(f"Saved sweep data to {sweep_path}")
    print(f"Saved interactive plot to {t_out}")
    print(f"Saved interactive plot to {u_out}")


if __name__ == "__main__":
    main()
