#!/usr/bin/env python3
"""Plot E vs t and E vs U per L from cached sweep data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

SERIES_ORDER = ["exact", "adapt", "uccsd"]
SERIES_LABELS = {
    "exact": "Exact",
    "adapt": "ADAPT",
    "uccsd": "UCCSD",
}
SERIES_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _series(payload: dict[str, Any], key: str, site: int) -> list[float] | None:
    store = payload.get(key, {})
    values = store.get(str(int(site)), [])
    if any(v is None for v in values):
        return None
    return [float(v) for v in values]


def _plot_line(
    *,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = plt.cm.tab10.colors
    for idx, (name, values) in enumerate(series):
        color = colors[idx % len(colors)]
        marker = SERIES_MARKERS[idx % len(SERIES_MARKERS)]
        style = {
            "color": color,
            "marker": marker,
            "linewidth": 2,
            "linestyle": "-" if name != "exact" else "--",
        }
        label = SERIES_LABELS.get(name, name)
        ax.plot(x_values, values, label=label, **style)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy E")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _available_series(payload: dict[str, Any], axis: str, site: int) -> list[tuple[str, list[float]]]:
    suffix = f"_vs_{axis}"
    found: list[tuple[str, list[float]]] = []
    seen: set[str] = set()

    for name in SERIES_ORDER:
        key = f"{name}{suffix}"
        if key not in payload:
            continue
        values = _series(payload, key, site)
        if values:
            found.append((name, values))
            seen.add(name)

    extra = sorted(
        key[:-len(suffix)] for key in payload.keys() if key.endswith(suffix) and key[:-len(suffix)] not in seen
    )
    for name in extra:
        key = f"{name}{suffix}"
        values = _series(payload, key, site)
        if values:
            found.append((name, values))

    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-json", type=str, default="runs/sweeps/sweep_t_u_multi.json")
    ap.add_argument("--sites", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--out-dir", type=str, default="runs/sweeps")
    args = ap.parse_args()

    payload = _load_payload(Path(args.sweep_json))
    t_values = [float(v) for v in payload.get("t_values", [])]
    u_values = [float(v) for v in payload.get("u_values", [])]

    if not t_values or not u_values:
        raise ValueError("Sweep payload missing t_values or u_values")

    out_dir = Path(args.out_dir)
    u_fixed = float(payload.get("u_fixed", 0.0))
    t_fixed = float(payload.get("t_fixed", 0.0))
    dv = float(payload.get("dv", 0.0))

    for site in sorted(set(int(s) for s in args.sites)):
        series_t = _available_series(payload, "t", site)
        series_u = _available_series(payload, "u", site)

        if series_t:
            _plot_line(
                x_values=t_values,
                series=series_t,
                x_label="t",
                title=f"L={site}: E vs t (U={u_fixed:g}, dv={dv:g})",
                out_path=out_dir / f"E_vs_t_L{site}.png",
            )
        else:
            print(f"Skipping E vs t for L={site} (missing data)")

        if series_u:
            _plot_line(
                x_values=u_values,
                series=series_u,
                x_label="U",
                title=f"L={site}: E vs U (t={t_fixed:g}, dv={dv:g})",
                out_path=out_dir / f"E_vs_U_L{site}.png",
            )
        else:
            print(f"Skipping E vs U for L={site} (missing data)")


if __name__ == "__main__":
    main()
