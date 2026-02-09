#!/usr/bin/env python3
"""Generate a simple SVG bar chart of comparison results by L and ansatz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

ANSATZ_ORDER = ["adapt_cse_hybrid", "adapt_uccsd_hybrid", "vqe_cse_ops", "uccsd"]
ANSATZ_LABELS = {
    "adapt_cse_hybrid": "ADAPT CSE (hybrid)",
    "adapt_uccsd_hybrid": "ADAPT UCCSD (hybrid)",
    "vqe_cse_ops": "VQE (CSE ops)",
    "uccsd": "UCCSD",
}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _compute_values(rows: list[dict], sites: list[int], metric: str) -> dict[int, dict[str, float]]:
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
        if metric in ("abs", "rel"):
            val = abs(val)
        if metric == "rel":
            exact = row.get("exact")
            if exact is None:
                continue
            denom = abs(float(exact))
            if denom == 0.0:
                continue
            val = val / denom
        table[s][ansatz] = val
    return table


def _add_text(svg: ET.Element, x: float, y: float, text: str, size: int = 12, anchor: str = "middle") -> None:
    el = ET.SubElement(svg, "text", {
        "x": f"{x:.2f}",
        "y": f"{y:.2f}",
        "font-size": str(size),
        "text-anchor": anchor,
        "font-family": "sans-serif",
        "fill": "#333333",
    })
    el.text = text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=str, default="runs/compare_vqe/compare_rows.json")
    ap.add_argument("--sites", nargs="*", type=int, default=[5])
    ap.add_argument("--metric", choices=["delta", "abs", "rel"], default="abs")
    ap.add_argument("--out", type=str, default="runs/compare_vqe/compare_bar_abs_L5.svg")
    args = ap.parse_args()

    rows = load_rows(Path(args.compare))
    sites = sorted(set(int(s) for s in args.sites))
    table = _compute_values(rows, sites, args.metric)

    values = [val for site in sites for val in table[site].values()]
    max_val = max(values) if values else 1.0
    if max_val <= 0:
        max_val = 1.0
    max_val *= 1.1

    width = 800
    height = 500
    margin_left = 80
    margin_right = 30
    margin_top = 50
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })

    ET.SubElement(svg, "rect", {
        "x": str(margin_left),
        "y": str(margin_top),
        "width": str(plot_width),
        "height": str(plot_height),
        "fill": "#ffffff",
        "stroke": "#cccccc",
    })

    # Axes
    ET.SubElement(svg, "line", {
        "x1": str(margin_left),
        "y1": str(margin_top + plot_height),
        "x2": str(margin_left + plot_width),
        "y2": str(margin_top + plot_height),
        "stroke": "#333333",
        "stroke-width": "1",
    })
    ET.SubElement(svg, "line", {
        "x1": str(margin_left),
        "y1": str(margin_top),
        "x2": str(margin_left),
        "y2": str(margin_top + plot_height),
        "stroke": "#333333",
        "stroke-width": "1",
    })

    # Bars
    group_count = len(sites)
    bars_per_group = len(ANSATZ_ORDER)
    group_width = plot_width / max(group_count, 1)
    bar_width = group_width / (bars_per_group + 1)

    for s_idx, site in enumerate(sites):
        group_start = margin_left + s_idx * group_width
        for a_idx, ansatz in enumerate(ANSATZ_ORDER):
            val = table[site].get(ansatz)
            if val is None:
                continue
            bar_height = (val / max_val) * plot_height
            x = group_start + (a_idx + 0.5) * bar_width
            y = margin_top + plot_height - bar_height
            ET.SubElement(svg, "rect", {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "width": f"{bar_width * 0.8:.2f}",
                "height": f"{bar_height:.2f}",
                "fill": COLORS[a_idx % len(COLORS)],
            })
            _add_text(svg, x + bar_width * 0.4, y - 6, f"{val:.3f}", size=10)

        label_x = group_start + (bars_per_group * bar_width) / 2
        _add_text(svg, label_x, margin_top + plot_height + 25, f"L={site}", size=12)

    # Title and y label
    title = "Comparison by L and ansatz"
    _add_text(svg, width / 2, 25, title, size=16)

    ylabel = "|ΔE|" if args.metric == "abs" else ("|ΔE| / |E_exact|" if args.metric == "rel" else "ΔE")
    _add_text(svg, 20, margin_top + plot_height / 2, ylabel, size=12, anchor="middle")

    # Legend
    legend_x = margin_left
    legend_y = height - 30
    for idx, ansatz in enumerate(ANSATZ_ORDER):
        x = legend_x + idx * 180
        ET.SubElement(svg, "rect", {
            "x": f"{x:.2f}",
            "y": f"{legend_y - 12:.2f}",
            "width": "12",
            "height": "12",
            "fill": COLORS[idx % len(COLORS)],
        })
        _add_text(svg, x + 18, legend_y - 2, ANSATZ_LABELS.get(ansatz, ansatz), size=11, anchor="start")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved SVG bar chart to {out_path}")


if __name__ == "__main__":
    main()
