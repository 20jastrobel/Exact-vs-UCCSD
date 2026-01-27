#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text())
    hist = []
    with open(run_dir / "history.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hist.append(json.loads(line))
    hist = sorted(hist, key=lambda r: r["iter"])
    return meta, hist


def log_base(x, base: float):
    if base == 10.0:
        return np.log10(x)
    return np.log(x) / np.log(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--log-base", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=1e-16)
    ap.add_argument("--save", type=str, default="runs/adapt3d_interactive.html")
    args = ap.parse_args()

    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("plotly is required for the interactive plot") from exc

    fig = go.Figure()

    for run in args.runs:
        run_dir = Path(run)
        meta, hist = load_run(run_dir)
        e_exact = meta.get("exact_energy", None)
        if e_exact is None:
            raise ValueError(f"{run_dir}: meta.json missing exact_energy")

        n = np.array([r["iter"] for r in hist], dtype=float)
        t = np.array([r["t_cum_s"] for r in hist], dtype=float)
        e = np.array([r["energy"] for r in hist], dtype=float)

        d_e = e - float(e_exact)
        d_e_abs = np.maximum(np.abs(d_e), args.eps)
        z = log_base(d_e_abs, args.log_base)

        L = meta.get("sites", "?")
        fig.add_trace(
            go.Scatter3d(
                x=n,
                y=t,
                z=z,
                mode="lines+markers",
                name=f"L={L}",
                hovertemplate="n=%{x}<br>t=%{y:.3f}s<br>log|ΔE|=%{z:.3f}<extra>L="
                + str(L)
                + "</extra>",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="iteration n",
            yaxis_title="cumulative wall time t^(n) [s]",
            zaxis_title=f"log_{args.log_base:g} |ΔE^(n)|",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="ADAPT Trajectories",
    )

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved interactive plot to {out}")


if __name__ == "__main__":
    main()
