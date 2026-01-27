#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
    ap.add_argument("--runs", nargs="+", required=True,
                    help="List of run directories containing meta.json and history.jsonl")
    ap.add_argument("--log-base", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=1e-16)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for run in args.runs:
        run_dir = Path(run)
        meta, hist = load_run(run_dir)

        E_exact = meta.get("exact_energy", None)
        if E_exact is None:
            raise ValueError(f"{run_dir}: meta.json missing exact_energy")

        n = np.array([r["iter"] for r in hist], dtype=float)
        t = np.array([r["t_cum_s"] for r in hist], dtype=float)
        E = np.array([r["energy"] for r in hist], dtype=float)

        dE = E - float(E_exact)
        dE_abs = np.maximum(np.abs(dE), args.eps)
        z = log_base(dE_abs, args.log_base)

        L = meta.get("sites", "?")
        ax.plot(n, t, z, label=f"L={L}")

    ax.set_xlabel("iteration n")
    ax.set_ylabel("cumulative wall time t^(n) [s]")
    ax.set_zlabel(f"log_{args.log_base:g} |ΔE^(n)|")

    ax.legend()
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
