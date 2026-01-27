#!/usr/bin/env python3
"""Sweep t and U for L=2 and plot exact vs ADAPT energy curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorEstimator

from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.symmetry import exact_ground_energy_sector
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
    reference = build_reference_state(qubit_op.num_qubits, [0, n_sites])
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


def main() -> None:
    out_dir = Path("runs/sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sites = 2
    n_up = 1
    n_down = 1
    dv = 0.5

    t_fixed = 1.0
    u_fixed = 4.0

    t_values = np.linspace(0.5, 1.5, 5)
    u_values = np.linspace(2.0, 6.0, 5)

    exact_vs_t = []
    adapt_vs_t = []
    for t in t_values:
        exact, adapt = run_case(n_sites=n_sites, t=float(t), u=u_fixed, dv=dv, n_up=n_up, n_down=n_down)
        exact_vs_t.append(exact)
        adapt_vs_t.append(adapt)

    exact_vs_u = []
    adapt_vs_u = []
    for u in u_values:
        exact, adapt = run_case(n_sites=n_sites, t=t_fixed, u=float(u), dv=dv, n_up=n_up, n_down=n_down)
        exact_vs_u.append(exact)
        adapt_vs_u.append(adapt)

    payload = {
        "t_values": list(map(float, t_values)),
        "u_values": list(map(float, u_values)),
        "exact_vs_t": list(map(float, exact_vs_t)),
        "adapt_vs_t": list(map(float, adapt_vs_t)),
        "exact_vs_u": list(map(float, exact_vs_u)),
        "adapt_vs_u": list(map(float, adapt_vs_u)),
        "u_fixed": u_fixed,
        "t_fixed": t_fixed,
        "dv": dv,
    }
    (out_dir / "sweep_t_u_L2.json").write_text(json.dumps(payload, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(t_values, exact_vs_t, "-o", label="Exact (sector)")
    axes[0].plot(t_values, adapt_vs_t, "-s", label="ADAPT (hybrid)")
    axes[0].set_title(f"E vs t (U={u_fixed:g}, dv={dv:g})")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Energy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(u_values, exact_vs_u, "-o", label="Exact (sector)")
    axes[1].plot(u_values, adapt_vs_u, "-s", label="ADAPT (hybrid)")
    axes[1].set_title(f"E vs U (t={t_fixed:g}, dv={dv:g})")
    axes[1].set_xlabel("U")
    axes[1].set_ylabel("Energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = out_dir / "E_vs_t_and_U_L2.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    print(f"Saved data to {out_dir / 'sweep_t_u_L2.json'}")


if __name__ == "__main__":
    main()
