#!/usr/bin/env python3
"""Compare ADAPT vs regular VQE ansaetze and plot delta-E by ansatz type per L."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from pydephasing.quantum.ansatz import build_ansatz
from pydephasing.quantum.hamiltonian.hubbard import (
    build_fermionic_hubbard,
    build_qubit_hamiltonian_from_fermionic,
    default_1d_chain_edges,
)
from pydephasing.quantum.symmetry import exact_ground_energy_sector
from pydephasing.quantum.vqe.adapt_vqe_meta import build_reference_state, run_meta_adapt_vqe


def sector_occ(n_sites: int, n_up: int, n_down: int) -> list[int]:
    return list(range(n_up)) + list(range(n_sites, n_sites + n_down))


def run_regular_vqe(
    *,
    qubit_op,
    mapper,
    ansatz_kind: str,
    reps: int = 2,
    seed: int = 7,
) -> float:
    estimator = StatevectorEstimator()
    ansatz = build_ansatz(ansatz_kind, qubit_op.num_qubits, reps, mapper)
    rng = np.random.default_rng(seed)
    initial_point = rng.random(ansatz.num_parameters) * 2 * np.pi
    optimizer = COBYLA(maxiter=150)
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    return float(np.real(result.eigenvalue))


def run_adapt(
    *,
    n_sites: int,
    ferm_op,
    qubit_op,
    mapper,
    n_up: int,
    n_down: int,
) -> float:
    estimator = StatevectorEstimator()
    reference = build_reference_state(qubit_op.num_qubits, sector_occ(n_sites, n_up, n_down))
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
    return float(result.energy)


def main() -> None:
    out_dir = Path("runs/compare_vqe")
    out_dir.mkdir(parents=True, exist_ok=True)

    t = 1.0
    u = 4.0
    dv = 0.5

    # Use the same sector for all sizes under symmetry enforcement.
    sectors = {2: (1, 1), 3: (1, 1), 4: (1, 1)}

    ansatz_kinds = ["clustered", "efficient_su2", "hea", "uccsd"]

    rows: list[dict] = []

    for n_sites in (2, 3, 4):
        n_up, n_down = sectors[n_sites]
        ferm_op = build_fermionic_hubbard(
            n_sites=n_sites,
            t=t,
            u=u,
            edges=default_1d_chain_edges(n_sites, periodic=False),
            v=[-dv / 2, dv / 2] if n_sites == 2 else None,
        )
        qubit_op, mapper = build_qubit_hamiltonian_from_fermionic(ferm_op)
        exact = exact_ground_energy_sector(qubit_op, n_sites, n_up + n_down, 0.5 * (n_up - n_down))

        adapt_e = run_adapt(
            n_sites=n_sites,
            ferm_op=ferm_op,
            qubit_op=qubit_op,
            mapper=mapper,
            n_up=n_up,
            n_down=n_down,
        )
        rows.append(
            {
                "sites": n_sites,
                "ansatz": "adapt_cse_hybrid",
                "energy": adapt_e,
                "exact": exact,
                "delta_e": adapt_e - exact,
            }
        )

        for kind in ansatz_kinds:
            try:
                e = run_regular_vqe(qubit_op=qubit_op, mapper=mapper, ansatz_kind=kind)
            except Exception as exc:
                rows.append(
                    {
                        "sites": n_sites,
                        "ansatz": kind,
                        "energy": None,
                        "exact": exact,
                        "delta_e": None,
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "sites": n_sites,
                    "ansatz": kind,
                    "energy": e,
                    "exact": exact,
                    "delta_e": e - exact,
                }
            )

    (out_dir / "compare_rows.json").write_text(json.dumps(rows, indent=2))

    # Per-L bar charts.
    for n_sites in (2, 3, 4):
        subset = [r for r in rows if r["sites"] == n_sites and r.get("delta_e") is not None]
        if not subset:
            continue
        labels = [r["ansatz"] for r in subset]
        deltas = np.array([float(r["delta_e"]) for r in subset])

        order = np.argsort(deltas)
        labels = [labels[i] for i in order]
        deltas = deltas[order]

        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(labels, deltas, color="#264653")
        ax.set_ylabel("ΔE = E_ansatz - E_exact_sector")
        ax.set_title(f"Delta-E by ansatz type (L={n_sites}, n_up=1, n_down=1)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.tick_params(axis="x", rotation=20)

        for idx, val in enumerate(deltas):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        out_path = out_dir / f"delta_e_hist_L{n_sites}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot to {out_path}")

    print(f"Saved comparison rows to {out_dir / 'compare_rows.json'}")


if __name__ == "__main__":
    main()
