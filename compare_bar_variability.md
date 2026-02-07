# Why The Compare Bar Chart Varies (Depth + Optimization Mismatch)

This note explains why the `runs/compare_vqe/compare_bar_*.png` results can look inconsistent across algorithms and across system size `L`. The comparisons are mixing methods with different circuit capacity, optimizer behavior, and compute.

## What Changes Between Methods (Main Drivers)

- **UCCSD(VQE) scales up in parameter count, but the optimizer budget stays fixed**
  - In `scripts/compare_vqe_hist.py`, UCCSD uses `reps=2` and `COBYLA(maxiter=150)`.
  - UCCSD parameter count for the cached sector (`n_up=n_down=1`, `reps=2`): L2=6, L3=16, L4=30, L5=48, L6=70.
  - Result: performance can swing sharply with `L` because larger-`L` runs are optimization-limited.

- **ADAPT is capped-depth**
  - In `scripts/compare_vqe_hist.py`, ADAPT runs use `max_depth=6` outer operator picks, independent of `L` (so ~6 parameters max).
  - Result: ADAPT can be easier to optimize (low-dim) but can become underexpressive as `L` grows and plateau.

- **VQE(CSE ops) is “ADAPT-picked circuit + VQE optimization”**
  - `vqe_cse_ops` reuses the operator list selected by ADAPT-CSE, then runs `COBYLA(maxiter=150)` on that small circuit.
  - Result: it often looks strong because it combines low parameter count with a large optimization budget.

## Optimization Budget Is Not Comparable

- ADAPT does pool-gradient scans + repeated inner solves; regular VQE does only `maxiter` objective evaluations.
- Example (L=4): ~124s for UCCSD(VQE) vs ~903s for ADAPT-UCCSD in the saved logs (`runs/compare_vqe/logs_uccsd_L4/history.jsonl`, `runs/adapt_uccsd_1769892048_L4_Nup1_Ndown1/history.jsonl`).

## Sector / Physics Regime Also Changes With L

The cached `runs/compare_vqe/compare_rows.json` corresponds to the fixed sector `n_up=n_down=1` (total `N=2`, `Sz=0`) for all `L` (see run directories under `runs/` like `*_L4_Nup1_Ndown1`).

That is **not** half-filling for `L>2`; as `L` increases, the electron density drops. Which restricted ansatz performs best can genuinely change with that regime.

## A Subtle Source Of Variability: “Last” vs “Best” Iteration

- When cached logs are reused, the script can take the **last logged energy** rather than the best-seen energy.
- COBYLA is not monotonic, and ADAPT outer steps can also be non-monotonic if the inner solve is approximate.

So the plotted value can drift depending on which snapshot is used.
