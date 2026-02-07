# Repository Description: Hubbard VQE + (Meta-)ADAPT-VQE (CSE + UCCSD Operator Pools)

Last updated: 2026-02-07

This repo is a focused sandbox for benchmarking variational quantum algorithms on the **spinful 1D Hubbard model** using Qiskit + Qiskit Nature. The two main ADAPT operator pools are:

- **CSE-inspired density-like fermionic terms** mapped to grouped Pauli generators (`pool_mode=cse_density_ops`).
- **Qiskit Nature UCCSD excitation generators** mapped to grouped Pauli generators (`pool_mode=uccsd_excitations`).

The practical goal is to understand:

- How **operator pool choice** + **selection rule** affects convergence and depth.
- When **ADAPT** outperforms/underperforms standard **VQE(UCCSD)** given the same simulation budget.
- Why performance can vary strongly across `L`, even with "similar" settings (optimizer budget, sector choice, ansatz depth).

This file is written to be consumed by an LLM for debugging advice. For deeper tables and results, see `deep_research_benchmarking.md`.

## Core Definitions (Repo Conventions)

- `L` / `n_sites`: number of spatial orbitals / Hubbard sites.
- Spinful mapping: number of qubits is `2 * L` (first `L` are spin-up orbitals, next `L` are spin-down orbitals).
- Particle-number sector is specified by `(n_up, n_down)`.
- `N_total = n_up + n_down`
- `Sz = (n_up - n_down) / 2`
- "Half filling" for the spinful Hubbard chain means `N_total = L` (not `2L`).

## Repository Layout (Where Things Live)

Primary entrypoints:

- `run_adapt_meta.py`: main CLI for running ADAPT/Meta-ADAPT on Hubbard models with pool selection, sector targeting, and logging.
- `run_vqe_local.py`: minimal local UCCSD-VQE runner (primarily for quick checks).

Core implementation:

- `pydephasing/quantum/vqe/adapt_vqe_meta.py`: pool builders (`build_operator_pool_from_hamiltonian`, `build_cse_density_pool_from_fermionic`, `build_uccsd_excitation_pool`), ADAPT driver (`run_meta_adapt_vqe`), and grouped-circuit builder (`build_adapt_circuit_grouped`).
- `pydephasing/quantum/symmetry.py`: sector operators `N` and `Sz` and the commutation check used for filtering pools.
- `pydephasing/quantum/utils_particles.py`: helpers for half-filling and reference occupations (Jordan-Wigner ordering).

Benchmarking + plots:

- `scripts/compare_vqe_hist.py`: produces the cached energy comparisons and plots under `runs/compare_vqe/`.
- `scripts/plot_compare_bars.py`: utility for generating grouped bar charts from `compare_rows.json`.
- `scripts/benchmark_circuit_metrics.py`: benchmarks circuit depth / gate counts (raw + transpiled) for the cached comparison methods.

Deep-research context:

- `deep_research_benchmarking.md`: comprehensive notes and tables for energy + circuit metrics benchmarking.
- `a.md`: notes focused on operator pools, selection mechanisms, and implementation details (historical working doc).
- `compare_bar_variability.md`: notes about why bar-chart comparisons can vary (sector, optimizer budget, reps/depth mismatch, etc.).

## Operator Pools (What They Actually Contain)

`pool_mode=ham_terms` / `ham_terms_plus_imag_partners`:

- Pools are Pauli strings extracted from the mapped qubit Hamiltonian.
- `ham_terms_plus_imag_partners` expands the pool with "imaginary partner" Pauli strings and also adds single-qubit `X`/`Y` terms to avoid trivial stagnation.

`pool_mode=cse_density_ops` (grouped pool):

- Built from eligible fermionic terms in the Hubbard Hamiltonian, including "density-like" `+_p -_q` (including diagonal `p==q` when enabled) and optionally additional diagonal-ish Hamiltonian terms when `include_diagonal=True`.
- Hermitian quadratures included (when enabled):
- `gamma_im(label)`: `-i * (gamma - gamma^dagger)` (Hermitian)
- `gamma_re(label)`: `(gamma + gamma^dagger)` (Hermitian)
- CLI flags for the CSE pool (`run_adapt_meta.py`):
- `--cse-include-diagonal / --no-cse-include-diagonal`
- `--cse-include-antihermitian / --no-cse-include-antihermitian`
- `--cse-include-hermitian / --no-cse-include-hermitian`

`pool_mode=uccsd_excitations` (grouped pool):

- Uses Qiskit Nature's built-in `UCCSD(...).operators` + `excitation_list`, then converts each generator to the repo's grouped-Pauli spec `{name, paulis}`.
- Important conversion detail: we canonicalize each operator to a numerically Hermitian form and require real Pauli coefficients after canonicalization (we do not silently drop imaginary components).
- CLI flags for the UCCSD pool (`run_adapt_meta.py`):
- `--uccsd-reps <int>`
- `--uccsd-include-imaginary / --no-uccsd-include-imaginary`
- `--uccsd-generalized / --no-uccsd-generalized`
- `--uccsd-preserve-spin / --no-uccsd-preserve-spin`

## ADAPT-VQE / Meta-ADAPT-VQE Mechanics (Important For Debugging)

In `run_meta_adapt_vqe(...)`:

- Outer loop: evaluate gradients of energy w.r.t. "adding one new operator at parameter 0" for every pool element; select the operator with maximum absolute gradient; append it; optimize all parameters.
- Stop conditions: `max_depth` reached, `max|gradient| < eps_grad`, or `|DeltaE| < eps_energy` (energy stall).
- `allow_repeats`: if `False` (default) selected operators are removed from the pool; if `True` selected operators remain selectable.

Sector enforcement (`enforce_sector=True`):

- The pool is filtered by commutation with mapped qubit `N` and `Sz` operators.
- Diagnostics logged to `history.jsonl` include: `N_mean`, `Sz_mean`, `VarN`, `VarSz`, `abs_N_err`, `abs_Sz_err`.
- This is important because the implementation of a "grouped generator" is a product of Pauli evolutions for each Pauli component, so exact sector preservation is not guaranteed even if the generator sum commutes as an operator.

## Benchmarking: What Exists And Where To Look

Energy benchmarking (cached):

- Main table: `runs/compare_vqe/compare_rows.json`
- Plots: `runs/compare_vqe/compare_bar_abs_L2_L3_L4_L5_L6.png` and `runs/compare_vqe/compare_bar_rel_L2_L3_L4_L5_L6.png`
- Driver: `scripts/compare_vqe_hist.py`

Circuit metrics benchmarking:

- Main table: `runs/compare_vqe/circuit_metrics.json`
- Plot: `runs/compare_vqe/circuit_metrics_depth.png`
- Driver: `scripts/benchmark_circuit_metrics.py`
- Rebuilds the final ansatz circuits for each method from cached ADAPT operator sequences.
- Computes circuit metrics both raw and after transpilation to a fixed basis (`rz,sx,x,cx`) with `optimization_level=1`.

Important caching/sector note:

- `runs/compare_vqe/compare_rows.json` currently stores `L`, ansatz kind, and energies, but does not include `n_up/n_down`.
- As of 2026-02-07, the cached comparison rows under `runs/compare_vqe/` correspond to the fixed sector `(n_up,n_down)=(1,1)` for `L=2..6` (not half-filling for `L>2`).
- The per-run ADAPT logs under `runs/<run_id>_L*_Nup*_Ndown*/meta.json` are the source of truth for sector and settings.

## Common Failure Modes / Why Comparisons Vary

If performance seems inconsistent across algorithms or across L, the most common causes in this repo are:

- Sector mismatch: comparing results computed in different `(n_up,n_down)` sectors will dominate all other effects, and cached compare tables may not reflect current "default sector" logic if they were generated earlier.
- UCCSD depth/expressivity mismatch: `UCCSD` baselines in `scripts/compare_vqe_hist.py` use `reps=2` (more parameters, much larger depth), while ADAPT-UCCSD uses a depth cap (`max_depth`) and (by default) a `reps=1` excitation generator pool.
- Optimizer budget mismatch: UCCSD parameter count grows quickly with `L` but the benchmark uses a fixed COBYLA budget (`maxiter=150`), and ADAPT uses fixed inner budgets (`inner_steps`, `warmup_steps`, `polish_steps`) that may not scale with size.
- Circuit depth effects: transpiled depth and CX counts can vary by orders of magnitude, and high-depth circuits are more sensitive to optimizer settings and numerical conditioning even in statevector simulations.
- Operator naming drift: older cached runs may contain `chosen_op` names like `gamma(...)` while the current CSE pool uses `gamma_im(...)` / `gamma_re(...)`; scripts that reconstruct operator sequences must map these names appropriately.

## Quick Commands (Local, Statevector)

Single ADAPT run (explicit sector):

```bash
python run_adapt_meta.py --sites 4 --n-up 1 --n-down 1 --pool cse_density_ops --inner-optimizer hybrid
```

Regenerate the comparison table/plots:

```bash
python scripts/compare_vqe_hist.py --sites 2 3 4 5 6 --force --allow-exact-compute
python scripts/plot_compare_bars.py --compare runs/compare_vqe/compare_rows.json --sites 2 3 4 5 6 --metric abs --out runs/compare_vqe/compare_bar_abs_L2_L3_L4_L5_L6.png
python scripts/plot_compare_bars.py --compare runs/compare_vqe/compare_rows.json --sites 2 3 4 5 6 --metric rel --out runs/compare_vqe/compare_bar_rel_L2_L3_L4_L5_L6.png
```

Benchmark circuit depth and CX counts for the cached runs:

```bash
python scripts/benchmark_circuit_metrics.py --sites 2 3 4 5 6
```

## IBM Runtime Integration

- `pydephasing/quantum/hubbard_jw_check.py` supports local VQE warm-start and IBM runtime evaluation modes with the UCCSD ansatz.
- Environment variables used in IBM modes:
- `QISKIT_IBM_TOKEN`
- `QISKIT_IBM_INSTANCE`
- `QISKIT_IBM_REGION`
- `QISKIT_IBM_PLANS`
- `IBM_BACKEND`
- `EXACT_TOL`

## Known Repo Warts (Worth Fixing If You Hit Them)

- `scripts/compare_adapt_reuse.py` currently contains leftover patch artifacts and is not runnable as-is.
- Some cached `runs/` entries were generated under older pool naming schemes and/or older defaults for sector selection.
