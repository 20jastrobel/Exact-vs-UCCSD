# How To Run This Repo (Algorithms, Pools, Benchmarks)

This file is meant to be fed to an LLM as operational context. It summarizes:

- which algorithms/ansaetze exist in the repo,
- what knobs/options they expose,
- what scripts generate the benchmarking artifacts under `runs/`.

Last updated: 2026-02-07

## Core Conventions (Important)

- `L` / `n_sites`: number of Hubbard sites (spatial orbitals).
- Spinful convention: `n_qubits = 2 * L`.
  - qubits `[0..L-1]` are spin-up orbitals
  - qubits `[L..2L-1]` are spin-down orbitals
- Symmetry sector is `(n_up, n_down)`.
  - `N_total = n_up + n_down`
  - `Sz = (n_up - n_down)/2`
- Half filling (repo convention for benchmarking): `N_total = L`.
  - even `L`: `(n_up,n_down)=(L/2,L/2)` (Sz=0)
  - odd `L`: Sz=0 impossible at half filling; choose minimal `|Sz|` by default, i.e. `((L+1)/2,(L-1)/2)` (Sz=+1/2).

## Algorithms / Ansaetze

### 1) ADAPT-VQE / Meta-ADAPT-VQE

Entry points:

- `run_adapt_meta.py` (CLI)
- core implementation: `pydephasing/quantum/vqe/adapt_vqe_meta.py` (`run_meta_adapt_vqe`)

High-level mechanics:

1. Build an operator pool (Pauli pool from Hamiltonian terms, or grouped Pauli generators from fermionic constructions).
2. Outer loop:
   - probe all pool operators by measuring the energy gradient if that operator were appended at parameter 0
   - pick operator with max `|gradient|`
   - append it to the ansatz
   - optimize all parameters (inner optimizer)
3. Stop when:
   - `max_depth` reached
   - `max|gradient| < eps_grad`
   - `|ΔE| < eps_energy`

Key knobs (from `run_adapt_meta.py` / `run_meta_adapt_vqe`):

- Pool choice: `--pool {ham_terms,ham_terms_plus_imag_partners,cse_density_ops,uccsd_excitations}`
- Sector filtering: `--enforce-sector` (default True in `run_adapt_meta.py`)
- Outer-loop budget:
  - `--max-depth`
  - `--eps-grad`
  - `--eps-energy`
- Inner optimizer selection:
  - `--inner-optimizer {lbfgs,meta,hybrid}`
  - L-BFGS-B knobs: `--inner-steps` (used as maxiter), `--lbfgs-restarts`, `--theta-bound {none,pi}`
  - Meta knobs: `--weights` (trained LSTM weights), `--meta-*` (hidden size, step scale, clipping, etc.)
  - Hybrid knobs: `--warmup-steps`, `--polish-steps`
- Operator reuse: `--allow-repeats`

Pools (ADAPT `--pool`):

- `ham_terms` / `ham_terms_plus_imag_partners`
  - pool is Pauli strings extracted from the mapped qubit Hamiltonian
  - `ham_terms_plus_imag_partners` also adds “imaginary partner” Pauli strings and enriches with single-qubit `X/Y`
- `cse_density_ops` (grouped generators)
  - built from fermionic terms found in the Hubbard Hamiltonian and mapped to qubits
  - include both Hermitian quadratures (when enabled):
    - `gamma_im(label)` = `-i (γ - γ†)` (Hermitian)
    - `gamma_re(label)` = `(γ + γ†)` (Hermitian)
  - CSE knobs:
    - `--cse-include-diagonal / --no-cse-include-diagonal`
    - `--cse-include-antihermitian / --no-cse-include-antihermitian`
    - `--cse-include-hermitian / --no-cse-include-hermitian`
- `uccsd_excitations` (grouped generators)
  - extracted from Qiskit Nature’s built-in `UCCSD(...).operators` list
  - UCCSD pool knobs:
    - `--uccsd-reps <int>`
    - `--uccsd-include-imaginary / --no-uccsd-include-imaginary`
    - `--uccsd-generalized / --no-uccsd-generalized`
    - `--uccsd-preserve-spin / --no-uccsd-preserve-spin`

Notes:

- “Grouped” generators are implemented as a product of Pauli evolutions for each Pauli term in the generator. This is a Trotter-like approximation to `exp(-i θ Σ c_j P_j / 2)`.
- ADAPT logs sector diagnostics (`N_mean`, `Sz_mean`, `VarN`, `VarSz`) and can optionally log energy variance `VarH` (`run_meta_adapt_vqe(..., compute_var_h=True)` or `run_adapt_meta.py --compute-var-h`).

### 2) Regular VQE baselines

Entry point (benchmark harness):

- `scripts/compare_vqe_hist.py`

Baselines currently used in compare runs:

- `uccsd`:
  - ansatz built via `pydephasing.quantum.ansatz.build_ansatz("uccsd", reps=2, ...)`
  - optimizer: `COBYLA(maxiter=150)` in the compare harness
- `vqe_cse_ops`:
  - fixed grouped-Pauli ansatz built from the operator sequence selected by cached ADAPT-CSE
  - optimizer: `COBYLA(maxiter=150)`

## Benchmark Pipelines (runs/compare_vqe/*)

### A) Energy benchmarking and caching

Script:

- `scripts/compare_vqe_hist.py`

What it produces under `runs/compare_vqe/`:

- `compare_rows.json` (per `(L,n_up,n_down,ansatz)` row: energy, exact energy, ΔE, sector metadata)
- per-L energy bar plots (ΔE, |ΔE|, relative error)
- per-run regular VQE logs under `runs/compare_vqe/logs_*/history.jsonl`

Important:

- The compare harness enforces half filling per `L` (with explicit odd-`L` policy).
- Cached directories and row keys include `Nup/Ndown` to avoid mixing sectors.
- Regular VQE runs write `result.json` in each `logs_*` dir with `optimal_point` so downstream scripts can reconstruct final states.
- ADAPT runs write `runs/<run_id>_L*_Nup*_Ndown*/result.json` with final `theta` and chosen operator names.

### B) Cost-normalized benchmarking (energy vs compute proxy)

Files:

- `pydephasing/quantum/vqe/cost_model.py`: `CountingEstimator` + `CostCounters`
- `scripts/plot_cost_curves.py`: plots best-so-far `|ΔE|` vs cost proxy

What is counted:

- `n_estimator_calls`: number of `estimator.run(...)` calls
- `n_circuits_executed`: number of pubs submitted across calls
- `n_pauli_terms_measured`: sum of non-identity Pauli terms measured
- coarse categories:
  - `n_energy_evals` (objective-category)
  - `n_grad_evals` (gradient-category)

Usage:

```bash
python scripts/plot_cost_curves.py --sites 2 3 4 5 6
```

### C) Circuit metrics benchmarking (depth, CX, etc.)

Script:

- `scripts/benchmark_circuit_metrics.py`

What it does:

- Rebuilds final circuits for each method and computes depth/CX counts (raw + transpiled).
- Uses cached ADAPT operator sequences to reconstruct ADAPT-style circuits.

### D) State-quality benchmarking (beyond energy)

Script:

- `scripts/benchmark_state_quality.py`

Metrics (statevector-only):

- `Var(H)` of the final state
- infidelity to the exact sector ground state: `1 - |<psi_exact|psi>|^2`
- physical observables:
  - total double occupancy `D = Σ_i <n_{i↑} n_{i↓}>`
  - site densities `<n_i>`
  - nearest-neighbor spin correlations `Σ_i <Sz_i Sz_{i+1}>`
- sector leakage:
  - `p_leak = 1 - p_sector`, where `p_sector` is probability mass in the target `(N,Sz)` sector

Artifacts under `runs/compare_vqe/`:

- `state_quality_summary.md`
- `state_quality_var_h.png`, `state_quality_infidelity.png`, `state_quality_p_leak.png`
- `leakage_traj.json` + per-L `leakage_vs_depth_L*.png` (ADAPT only)

### E) Grouped-generator approximation microbenchmark (sum vs product)

Script:

- `scripts/benchmark_grouped_generator_approx.py`

Purpose:

- Quantify the implementation error from using a product of Pauli evolutions instead of the exact sum-form evolution.

Usage example:

```bash
python scripts/benchmark_grouped_generator_approx.py --run-dir runs/<adapt_run_dir> --theta 0.5 --approx prod --order sorted
python scripts/benchmark_grouped_generator_approx.py --run-dir runs/<adapt_run_dir> --theta 0.5 --approx symmetric --order random
```

Outputs:

- JSON report per selected operator (distance proxy + sector leakage induced).
- A simple bar plot summarizing the operators.

## Common Debugging Questions

- If ADAPT(UCCSD) is worse than regular UCCSD:
  - ADAPT is usually capped by `max_depth` and may use a smaller pool (`uccsd_reps=1` by default) than the baseline UCCSD ansatz (`reps=2` in the compare harness).
  - Sector leakage can occur even when generators commute as a sum, due to product-form implementation.
  - Fixed optimizer budgets (COBYLA maxiter) can under-optimize large-parameter UCCSD circuits for larger `L`.
- If energy is close but fidelity/observables are bad:
  - near-degeneracy and/or sector leakage; use `infidelity`, `Var(H)`, and observables to disambiguate.
