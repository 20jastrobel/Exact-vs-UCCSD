# Benchmarking: ADAPT vs VQE vs UCCSD (Energy + Circuit Metrics)

Last updated: 2026-02-07

This document is meant to be fed to a Deep Research LLM. It explains how we benchmark the different algorithms/ansaetze in this repo, and summarizes the latest cached results (both energy accuracy and circuit-structure metrics like transpiled depth and CX count).

Artifacts (latest in this workspace):

- Energy comparison rows: `runs/compare_vqe/compare_rows.json`
- Energy bar plots: `runs/compare_vqe/compare_bar_abs_L2_L3_L4_L5_L6.png`, `runs/compare_vqe/compare_bar_rel_L2_L3_L4_L5_L6.png`
- Cost curves (error vs compute proxy): `runs/compare_vqe/cost_curve_abs_delta_e_vs_n_circuits_executed_L*_Nup*_Ndown*.png`
- Circuit metrics rows: `runs/compare_vqe/circuit_metrics.json`
- Circuit metrics plot: `runs/compare_vqe/circuit_metrics_depth.png`
- State-quality summary: `runs/compare_vqe/state_quality_summary.md`
- State-quality plots: `runs/compare_vqe/state_quality_var_h.png`, `runs/compare_vqe/state_quality_infidelity.png`, `runs/compare_vqe/state_quality_p_leak.png`
- ADAPT leakage trajectories: `runs/compare_vqe/leakage_traj.json` and `runs/compare_vqe/leakage_vs_depth_L*.png`
- Energy comparison driver: `scripts/compare_vqe_hist.py`
- Cost curve driver: `scripts/plot_cost_curves.py`
- Circuit metrics driver: `scripts/benchmark_circuit_metrics.py`
- State-quality driver: `scripts/benchmark_state_quality.py`
- Grouped-generator microbenchmark: `scripts/benchmark_grouped_generator_approx.py`
- Cost counters implementation: `pydephasing/quantum/vqe/cost_model.py`

## What We Benchmark

We benchmark two main things:

- Energy accuracy: energy error versus the exact ground-state energy in the same symmetry sector.
- Circuit structure: circuit depth and two-qubit gate counts (CX) for the final ansatz circuits.
- Compute budget (cost-normalized): error versus *true primitive workload proxies* (estimator calls, circuits executed, Pauli terms measured).

## Model / Setup (Applies to All Results Below)

Hamiltonian:

- Spinful 1D Hubbard model on an open chain (non-periodic edges).
- Parameters used in cached benchmarks: `t=1.0`, `U=4.0`, `dv=0.5`.
- Potential term usage: for `L=2`, we include a staggered potential `v=[-dv/2, +dv/2]`; for `L>2` we used `v=None` (no site potential).

Mapping / simulation:

- Fermions are mapped to qubits with Jordan-Wigner: `qiskit_nature.second_q.mappers.JordanWignerMapper`.
- Energies are evaluated with `qiskit.primitives.StatevectorEstimator` (statevector expectation, no shot noise).

Symmetry sector (important):

- The intended comparison regime is **half filling** for the spinful Hubbard convention used here:
  - `N_total = n_up + n_down = L`
  - if `L` is even: `n_up = n_down = L/2` (Sz=0)
  - if `L` is odd: Sz=0 is impossible at half filling; we choose minimal `|Sz|` (default Sz=+1/2), i.e. `n_up=(L+1)/2`, `n_down=(L-1)/2`
- Legacy cached results may have been produced in a fixed sector `(n_up,n_down)=(1,1)` for all `L` (constant `N=2`), which is **not** half-filling for `L>2`. Always check `runs/compare_vqe/compare_rows.json` for `n_up/n_down/N/filling`.

Exact reference energy:

- We compute the exact ground energy in the same sector using `pydephasing.quantum.symmetry.exact_ground_energy_sector`.
- The comparison rows store this in the `exact` field, and store `delta_e = energy - exact`.

## Algorithms / Ansaetze Being Compared

The comparison plots/tables include 4 entries per `L`:

1. `ADAPT CSE (hybrid)` (`ansatz=adapt_cse_hybrid`)

- Outer loop: ADAPT-VQE selecting from a grouped operator pool built by `build_cse_density_pool_from_fermionic(...)` in `pydephasing/quantum/vqe/adapt_vqe_meta.py`.
- Selection rule: pick the operator with the largest absolute energy gradient at the current point (gradient evaluated via parameter-shift probing of a new operator at parameter 0).
- Symmetry enforcement: `enforce_sector=True` in the ADAPT run; the pool is filtered by commutation with qubit `N` and `Sz`.
- Inner optimizer: `inner_optimizer="hybrid"` (hybrid of meta warmup and LBFGS polishing).
- In the cached benchmark harness: `max_depth=6`, `inner_steps=25`, `warmup_steps=5`, `polish_steps=3`, `eps_grad=1e-4`, `eps_energy=1e-3`.

2. `ADAPT UCCSD (hybrid)` (`ansatz=adapt_uccsd_hybrid`)

- Outer loop: ADAPT-VQE selecting from the grouped pool `uccsd_excitations`.
- The pool is extracted from Qiskit Nature's UCCSD generator list (`UCCSD(...).operators` with associated `excitation_list`) and converted into grouped Pauli-generator specs.
- Symmetry enforcement and ADAPT selection mechanics are the same as in ADAPT CSE.
- The cached benchmark uses the same ADAPT budgets as above (`max_depth=6`, hybrid inner optimizer, etc.).

3. `VQE (CSE ops)` (`ansatz=vqe_cse_ops`)

- This is a standard VQE run that uses the *operator sequence selected by ADAPT CSE* (from cached ADAPT logs) as a fixed ansatz circuit.
- The circuit is built with `build_adapt_circuit_grouped(reference_state, chosen_operator_specs)`.
- Optimizer: `COBYLA(maxiter=150)` with initial parameters all zeros.

4. `UCCSD (VQE, reps=2)` (`ansatz=uccsd`)

- This is Qiskit Nature UCCSD used directly as the VQE ansatz, created by `pydephasing.quantum.ansatz.build_ansatz("uccsd", reps=2, ...)`.
- It includes a Hartree-Fock initial state in the circuit.
- Optimizer: `COBYLA(maxiter=150)` with initial parameters all zeros.

Note on UCCSD `reps`:

- In Qiskit Nature, `reps` controls the number of Trotter repetitions of the UCC operator list.
- Increasing `reps` increases depth and parameter count, and can also increase expressivity.

## How The Energy Benchmark Was Computed

Primary driver:

- `scripts/compare_vqe_hist.py` (this script manages caching and produces `runs/compare_vqe/compare_rows.json`).

Per system size `L`:

- Build the fermionic Hubbard Hamiltonian (`build_fermionic_hubbard`), map to qubit Hamiltonian (`build_qubit_hamiltonian_from_fermionic`).
- Determine exact sector ground energy with `exact_ground_energy_sector` (loaded from cached run metadata when available).
- For each ansatz kind:
  - ADAPT runs: `run_meta_adapt_vqe(...)` with the given pool mode and `StatevectorEstimator`.
  - VQE runs: `qiskit_algorithms.VQE` with `StatevectorEstimator` and `COBYLA`.
- Store comparison rows in `runs/compare_vqe/compare_rows.json` including at least:
  - `sites` (L), `n_up`, `n_down`, `N`, `Sz`, `filling`
  - `ansatz`, `energy`, `exact`, `delta_e` (plus optional `error`)
- Generate bar plots of `abs(delta_e)` and `abs(delta_e)/abs(exact)` in `runs/compare_vqe/`.

## Cost-Normalized Benchmarking (Error vs Compute Proxy)

Motivation:

- ADAPT has a large fixed overhead per outer iteration due to *pool-wide gradient probes* (scales with pool size).
- Regular VQE spends most of its budget on optimizer-driven energy evaluations (scales with parameter count and optimizer behavior).

Implementation:

- We wrap `StatevectorEstimator` (and any `BaseEstimatorV2`) with `CountingEstimator` (`pydephasing/quantum/vqe/cost_model.py`) to count primitive-level work:
  - `n_estimator_calls`: number of `estimator.run(...)` calls
  - `n_circuits_executed`: total pubs submitted across those calls (proxy for circuit executions)
  - `n_pauli_terms_measured`: sum of non-identity Pauli terms across all submitted observables (proxy for measurement workload)
- We also record coarse algorithm-level categories:
  - `n_energy_evals`: objective-category energy evaluations (VQE objective calls; ADAPT energy evals of the current ansatz)
  - `n_grad_evals`: gradient-category energy evaluations (ADAPT pool probes + parameter-shift energies; typically 0 for COBYLA VQE)

Where it shows up:

- Regular VQE histories: `runs/compare_vqe/logs_*/history.jsonl` now include the above counters on each callback line.
- ADAPT histories: `runs/<run_id>_L*_Nup*_Ndown*/history.jsonl` include the counters each outer iteration.

Plots:

- `scripts/plot_cost_curves.py` generates best-so-far curves of `|ΔE|` versus a chosen x-axis cost proxy:
  - default: `n_circuits_executed` (log-x)
  - optional: `n_pauli_terms_measured`, `n_estimator_calls`, or `t_cum_s`

Important note:

- Old cached logs will not have these counters; to get cost curves under `runs/compare_vqe/`, re-run the benchmark after this instrumentation is present.

## Correctness Beyond Energy (Variance, Fidelity, Observables, Leakage)

Energy alone can be misleading (sector leakage, near-degeneracy, under-optimized circuits). We therefore compute additional state-quality metrics in statevector mode.

Driver:

- `scripts/benchmark_state_quality.py`

Computed metrics per `(L, ansatz)` (when the run directory contains enough info to reconstruct the final state):

- Energy variance: `Var(H) = <H^2> - <H>^2` (exact eigenstates have Var(H)=0).
- Fidelity to the exact sector ground state: `1 - |<psi_exact|psi_ansatz>|^2`.
- Physical observables (Hubbard-relevant):
  - total double occupancy: `D = Σ_i <n_{i↑} n_{i↓}>`
  - site densities: `<n_i>`
  - nearest-neighbor spin correlations: `Σ_i <Sz_i Sz_{i+1}>`
- Sector leakage: `p_leak = 1 - p_sector`, where `p_sector` is the probability mass in the target `(N,Sz)` sector.

Outputs:

- `runs/compare_vqe/state_quality_summary.md` and the associated plots under `runs/compare_vqe/`.
- ADAPT trajectory summaries for `VarN/VarSz` are written to `runs/compare_vqe/leakage_traj.json` and plotted as `runs/compare_vqe/leakage_vs_depth_L*.png`.

Prerequisites / caveats:

- `benchmark_state_quality.py` expects `compare_rows.json` to include sector fields (`n_up`, `n_down`, `N`, `Sz`).
- It also expects each cached run directory to contain a `result.json` with the final parameters (regular VQE logs store `optimal_point`; ADAPT runs store `theta` + chosen operator names). Old cached runs may lack these files and will be skipped with a `state_quality_error`.

Microbenchmark for grouped-generator implementation error:

- `scripts/benchmark_grouped_generator_approx.py` compares the exact sum-form evolution `exp(-i (theta/2) Σ c_j P_j)` against the product-form implementation `Π exp(-i (theta/2) c_j P_j)` (and a symmetric 2nd-order option).
- It reports a distance proxy and induced sector leakage per selected operator, helping isolate whether leakage is coming from the implementation (product-form) rather than the mathematical generator (sum-form).

## How Circuit Metrics (Depth / CX) Were Computed

Primary driver:

- `scripts/benchmark_circuit_metrics.py` (added to benchmark circuit structure consistently).

Key idea:

- We rebuild the final ansatz circuits for each method (using cached chosen operator sequences for ADAPT-based methods), then compute circuit metrics both before and after transpilation.

Steps:

- Use the same per-L half-filling sector policy as the energy benchmark (do not infer sectors from unrelated cached runs).
- For ADAPT CSE and ADAPT UCCSD:
  - Find cached run dirs under `runs/` for `pool="cse_density_ops"` and `pool="uccsd_excitations"`.
  - Parse the chosen operator names from `history.jsonl` (`chosen_op` entries).
  - Rebuild the operator pool in the current code and map operator names back to operator specs.
  - Build the grouped-Pauli ansatz circuit with `build_adapt_circuit_grouped(reference_state, operator_specs)`.
- For `VQE (CSE ops)`:
  - Use the same circuit as ADAPT CSE (because the ansatz is defined as the ADAPT-selected operator sequence).
- For `UCCSD`:
  - Build the circuit via `build_ansatz("uccsd", reps=2, ...)`.

Metrics:

- Raw metrics from the circuit object:
  - `qc.depth()`, `qc.size()`, `qc.num_parameters`, and `qc.count_ops()` (we also store a `cx` count).
- Transpiled metrics:
  - We transpile to a fixed basis gate set `['rz','sx','x','cx']` using `qiskit.transpile(...)` with `optimization_level=1` and `seed_transpiler=7`.
  - No coupling map is provided (all-to-all connectivity). Depth/CX under realistic hardware constraints will generally be higher.

Outputs:

- `runs/compare_vqe/circuit_metrics.json` with raw + transpiled metrics per `(L, ansatz)`.
- `runs/compare_vqe/circuit_metrics_depth.png` (bar chart of transpiled depth by `L` and ansatz).

## Results: Energy Accuracy (Legacy Cached Numbers)

Source: `runs/compare_vqe/compare_rows.json` (as cached at the time this table was generated).

These numbers were from a cached benchmark where the sector was fixed to `(n_up,n_down)=(1,1)` for all L (constant `N=2`). Treat this as **legacy/non-half-filling** for `L>2`.

| L | Ansatz | Energy | Exact (sector) | dE = E - Exact | abs(dE) | rel = abs(dE)/abs(Exact) |
|---:|---|---:|---:|---:|---:|---:|
| 2 | ADAPT CSE (hybrid) | -0.506312970694 | -0.836057118155 | 0.329744147461 | 0.329744147461 | 0.394404 |
| 2 | ADAPT UCCSD (hybrid) | -0.505387782455 | -0.836057118155 | 0.330669335700 | 0.330669335700 | 0.395510 |
| 2 | VQE (CSE ops) | -0.506312950819 | -0.836057118155 | 0.329744167336 | 0.329744167336 | 0.394404 |
| 2 | UCCSD (VQE, reps=2) | -0.836043075163 | -0.836057118155 | 0.000014042992 | 0.000014042992 | 0.000017 |
| 3 | ADAPT CSE (hybrid) | -1.650405666771 | -2.000000000000 | 0.349594333229 | 0.349594333229 | 0.174797 |
| 3 | ADAPT UCCSD (hybrid) | -1.934553808491 | -2.000000000000 | 0.065446191509 | 0.065446191509 | 0.032723 |
| 3 | VQE (CSE ops) | -1.652391528974 | -2.000000000000 | 0.347608471026 | 0.347608471026 | 0.173804 |
| 3 | UCCSD (VQE, reps=2) | -1.979459960213 | -2.000000000000 | 0.020540039787 | 0.020540039787 | 0.010270 |
| 4 | ADAPT CSE (hybrid) | -2.343824284430 | -2.624942271511 | 0.281117987081 | 0.281117987081 | 0.107095 |
| 4 | ADAPT UCCSD (hybrid) | -2.411393906094 | -2.624942271511 | 0.213548365417 | 0.213548365417 | 0.081354 |
| 4 | VQE (CSE ops) | -2.351220097542 | -2.624942271511 | 0.273722173969 | 0.273722173969 | 0.104277 |
| 4 | UCCSD (VQE, reps=2) | -1.831297201866 | -2.624942271511 | 0.793645069645 | 0.793645069645 | 0.302348 |
| 5 | ADAPT CSE (hybrid) | -2.719969158896 | -2.995181761882 | 0.275212602986 | 0.275212602986 | 0.091885 |
| 5 | ADAPT UCCSD (hybrid) | -2.411393906094 | -2.995181761882 | 0.583787855788 | 0.583787855788 | 0.194909 |
| 5 | VQE (CSE ops) | -2.742564035920 | -2.995181761882 | 0.252617725961 | 0.252617725961 | 0.084341 |
| 5 | UCCSD (VQE, reps=2) | -2.690633871096 | -2.995181761882 | 0.304547890786 | 0.304547890786 | 0.101679 |
| 6 | ADAPT CSE (hybrid) | -2.717301378320 | -3.232781389062 | 0.515480010742 | 0.515480010742 | 0.159454 |
| 6 | ADAPT UCCSD (hybrid) | -2.411393906093 | -3.232781389062 | 0.821387482969 | 0.821387482969 | 0.254081 |
| 6 | VQE (CSE ops) | -2.742564029054 | -3.232781389062 | 0.490217360009 | 0.490217360009 | 0.151640 |
| 6 | UCCSD (VQE, reps=2) | -2.651862263213 | -3.232781389062 | 0.580919125849 | 0.580919125849 | 0.179696 |

## Results: Circuit Structure Metrics

Source: `runs/compare_vqe/circuit_metrics.json`

Numbers shown below are **transpiled** depth and **transpiled** CX counts after transpiling to `['rz','sx','x','cx']` with `optimization_level=1` and `seed_transpiler=7`.

| L | Sector (n_up,n_down) | Ansatz | n_qubits | n_params | n_ops | transpiled_depth | transpiled_cx |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | (1,1) | ADAPT CSE (hybrid) | 4 | 3 | 3 | 27 | 12 |
| 2 | (1,1) | ADAPT UCCSD (hybrid) | 4 | 3 | 3 | 86 | 48 |
| 2 | (1,1) | VQE (CSE ops) | 4 | 3 | 3 | 27 | 12 |
| 2 | (1,1) | UCCSD (VQE, reps=2) | 4 | 6 |  | 179 | 112 |
| 3 | (1,1) | ADAPT CSE (hybrid) | 6 | 6 | 6 | 36 | 24 |
| 3 | (1,1) | ADAPT UCCSD (hybrid) | 6 | 6 | 6 | 195 | 124 |
| 3 | (1,1) | VQE (CSE ops) | 6 | 6 | 6 | 36 | 24 |
| 3 | (1,1) | UCCSD (VQE, reps=2) | 6 | 16 |  | 771 | 544 |
| 4 | (1,1) | ADAPT CSE (hybrid) | 8 | 6 | 6 | 36 | 24 |
| 4 | (1,1) | ADAPT UCCSD (hybrid) | 8 | 6 | 6 | 138 | 92 |
| 4 | (1,1) | VQE (CSE ops) | 8 | 6 | 6 | 36 | 24 |
| 4 | (1,1) | UCCSD (VQE, reps=2) | 8 | 30 |  | 1946 | 1464 |
| 5 | (1,1) | ADAPT CSE (hybrid) | 10 | 6 | 6 | 42 | 24 |
| 5 | (1,1) | ADAPT UCCSD (hybrid) | 10 | 6 | 6 | 138 | 92 |
| 5 | (1,1) | VQE (CSE ops) | 10 | 6 | 6 | 42 | 24 |
| 5 | (1,1) | UCCSD (VQE, reps=2) | 10 | 48 |  | 3873 | 3040 |
| 6 | (1,1) | ADAPT CSE (hybrid) | 12 | 5 | 5 | 33 | 20 |
| 6 | (1,1) | ADAPT UCCSD (hybrid) | 12 | 6 | 6 | 126 | 80 |
| 6 | (1,1) | VQE (CSE ops) | 12 | 5 | 5 | 33 | 20 |
| 6 | (1,1) | UCCSD (VQE, reps=2) | 12 | 70 |  | 6720 | 5440 |

Notes:

- `n_ops` is the number of grouped generators in the final ADAPT-style ansatz (also equal to the number of parameters in those circuits). It is empty for regular UCCSD because that circuit is defined differently.
- For the UCCSD circuits, the **raw** (pre-transpile) `QuantumCircuit.depth()` is not meaningful because Qiskit stores large composite instructions; use the transpiled metrics for fair comparison.

## Key Observations (Hypotheses For A Research LLM To Investigate)

- Optimization budget vs parameter count:
  - UCCSD(reps=2) parameter count grows quickly with L (6, 16, 30, 48, 70 for L=2..6 here), but we used a fixed COBYLA budget (`maxiter=150`). Under-optimization is a plausible reason for large UCCSD energy errors at larger L in these cached results.
- Expressivity mismatch in the comparison:
  - ADAPT-UCCSD here is not constructed to be equivalent to UCCSD(reps=2). It is an ADAPT-selected sequence from a (by-default) reps=1 excitation-generator pool, capped at `max_depth=6`.
- Circuit structure scaling:
  - Transpiled depth/CX for UCCSD(reps=2) grows very rapidly with L (179/112 at L=2 to 6720/5440 at L=6).
  - The ADAPT-style circuits (including VQE(CSE ops)) remain comparatively shallow because `max_depth` is capped and the number of selected operators stayed small in the cached runs.

## Additional Diagnostics Available In Logs (Not Included In Tables Above)

ADAPT logging (`runs/<run_id>_L*_Nup*_Ndown*/history.jsonl`) includes:

- `energy`, `max_grad`, `chosen_op`, `ansatz_len`, `n_params`, `pool_size`
- Per-outer-iteration timing: `t_iter_s`, `t_cum_s`
- Sector diagnostics when the run is configured with `(mapper, n_sites, n_up, n_down)`:
  - `N_mean`, `Sz_mean`
  - `VarN`, `VarSz`
  - `abs_N_err`, `abs_Sz_err`

These diagnostics were added to help distinguish "the generator sum commutes with N/Sz" from "the implemented product-of-Pauli-evolutions preserves the sector in practice" (trotterization and numerical effects can cause leakage).

## Reproducing / Updating The Benchmark

Energy benchmark:

- The current energy rows are already in `runs/compare_vqe/compare_rows.json`.
- To recompute comparisons from scratch, use `scripts/compare_vqe_hist.py` (note: current defaults in that script may target half-filling sectors, which will produce different cached data than the fixed (1,1) sector reflected above).

Circuit metrics benchmark:

- Run:

```bash
python scripts/benchmark_circuit_metrics.py --sites 2 3 4 5 6
```

- Outputs will be written to:
  - `runs/compare_vqe/circuit_metrics.json`
  - `runs/compare_vqe/circuit_metrics_depth.png`
