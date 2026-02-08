# VQE/ADAPT Comparison (L=2,3,4)

Generated: 2026-02-07T21:05:47-0600
Source: `runs/compare_vqe/compare_rows.json`

## Run Setup

- Model: spinful 1D Hubbard chain (2L spin-orbitals), open boundary (`periodic=False`)
- Parameters: `t=1.0`, `U=4.0`, `dv=0.5` (dimer only)
- Sectors: half filling (N=L); for odd L the sector uses the “min |Sz|” policy:
  - L=2: (n_up,n_down)=(1,1), N=2, Sz=0
  - L=3: (n_up,n_down)=(2,1), N=3, Sz=+1/2
  - L=4: (n_up,n_down)=(2,2), N=4, Sz=0

Run config (from ADAPT meta.json, best-effort):

```
budget_k=2000.0, max_pauli_terms_measured=8000, max_circuits_executed=None, max_time_s=200.0
```

Budget notes:

- Per-L pauli-term budget (derived as `budget_k * L^2`):
  - L=2: 8000
  - L=3: 18000
  - L=4: 32000
- Wall-time cap: `max_time_s=200` applies to both ADAPT and VQE baselines.

## Plots

- Absolute error bars: `runs/compare_vqe/compare_bar_abs_L2_L3_L4.png`
- Relative error bars: `runs/compare_vqe/compare_bar_rel_L2_L3_L4.png`
- Circuit metrics (depth/CX/params): `runs/compare_vqe/bench_circuit_metrics_L2_L3_L4.png`
- Cost counters (runtime/estimator calls/circuits/Pauli terms): `runs/compare_vqe/bench_cost_metrics_L2_L3_L4.png`
- Budget fractions (used/budget): `runs/compare_vqe/bench_budget_fractions_L2_L3_L4.png`

![Absolute error bars](compare_bar_abs_L2_L3_L4.png)

![Relative error bars](compare_bar_rel_L2_L3_L4.png)

### Hardware-Proxy Benchmarks

These plots are intended to answer “are the runs comparable under the same budget constraints?”:

- Circuit metrics are taken from `scripts/benchmark_circuit_metrics.py` and are transpiled to a fixed basis (`rz,sx,x,cx`).
- Cost metrics and budget fractions are taken from the **final** `history.jsonl` row for each run.

![Circuit metrics](bench_circuit_metrics_L2_L3_L4.png)

![Cost metrics](bench_cost_metrics_L2_L3_L4.png)

![Budget fractions](bench_budget_fractions_L2_L3_L4.png)

#### Cost-Normalized Error Curves (Best-So-Far)

- `|ΔE|` vs `n_pauli_terms_measured`:
  - `runs/compare_vqe/cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L2_Nup1_Ndown1.png`
  - `runs/compare_vqe/cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L3_Nup2_Ndown1.png`
  - `runs/compare_vqe/cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L4_Nup2_Ndown2.png`

![L=2 cost curve vs n_pauli_terms_measured](cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L2_Nup1_Ndown1.png)

![L=3 cost curve vs n_pauli_terms_measured](cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L3_Nup2_Ndown1.png)

![L=4 cost curve vs n_pauli_terms_measured](cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L4_Nup2_Ndown2.png)

- L=2 per-ansatz: `runs/compare_vqe/delta_e_abs_hist_L2_Nup1_Ndown1.png`, `runs/compare_vqe/delta_e_rel_hist_L2_Nup1_Ndown1.png`
- L=3 per-ansatz: `runs/compare_vqe/delta_e_abs_hist_L3_Nup2_Ndown1.png`, `runs/compare_vqe/delta_e_rel_hist_L3_Nup2_Ndown1.png`
- L=4 per-ansatz: `runs/compare_vqe/delta_e_abs_hist_L4_Nup2_Ndown2.png`, `runs/compare_vqe/delta_e_rel_hist_L4_Nup2_Ndown2.png`

## Summary Table (Best-So-Far Energies)

| L | (n_up,n_down) | Ansatz | E_best | E_exact(sector) | ΔE_best | |ΔE| | |ΔE|/|E_exact| | Stop reason | Run path |
|---:|:------------:|:------|------:|---------------:|------:|-----:|--------------:|:----------|:--------|
| 2 | (1,1) | ADAPT CSE (hybrid) | -0.50631297 | -0.83605712 | 0.32974415 | 0.329744 | 0.394404 | eps_grad | `runs/adapt_cse_1770516184_L2_Nup1_Ndown1` |
| 2 | (1,1) | ADAPT UCCSD (hybrid) | -0.50631297 | -0.83605712 | 0.32974415 | 0.329744 | 0.394404 | pool_exhausted | `runs/adapt_uccsd_1770516186_L2_Nup1_Ndown1` |
| 2 | (1,1) | VQE (CSE ops) | -0.50631297 | -0.83605712 | 0.32974415 | 0.329744 | 0.394404 |  | `runs/compare_vqe/logs_vqe_cse_ops_L2_Nup1_Ndown1` |
| 2 | (1,1) | VQE (UCCSD) | -0.83605700 | -0.83605712 | 0.00000012 | 1.15854e-07 | 1.38572e-07 |  | `runs/compare_vqe/logs_uccsd_L2_Nup1_Ndown1` |
| 3 | (2,1) | ADAPT CSE (hybrid) | -0.96433640 | -1.23606798 | 0.27173158 | 0.271732 | 0.219835 | budget:max_pauli_terms_measured | `runs/adapt_cse_1770516207_L3_Nup2_Ndown1` |
| 3 | (2,1) | ADAPT UCCSD (hybrid) | -0.96471164 | -1.23606798 | 0.27135634 | 0.271356 | 0.219532 | budget:max_pauli_terms_measured | `runs/adapt_uccsd_1770516250_L3_Nup2_Ndown1` |
| 3 | (2,1) | VQE (CSE ops) | -0.96471223 | -1.23606798 | 0.27135575 | 0.271356 | 0.219531 |  | `runs/compare_vqe/logs_vqe_cse_ops_L3_Nup2_Ndown1` |
| 3 | (2,1) | VQE (UCCSD) | -1.23606786 | -1.23606798 | 0.00000011 | 1.13228e-07 | 9.16033e-08 |  | `runs/compare_vqe/logs_uccsd_L3_Nup2_Ndown1` |
| 4 | (2,2) | ADAPT CSE (hybrid) | -0.73501043 | -1.95314531 | 1.21813488 | 1.21813 | 0.623679 | budget:max_time_s | `runs/adapt_cse_1770516350_L4_Nup2_Ndown2` |
| 4 | (2,2) | ADAPT UCCSD (hybrid) | 3.50000000 | -1.95314531 | 5.45314531 | 5.45315 | 2.79198 | budget:max_pauli_terms_measured | `runs/adapt_uccsd_1770516550_L4_Nup2_Ndown2` |
| 4 | (2,2) | VQE (CSE ops) | -0.73501110 | -1.95314531 | 1.21813421 | 1.21813 | 0.623678 |  | `runs/compare_vqe/logs_vqe_cse_ops_L4_Nup2_Ndown2` |
| 4 | (2,2) | VQE (UCCSD) | -0.34978323 | -1.95314531 | 1.60336208 | 1.60336 | 0.820913 | budget:max_time_s | `runs/compare_vqe/logs_uccsd_L4_Nup2_Ndown2` |

## Cost / Benchmarking Counters

All cost counters come from `CountingEstimator` snapshots logged into `history.jsonl`.
The “best” row is the log row at the minimum energy observed; the “final” row is the last log row.

| L | Ansatz | n_params(best) | depth(best) | estimator_calls(best/final) | circuits(best/final) | pauli_terms(best/final) | energy_evals(best/final) | grad_evals(best/final) | t_cum_s(best/final) |
|---:|:------|--------------:|-----------:|----------------------------:|--------------------:|------------------------:|-------------------------:|-----------------------:|--------------------:|
| 2 | ADAPT CSE (hybrid) | 2 | 2 | 184/213 | 184/213 | 1855/2166 | 21/21 | 148/176 | 1.482/1.86 |
| 2 | ADAPT UCCSD (hybrid) | 2 | 2 | 172/689 | 172/689 | 1735/6910 | 21/85 | 136/584 | 1.386/16.79 |
| 2 | VQE (CSE ops) | 1 |  | 54/55 | 54/55 | 540/550 | 54/55 | 0/0 | 0.598/0.6085 |
| 2 | VQE (UCCSD) | 1 |  | 197/197 | 197/197 | 1970/1970 | 197/197 | 0/0 | 3.221/3.221 |
| 3 | ADAPT CSE (hybrid) | 4 | 4 | 1033/1040 | 1033/1040 | 17896/18015 | 90/90 | 918/925 | 42.86/43.28 |
| 3 | ADAPT UCCSD (hybrid) | 4 | 4 | 782/1040 | 782/1040 | 13629/18015 | 49/67 | 708/948 | 27.9/51.9 |
| 3 | VQE (CSE ops) | 1 |  | 74/76 | 74/76 | 1258/1292 | 74/76 | 0/0 | 3.863/3.968 |
| 3 | VQE (UCCSD) | 1 |  | 411/419 | 411/419 | 6987/7123 | 411/419 | 0/0 | 43.29/44.15 |
| 4 | ADAPT CSE (hybrid) | 4 | 4 | 997/1282 | 997/1282 | 24858/31698 | 80/88 | 892/1169 | 132.8/200 |
| 4 | ADAPT UCCSD (hybrid) | 2 | 2 | 764/1311 | 764/1311 | 18894/32022 | 21/47 | 728/1249 | 41.29/113.3 |
| 4 | VQE (CSE ops) | 1 |  | 83/83 | 83/83 | 1992/1992 | 83/83 | 0/0 | 16.23/16.23 |
| 4 | VQE (UCCSD) | 1 |  | 168/170 | 168/170 | 4032/4080 | 168/170 | 0/0 | 198.3/200.7 |

## Notes

- `compare_rows.json` stores “best-so-far” energies (minimum over `history.jsonl`), not the last-logged energy.
- Budget aborts for COBYLA may emit SciPy messages like `capi_return is NULL`; these correspond to intentional early-stops via callback.
