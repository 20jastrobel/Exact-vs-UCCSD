# VQE/ADAPT Comparison (L=2,3,4)

Generated: 2026-02-08T04:01:42-0600
Source: `runs/compare_vqe_L2_L3_L4_vqeparamsfix/compare_rows.json`

## Run Setup

- Model: spinful 1D Hubbard chain (2L spin-orbitals), open boundary (`periodic=False`)
- Parameters: `t=1.0`, `U=4.0`, `dv=0.5` (dimer only)
- Sectors: half filling (N=L); for odd L the sector uses the “min |Sz|” policy:
  - L=2: (n_up,n_down)=(1,1), N=2, Sz=+0.0
  - L=3: (n_up,n_down)=(2,1), N=3, Sz=+0.5
  - L=4: (n_up,n_down)=(2,2), N=4, Sz=+0.0

Run config (derived budgets):

```
budget_k=2000.0, max_pauli_terms_measured=budget_k*L^2, max_time_s=200.0
```

## Plots

- Absolute error bars: `compare_bar_abs_L2_L3_L4.png`
- Relative error bars: `compare_bar_rel_L2_L3_L4.png`
- Circuit metrics (depth/CX/params): `bench_circuit_metrics_L2_L3_L4.png`
- Cost counters (runtime/estimator calls/circuits/Pauli terms): `bench_cost_metrics_L2_L3_L4.png`
- Budget fractions (used/budget): `bench_budget_fractions_L2_L3_L4.png`

![Absolute error bars](compare_bar_abs_L2_L3_L4.png)

![Relative error bars](compare_bar_rel_L2_L3_L4.png)

### Hardware-Proxy Benchmarks

Circuit depth is computed by transpiling to a fixed basis (`rz,sx,x,cx`) on an all-to-all backend (no coupling map).

![Circuit metrics](bench_circuit_metrics_L2_L3_L4.png)

![Cost metrics](bench_cost_metrics_L2_L3_L4.png)

![Budget fractions](bench_budget_fractions_L2_L3_L4.png)

#### Cost-Normalized Error Curves (Best-So-Far)

- `|ΔE|` vs `n_pauli_terms_measured`:
  - `cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L2_Nup1_Ndown1.png`
  - `cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L3_Nup2_Ndown1.png`
  - `cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L4_Nup2_Ndown2.png`

![L=2 cost curve vs n_pauli_terms_measured](cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L2_Nup1_Ndown1.png)

![L=3 cost curve vs n_pauli_terms_measured](cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L3_Nup2_Ndown1.png)

![L=4 cost curve vs n_pauli_terms_measured](cost_curve_abs_delta_e_vs_n_pauli_terms_measured_L4_Nup2_Ndown2.png)

## Summary Table (Best-So-Far Energies)

| L | (n_up,n_down) | Ansatz | E_best | E_exact(sector) | ΔE_best | |ΔE| | |ΔE|/|E_exact| | Stop reason | Run path |
|---:|:------------:|:------|------:|---------------:|------:|-----:|--------------:|:----------|:--------|
| 2 | (1,1) | ADAPT CSE (hybrid) | -0.50631297 | -0.83605712 | 0.32974415 | 3.2974e-01 | 3.9440e-01 | eps_grad | `runs/adapt_cse_1770520649_L2_Nup1_Ndown1` |
| 2 | (1,1) | ADAPT UCCSD (hybrid) | -0.50631297 | -0.83605712 | 0.32974415 | 3.2974e-01 | 3.9440e-01 | pool_exhausted | `runs/adapt_uccsd_1770520650_L2_Nup1_Ndown1` |
| 2 | (1,1) | VQE (CSE ops) | -0.50631297 | -0.83605712 | 0.32974415 | 3.2974e-01 | 3.9440e-01 |  | `runs/compare_vqe_L2_L3_L4_vqeparamsfix/logs_vqe_cse_ops_L2_Nup1_Ndown1` |
| 2 | (1,1) | VQE (UCCSD) | -0.83605700 | -0.83605712 | 0.00000012 | 1.1585e-07 | 1.3857e-07 |  | `runs/compare_vqe_L2_L3_L4_vqeparamsfix/logs_uccsd_L2_Nup1_Ndown1` |
| 3 | (2,1) | ADAPT CSE (hybrid) | -0.96433640 | -1.23606798 | 0.27173158 | 2.7173e-01 | 2.1984e-01 | budget:max_pauli_terms_measured | `runs/adapt_cse_1770520671_L3_Nup2_Ndown1` |
| 3 | (2,1) | ADAPT UCCSD (hybrid) | -0.96471164 | -1.23606798 | 0.27135634 | 2.7136e-01 | 2.1953e-01 | budget:max_pauli_terms_measured | `runs/adapt_uccsd_1770516250_L3_Nup2_Ndown1` |
| 3 | (2,1) | VQE (CSE ops) | -0.96471223 | -1.23606798 | 0.27135575 | 2.7136e-01 | 2.1953e-01 |  | `runs/compare_vqe_L2_L3_L4_vqeparamsfix/logs_vqe_cse_ops_L3_Nup2_Ndown1` |
| 3 | (2,1) | VQE (UCCSD) | -1.23606786 | -1.23606798 | 0.00000011 | 1.1323e-07 | 9.1603e-08 |  | `runs/compare_vqe_L2_L3_L4_vqeparamsfix/logs_uccsd_L3_Nup2_Ndown1` |
| 4 | (2,2) | ADAPT CSE (hybrid) | -0.73501043 | -1.95314531 | 1.21813488 | 1.2181e+00 | 6.2368e-01 | budget:max_time_s | `runs/adapt_cse_1770516350_L4_Nup2_Ndown2` |
| 4 | (2,2) | ADAPT UCCSD (hybrid) | 3.50000000 | -1.95314531 | 5.45314531 | 5.4531e+00 | 2.7920e+00 | budget:max_pauli_terms_measured | `runs/adapt_uccsd_1770516550_L4_Nup2_Ndown2` |
| 4 | (2,2) | VQE (CSE ops) | -0.73501110 | -1.95314531 | 1.21813421 | 1.2181e+00 | 6.2368e-01 |  | `runs/compare_vqe_L2_L3_L4_vqeparamsfix/logs_vqe_cse_ops_L4_Nup2_Ndown2` |
| 4 | (2,2) | VQE (UCCSD) | -0.34535484 | -1.95314531 | 1.60779047 | 1.6078e+00 | 8.2318e-01 | budget:max_time_s | `runs/compare_vqe_L2_L3_L4_vqeparamsfix/logs_uccsd_L4_Nup2_Ndown2` |

## Cost / Benchmarking Counters

All cost counters come from `CountingEstimator` snapshots logged into `history.jsonl`.
The “best” row is the log row at the minimum energy observed; the “final” row is the last log row.

| L | Ansatz | n_params(best) | transpiled_depth | transpiled_cx | estimator_calls(best/final) | circuits(best/final) | pauli_terms(best/final) | energy_evals(best/final) | grad_evals(best/final) | t_cum_s(best/final) |
|---:|:------|--------------:|----------------:|-------------:|----------------------------:|--------------------:|------------------------:|-------------------------:|-----------------------:|--------------------:|
| 2 | ADAPT CSE (hybrid) | 2 | 15 | 8 | 184/213 | 184/213 | 1855/2166 | 21/21 | 148/176 | 1.52/1.9 |
| 2 | ADAPT UCCSD (hybrid) | 2 | 86 | 48 | 172/689 | 172/689 | 1735/6910 | 21/85 | 136/584 | 1.39/16.6 |
| 2 | VQE (CSE ops) | 2 | 15 | 8 | 54/55 | 54/55 | 540/550 | 54/55 | 0/0 | 0.598/0.608 |
| 2 | VQE (UCCSD) | 6 | 179 | 112 | 197/197 | 197/197 | 1970/1970 | 197/197 | 0/0 | 3.27/3.27 |
| 3 | ADAPT CSE (hybrid) | 4 | 24 | 16 | 1033/1040 | 1033/1040 | 17896/18015 | 90/90 | 918/925 | 44.6/45 |
| 3 | ADAPT UCCSD (hybrid) | 4 | 31 | 24 | 782/1040 | 782/1040 | 13629/18015 | 49/67 | 708/948 | 27.9/51.9 |
| 3 | VQE (CSE ops) | 4 | 24 | 16 | 74/76 | 74/76 | 1258/1292 | 74/76 | 0/0 | 3.83/3.93 |
| 3 | VQE (UCCSD) | 16 | 771 | 544 | 411/419 | 411/419 | 6987/7123 | 411/419 | 0/0 | 41.4/42.2 |
| 4 | ADAPT CSE (hybrid) | 4 | 24 | 16 | 997/1282 | 997/1282 | 24858/31698 | 80/88 | 892/1169 | 133/200 |
| 4 | ADAPT UCCSD (hybrid) | 2 | 15 | 8 | 764/1311 | 764/1311 | 18894/32022 | 21/47 | 728/1249 | 41.3/113 |
| 4 | VQE (CSE ops) | 4 | 24 | 16 | 83/83 | 83/83 | 1992/1992 | 83/83 | 0/0 | 15.4/15.4 |
| 4 | VQE (UCCSD) | 52 | 3639 | 2752 | 162/165 | 162/165 | 3888/3960 | 162/165 | 0/0 | 198/201 |

## Notes

- VQE baseline callbacks currently report a `callback_params_len` of 1 due to an upstream Qiskit Algorithms callback behavior; we log the correct `n_params` from the ansatz itself.
- ADAPT grouped operators are implemented as products of Pauli evolutions for each Pauli component (a Trotterized/product-form generator), which can differ from `exp(-i θ Σ_j c_j P_j)` and can affect convergence/leakage.
