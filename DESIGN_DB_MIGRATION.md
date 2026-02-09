# DESIGN: DB Migration (JSON/JSONL Runs -> SQLite Ledger)

This repo currently logs runs as `meta.json` + `history.jsonl` (+ `result.json`) under `runs/`, and drives comparisons/plots by scanning those folders. This document summarizes the *current* run layout + schemas (from code), then proposes a migration to an authoritative SQLite “run ledger” with artifact references.

No implementation here; this is a plan only.

## Current Pipeline (Repo Reality)

### Canonical Run Directory Layout + Naming

ADAPT runs (produced by both `run_adapt_meta.py` and `scripts/compare_vqe_hist.py`):

- `runs/<run_id>_L{L}_Nup{n_up}_Ndown{n_down}/`
- Files: `meta.json`, `history.jsonl`, `result.json`
- Naming is constructed in:
  - `run_adapt_meta.py:make_run_dir_and_meta`
  - `scripts/compare_vqe_hist.py:make_adapt_run_dir`

Regular-VQE baseline logs (produced by `scripts/compare_vqe_hist.py` under a compare output dir):

- `runs/compare_vqe*/logs_uccsd_L{L}_Nup{n_up}_Ndown{n_down}/`
- `runs/compare_vqe*/logs_vqe_cse_ops_L{L}_Nup{n_up}_Ndown{n_down}/`
- Files: `history.jsonl`, `result.json`

Compare harness outputs (produced by `scripts/compare_vqe_hist.py` and plot/benchmark scripts):

- `runs/compare_vqe*/compare_rows.json`
- Plots and derived artifacts under the same `runs/compare_vqe*/` folder.

### JSON / JSONL Schemas (Discovered From Writers)

#### ADAPT: `meta.json`

Two variants exist today:

- Superset (written by `run_adapt_meta.py:make_run_dir_and_meta`):
  - `run_id`, `sites`, `n_up`, `n_down`, `sz_target`, `pool`
  - `cse.{include_diagonal, include_antihermitian_part, include_hermitian_part}`
  - `uccsd.{reps, include_imaginary, generalized, preserve_spin}`
  - `inner_optimizer`, `max_depth`, `eps_grad`, `eps_energy`
  - `ham_params.{t,u,dv}`
  - `budget` (dict; populated in `run_adapt_meta.py` before writing)
  - `exact_energy` (updated post-run via `run_adapt_meta.py:update_meta_exact_energy`)
  - `python`, `platform`
- Subset (written by `scripts/compare_vqe_hist.py:make_adapt_run_dir`):
  - `run_id`, `sites`, `n_up`, `n_down`, `pool`, `inner_optimizer`, `max_depth`, `eps_grad`, `eps_energy`
  - `budget.{budget_k,max_pauli_terms_measured,max_circuits_executed,max_time_s}`
  - `ham_params.{t,u,dv}`, `exact_energy`, `python`, `platform`

#### ADAPT: `history.jsonl` (iterative metrics)

Format: JSON Lines (one JSON object per line). Rows are written via a `RunLogger`:

- Base row keys (from `run_adapt_meta.py:RunLogger.log_point` and `scripts/compare_vqe_hist.py:RunLogger.log_point`):
  - `iter`, `energy`, `max_grad`, `chosen_op`, `t_iter_s`, `t_cum_s`
- Extra keys supplied by `pydephasing/quantum/vqe/adapt_vqe_meta.py:run_meta_adapt_vqe`:
  - Algorithm state: `ansatz_len`, `n_params`, `pool_size`, `stop_reason`
  - Cost counters: `CostCounters.snapshot()` in `pydephasing/quantum/vqe/cost_model.py`
    - `n_energy_evals`, `n_grad_evals`, `n_estimator_calls`, `n_circuits_executed`, `n_pauli_terms_measured`, `total_shots`
  - Optional correctness diagnostics:
    - `VarH` (when `compute_var_h=True`)
    - Sector diagnostics (from `pydephasing/quantum/vqe/adapt_vqe_meta.py:compute_sector_diagnostics`):
      - `N_mean`, `Sz_mean`, `VarN`, `VarSz`, `abs_N_err`, `abs_Sz_err`

Iteration semantics:

- ADAPT logs an `iter=0` baseline row, then logs outer iterations (`iter=1..max_depth`) when `log_every=1`. (`pydephasing/quantum/vqe/adapt_vqe_meta.py`)

#### ADAPT: `result.json`

Two variants exist today:

- Minimal (written by `run_adapt_meta.py:_save_result`): `energy`, `theta`, `operators`
- Compare-harness (written by `scripts/compare_vqe_hist.py`): adds `best_energy`, `pool`, `stop_reason`, `budget`

#### Regular VQE baselines: `history.jsonl`

Written by `scripts/compare_vqe_hist.py` inside the VQE optimizer callback (`run_regular_vqe(...)` and `run_vqe_cse_ops(...)`).

Row keys:

- `iter` (COBYLA objective eval count)
- `energy`, `delta_e` (relative to `exact_energy` passed into the run)
- `t_iter_s`, `t_cum_s`
- `ansatz`, `n_params`, `callback_params_len`
- Cost counters: `CostCounters.snapshot()` (`pydephasing/quantum/vqe/cost_model.py`)
- Optional: `stop_reason` (only on the row that triggers budget abort)

#### Regular VQE baselines: `result.json`

Written by `scripts/compare_vqe_hist.py` at the end of `run_regular_vqe(...)` / `run_vqe_cse_ops(...)`.

- `ansatz`, `n_params`, `n_sites`, `num_particles` (`[n_up,n_down]`)
- `reps` (UCCSD only)
- `energy`, `optimal_point`
- `optimizer.{name,maxiter}`, `seed`
- `stop_reason`, `budget.{max_pauli_terms_measured,max_circuits_executed,max_time_s}`

#### Compare harness table: `compare_rows.json`

Written by `scripts/compare_vqe_hist.py` to `<out_dir>/compare_rows.json` (default `runs/compare_vqe/compare_rows.json`).

Each row includes (minimum):

- `sites` and `L` (both set)
- `n_up`, `n_down`, `N`, `Sz`, `filling`
- `ansatz`
- `energy`, `exact`, `delta_e`
- optional `error` if a run failed.

#### Derived JSON artifacts (under `runs/compare_vqe*/` and run dirs)

- Circuit metrics: `scripts/benchmark_circuit_metrics.py` writes `runs/compare_vqe/circuit_metrics.json` (default) with:
  - `meta` and `rows[]` where each row contains `sites,n_up,n_down,ansatz,n_qubits,raw{...},transpiled{...}`.
- Leakage trajectory summary: `scripts/benchmark_state_quality.py` writes `runs/compare_vqe/leakage_traj.json`:
  - list entries with keys: `sites,n_up,n_down,ansatz,depth,abs_N_err,abs_Sz_err,VarN,VarSz,VarH,max_abs_N_err,max_abs_Sz_err,max_VarN,max_VarSz,max_VarH`.
- Grouped-generator approximation benchmark: `scripts/benchmark_grouped_generator_approx.py` writes `grouped_generator_approx_theta{...}_{approx}_{order}.json` (default inside a run dir):
  - keys: `run_dir,sites,n_up,n_down,pool,theta,approx,order,state,seed,operators[]` (each operator has `name,n_terms,dist_proxy,p_leak_*`, etc.).

### Plot Outputs (Where, Names, Formats)

Format: all `savefig` calls in repo are `.png` (no `.pdf`/`.svg` save paths found); some optional interactive `.html` outputs exist.

Core compare pipeline plots (all under `runs/compare_vqe*/`):

- Per-L plots written by `scripts/compare_vqe_hist.py`:
  - `delta_e_hist_L{L}_Nup{n_up}_Ndown{n_down}.png`
  - `delta_e_abs_hist_L{L}_Nup{n_up}_Ndown{n_down}.png`
  - `delta_e_rel_hist_L{L}_Nup{n_up}_Ndown{n_down}.png`
- Grouped bars (written by `scripts/plot_compare_bars.py`): output path is CLI-controlled (commonly `compare_bar_abs_*.png`, `compare_bar_rel_*.png`).
- Bench bars (written by `scripts/plot_benchmark_bars.py`): `bench_circuit_metrics_{sites_tag}.png`, `bench_cost_metrics_{sites_tag}.png`, `bench_budget_fractions_{sites_tag}.png`.
- Cost curves (written by `scripts/plot_cost_curves.py`): `cost_curve_abs_delta_e_vs_{x}_L{L}_Nup{n_up}_Ndown{n_down}.png`.
- Circuit-metrics plot (written by `scripts/benchmark_circuit_metrics.py`): `runs/compare_vqe/circuit_metrics_depth.png` (default).
- State-quality plots (written by `scripts/benchmark_state_quality.py`): `state_quality_*.png`, `leakage_vs_depth_L{L}.png`.

Interactive HTML (non-core): `plot_adapt_3d_interactive.py` defaults to `runs/adapt3d_L2_to_L6_interactive.html`; `scripts/plot_adapt_hist3d_blocks.py` can emit `*_interactive.html`.

## Target State (SQLite Run Ledger + Artifact References)

### Rules

- **SQL is authoritative:** once enabled, SQLite is the source of truth; JSON is legacy/export only (align with `AGENTS.md`).
- Store artifacts by *reference* (path + checksum), not as DB blobs.
- Store raw JSON rows as a JSON column to tolerate schema drift.

### Proposed Location

- Default DB: `runs/run_ledger.sqlite3` (override via `--db <path>`).

### Minimal Schema Proposal

`runs` (one row per run):

- identity/provenance: `run_uuid` (PK), `run_dir`, `created_at_utc`, `git_commit`, `git_dirty`, `command_line`, `python_version`, `platform`, `hostname`
- system/sector: `n_sites,n_up,n_down,n_total,sz_target,filling,ham_t,ham_u,ham_dv`
- method config: `algorithm` (`adapt|vqe`), `ansatz`, `pool`, `inner_optimizer`, `optimizer_name`, `optimizer_maxiter`, `reps`, `seed`, plus a JSON `config_json`
- stop/budget: `stop_reason`, `status`, and budget columns mirroring current `meta.json`/`result.json`
- results: `exact_energy_sector`, `energy_best`, `energy_final`, and deltas; plus JSON `meta_json`/`result_json` for lossless import/export

`steps` (one row per history entry):

- `run_uuid`, `iter`, `energy`
- optional columns for common keys: `max_grad,chosen_op,t_iter_s,t_cum_s,VarH,N_mean,Sz_mean,VarN,VarSz,abs_N_err,abs_Sz_err`
- cost counters: `n_energy_evals,n_grad_evals,n_estimator_calls,n_circuits_executed,n_pauli_terms_measured,total_shots`
- `stop_reason` (if present)
- `row_json` (full original row)

`artifacts` (files on disk):

- `artifact_uuid` (PK), `run_uuid` (nullable), `kind`, `path`, `sha256`, `bytes`, `created_at_utc`, optional `mime`

Optional: `ops` (normalized operator sequence):

- `run_uuid, op_index, op_name, op_json`

## Compat Mode Plan (Keep JSON Behind a Flag)

1) “Compat release”: default write to SQLite; keep live JSON writing behind `--write-json` (or `--log-format both`). Provide a DB->JSON exporter so existing plot scripts can keep working with old inputs if needed.

2) Next release: default to SQL-only; JSON is export-only.

## Migration Steps (Ordered) + Risks/Mitigations

1) Add a tiny SQLite module (schema + WAL + insert helpers).

2) Add a DB-backed logger implementing the existing `log_point(...)` interface used by:

- `pydephasing/quantum/vqe/adapt_vqe_meta.py:run_meta_adapt_vqe` (ADAPT)
- `scripts/compare_vqe_hist.py` VQE callbacks

3) Wire producers (`run_adapt_meta.py`, `scripts/compare_vqe_hist.py`) to create/finalize `runs` rows and stream `steps`.

4) Add an importer to backfill existing `runs/**/{meta.json,history.jsonl,result.json}` and `runs/compare_vqe*/logs_*/{history.jsonl,result.json}` into the DB (idempotent by checksum).

5) Update plot/benchmark scripts to prefer SQL (`--source sql`) with JSON fallback for one release.

6) Flip defaults to SQL-only; keep JSON as export-only.

Risks:

- SQLite locking with concurrent runs: mitigate with WAL, short transactions, batched inserts, and optional `--db` scoping (per-compare-dir DB).
- Performance overhead for per-iteration inserts (VQE): mitigate by batching.
- Schema drift across older logs: mitigate with `row_json` and nullable extracted columns.
- Partial/incomplete runs: mitigate with explicit `status` in DB (don’t infer from file presence).

