# Codex Agent Guide (Re_copy)

## Commands (Run These First)

### Install (Python)

This repo is not packaged (no `pyproject.toml`). Scripts add the repo root to `sys.path`, so installs are only for dependencies.

Minimal dev install (enough to run the quantum benchmarks + tests):

```bash
python -m pip install -U pip
python -m pip install numpy scipy matplotlib pytest
python -m pip install qiskit qiskit-aer qiskit-nature qiskit-algorithms
```

Full/legacy install (broad list; may pull in heavy/native deps):

```bash
python -m pip install -U pip
python -m pip install -r dependencies/requirements.txt
```

Known-good versions in this workspace (for reference, not a guarantee):

```bash
python -c "import qiskit,qiskit_aer,qiskit_nature,qiskit_algorithms; \
print('qiskit',qiskit.__version__); \
print('qiskit-aer',qiskit_aer.__version__); \
print('qiskit-nature',qiskit_nature.__version__); \
print('qiskit-algorithms',qiskit_algorithms.__version__)"
```

### Tests

```bash
python -m pytest -q
```

### Lint / Static Checks

No repo-wide linter config is present (no Ruff/Black config checked in).

Minimal “smoke lint” that always works:

```bash
python -m compileall -q pydephasing scripts run_adapt_meta.py run_vqe_local.py
```

If you install Ruff locally, you may run:

```bash
ruff check .
```

## Strict Boundaries (Non-Negotiable)

- Do not fabricate paths, results, performance claims, or “what the code does”. Verify by reading files or running commands.
- Do not edit large/generated artifacts under `runs/` unless explicitly asked.
- Prefer additive changes (new flags, new helpers, new tests) over refactors.
- Any change that may alter numerical behavior must be gated by:
  - deterministic seeds where applicable,
  - unit tests (add/adjust tests in `pydephasing/quantum/vqe/test_*.py`),
  - and a short note in the PR/commit message describing the numerical impact.
- Keep cached-run semantics stable: if changing logging/caching rules, update the harness and add a regression test.
- Avoid “cleanup-only” formatting churn (no mass reformatting, no mechanical renames) unless requested.

## Repo Map (What’s Where)

Core library:

- `pydephasing/`
- `pydephasing/quantum/hamiltonian/`: Hubbard model builders + exact diagonalization helpers.
- `pydephasing/quantum/ansatz/`: ansatz builders (notably UCCSD wrapper).
- `pydephasing/quantum/vqe/adapt_vqe_meta.py`: operator pools + ADAPT driver + logging hooks + budget stopping.
- `pydephasing/quantum/vqe/cost_model.py`: cost instrumentation (`CostCounters`, `CountingEstimator`).
- `pydephasing/quantum/symmetry.py`: `N`, `S_z` mapping + commutation checks for sector filtering.
- `pydephasing/quantum/utils_particles.py`: half-filling / sector utilities.

Entry points / CLIs:

- `run_adapt_meta.py`: main ADAPT CLI (pool selection, sector enforcement, budgets, logging).
- `run_vqe_local.py`: small UCCSD-VQE runner (mainly for quick checks).

Benchmark + plotting scripts:

- `scripts/compare_vqe_hist.py`: generates per-L comparison rows + per-L plots; writes to `runs/compare_vqe*/`.
- `scripts/plot_compare_bars.py`: absolute/relative error bar plots from `compare_rows.json`.
- `scripts/benchmark_circuit_metrics.py`: transpiled depth/CX/params for final circuits.
- `scripts/plot_benchmark_bars.py`: grouped bar charts for cost counters + circuit metrics.
- `scripts/plot_cost_curves.py`: best-so-far `|ΔE|` vs cost proxy curves.

Outputs (generated data/artifacts):

- `runs/`: all experiment outputs.
  - ADAPT runs: `runs/adapt_*_L{L}_Nup{n_up}_Ndown{n_down}/` (contains `meta.json`, `history.jsonl`, `result.json`).
  - Compare harness outputs: `runs/compare_vqe*/` (contains `compare_rows.json`, `logs_*` folders, plots, reports).

Config/data:

- `dependencies/requirements.txt`: dependency list (broad; may not reflect the minimal quantum-only set).
- `hubbard_params.json`: commonly used Hubbard parameter presets.

Docs (LLM context, keep updated when behavior changes):

- `REPO_DESCRIPTION.md`: high-level project context + key file pointers.
- `deep_research_benchmarking.md`: benchmarking methodology + artifact inventory.
- `a.md`: operator pool definition/selection/implementation details.
- `instructional.md`: “how to run” options and script knobs.
- `compare_bar_variability.md`: analysis of compare-bar variability sources.

## Runs Data Contract (Directories + Fields)

### Definition: “Run”

A **run** is a reproducible execution of a specific algorithm/ansatz on a fixed Hubbard instance and symmetry sector, producing:

- provenance (`meta.json` or `result.json`),
- step-by-step metrics (`history.jsonl`),
- final/best summary (`result.json`),
- optional plots/artifacts with checksums.

### Required Provenance Fields

Every run must record (in `meta.json` and/or `result.json`):

- `run_id`: unique identifier (timestamp-based ok)
- `created_at`: ISO-8601 timestamp
- `git`: commit hash and dirty flag
- `platform`: OS/arch + `python` version
- `problem`:
  - `model`: `"hubbard_1d_spinful"`
  - `L` / `n_sites`
  - `ham_params`: `t`, `u`, `dv`, `periodic` and any site potentials `v`
  - `sector`: `(n_up, n_down)` and derived `(N, Sz)`
- `method`:
  - algorithm name (e.g. `adapt_*`, `vqe_*`)
  - ansatz/pool mode (e.g. `cse_density_ops`, `uccsd_excitations`)
  - optimizer + hyperparameters
  - `seed`(s) used (optimizer seed, transpiler seed if relevant)
- `budgets` (when enabled): `max_time_s`, `max_pauli_terms_measured`, `max_circuits_executed`, `max_estimator_calls`

Note: older cached runs may not include all fields; new code should add missing fields additively.

### Step Metrics (`history.jsonl`)

Each JSONL row should include (when applicable):

- `iter` (outer iter for ADAPT, eval count for VQE)
- `energy` (and `delta_e` if exact is known)
- `chosen_op` and `max_grad` (ADAPT)
- cost counters snapshot:
  - `n_estimator_calls`, `n_circuits_executed`, `n_pauli_terms_measured`, `total_shots` (if shot-based)
  - `n_energy_evals`, `n_grad_evals`
- runtime:
  - `t_iter_s`, `t_cum_s`
- correctness diagnostics (optional but preferred for statevector runs):
  - `VarH`, `N_mean`, `Sz_mean`, `VarN`, `VarSz`, `abs_N_err`, `abs_Sz_err`
- `stop_reason` when stopping/aborting

### Artifact References by Checksum

If a run produces nontrivial artifacts (plots, serialized circuits, tables), it should also write:

- `artifacts.json` containing a list of `{relpath, sha256, bytes}` for each artifact.

Do not “trust by filename”; use checksums for comparison/caching.

## SQL Is Authoritative (Future-Proofing Rule)

Once SQL logging is enabled (e.g. a SQLite DB under `runs/`), the SQL database becomes the **source of truth**:

- ingestion, filtering, and plotting must read from SQL
- JSON (`history.jsonl`, `compare_rows.json`) becomes legacy/export-only
- any script that writes JSON must be considered an exporter from SQL, not the primary logger

Until SQL exists, JSON remains the canonical store.

## Experiment Output Conventions

- Default write location is under `runs/`.
- Prefer writing new comparisons to a new folder (e.g. `runs/compare_vqe_<tag>/`) rather than mutating an old comparison in-place.
- Keep budget caps on by default for scaling studies; avoid unbounded runs in scripts.

