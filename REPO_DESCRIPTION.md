# Repository Description: Hubbard VQE + ADAPT (CSE & UCCSD)

This repository is a focused sandbox for VQE and ADAPT-VQE experiments on the spinful Hubbard model, centered on the 2-site dimer but extensible to small chains. The code targets two operator families:

- **CSE density-operator pool** for ADAPT/Meta-ADAPT.
- **UCCSD excitation pool** for ADAPT/Meta-ADAPT and a standard UCCSD VQE baseline.

The goal is to compare:

- Conventional VQE with **UCCSD** versus **VQE using the fixed operator list from ADAPT-CSE**.
- Meta-ADAPT-VQE with **CSE** versus **UCCSD** operator pools.

The repo includes cached runs and plotting utilities to visualize these comparisons.


## Repository layout (high level)

- `run_adapt_meta.py`
  - Meta-ADAPT-VQE entrypoint with pool selection (`cse_density_ops` or `uccsd_excitations`).
- `run_vqe_local.py`
  - Minimal local UCCSD-VQE entrypoint for the Hubbard dimer.
- `pydephasing/`
  - Python package containing Hamiltonian builders, ADAPT-VQE, VQE runners, and IBM runtime utilities.
- `scripts/compare_vqe_hist.py`
  - Comparison driver that produces cached results and plots:
    - ADAPT-CSE vs ADAPT-UCCSD
    - VQE(UCCSD) vs VQE(CSE-operator list)
- `runs/`
  - Cached ADAPT runs and VQE logs.
- `scripts/plot_adapt_hist3d_blocks.py`
  - 3D block visualizations using cached runs/compare rows.


## Core concepts

- **FermionicOp**: second-quantized Hubbard Hamiltonian (qiskit-nature).
- **SparsePauliOp**: qubit Hamiltonian after Jordan-Wigner mapping.
- **Operator pools**:
  - CSE density operators (grouped Pauli generators).
  - UCCSD excitation generators mapped to grouped Pauli operators.
- **Meta-ADAPT-VQE**: adaptive selection of operators with an LSTM meta-optimizer or hybrid warm-start.


## Typical workflows

### 1) Meta-ADAPT-VQE (CSE or UCCSD pools)

- Use `run_adapt_meta.py`:
  - `--pool cse_density_ops` for CSE operators.
  - `--pool uccsd_excitations` for UCCSD excitation operators.
  - `--inner-optimizer hybrid` for the hybrid meta/L-BFGS workflow.

### 2) Comparison runs and plots

- `scripts/compare_vqe_hist.py`:
  - Produces/updates `runs/compare_vqe/compare_rows.json`.
  - Creates regular VQE logs under `runs/compare_vqe/logs_*`.
  - Uses cached ADAPT runs when available.

- `scripts/plot_adapt_hist3d_blocks.py`:
  - Reads cached ADAPT-CSE history + compare rows to generate 3D plots.

### 3) Local UCCSD VQE quick check

- `run_vqe_local.py`:
  - Builds the Hubbard dimer Hamiltonian.
  - Runs UCCSD VQE with COBYLA on a statevector estimator.


## IBM Runtime integration

- `pydephasing/quantum/hubbard_jw_check.py` supports local VQE warm-start and IBM runtime evaluation modes with the UCCSD ansatz.

Environment variables used in IBM modes:

- `QISKIT_IBM_TOKEN`
- `QISKIT_IBM_INSTANCE`
- `QISKIT_IBM_REGION`
- `QISKIT_IBM_PLANS`
- `IBM_BACKEND`
- `EXACT_TOL`


## Cached runs

- `runs/` holds cached ADAPT and VQE results for repeatable comparisons.
- `runs/compare_vqe/compare_rows.json` is the main comparison table used by plotting scripts.
