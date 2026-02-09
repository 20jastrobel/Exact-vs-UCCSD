# Re_copy (Hubbard VQE/ADAPT Benchmarks)

This repo benchmarks multiple VQE/ADAPT variants (UCCSD, ADAPT-CSE, ADAPT-UCCSD, etc.) on small 1D spinful Hubbard systems, with cost instrumentation and run logging.

## Quickstart (SQLite Run DB)

```bash
# 1) Install deps (minimal)
python -m pip install -U pip
python -m pip install numpy scipy matplotlib pytest qiskit qiskit-aer qiskit-nature qiskit-algorithms

# 2) Initialize the run DB (default path is data/runs.db)
python scripts/run_store_db.py db init --db data/runs.db

# 3) Run an experiment (defaults: --store sqlite, --db data/runs.db)
python run_adapt_meta.py --sites 2 --pool cse_density_ops
python scripts/compare_vqe_hist.py --sites 2 3 4 --allow-exact-compute

# 4) Query runs / metrics
python scripts/run_store_db.py db list --db data/runs.db
python scripts/run_store_db.py db show --db data/runs.db --run-id <RUN_ID>
python scripts/run_store_db.py db steps --db data/runs.db --run-id <RUN_ID> --metric energy
```

Notes:
- Legacy JSON runs are written under `runs/` for compatibility; switch via `--store json` (or env `RUN_STORE=json`).
- Override DB path via `--db PATH` (or env `RUNS_DB_PATH=/path/to/runs.db`).

