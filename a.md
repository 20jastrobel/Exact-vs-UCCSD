# Operator Pool, Selection, and Implementation (ADAPT/Meta-ADAPT-VQE)

This note is intended to be fed into a Deep Research LLM. It documents how this repository defines operator pools, filters them, evaluates selection gradients, and appends selected operators to form an ADAPT ansatz.

## Key Files

- `pydephasing/quantum/vqe/adapt_vqe_meta.py`: operator pool construction + ADAPT outer loop + gradient evaluation.
- `pydephasing/quantum/symmetry.py`: symmetry operators (`N`, `S_z`) and commutation check used for sector filtering (`enforce_sector`).
- `run_adapt_meta.py`: CLI defaults controlling pool choice and whether sector filtering is enabled.

## Verbatim Answer (From Chat)

Other than “pick the max |gradient|”, the main restrictions come from how the **UCCSD pool is defined and filtered**:

- **We only consider Qiskit Nature’s UCCSD generator set with conservative defaults**: singles+doubles only (UCCSD), `generalized=False`, `preserve_spin=True`, `include_imaginary=False`, `reps=1` when building the pool (`build_uccsd_excitation_pool`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:150`).
- **(Usually) symmetry-sector filtering**: if `enforce_sector=True`, we drop any candidate whose grouped Pauli operator does *not* commute with the mapped particle-number and `S_z` symmetry ops (`pydephasing/quantum/vqe/adapt_vqe_meta.py:625`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:656`). In the CLI, this is **on by default** (`run_adapt_meta.py:180`).
- **Pool cleanup**: we discard identity terms, discard Pauli components with nontrivial imaginary coefficients / tiny weights, and deduplicate operators with identical Pauli decompositions (`pydephasing/quantum/vqe/adapt_vqe_meta.py:204`).

And during selection:
- **No repeats by default**: once an operator is chosen it’s removed from the pool unless `allow_repeats=True` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:807`).
- **Early stop conditions**: the loop stops if `max|g| < eps_grad` or if the outer-step energy change drops below `eps_energy` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:744`).

## Operator Pool Representations

The code supports two pool representations inside `run_meta_adapt_vqe` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:579`):

- **Ungrouped pool** (`pool_labels: list[str]`): each candidate operator is a single Pauli string label (e.g. `"IXYZ..."`).
- **Grouped pool** (`pool_specs: list[dict]`): each candidate is a dict:
  - `{"name": str, "paulis": list[tuple[label: str, weight: float]]}`
  - This represents a *generator* that is a weighted sum of Pauli strings; ADAPT treats it as one parameter `theta_k` but implements it as multiple Pauli evolution gates weighted by `weight`.

Grouped pool modes are controlled by:

- `GROUPED_POOL_MODES = {"cse_density_ops", "uccsd_excitations"}` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:22`).

## Pool Construction Modes

Pool construction happens at the start of `run_meta_adapt_vqe`:

### 1) Hamiltonian-derived pools (ungrouped)

`build_operator_pool_from_hamiltonian` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:33`):

- `mode="ham_terms"`: pool is exactly the non-identity Pauli labels present in the Hamiltonian.
- `mode="ham_terms_plus_imag_partners"` (default): starts from `ham_terms`, then:
  - adds "imaginary partners" for each Pauli term by swapping `X <-> Y` per qubit (`_imaginary_partners`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:336`).
  - adds single-qubit `X` and `Y` terms on each qubit to avoid trivial stagnation (`pydephasing/quantum/vqe/adapt_vqe_meta.py:61`).

### 2) CSE density operator pool (grouped)

`build_cse_density_pool_from_fermionic` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:73`):

- Iterates fermionic terms and selects density-like patterns (e.g. `+_p -_q`), optionally including diagonals.
- For each eligible fermionic term `gamma`, it can include **both Hermitian quadratures**:
  - `gamma_im(...)`: `A = i(gamma - gamma^\dagger)` (imag/current-like quadrature)
  - `gamma_re(...)`: `B = gamma + gamma^\dagger` (real/hopping-like quadrature)
  These are controlled by flags `include_antihermitian_part` and `include_hermitian_part` (both default True).
- Each quadrature is mapped to qubits via the provided mapper (`JordanWignerMapper`) and converted into a grouped Pauli spec.
- Filters out identity terms, coerces numerically-real coefficients to real weights after Hermitian canonicalization, drops near-zero weights, then deduplicates by a rounded Pauli+weight signature (dedup spans both quadratures).
- Optional fermionic-sector prefiltering (`enforce_symmetry`/`_fermionic_term_in_sector`) rejects terms that change `N` or `S_z` before mapping.

Note on diagonal terms:

- “Diagonal” here covers both `+_p -_p` (number operators) and 4-fermion diagonal densities like `+_{p_up} -_{p_up} +_{p_dn} -_{p_dn}` (Hubbard on-site interaction).
- In `run_meta_adapt_vqe`, diagonal inclusion is controlled by `cse_include_diagonal` (now default True) and is exposed in the CLI as `--cse-include-diagonal/--no-cse-include-diagonal`.

### 3) UCCSD excitation generator pool (grouped)

`build_uccsd_excitation_pool` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:150`):

- Instantiates Qiskit Nature’s built-in `HartreeFock` and `UCCSD`, then reads:
  - `ansatz.operators` (SparsePauliOp excitation generators)
  - `ansatz.excitation_list` (metadata for naming)
- Default restrictions (because `run_meta_adapt_vqe` calls this function with defaults):
  - `reps=1`
  - `preserve_spin=True`
  - `generalized=False`
  - `include_imaginary=False`
- `run_meta_adapt_vqe` now plumbs UCCSD pool knobs through as `uccsd_reps`, `uccsd_include_imaginary`, `uccsd_generalized`, `uccsd_preserve_spin` (see call-site logic in `pydephasing/quantum/vqe/adapt_vqe_meta.py:579`).
- Each operator is converted to a grouped spec by:
  - canonicalizing to a Hermitian operator (or rotating a dominant anti-Hermitian part by `-i`),
  - removing the identity component,
  - coercing numerically-real coefficients to real weights (and raising on truly complex residuals),
  - dropping tiny magnitudes (`abs(real) > dedup_tol`),
  - sorting labels and deduplicating by a rounded signature.

Note: `build_uccsd_excitation_pool` exposes knobs (`preserve_spin`, `generalized`, `include_imaginary`, `reps`) but the current ADAPT entrypoint does not pass non-default values, so the pool is effectively fixed unless code is changed.

## Sector/Symmetry Filtering (`enforce_sector`)

When `enforce_sector=True`, `run_meta_adapt_vqe` filters the pool by commutation with the mapped particle-number and `S_z` operators:

- `n_q, sz_q = map_symmetry_ops_to_qubits(mapper, n_sites)` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:625`), implemented in `pydephasing/quantum/symmetry.py:24`.
- For **grouped pools**, each spec is converted into a `SparsePauliOp` via `SparsePauliOp.from_list(spec["paulis"])` and kept only if it commutes with both `n_q` and `sz_q` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:656`).
- For **ungrouped pools**, each Pauli label is wrapped as `SparsePauliOp.from_list([(label, 1.0)])` and filtered similarly (`pydephasing/quantum/vqe/adapt_vqe_meta.py:671`).

Commutation is checked by building the commutator `(op @ sym - sym @ op)` and verifying all coefficients are below tolerance (`commutes`, `pydephasing/quantum/symmetry.py:77`).

CLI default: `run_adapt_meta.py` enables sector filtering by default (`--enforce-sector` default True; see `run_adapt_meta.py:180`).

## Selection Gradient Definition (Probe-Based)

Operator selection is gradient-based at each ADAPT outer iteration (`pydephasing/quantum/vqe/adapt_vqe_meta.py:715`):

- The current circuit is `reference_state` composed with all previously selected operators (`build_adapt_circuit` / `build_adapt_circuit_grouped`).
- For each candidate operator, the code estimates a “selection gradient” by appending a *probe* Pauli evolution gate and measuring an energy difference.

Probe gate construction (`_append_probe_gate`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:322`):

- Appends `PauliEvolutionGate(Pauli(label), time=angle/2)` to the circuit.

Probe convention (`_probe_shift_and_coeff`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:355`):

- `half_angle`: `shift = pi/2`, `coeff = 0.5` (default in `run_meta_adapt_vqe`).
- `full_angle`: `shift = pi/4`, `coeff = 1.0`.

Ungrouped gradient estimate (`pydephasing/quantum/vqe/adapt_vqe_meta.py:745`):

- For candidate Pauli label `P`:
  - `g(P) = coeff * (E(theta; +shift probe P) - E(theta; -shift probe P))`
- The code scans all labels, stores `pool_gradients`, and tracks the maximum absolute value.

Grouped gradient estimate (`_compute_grouped_pool_gradients`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:363`):

- Each grouped generator is a weighted sum `G_k = sum_j w_{k,j} P_{k,j}`.
- The code:
  - builds a set of all Pauli labels across the entire grouped pool,
  - computes `g(P)` for each unique label once (cached),
  - computes `g(G_k) = sum_j w_{k,j} * g(P_{k,j})`.

Selection rule (`pydephasing/quantum/vqe/adapt_vqe_meta.py:743`):

- Choose `max_idx = argmax_k |g_k|`.

## Operator Acceptance, Removal, and Stop Conditions

After computing gradients, `run_meta_adapt_vqe` can stop before selecting anything further:

- Stop if pool exhausted (`pydephasing/quantum/vqe/adapt_vqe_meta.py:718`).
- Stop if outer-step energy change is small: `|E_current - E_prev| < eps_energy` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:760`).
- Stop if `max|g| < eps_grad` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:782`).
- Stop after `max_depth` selections (outer loop bound).

If continuing, it appends the chosen operator:

- Grouped: `ops.append(chosen_spec)`; ungrouped: `ops.append(chosen_op_label)` (`pydephasing/quantum/vqe/adapt_vqe_meta.py:803`).
- If `allow_repeats` is False (default), it removes the chosen operator from the pool (`pool_specs.pop(...)` / `pool_labels.pop(...)`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:807`).

## Sector Leakage Diagnostics (N, Sz Moments)

After each successful outer iteration (i.e., after adding an operator and re-optimizing parameters), `run_meta_adapt_vqe` computes and stores sector diagnostics when `mapper`, `n_sites`, `n_up`, and `n_down` are available:

- `N_mean = <N>`, `Sz_mean = <Sz>`
- variances: `VarN = <N^2> - <N>^2`, `VarSz = <Sz^2> - <Sz>^2`
- absolute errors vs the requested target sector: `abs_N_err`, `abs_Sz_err`

Implementation:

- `estimate_expectation(...)` and `compute_sector_diagnostics(...)` live in `pydephasing/quantum/vqe/adapt_vqe_meta.py`.
- Values are stored under `result.diagnostics["outer"][k]["sector"]` and also written into `history.jsonl` via `logger.log_point(..., extra={...})` (so they appear alongside energy/max_grad per iteration).

## How Selected Operators Become the ADAPT Ansatz Circuit

The ADAPT ansatz is constructed by repeated appending of Pauli evolutions:

- Ungrouped operators (`build_adapt_circuit`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:256`):
  - each selected Pauli label becomes `PauliEvolutionGate(Pauli(label), time=theta_k/2)`.
- Grouped operators (`build_adapt_circuit_grouped`, `pydephasing/quantum/vqe/adapt_vqe_meta.py:274`):
  - each selected spec becomes multiple gates:
    - for each component `(label, weight)`:
      - `PauliEvolutionGate(Pauli(label), time=theta_k * weight / 2)`.

This is the implementation reason the grouped pool is a restriction: a single UCCSD excitation generator (from Qiskit Nature) becomes a grouped spec; ADAPT chooses a subset of these generators and assigns one parameter per chosen generator.

## Minimal Pseudocode (Outer Loop)

Conceptually, `run_meta_adapt_vqe` does:

```text
pool = build_pool(pool_mode)
if enforce_sector: pool = [op in pool if commutes(op, N) and commutes(op, Sz)]

ops = []
theta = []
E_current = E(reference_state, theta=[])

for iter in range(max_depth):
  circuit = build_circuit(reference_state, ops)
  gradients = selection_gradients(circuit, theta, pool)  # probe-based
  if stop_condition(E_current, E_prev, max(|gradients|), pool): break

  k = argmax(|gradients|)
  ops.append(pool[k])
  if not allow_repeats: pool.pop(k)

  theta.append(0.0)
  theta = inner_optimize(theta)  # re-optimizes parameters after adding op
  E_prev = E_current
  E_current = E(circuit(theta), theta)
```

## Latest Implementation Status (Operator-Pool Work)

This section summarizes the current state of the operator pools and what changed most recently.

### What Changed (Code-Level)

- **CSE pool now includes both quadratures by default**:
  - `gamma_im(...) = i(gamma - gamma^\dagger)` and `gamma_re(...) = gamma + gamma^\dagger`
  - Controlled by `include_antihermitian_part` / `include_hermitian_part` in `build_cse_density_pool_from_fermionic`.
- **CSE diagonal fermionic terms are now included by default in ADAPT**:
  - `run_meta_adapt_vqe(..., cse_include_diagonal=True)` default is now `True`.
  - CLI toggle: `run_adapt_meta.py` supports `--cse-include-diagonal/--no-cse-include-diagonal` (default include).
  - This enables number operators `+_p -_p` and 4-fermion on-site density terms like `+_{p_up} -_{p_up} +_{p_dn} -_{p_dn}` to appear in the CSE pool (as `gamma_re(...)` specs).
- **UCCSD pool knobs are now wired end-to-end**:
  - `uccsd_reps`, `uccsd_include_imaginary`, `uccsd_generalized`, `uccsd_preserve_spin`
  - CLI flags in `run_adapt_meta.py`: `--uccsd-reps`, `--uccsd-include-imaginary`, `--uccsd-generalized`, `--uccsd-preserve-spin` (and `--no-*` forms).
- **No silent dropping of complex Pauli coefficients** in grouped pools:
  - Operators are canonicalized to Hermitian (or rotated from anti-Hermitian by `-i`) before extracting real weights.
  - If coefficients remain non-real beyond tolerance after canonicalization, the code raises a clear error.
- **Sector leakage diagnostics are computed and logged**:
  - After each outer step, if `mapper/n_sites/n_up/n_down` are provided, we compute and store:
    - `N_mean`, `Sz_mean`, `VarN`, `VarSz`, `abs_N_err`, `abs_Sz_err`
  - These are stored in `result.diagnostics["outer"][k]["sector"]` and are also written into `history.jsonl` via `logger.log_point(..., extra={...})`.

### Empirical Result: Enriching the CSE Pool Did Not Change ADAPT Selections (L=2/3/4/5, Sector (1,1))

We reran ADAPT with `pool_mode="cse_density_ops"` and diagonal+quadrature inclusion enabled. The pool size increased substantially, but the max-|grad| selection rule continued to pick the same current-like/hopping-like `gamma_im(...)` operators as before, yielding the same energies (to the shown precision).

Reason (observed): for the tested reference states and early iterations, the newly added `gamma_re(...)` (including diagonal density terms) had **zero probe gradient**, so they were never selected under `argmax |g|`.

Runs (t=1, U=4, dv=0.5, `max_depth=6`, `inner_optimizer="hybrid"`, `inner_steps=25`, sector `n_up=1, n_down=1`):

| L | Old Pool Size | New Pool Size (diag+quadratures) | E_final (new) | E_exact_sector | abs(E-E_exact) | Run Dir |
|---|--------------:|----------------------------------:|--------------:|---------------:|---------------:|--------|
| 2 | 4             | 12                                | -0.50628430   | -0.83605712    | 0.3298         | `runs/L2_cse_diag_L2_Nup1_Ndown1` |
| 3 | 8             | 21                                | -1.64939950   | -2.00000000    | 0.3506         | `runs/L3_cse_diag_L3_Nup1_Ndown1` |
| 4 | 12            | 30                                | -2.33739114   | -2.62494227    | 0.2876         | `runs/L4_cse_diag_L4_Nup1_Ndown1` |
| 5 | 16            | 39                                | -2.71730138   | -2.99518176    | 0.2779         | `runs/L5_cse_diag_L5_Nup1_Ndown1` |

Notes:

- Old pool sizes above come from the corresponding `history.jsonl` `iter=0` rows in the pre-change runs (`runs/L2_run_L2_Nup1_Ndown1`, `runs/L3_run_L3_Nup1_Ndown1`, `runs/L4_run_L4_Nup1_Ndown1`, `runs/L5_long_L5_Nup1_Ndown1`).
- New runs confirm diagonal terms are present in the pool (e.g. `gamma_re(+_0 -_0 +_2 -_2)` for L=2), but were not selected in these runs.
- The new `history.jsonl` rows include N/Sz moments; in these CSE runs, `VarN` and `VarSz` were ~0, indicating essentially no sector leakage for the selected operators (within numerical tolerance).

### Current Project State (Operator-Pool Perspective)

- The code now supports richer pools (CSE: diagonal + both quadratures; UCCSD: configurable “include imaginary/generalized/spin-preserving/reps”).
- The selection mechanism is still “probe-based max-|gradient|”. With the current probe definition and reference states, enriched CSE pools can be effectively inert if the added generators have near-zero probe gradients.
- Sector diagnostics are in place to quantify whether grouped-operator implementations leak out of (N, Sz) sectors (important for UCCSD-in-ADAPT due to Trotterization over Pauli components).
- Optional statevector-only “probability leakage” metrics (e.g., `p_leak = 1 - p_sector`) are not implemented yet; only moment-based leakage is currently available.
