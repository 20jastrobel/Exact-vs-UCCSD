# Repository Description: Hubbard Dimer VQE + IBM Runtime Tooling

This repository is a small, focused quantum-chemistry / condensed-matter sandbox for Variational Quantum Eigensolver (VQE) experiments on the 2-site Hubbard model (a Hubbard dimer). It provides:

- A clean Hubbard Hamiltonian builder (fermionic then mapped to qubits via Jordan-Wigner).
- Multiple ansatz builders (hardware efficient, clustered EfficientSU2, UCCSD).
- Local VQE (statevector) workflows for quick iteration and warm starts.
- IBM Runtime workflows for evaluation or simple search/optimization on real hardware.
- IBM-like local simulation using Aer + fake backends for noise realism.

The repository is designed to compare exact ground state energies vs VQE results and to stage parameters for IBM runs. The scripts are CLI-driven and meant for experimentation, not a full library.


## Repository layout (high level)

- `run_vqe_local.py`
  - Minimal local VQE entrypoint for the Hubbard dimer.
- `pydephasing/`
  - Python package containing quantum utilities.
  - `quantum/` holds ansatz, Hamiltonian, VQE runner, and IBM tools.
- `hubbard_params.json`, `hea_ibm_params.json`
  - Example parameter sets saved from runs.
- `dependencies/requirements.txt`
  - Full dependency list (many are not directly used by this tiny repo).


## Domain and objective

Core scientific intent:

- Construct a 2-site Hubbard Hamiltonian in second quantization.
- Map the fermionic Hamiltonian to qubits via Jordan-Wigner.
- Find ground state energy exactly (dense diagonalization, with sparse fallback).
- Approximate ground state energy using VQE with various ansatz families.
- Optionally evaluate or optimize ansatz parameters with IBM Runtime backends.

The scripts are a convenient experimental harness, not a polished product. Inputs are small; most workflows are tailored to the 4-qubit Hubbard dimer.


## Key concepts and data objects

- FermionicOp: second-quantized Hamiltonian (qiskit-nature).
- SparsePauliOp: qubit Hamiltonian (qiskit), after Jordan-Wigner mapping.
- QuantumCircuit: ansatz circuit.
- VQE: qiskit-algorithms VQE routine.
- Estimator / EstimatorV2: energy estimation primitive.


## Environment variables used (IBM runtime modes)

- `QISKIT_IBM_TOKEN`: IBM Quantum token for authentication.
- `QISKIT_IBM_INSTANCE`: runtime instance string.
- `QISKIT_IBM_REGION`: optional region override.
- `QISKIT_IBM_PLANS`: if missing, defaults to `plans_preference = ["open"]`.
- `IBM_BACKEND`: optional backend name shortcut.
- `EXACT_TOL`: optional tolerance override for simulated VQE energy mismatch.


## Entry points and typical workflows

### 1) Local VQE quick check (minimal)

- `run_vqe_local.py` is a minimal script that:
  - builds the qubit Hamiltonian for a Hubbard dimer,
  - runs a clustered EfficientSU2 ansatz,
  - runs COBYLA on statevector estimator,
  - prints exact energy, VQE energy, and timing.

### 2) Comprehensive CLI workflow

- `pydephasing/quantum/hubbard_jw_check.py` is the main script.
- It supports:
  - local VQE warm start,
  - IBM eval-only (estimate energy for fixed parameters),
  - IBM search (evaluate a jittered neighborhood of parameters),
  - IBM hardware optimization (simple iterative stochastic search),
  - IBM-like noisy simulation using fake backends.


## Pseudocode summary for all modules

Below is a detailed pseudocode rendition of nearly all logic in the repository. It is written to be read by a deep research LLM and should map closely to the real implementation.


# =============================
# run_vqe_local.py (entrypoint)
# =============================

```pseudo
function run_vqe(qubit_op, maxiter, reps, seed) -> (vqe_energy, elapsed_seconds):
    estimator = StatevectorEstimator()
    ansatz = build_clustered_ansatz(num_qubits=qubit_op.num_qubits, reps=reps)
    optimizer = COBYLA(maxiter=maxiter)
    rng = np.random.default_rng(seed)
    initial_point = rng.random(ansatz.num_parameters) * 2*pi

    vqe = VQE(estimator=estimator, ansatz=ansatz,
              optimizer=optimizer, initial_point=initial_point)

    start_timer
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    elapsed = stop_timer

    vqe_energy = real_part(result.eigenvalue)
    return (vqe_energy, elapsed)

function main():
    parse CLI args: t, u, dv, maxiter, reps, seed

    qubit_op, mapper = build_qubit_hamiltonian(t, u, dv)
    exact = exact_ground_energy(qubit_op)

    vqe_energy, elapsed = run_vqe(qubit_op, maxiter, reps, seed)

    diff = vqe_energy - exact
    print summary

if __name__ == "__main__":
    main()
```


# ===============================================
# pydephasing/quantum/ansatz/__init__.py
# ===============================================

```pseudo
ANSATZ_BUILDERS = {
    "clustered": build_clustered_ansatz,
    "efficient_su2": build_efficient_su2_ansatz,
    "uccsd": build_uccsd_ansatz,
    "hea": build_hardware_efficient_ansatz,
}

function build_ansatz(kind, num_qubits, reps, mapper,
                      hea_rotation="ry",
                      hea_entanglement="linear",
                      hea_occ=None):
    if kind not in ANSATZ_BUILDERS:
        raise ValueError

    if kind == "hea":
        return build_hardware_efficient_ansatz(
            n_qubits=num_qubits,
            reps=reps,
            entanglement=hea_entanglement,
            initial_occupations=hea_occ,
            rotation=hea_rotation,
        )

    if kind == "uccsd":
        return build_uccsd_ansatz(num_qubits=num_qubits,
                                  reps=reps,
                                  mapper=mapper)

    return ANSATZ_BUILDERS[kind](num_qubits=num_qubits, reps=reps)
```


# =======================================
# pydephasing/quantum/ansatz/clustered.py
# =======================================

```pseudo
function build_ansatz(num_qubits, reps=2):
    return EfficientSU2(
        num_qubits,
        su2_gates=["ry", "rz"],
        reps=reps,
        entanglement="linear"
    )

function build_clustered_ansatz(num_qubits, reps=2):
    return build_ansatz(num_qubits, reps)
```


# ===========================================
# pydephasing/quantum/ansatz/efficient_su2.py
# ===========================================

```pseudo
function build_ansatz(num_qubits, reps=2):
    return EfficientSU2(
        num_qubits,
        su2_gates=["ry", "rz"],
        reps=reps,
        entanglement="full"
    )

function build_efficient_su2_ansatz(num_qubits, reps=2):
    return build_ansatz(num_qubits, reps)
```


# ====================================
# pydephasing/quantum/ansatz/hea.py
# ====================================

```pseudo
function _entanglement_pairs(n_qubits, entanglement):
    if n_qubits < 2: return []
    if entanglement == "linear":
        return [(0,1), (1,2), ..., (n-2, n-1)]
    if entanglement == "circular":
        return linear pairs + (n-1, 0)
    if entanglement == "full":
        return all (i, j) with i < j
    else:
        raise ValueError

function _apply_manual_hea(qc, reps, entanglement, rotation, add_barriers):
    if n_qubits == 0: return

    if rotation == "ry":
        params = ParameterVector("theta", reps * n_qubits)
        param_idx = 0
        repeat reps times:
            for each qubit q:
                qc.ry(params[param_idx], q)
                param_idx += 1
            for (src, dst) in entanglement pairs:
                qc.cx(src, dst)
            if add_barriers: qc.barrier()
        return

    if rotation == "ryrz":
        params = ParameterVector("theta", reps * n_qubits * 2)
        param_idx = 0
        repeat reps times:
            for each qubit q:
                qc.ry(params[param_idx], q); param_idx += 1
                qc.rz(params[param_idx], q); param_idx += 1
            for (src, dst) in entanglement pairs:
                qc.cx(src, dst)
            if add_barriers: qc.barrier()
        return

    raise ValueError

function build_ansatz(n_qubits, reps=1, entanglement="linear",
                      initial_occupations=None,
                      rotation="ry", add_barriers=False):
    rotation = rotation.lower()
    if rotation not in {"ry", "ryrz"}:
        raise ValueError

    qc_init = QuantumCircuit(n_qubits)
    if initial_occupations is None:
        initial_occupations = [0, 2]

    for idx in initial_occupations:
        validate 0 <= idx < n_qubits
        qc_init.x(idx)

    try load RealAmplitudes and TwoLocal from qiskit

    if rotation == "ry" and RealAmplitudes exists:
        hea = RealAmplitudes(n_qubits, entanglement=entanglement,
                             reps=reps, insert_barriers=add_barriers)
        return qc_init.compose(hea)

    if TwoLocal exists:
        rotation_blocks = ["ry"] or ["ry", "rz"]
        hea = TwoLocal(n_qubits,
                       rotation_blocks=rotation_blocks,
                       entanglement_blocks="cx",
                       entanglement=entanglement,
                       reps=reps,
                       insert_barriers=add_barriers,
                       skip_final_rotation_layer=False)
        return qc_init.compose(hea)

    qc = copy(qc_init)
    _apply_manual_hea(qc, reps, entanglement, rotation, add_barriers)
    return qc

function build_hardware_efficient_ansatz(...):
    return build_ansatz(...)
```


# =====================================
# pydephasing/quantum/ansatz/uccsd.py
# =====================================

```pseudo
function build_ansatz(num_qubits, reps=1, mapper,
                      num_spatial_orbitals=2,
                      num_particles=(1,1),
                      initial_state=None):
    import HartreeFock, UCCSD from qiskit_nature

    if num_spatial_orbitals * 2 != num_qubits:
        raise ValueError

    if initial_state is None:
        try HartreeFock(num_spatial_orbitals, num_particles, mapper)
        except TypeError: HartreeFock(..., qubit_mapper=mapper)

    try:
        ansatz = UCCSD(num_spatial_orbitals, num_particles, mapper,
                       initial_state=initial_state,
                       reps=reps, preserve_spin=True)
    except TypeError:
        ansatz = UCCSD(..., qubit_mapper=mapper, ...)

    return ansatz

function build_uccsd_ansatz(...):
    return build_ansatz(...)
```


# ============================================
# pydephasing/quantum/hamiltonian/hubbard.py
# ============================================

```pseudo
function _spin_orbital_index(site, spin, n_sites):
    validate site in [0, n_sites)
    validate spin in {0, 1}
    return site + spin * n_sites

function _add_term(data_dict, label, coeff, atol=0.0):
    if atol and |coeff| <= atol: return
    data_dict[label] = data_dict.get(label, 0) + coeff
    if atol and |data_dict[label]| <= atol: delete entry

function _normalize_onsite_potential(v, n_sites):
    if v is None:
        return zeros(n_sites, 2)

    arr = asarray(v)
    if arr.ndim == 1:
        ensure len == n_sites
        return column_stack([arr, arr])   # same for both spins
    if arr.ndim == 2:
        ensure shape == (n_sites, 2)
        return arr

    raise ValueError

function _normalize_u(u, n_sites):
    if u is scalar:
        return array of length n_sites filled with u
    else:
        arr = asarray(u)
        ensure arr shape == (n_sites,)
        return arr

function _default_1d_chain_edges(n_sites, periodic=False):
    edges = [(0,1), (1,2), ..., (n-2, n-1)]
    if periodic and n_sites > 2:
        edges += [(n-1, 0)]
    return edges

function default_1d_chain_edges(n_sites, periodic=False):
    return _default_1d_chain_edges(n_sites, periodic)

function build_fermionic_hubbard(n_sites, t=1.0, u=4.0,
                                 edges=None, v=None, atol=0.0):
    validate n_sites >= 1
    v_mat = _normalize_onsite_potential(v, n_sites)
    u_vec = _normalize_u(u, n_sites)

    if edges is None:
        edges_list = default 1d chain, open boundary
    else:
        edges_list = list(edges)

    data = empty dict

    # onsite potential term: sum_i,sum_spin v_i,spin n_i,spin
    for i in range(n_sites):
        for spin in (0,1):
            p = _spin_orbital_index(i, spin, n_sites)
            add term "+_p -_p" with coeff v_mat[i, spin]

    # hopping terms: -t (c_i^dagger c_j + h.c.) per spin
    for edge in edges_list:
        parse edge as (i, j) or (i, j, t_ij)
        validate i != j and indices in range
        for spin in (0,1):
            p_i = _spin_orbital_index(i, spin, n_sites)
            p_j = _spin_orbital_index(j, spin, n_sites)
            add "+_p_i -_p_j" with coeff -t_ij
            add "+_p_j -_p_i" with coeff -conj(t_ij)

    # onsite repulsion: sum_i U_i n_i,up n_i,down
    for i in range(n_sites):
        p_up = _spin_orbital_index(i, 0, n_sites)
        p_dn = _spin_orbital_index(i, 1, n_sites)
        add term "+_p_up -_p_up +_p_dn -_p_dn" with coeff U_i

    return FermionicOp(data, num_spin_orbitals=2*n_sites)

function _legacy_dimer_fermionic(t, u, dv):
    return explicit 2-site (4 orbital) dictionary of terms

function build_qubit_hamiltonian_from_fermionic(ferm_op, simplify_atol=1e-12):
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(ferm_op)
    if simplify_atol is not None:
        qubit_op = qubit_op.simplify(atol=simplify_atol)
    return (qubit_op, mapper)

function build_qubit_hamiltonian(t, u, dv):
    ferm_op = build_fermionic_hubbard(
        n_sites=2, t=t, u=u,
        edges=[(0,1)],
        v=[-dv/2, dv/2]
    )
    return build_qubit_hamiltonian_from_fermionic(ferm_op)
```


# ========================================
# pydephasing/quantum/hamiltonian/exact.py
# ========================================

```pseudo
function exact_ground_energy(qubit_op):
    n = qubit_op.num_qubits if present
    dim = 2^n if n known

    if dim is None or dim <= 4096:
        mat = qubit_op.to_matrix()
        evals = eigvalsh(mat)
        return min(real(evals))

    try:
        mat = qubit_op.to_matrix(sparse=True)   # if supported
        val = eigsh(mat, k=1, which="SA")
        return real(val)
    except:
        mat = qubit_op.to_matrix()
        evals = eigvalsh(mat)
        return min(real(evals))
```


# ==============================================
# pydephasing/quantum/vqe/runner.py
# ==============================================

```pseudo
class LocalVQEResult:
    energy: float
    params: list[float]
    seconds: float

function run_vqe_with_estimator(qubit_op, ansatz, estimator,
                                restarts, maxiter, optimizer_name, seed):
    rng = np.random.default_rng(seed)
    best_energy = None
    best_params = []
    start_timer

    for restart_idx in range(restarts):
        optimizer = build_optimizer(optimizer_name, maxiter)
        initial_point = rng.random(ansatz.num_parameters) * 2*pi

        vqe = VQE(estimator=estimator, ansatz=ansatz,
                  optimizer=optimizer, initial_point=initial_point)
        result = vqe.compute_minimum_eigenvalue(qubit_op)
        energy = real(result.eigenvalue)

        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_params = list(result.optimal_point)

        print("local-vqe restart i: energy=...")

    if best_energy is None:
        raise RuntimeError

    elapsed = stop_timer
    return LocalVQEResult(best_energy, best_params, elapsed)

function run_local_vqe(qubit_op, ansatz, restarts, maxiter, optimizer_name, seed):
    estimator = StatevectorEstimator()
    return run_vqe_with_estimator(qubit_op, ansatz, estimator,
                                  restarts, maxiter, optimizer_name, seed)
```


# ==============================================
# pydephasing/quantum/vqe/optimizers/__init__.py
# ==============================================

```pseudo
_OPTIMIZERS = {
    "COBYLA": build_cobyla,
    "L_BFGS_B": build_l_bfgs_b,
    "SLSQP": build_slsqp,
}

function build_optimizer(name, maxiter):
    if name not in _OPTIMIZERS:
        raise ValueError
    return _OPTIMIZERS[name](maxiter)
```


# =============================================
# pydephasing/quantum/vqe/optimizers/cobyla.py
# =============================================

```pseudo
function build(maxiter):
    return COBYLA(maxiter=maxiter)
```

# ==============================================
# pydephasing/quantum/vqe/optimizers/l_bfgs_b.py
# ==============================================

```pseudo
function build(maxiter):
    return L_BFGS_B(maxiter=maxiter)
```

# =============================================
# pydephasing/quantum/vqe/optimizers/slsqp.py
# =============================================

```pseudo
function build(maxiter):
    return SLSQP(maxiter=maxiter)
```


# =============================================
# pydephasing/quantum/ibm_runtime_tools.py
# =============================================

```pseudo
function get_runtime_service():
    import QiskitRuntimeService
    read env QISKIT_IBM_TOKEN, QISKIT_IBM_INSTANCE
    kwargs = {
        channel: "ibm_quantum_platform",
        token: token,
        instance: instance
    }
    if QISKIT_IBM_REGION set: kwargs["region"] = region
    if QISKIT_IBM_PLANS not set: kwargs["plans_preference"] = ["open"]
    return QiskitRuntimeService(**kwargs)

function choose_backend(service, backend_name=None, no_scan=True, force_hardware=False):
    name = backend_name or env IBM_BACKEND
    if name: return service.backend(name)

    if no_scan:
        raise RuntimeError("No backend specified... avoid scans")

    sims = service.backends(simulator=True, operational=True)
    if sims:
        return "ibmq_qasm_simulator" if present else first simulator

    if not force_hardware:
        raise RuntimeError("No simulator available...")

    return service.least_busy(operational=True, simulator=False, min_num_qubits=4)

contextmanager runtime_context(service, backend, prefer_batch=True, prefer_session=False):
    try import Batch, Session
    if prefer_session:
        try to open Session(service, backend) and yield
    if prefer_batch:
        try to open Batch(backend) and yield
    yield backend as fallback

function make_estimator(mode_obj, shots, resilience, max_exec):
    estimator = Estimator(mode=mode_obj)
    estimator.options.default_shots = shots
    estimator.options.resilience_level = resilience
    estimator.options.max_execution_time = max_exec
    return estimator
```


# =============================================
# pydephasing/quantum/ibm_sim_tools.py
# =============================================

```pseudo
function _load_fake_backend_from(module, name):
    try: return module.<name>()
    except AttributeError: return None

function load_fake_backend(name):
    try import qiskit_ibm_runtime.fake_provider as runtime_fake_provider
    if runtime_fake_provider exists:
        backend = _load_fake_backend_from(runtime_fake_provider, name)
    if backend is None:
        import qiskit.providers.fake_provider as terra_fake_provider
        backend = _load_fake_backend_from(terra_fake_provider, name)
    if backend is None:
        raise ValueError
    return backend

function _backend_basis_gates(backend):
    try: return backend.configuration().basis_gates
    except: return backend.basis_gates or None

function build_aer_estimator_for_backend(backend, shots=4096, seed=None,
                                         noisy=True, coupling_map=None,
                                         basis_gates=None):
    noise_model = NoiseModel.from_backend(backend) if noisy else None
    backend_options = {}
    run_options = {}

    if noise_model:
        backend_options["noise_model"] = noise_model
    if basis_gates is None and noise_model:
        basis_gates = noise_model.basis_gates
    if basis_gates is None:
        basis_gates = _backend_basis_gates(backend)
    if basis_gates is not None:
        backend_options["basis_gates"] = basis_gates

    if coupling_map is None:
        coupling_map = backend.coupling_map if exists
    if coupling_map is not None:
        backend_options["coupling_map"] = coupling_map

    if seed is not None:
        run_options["seed_simulator"] = seed
    if shots > 0:
        run_options["shots"] = shots

    options = {"backend_options": backend_options, "run_options": run_options}
    estimator = EstimatorV2(options=options)
    return (estimator, None)
```


# =====================================================
# pydephasing/quantum/hubbard_jw_check.py (MAIN SCRIPT)
# =====================================================

This is the main, comprehensive command line driver. It wires together Hamiltonian construction, ansatz selection, local warm starts, IBM eval/search/opt flows, and fake-backend simulation.

Below is the main pseudocode script in detail.

```pseudo
function _str_to_bool(value):
    normalize to lower-case
    if in {"1","true","yes","y"} -> True
    if in {"0","false","no","n"} -> False
    else raise argparse error

function _parse_occ_list(value):
    if None or empty -> []
    split by comma, trim, parse ints

function _normalize_ansatz(name):
    if name in {"hea","hardware","hardware-efficient"}: return "hea"
    else return name

function _default_reps(ansatz_name, requested):
    if requested is not None: return requested
    if ansatz_name in {"clustered","efficient_su2"}: return 2
    return 1

function save_params(path, params, meta):
    payload = {"theta": params, ...meta}
    write JSON to path

function load_params(path) -> list[float]:
    read JSON
    if "theta" in data: return data["theta"]
    if "params" in data: return data["params"]
    else raise KeyError

function _params_path_for_ansatz(path, ansatz_name):
    if path has extension: insert "_<ansatz>" before extension
    else: return path + "_<ansatz>"

function compile_ansatz_and_op(ansatz, qubit_op, backend, opt_level=1):
    pm = generate_preset_pass_manager(backend, optimization_level=opt_level)
    isa_circ = pm.run(ansatz)
    isa_op = qubit_op.apply_layout(isa_circ.layout)
    return (isa_circ, isa_op)

function compile_ansatz_and_op_for_sim(ansatz, qubit_op, backend, opt_level=1):
    coupling_map = backend.coupling_map if exists
    reduce coupling map to used qubits if possible
    basis_gates = backend.configuration().basis_gates or backend.basis_gates

    isa_circ = transpile(ansatz,
                         coupling_map=coupling_map,
                         basis_gates=basis_gates,
                         optimization_level=opt_level)
    isa_op = qubit_op.apply_layout(isa_circ.layout)
    return (isa_circ, isa_op, coupling_map, basis_gates)

function ibm_estimate_energies(qubit_op, ansatz, theta_list, backend,
                               shots, resilience, max_exec, opt_level=1):
    isa_circ, isa_op = compile_ansatz_and_op(ansatz, qubit_op, backend, opt_level)
    estimator = make_estimator(backend, shots, resilience, max_exec)
    job = estimator.run([(isa_circ, isa_op, theta_list)])
    result = job.result(timeout=max_exec + 30)
    evs = list(result[0].data.evs)
    return (evs, job.job_id())

function ibm_optimize_on_hardware(qubit_op, ansatz, backend, service,
                                  rounds, k, sigma, shrink, seed,
                                  shots_opt, shots_final,
                                  resilience_opt, resilience_final,
                                  max_exec_opt, max_exec_final,
                                  opt_level=1):
    validate rounds >= 1 and k >= 1

    isa_circ, isa_op = compile_ansatz_and_op(ansatz, qubit_op, backend, opt_level)
    num_params = isa_circ.num_parameters
    if num_params == 0: raise

    rng = random(seed)
    center = zeros(num_params)
    job_ids = []
    best_energy = None

    with runtime_context(service, backend, prefer_batch=True) as mode:
        for round_idx in range(rounds):
            candidates = [center]
            for _ in range(k-1):
                candidates.append(center + sigma * normal_noise)
            theta_list = list(map(float, each candidate))

            estimator = make_estimator(mode, shots_opt, resilience_opt, max_exec_opt)
            job = estimator.run([(isa_circ, isa_op, theta_list)])
            job_ids.append(job.job_id())

            result = job.result(timeout=max_exec_opt + 30)
            evs = array(result[0].data.evs)
            idx = argmin(evs)
            center = candidates[idx]
            best_energy = evs[idx]
            print progress
            sigma *= shrink

        final_theta = [center]
        estimator = make_estimator(mode, shots_final, resilience_final, max_exec_final)
        job = estimator.run([(isa_circ, isa_op, final_theta)])
        job_ids.append(job.job_id())
        result = job.result(timeout=max_exec_final + 30)
        final_energy = result[0].data.evs[0]

    return (final_energy, best_energy, list(center), job_ids)

function _run_self_test():
    build 2-site fermionic hubbard using builder
    compare to legacy explicit form (ferm2.equiv)
    compute exact ground energy and print

    for n_sites in 3..6:
        build fermionic hubbard for 1d chain
        map to qubits, compute exact energy, print

    for dimer example:
        build qubit hamiltonian, exact energy
        build clustered ansatz
        run_local_vqe with short settings
        print results

function _run_vqe_on_ibm_sim(qubit_op, ansatz, fake_backend_name,
                             shots, seed_sim, noisy,
                             restarts, maxiter, optimizer_name,
                             seed, exact):
    backend = load_fake_backend(fake_backend_name)
    isa_circ, isa_op, coupling_map, basis_gates = compile_ansatz_and_op_for_sim(...)
    estimator = build_aer_estimator_for_backend(backend,
                                                shots=shots, seed=seed_sim,
                                                noisy=noisy,
                                                coupling_map=coupling_map,
                                                basis_gates=basis_gates)

    run_vqe_with_estimator(isa_op, isa_circ, estimator, ...)
    delta = vqe_energy - exact
    print results

    tol = EXACT_TOL env or default (0.5 if noisy else 0.05)
    if |delta| > tol:
        if EXACT_TOL set: raise RuntimeError
        else: print warning

function _build_theta_list(theta_best, k, sigma, seed):
    validate k >= 1
    thetas = [theta_best]
    for i in range(k-1):
        jitter = sigma * normal_noise
        candidate = theta_best + jitter
        wrapped = (candidate + pi) mod (2*pi) - pi
        thetas.append(wrapped)
    return list of float lists

function _print_queue_warning(backend):
    status = backend.status()
    if pending_jobs > 100: print warning

function main():
    parse CLI args:
        model: t, u, dv
        ansatz selection: ansatz, reps, hea settings, etc.
        local vqe: restarts, maxiter, optimizer, seed, save/load params
        ibm modes: ibm_eval, ibm_search, ibm_opt, ibm_sim
        sim settings: fake backend, noise
        ibm run settings: shots/resilience/max exec, etc.
        backend selection: backend, no_backend_scan, force_hardware

    if self_test: _run_self_test(); return

    validate incompatible args:
        ibm_sim cannot combine with other IBM modes or local_vqe
        ibm_opt cannot combine with ibm_eval or ibm_search or local_vqe

    ibm_mode = ibm_sim or ibm_opt or ibm_eval or ibm_search

    # choose default ansatz if user didn't specify
    if ansatz not set:
        if ibm_opt: ansatz = "hea"
        elif ibm_sim: ansatz = "uccsd"
        else: ansatz = "clustered"
    ansatz = normalize ansatz

    qubit_op, mapper = build_qubit_hamiltonian(t, u, dv)
    exact = exact_ground_energy(qubit_op)
    hea_occ = parse occ list

    if ansatz_choice in {"both", "all"} and not ibm_mode:
        multi_kinds = ["clustered", "uccsd"] (+ "efficient_su2" if all)
        for each kind in multi_kinds:
            reps = _default_reps(kind, args.reps)
            ansatz = build_ansatz(kind, num_qubits, reps, mapper, hea params)
            local_result = run_local_vqe(...)
            print results
            if save_params: save to kind-specific path
        return

    # otherwise build single ansatz
    reps = _default_reps(ansatz_choice, args.reps)
    try:
        ansatz = build_ansatz(ansatz_choice, num_qubits, reps, mapper, hea params)
    except Exception:
        if ibm_sim and ansatz_choice == "uccsd":
            fallback to clustered
        else: re-raise

    if ibm_opt and save_params_path not given:
        save_params_path = "hea_ibm_params.json"

    if ibm_sim:
        _run_vqe_on_ibm_sim(...)
        return

    theta_best = load_params if args.load_params else None

    needs_warm_start = local_vqe requested
        OR (ibm_eval or ibm_search) and theta_best is None
    if needs_warm_start:
        local_result = run_local_vqe(...)
        theta_best = local_result.params
        print local-vqe results
        if save_params: save

    if ibm_opt:
        require backend name if no_backend_scan
        warn if ansatz not hea
        service = get_runtime_service()
        backend = choose_backend(...)
        print active instance, backend name, queue warning

        final_energy, best_energy, theta_best, job_ids = ibm_optimize_on_hardware(...)
        print job IDs, energies, delta
        if save_params_path: save with metadata
        return

    if ibm_eval or ibm_search:
        require backend name if no_backend_scan
        ensure theta_best exists

        theta_list = _build_theta_list(theta_best,
                                       k = ibm_k if search else 1,
                                       sigma = ibm_sigma,
                                       seed = seed)

        service = get_runtime_service()
        backend = choose_backend(...)
        print active instance, backend name, queue warning

        evs, job_id = ibm_estimate_energies(...)

        if ibm_eval:
            energy = evs[0]
            print job id, energy, exact, delta
        else:
            idx = argmin(evs)
            energy = evs[idx]
            print job id, min energy, best index, exact, delta

if __name__ == "__main__":
    main()
```


# ======================================
# JSON parameter artifacts (examples)
# ======================================

`hubbard_params.json` and `hea_ibm_params.json` store ansatz parameters and metadata.

Pseudocode representation of their structure:

```pseudo
{
  "theta": [ list of floats ],
  "t": float,
  "u": float,
  "dv": float,
  "ansatz": "clustered" | "hea" | ...,
  "reps": int,
  # optional IBM metadata:
  "backend": string,
  "E_final": float,
  "E_opt": float,
  "shots_final": int,
  "resilience_final": int
}
```


# ==================================
# Dependencies (from requirements)
# ==================================

Key dependencies actually used by the code in this repo:

- `qiskit` (core Terra + algorithms),
- `qiskit-aer` (noise + Aer estimators),
- `qiskit-nature` (FermionicOp, UCCSD, HartreeFock),
- `numpy`, `scipy`.

The requirements file lists more libraries than the code currently uses.


# =============================
# Suggested mental model
# =============================

A simple way to think about the repository:

1) Build a Hubbard model in second quantization.
2) Map to qubits using Jordan-Wigner.
3) Select a parametric ansatz circuit.
4) Use VQE to minimize energy using local statevector or IBM runtime estimators.
5) Compare to exact diagonalization for validation.

The main script (`hubbard_jw_check.py`) is the orchestrator for all of this.
It is also the most important file to read to understand how the project fits together.


# =============================
# Notes for deep research LLMs
# =============================

- The code is short but highly modular; it is intended to let you swap ansatz,
  VQE optimizer, and backend mode without rewriting logic.
- The repo is not a general-purpose simulation engine; it is tailored to
  small Hubbard systems (especially the 2-site dimer).
- IBM runtime flows assume a valid account and a known backend name; the
  CLI avoids backend scanning by default to reduce API calls.
- The optimization on hardware is a simple stochastic search (not gradient-based).
- The local statevector VQE is used both for a quick check and as a warm start
  to set parameters for IBM runs.

