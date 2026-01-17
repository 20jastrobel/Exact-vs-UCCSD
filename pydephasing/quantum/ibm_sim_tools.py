"""Local IBM-like simulation tools (Aer + Fake backends)."""

from __future__ import annotations

from typing import Optional

from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2


def _load_fake_backend_from(module, name: str):
    try:
        return getattr(module, name)()
    except AttributeError:
        return None


def load_fake_backend(name: str):
    """Load a fake backend class by name from IBM runtime or Terra fake provider."""
    try:
        from qiskit_ibm_runtime import fake_provider as runtime_fake_provider
    except Exception:
        runtime_fake_provider = None

    backend = None
    if runtime_fake_provider is not None:
        backend = _load_fake_backend_from(runtime_fake_provider, name)

    if backend is None:
        from qiskit.providers import fake_provider as terra_fake_provider

        backend = _load_fake_backend_from(terra_fake_provider, name)

    if backend is None:
        raise ValueError(f"Unknown fake backend: {name}")

    return backend


def _backend_basis_gates(backend) -> Optional[list[str]]:
    try:
        cfg = backend.configuration()
        return list(cfg.basis_gates) if cfg is not None else None
    except Exception:
        return getattr(backend, "basis_gates", None)


def build_aer_estimator_for_backend(
    backend,
    *,
    shots: int = 4096,
    seed: Optional[int] = None,
    noisy: bool = True,
    coupling_map=None,
    basis_gates=None,
):
    """Build an Aer EstimatorV2 configured to match a fake backend."""
    noise_model = NoiseModel.from_backend(backend) if noisy else None
    backend_options = {}
    run_options = {}

    if noise_model is not None:
        backend_options["noise_model"] = noise_model

    if basis_gates is None and noise_model is not None:
        basis_gates = noise_model.basis_gates
    if basis_gates is None:
        basis_gates = _backend_basis_gates(backend)
    if basis_gates is not None:
        backend_options["basis_gates"] = basis_gates

    if coupling_map is None:
        coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is not None:
        backend_options["coupling_map"] = coupling_map

    if seed is not None:
        run_options["seed_simulator"] = int(seed)
    if shots > 0:
        run_options["shots"] = int(shots)

    options = {"backend_options": backend_options, "run_options": run_options}
    estimator = EstimatorV2(options=options)
    return estimator, None
