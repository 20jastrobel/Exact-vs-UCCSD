"""IBM Runtime helpers with minimal API calls."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional


def get_runtime_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    import os

    token = os.environ.get("QISKIT_IBM_TOKEN", None)
    instance = os.environ.get("QISKIT_IBM_INSTANCE", None)

    kwargs = {
        "channel": "ibm_quantum_platform",
        "token": token,
        "instance": instance,
    }

    region = os.environ.get("QISKIT_IBM_REGION", None)
    if region is not None:
        kwargs["region"] = region

    if os.environ.get("QISKIT_IBM_PLANS") is None:
        kwargs["plans_preference"] = ["open"]

    return QiskitRuntimeService(**kwargs)


def choose_backend(
    service,
    backend_name: Optional[str] = None,
    *,
    no_scan: bool = True,
    force_hardware: bool = False,
):
    import os

    name = backend_name or os.environ.get("IBM_BACKEND", None)
    if name:
        return service.backend(name)

    if no_scan:
        raise RuntimeError(
            "No backend specified. Set --backend or env IBM_BACKEND to avoid scans."
        )

    sims = service.backends(simulator=True, operational=True)
    if sims:
        for backend in sims:
            if backend.name == "ibmq_qasm_simulator":
                return backend
        return sims[0]

    if not force_hardware:
        raise RuntimeError(
            "No simulator available. Re-run with --force-hardware or specify --backend."
        )

    return service.least_busy(operational=True, simulator=False, min_num_qubits=4)


@contextmanager
def runtime_context(service, backend, *, prefer_batch: bool = True, prefer_session: bool = False):
    try:
        from qiskit_ibm_runtime import Batch, Session
    except Exception:
        yield backend
        return

    if prefer_session:
        try:
            with Session(service=service, backend=backend) as session:
                yield session
                return
        except Exception:
            pass

    if prefer_batch:
        try:
            with Batch(backend=backend) as batch:
                yield batch
                return
        except Exception:
            pass

    yield backend


def make_estimator(mode_obj, shots: int, resilience: int, max_exec: int):
    from qiskit_ibm_runtime import Estimator

    estimator = Estimator(mode=mode_obj)
    estimator.options.default_shots = int(shots)
    estimator.options.resilience_level = int(resilience)
    estimator.options.max_execution_time = int(max_exec)
    return estimator
