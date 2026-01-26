"""Ansatz builders for VQE."""

from .clustered import build_ansatz as build_clustered_ansatz
from .efficient_su2 import build_ansatz as build_efficient_su2_ansatz
from .hea import build_ansatz as build_hardware_efficient_ansatz
from .uccsd import build_ansatz as build_uccsd_ansatz

ANSATZ_BUILDERS = {
    "clustered": build_clustered_ansatz,
    "efficient_su2": build_efficient_su2_ansatz,
    "uccsd": build_uccsd_ansatz,
    "hea": build_hardware_efficient_ansatz,
}


def build_ansatz(
    kind: str,
    num_qubits: int,
    reps: int,
    mapper,
    *,
    hea_rotation: str = "ry",
    hea_entanglement: str = "linear",
    hea_occ: list[int] | None = None,
):
    """Dispatch ansatz construction by name."""
    if kind not in ANSATZ_BUILDERS:
        raise ValueError(f"Unknown ansatz: {kind}")

    if kind == "hea":
        return ANSATZ_BUILDERS[kind](
            n_qubits=num_qubits,
            reps=reps,
            entanglement=hea_entanglement,
            initial_occupations=hea_occ,
            rotation=hea_rotation,
        )

    if kind == "uccsd":
        return ANSATZ_BUILDERS[kind](
            num_qubits=num_qubits,
            reps=reps,
            mapper=mapper,
        )

    return ANSATZ_BUILDERS[kind](num_qubits=num_qubits, reps=reps)

