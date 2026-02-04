"""Ansatz builders for VQE."""

from .uccsd import build_ansatz as build_uccsd_ansatz

ANSATZ_BUILDERS = {
    "uccsd": build_uccsd_ansatz,
}


def build_ansatz(
    kind: str,
    num_qubits: int,
    reps: int,
    mapper,
    *,
    n_sites: int | None = None,
    num_particles: int | tuple[int, int] | None = None,
):
    """Dispatch ansatz construction by name."""
    if kind not in ANSATZ_BUILDERS:
        raise ValueError(f"Unknown ansatz: {kind}")

    if kind == "uccsd":
        # Spinful Hubbard convention: 2 spin orbitals per spatial orbital/site.
        if n_sites is None:
            if num_qubits % 2 != 0:
                raise ValueError("UCCSD requires an even number of qubits for spinful mapping.")
            n_sites = num_qubits // 2

        if num_particles is None:
            # Default to half-filling, Sz=0 when not provided.
            if n_sites % 2 != 0:
                raise ValueError(
                    "Default UCCSD num_particles assumes even n_sites. "
                    "Pass num_particles explicitly for odd n_sites."
                )
            num_particles = (n_sites // 2, n_sites // 2)

        return ANSATZ_BUILDERS[kind](
            n_sites=n_sites,
            num_particles=num_particles,
            reps=reps,
            qubit_mapper=mapper,
        )

    raise ValueError(f"Unsupported ansatz kind: {kind}")
