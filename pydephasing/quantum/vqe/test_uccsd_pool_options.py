from qiskit_nature.second_q.mappers import JordanWignerMapper

from pydephasing.quantum.vqe.adapt_vqe_meta import build_uccsd_excitation_pool


def test_uccsd_pool_include_imaginary_does_not_shrink() -> None:
    mapper = JordanWignerMapper()
    pool_real = build_uccsd_excitation_pool(
        n_sites=2,
        num_particles=(1, 1),
        mapper=mapper,
        include_imaginary=False,
    )
    pool_imag = build_uccsd_excitation_pool(
        n_sites=2,
        num_particles=(1, 1),
        mapper=mapper,
        include_imaginary=True,
    )
    assert len(pool_real) > 0
    assert len(pool_imag) >= len(pool_real)

