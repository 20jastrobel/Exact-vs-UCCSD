"""COBYLA optimizer wrapper."""

from qiskit_algorithms.optimizers import COBYLA


def build(*, maxiter: int):
    return COBYLA(maxiter=maxiter)

