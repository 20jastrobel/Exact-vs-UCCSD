"""SLSQP optimizer wrapper."""

from qiskit_algorithms.optimizers import SLSQP


def build(*, maxiter: int):
    return SLSQP(maxiter=maxiter)

