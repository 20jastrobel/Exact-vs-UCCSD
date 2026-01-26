"""L-BFGS-B optimizer wrapper."""

from qiskit_algorithms.optimizers import L_BFGS_B


def build(*, maxiter: int):
    return L_BFGS_B(maxiter=maxiter)

