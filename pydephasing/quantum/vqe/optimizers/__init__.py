"""Optimizer factory for VQE."""

from .cobyla import build as build_cobyla
from .l_bfgs_b import build as build_l_bfgs_b
from .slsqp import build as build_slsqp

_OPTIMIZERS = {
    "COBYLA": build_cobyla,
    "L_BFGS_B": build_l_bfgs_b,
    "SLSQP": build_slsqp,
}


def build_optimizer(name: str, *, maxiter: int):
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return _OPTIMIZERS[name](maxiter=maxiter)

