"""Exact energy evaluation utilities."""

from __future__ import annotations

import numpy as np


def exact_ground_energy(qubit_op) -> float:
    n = getattr(qubit_op, "num_qubits", None)
    if n is None:
        try:
            n = qubit_op.num_qubits
        except Exception:
            n = None

    if n is not None:
        dim = 2 ** n
    else:
        dim = None

    if dim is None or dim <= 4096:
        mat = qubit_op.to_matrix()
        evals = np.linalg.eigvalsh(mat)
        return float(np.min(np.real(evals)))

    try:
        from scipy.sparse.linalg import eigsh

        try:
            mat = qubit_op.to_matrix(sparse=True)
        except TypeError:
            mat = qubit_op.to_matrix()
        val = eigsh(mat, k=1, which="SA", return_eigenvectors=False)[0]
        return float(np.real(val))
    except Exception:
        mat = qubit_op.to_matrix()
        evals = np.linalg.eigvalsh(mat)
        return float(np.min(np.real(evals)))

