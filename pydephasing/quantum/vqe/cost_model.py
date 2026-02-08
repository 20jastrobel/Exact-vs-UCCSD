"""Lightweight cost accounting for estimator-based VQE/ADAPT runs.

This is intentionally simple and hardware-agnostic:
  - counts estimator calls and submitted pubs ("circuits executed")
  - counts measured Pauli terms as a proxy for measurement cost
  - optionally tracks objective vs gradient energy-evaluation categories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp


@dataclass
class CostCounters:
    """Cumulative counters for a single run."""

    # Algorithm-level categories (set by callers, not by the estimator wrapper).
    n_energy_evals: int = 0
    n_grad_evals: int = 0

    # Primitive-level accounting (maintained by CountingEstimator).
    n_estimator_calls: int = 0
    n_circuits_executed: int = 0
    n_pauli_terms_measured: int = 0

    # Shot-based runs can populate this later (EstimatorV2 is precision-based).
    total_shots: int | None = None

    def snapshot(self) -> dict[str, int | float | None]:
        return {
            "n_energy_evals": int(self.n_energy_evals),
            "n_grad_evals": int(self.n_grad_evals),
            "n_estimator_calls": int(self.n_estimator_calls),
            "n_circuits_executed": int(self.n_circuits_executed),
            "n_pauli_terms_measured": int(self.n_pauli_terms_measured),
            "total_shots": None if self.total_shots is None else int(self.total_shots),
        }


def _count_nonidentity_terms_in_observable(obs: Any) -> int:
    """Best-effort term counting across common observable representations."""
    if obs is None:
        return 0

    if isinstance(obs, SparsePauliOp):
        identity = "I" * obs.num_qubits
        return int(sum(1 for lbl in obs.paulis.to_labels() if lbl != identity))

    # Qiskit primitives sometimes serialize observables as a {label: coeff} dict.
    if isinstance(obs, dict):
        if not obs:
            return 0
        # Infer identity label length from any key.
        first_key = next(iter(obs.keys()))
        identity = "I" * len(str(first_key))
        return int(sum(1 for k in obs.keys() if str(k) != identity))

    # Unknown representation: fall back to "1 term".
    return 1


def _count_terms_in_observables_array(arr: ObservablesArray) -> int:
    if arr is None:
        return 0
    flat = arr.ravel()
    total = 0
    for idx in range(int(flat.size)):
        total += _count_nonidentity_terms_in_observable(flat[idx])
    return int(total)


class CountingEstimator(BaseEstimatorV2):
    """Wrap a BaseEstimatorV2 to count primitive-level execution cost."""

    def __init__(self, base: BaseEstimatorV2, counters: CostCounters | None = None):
        self._base = base
        self.counters = counters if counters is not None else CostCounters()

    def run(self, pubs: Iterable, *, precision: float | None = None):
        pubs_list = list(pubs)
        self.counters.n_estimator_calls += 1
        self.counters.n_circuits_executed += len(pubs_list)

        for pub in pubs_list:
            # EstimatorPubLike can be:
            # - EstimatorPub (has .observables)
            # - tuple/list (circuit, observable, param_values)
            if hasattr(pub, "observables"):
                try:
                    self.counters.n_pauli_terms_measured += _count_terms_in_observables_array(pub.observables)
                except Exception:
                    # Don't fail runs due to accounting; just be conservative.
                    self.counters.n_pauli_terms_measured += 0
                continue

            if isinstance(pub, (tuple, list)) and len(pub) >= 2:
                obs = pub[1]
                if isinstance(obs, (tuple, list)):
                    for o in obs:
                        self.counters.n_pauli_terms_measured += _count_nonidentity_terms_in_observable(o)
                else:
                    self.counters.n_pauli_terms_measured += _count_nonidentity_terms_in_observable(obs)
                continue

        return self._base.run(pubs_list, precision=precision)

    def __getattr__(self, name: str):
        # Delegate all other attributes/options to the wrapped estimator.
        return getattr(self._base, name)

