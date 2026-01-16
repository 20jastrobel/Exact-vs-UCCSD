import os

from pydephasing.quantum.pauli_polynomial_class import (
	PauliPolynomial,
	fermion_minus_operator,
	fermion_plus_operator,
)
from pydephasing.quantum.qubitization_module import PauliTerm


def _debug_enabled():
	raw = os.environ.get("DEBUG_POLY", "")
	return raw.strip().lower() in {"1", "true", "yes", "y"}


def _clone_polynomial(pp, repr_mode):
	terms = []
	for label, coeff in pp.to_term_list():
		terms.append(PauliTerm(len(label), ps=label, pc=coeff))
	return PauliPolynomial(repr_mode, terms)


def build_reference_hubbard_image_polynomial(t, U, dv, repr_mode="JW"):
	nq = 4
	terms = [
		PauliTerm(nq, ps="eexx", pc=-t / 2.0),
		PauliTerm(nq, ps="eeyy", pc=-t / 2.0),
		PauliTerm(nq, ps="xxee", pc=-t / 2.0),
		PauliTerm(nq, ps="yyee", pc=-t / 2.0),
		PauliTerm(nq, ps="eeez", pc=+dv / 4.0),
		PauliTerm(nq, ps="ezee", pc=+dv / 4.0),
		PauliTerm(nq, ps="eeze", pc=-dv / 4.0),
		PauliTerm(nq, ps="zeee", pc=-dv / 4.0),
		PauliTerm(nq, ps="ezez", pc=+U / 4.0),
		PauliTerm(nq, ps="zeze", pc=+U / 4.0),
		PauliTerm(nq, ps="eeez", pc=-U / 4.0),
		PauliTerm(nq, ps="eeze", pc=-U / 4.0),
		PauliTerm(nq, ps="ezee", pc=-U / 4.0),
		PauliTerm(nq, ps="zeee", pc=-U / 4.0),
		PauliTerm(nq, ps="eeee", pc=+U / 2.0),
	]
	return PauliPolynomial(repr_mode, terms)


def build_hubbard_dimer_jw_polynomial(t, U, dv, repr_mode="JW"):
	nq = 4
	c_dag = [fermion_plus_operator(repr_mode, nq, j) for j in range(nq)]
	c = [fermion_minus_operator(repr_mode, nq, j) for j in range(nq)]
	if _debug_enabled():
		plus_terms = [pt.pw2strng() for pt in c_dag[0].return_polynomial()]
		minus_terms = [pt.pw2strng() for pt in c[0].return_polynomial()]
		print(f"DEBUG_POLY fermion_plus[0] terms: {plus_terms[:3]}")
		print(f"DEBUG_POLY fermion_minus[0] terms: {minus_terms[:3]}")
	n = [c_dag[j] * c[j] for j in range(nq)]

	hopping = (-t) * (
		c_dag[0] * c[1] + c_dag[1] * c[0] + c_dag[2] * c[3] + c_dag[3] * c[2]
	)
	n_for_potential = [_clone_polynomial(op, repr_mode) for op in n]
	potential = (dv / 2.0) * (
		(n_for_potential[1] + n_for_potential[3])
		- (n_for_potential[0] + n_for_potential[2])
	)
	interaction = U * (
		(_clone_polynomial(n[0], repr_mode) * _clone_polynomial(n[2], repr_mode))
		+ (_clone_polynomial(n[1], repr_mode) * _clone_polynomial(n[3], repr_mode))
	)

	H = hopping + potential + interaction
	H._reduce()
	if _debug_enabled():
		print(f"DEBUG_POLY H is PauliPolynomial: {isinstance(H, PauliPolynomial)}")
		print(f"DEBUG_POLY H terms: {H.count_number_terms()}")
	return H


def build_hubbard_dimer_qubit_polynomial(t, U, dv, *, mode="JW", repr_mode="JW"):
	mode_norm = mode.strip().lower()
	if mode_norm == "jw":
		return build_hubbard_dimer_jw_polynomial(t, U, dv, repr_mode=repr_mode)
	if mode_norm in {"image_ref", "ref"}:
		return build_reference_hubbard_image_polynomial(t, U, dv, repr_mode=repr_mode)
	raise ValueError(f"Unknown mode: {mode}")


def build_hubbard_dimer_sparse_pauli_op(t, U, dv, source="jw_poly", repr_mode="JW"):
	if source == "jw_poly":
		poly = build_hubbard_dimer_jw_polynomial(t, U, dv, repr_mode=repr_mode)
	elif source == "image_ref":
		poly = build_reference_hubbard_image_polynomial(t, U, dv, repr_mode=repr_mode)
	else:
		raise ValueError(f"Unknown source: {source}")

	from pydephasing.quantum.qiskit_bridge import pauli_polynomial_to_sparse_pauli_op

	return pauli_polynomial_to_sparse_pauli_op(poly)


def build_hubbard_dimer_qiskit_nature_jw(t, U, dv):
	try:
		from qiskit_nature.second_q.mappers import JordanWignerMapper
		from qiskit_nature.second_q.operators import FermionicOp
	except Exception as exc:
		raise ImportError("qiskit_nature is required for JW mapping") from exc

	terms = {
		"+_0 -_1": -t,
		"+_1 -_0": -t,
		"+_2 -_3": -t,
		"+_3 -_2": -t,
		"+_1 -_1": dv / 2.0,
		"+_3 -_3": dv / 2.0,
		"+_0 -_0": -dv / 2.0,
		"+_2 -_2": -dv / 2.0,
		"+_0 -_0 +_2 -_2": U,
		"+_1 -_1 +_3 -_3": U,
	}
	try:
		fermionic_op = FermionicOp(terms, num_spin_orbitals=4)
	except TypeError:
		fermionic_op = FermionicOp(terms, register_length=4)
	try:
		fermionic_op = fermionic_op.simplify()
	except Exception:
		pass

	mapper = JordanWignerMapper()
	qubit_op = mapper.map(fermionic_op)
	if hasattr(qubit_op, "primitive"):
		qubit_op = qubit_op.primitive
	try:
		return qubit_op.simplify(atol=1e-12)
	except TypeError:
		return qubit_op.simplify()
