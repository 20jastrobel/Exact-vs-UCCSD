from pydephasing.quantum.pauli_polynomial_class import PauliPolynomial
from pydephasing.quantum.qubitization_module import PauliTerm

try:
	from qiskit.quantum_info import SparsePauliOp
except Exception as exc:
	raise ImportError("qiskit is required for qiskit_bridge") from exc


def letters_to_qiskit_label(ps):
	mapping = {"e": "I", "x": "X", "y": "Y", "z": "Z"}
	return "".join(mapping[ch] for ch in ps)


def qiskit_label_to_letters(label):
	mapping = {"I": "e", "X": "x", "Y": "y", "Z": "z"}
	return "".join(mapping[ch] for ch in label)


def _simplify_sparse_pauli(op, atol=1e-12):
	try:
		return op.simplify(atol=atol)
	except TypeError:
		return op.simplify()


def pauli_polynomial_to_sparse_pauli_op(pp):
	terms = []
	for label, coeff in pp.to_term_list():
		terms.append((letters_to_qiskit_label(label), coeff))
	op = SparsePauliOp.from_list(terms)
	return _simplify_sparse_pauli(op, atol=1e-12)


def sparse_pauli_op_to_pauli_polynomial(op, repr_mode="JW"):
	terms = []
	for label, coeff in op.to_list():
		letters = qiskit_label_to_letters(label)
		terms.append(PauliTerm(len(letters), ps=letters, pc=coeff))
	return PauliPolynomial(repr_mode, terms)


def _term_dict(term_list):
	items = {}
	for label, coeff in term_list:
		items[label] = items.get(label, 0) + coeff
	return items


def _compare_term_lists(a_terms, b_terms, atol=1e-12):
	a_dict = _term_dict(a_terms)
	b_dict = _term_dict(b_terms)
	if set(a_dict.keys()) != set(b_dict.keys()):
		return False
	for label in a_dict:
		if abs(a_dict[label] - b_dict[label]) > atol:
			return False
	return True


def self_test_roundtrip(atol=1e-12):
	pp = PauliPolynomial(
		"JW",
		[
			PauliTerm(4, ps="eexx", pc=0.5),
			PauliTerm(4, ps="zeee", pc=-0.25j),
			PauliTerm(4, ps="eeee", pc=1.25),
		],
	)
	op = pauli_polynomial_to_sparse_pauli_op(pp)
	back = sparse_pauli_op_to_pauli_polynomial(op, repr_mode="JW")
	return _compare_term_lists(pp.to_term_list(), back.to_term_list(), atol=atol)


def self_test_label_convention():
	if letters_to_qiskit_label("exyz") != "IXYZ":
		return False
	if qiskit_label_to_letters("IXYZ") != "exyz":
		return False
	if letters_to_qiskit_label("eeex") != "IIIX":
		return False
	if qiskit_label_to_letters("IIIX") != "eeex":
		return False
	if letters_to_qiskit_label("xeee") != "XIII":
		return False
	if qiskit_label_to_letters("XIII") != "xeee":
		return False
	return True
