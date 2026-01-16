import numpy as np

from pydephasing.quantum.pauli_polynomial_class import PauliPolynomial
from pydephasing.quantum.clustered_ansatz import build_hubbard_clustered_ansatz

#   Hamiltonian ground state calculation


def sector_indices(
		n_qubits,
		n_electrons=None,
		sz=None,
		up_qubits=None,
		down_qubits=None,
):
	if up_qubits is None:
		up_qubits = []
	if down_qubits is None:
		down_qubits = []

	idx = []
	for bitstring in range(1 << n_qubits):
		if n_electrons is not None and bitstring.bit_count() != n_electrons:
			continue
		if sz is not None:
			n_up = sum((bitstring >> q) & 1 for q in up_qubits)
			n_dn = sum((bitstring >> q) & 1 for q in down_qubits)
			if 0.5 * (n_up - n_dn) != sz:
				continue
		idx.append(bitstring)
	return idx


def dense_hermitian(matrix):
	matrix = np.asarray(matrix, dtype=complex)
	return 0.5 * (matrix + matrix.conj().T)


_PAULI_MATS = {
	"e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
	"x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
	"y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
	"z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _evaluate_energy(estimator, ansatz, operator, parameters):
	params = np.asarray(parameters, dtype=float)
	try:
		job = estimator.run([(ansatz, operator, params)])
		result = job.result()
		data = result[0].data
		if hasattr(data, "evs"):
			return float(np.real(np.asarray(data.evs).item()))
	except Exception:
		pass

	try:
		job = estimator.run(ansatz, operator, params)
		result = job.result()
		values = getattr(result, "values", None)
		if values is not None:
			return float(np.real(values[0]))
	except Exception:
		pass

	raise RuntimeError("Estimator energy evaluation failed.")


def _summarize_energy_trace(trace):
	if not trace:
		print("VQE energy trace: []")
		return
	head = trace[:5]
	tail = trace[-5:] if len(trace) > 5 else trace
	print(f"VQE energy trace (first 5): {head}")
	print(f"VQE energy trace (last 5): {tail}")


def _optimizer_status(result):
	opt = getattr(result, "optimizer_result", None)
	if opt is None:
		return "nfev=None, nit=None, success=None, message=None"
	try:
		nfev = opt.get("nfev")
		nit = opt.get("nit")
		success = opt.get("success")
		message = opt.get("message")
	except Exception:
		nfev = getattr(opt, "nfev", None)
		nit = getattr(opt, "nit", None)
		success = getattr(opt, "success", None)
		message = getattr(opt, "message", None)
	return f"nfev={nfev}, nit={nit}, success={success}, message={message}"


def _pauli_word_matrix(label):
	mat = np.array([[1.0]], dtype=complex)
	for ch in label:
		mat = np.kron(mat, _PAULI_MATS[ch])
	return mat


def _pauli_polynomial_to_dense(pp):
	terms = pp.to_term_list()
	if not terms:
		return np.zeros((1, 1), dtype=complex)
	n_qubits = len(terms[0][0])
	mat = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
	for label, coeff in terms:
		mat += coeff * _pauli_word_matrix(label)
	return mat


def _extract_occupied_qubits(circuit):
	occupied = []
	for inst in circuit.data:
		operation = getattr(inst, "operation", None)
		qubits = getattr(inst, "qubits", None)
		if operation is None:
			operation = inst[0]
		if qubits is None:
			qubits = inst[1]
		if getattr(operation, "name", None) == "x" and qubits:
			occupied.append(circuit.find_bit(qubits[0]).index)
	return sorted(occupied)


def _uccsd_initial_point(ansatz, seed=0, scale=0.1):
	n_params = getattr(ansatz, "num_parameters", 0) or 0
	if n_params == 0:
		return None, "empty"
	rng = np.random.default_rng(seed)
	return rng.uniform(-scale, scale, size=n_params), "random_small"


def exact_ground_energy(qubit_op, sector=None, n_qubits=None):
	matrix = None
	if isinstance(qubit_op, PauliPolynomial):
		try:
			from pydephasing.quantum.qiskit_bridge import (
				pauli_polynomial_to_sparse_pauli_op,
			)
			qubit_op = pauli_polynomial_to_sparse_pauli_op(qubit_op)
		except Exception:
			matrix = _pauli_polynomial_to_dense(qubit_op)

	if matrix is None:
		if hasattr(qubit_op, "to_matrix"):
			matrix = qubit_op.to_matrix()
		elif hasattr(qubit_op, "data"):
			matrix = qubit_op.data
		else:
			raise TypeError("Operator must expose to_matrix() or data.")

	matrix = dense_hermitian(matrix)
	dim = matrix.shape[0]
	if n_qubits is None:
		n_qubits = int(round(np.log2(dim)))

	if sector is None:
		return float(np.linalg.eigvalsh(matrix).min().real)

	n_electrons = sector.get("n_electrons")
	sz = sector.get("sz")
	up_qubits = sector.get("up_qubits")
	down_qubits = sector.get("down_qubits")
	idx = sector_indices(
		n_qubits=n_qubits,
		n_electrons=n_electrons,
		sz=sz,
		up_qubits=up_qubits,
		down_qubits=down_qubits,
	)
	reduced = matrix[np.ix_(idx, idx)]
	return float(np.linalg.eigvalsh(reduced).min().real)


def ansatz_factory(kind, *, t, U, dv, reps=1, initial_occupations=None):
	kind_norm = kind.strip().lower()
	if kind_norm == "clustered":
		return (
			build_hubbard_clustered_ansatz(
				t=t,
				U=U,
				dv=dv,
				reps=reps,
				initial_occupations=initial_occupations,
			),
			"clustered",
		)
	if kind_norm == "efficient_su2":
		try:
			from qiskit.circuit.library import EfficientSU2
		except Exception as exc:
			raise ImportError("qiskit is required for EfficientSU2") from exc
		return EfficientSU2(num_qubits=4, reps=reps, entanglement="full"), "efficient_su2"
	if kind_norm == "uccsd":
		try:
			from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
			from qiskit_nature.second_q.mappers import JordanWignerMapper
		except Exception:
			return ansatz_factory(
				"clustered",
				t=t,
				U=U,
				dv=dv,
				reps=reps,
				initial_occupations=initial_occupations,
			)

		mapper = JordanWignerMapper()
		num_spatial_orbitals = 2
		num_particles = (1, 1)
		if num_spatial_orbitals != 2 or num_particles != (1, 1):
			raise AssertionError("UCCSD configuration mismatch for Hubbard dimer.")
		try:
			init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
		except TypeError:
			init_state = HartreeFock(
				num_spatial_orbitals, num_particles, qubit_mapper=mapper
			)
		try:
			ansatz = UCCSD(
				num_spatial_orbitals,
				num_particles,
				mapper,
				initial_state=init_state,
				reps=reps,
				preserve_spin=True,
			)
		except TypeError:
			ansatz = UCCSD(
				num_spatial_orbitals,
				num_particles,
				qubit_mapper=mapper,
				initial_state=init_state,
				reps=reps,
				preserve_spin=True,
			)

		expected_hf = [0, 2]
		hf_occupied = _extract_occupied_qubits(init_state)
		if hf_occupied != expected_hf:
			raise AssertionError(
				"UCCSD Hartree-Fock occupation mismatch: "
				f"expected {expected_hf}, got {hf_occupied}."
			)

		metadata = ansatz.metadata or {}
		metadata.update(
			{
				"ansatz_kind": "uccsd",
				"num_spatial_orbitals": num_spatial_orbitals,
				"num_particles": num_particles,
				"hf_occupations": hf_occupied,
				"initial_point_range": float(np.pi),
			}
		)
		excitation_list = getattr(ansatz, "excitation_list", None)
		if excitation_list is not None:
			metadata["excitation_list"] = excitation_list
		initial_point, source = _uccsd_initial_point(ansatz)
		if initial_point is not None:
			metadata["initial_point"] = initial_point
			metadata["initial_point_source"] = source
		ansatz.metadata = metadata
		return ansatz, "uccsd"

	raise ValueError(f"Unknown ansatz kind: {kind}")


def vqe_ground_energy(
		qubit_op,
		*,
		ansatz,
		optimizer=None,
		estimator=None,
		transpiler=None,
		initial_point=None,
):
	from qiskit_algorithms.minimum_eigensolvers import VQE
	from qiskit_algorithms.optimizers import COBYLA

	if optimizer is None:
		optimizer = COBYLA()
	if estimator is None:
		from qiskit.primitives import Estimator

		estimator = Estimator()

	energy_trace = []

	def _callback(eval_count, parameters, mean, std):
		energy_trace.append(float(np.real(mean)))

	initial_energy = None
	if initial_point is not None:
		try:
			initial_energy = _evaluate_energy(estimator, ansatz, qubit_op, initial_point)
		except Exception:
			initial_energy = None

	try:
		if transpiler is None:
			vqe = VQE(
				estimator,
				ansatz,
				optimizer=optimizer,
				initial_point=initial_point,
				callback=_callback,
			)
		else:
			vqe = VQE(
				estimator,
				ansatz,
				optimizer=optimizer,
				transpiler=transpiler,
				initial_point=initial_point,
				callback=_callback,
			)
	except TypeError:
		if transpiler is None:
			vqe = VQE(estimator, ansatz, optimizer=optimizer)
		else:
			vqe = VQE(estimator, ansatz, optimizer=optimizer, transpiler=transpiler)
		if initial_point is not None:
			vqe.initial_point = initial_point
		try:
			vqe.callback = _callback
		except Exception:
			pass

	result = vqe.compute_minimum_eigenvalue(qubit_op)
	energy = float(np.real(result.eigenvalue))
	if initial_energy is not None:
		print(f"VQE initial energy: {initial_energy}")
	else:
		print("VQE initial energy: n/a (no initial point provided)")
	print(f"VQE final energy: {energy}")
	print(f"VQE optimizer status: {_optimizer_status(result)}")

	optimal_point = getattr(result, "optimal_point", None)
	if optimal_point is None:
		opt_params = getattr(result, "optimal_parameters", None)
		if opt_params is not None:
			optimal_point = np.array([opt_params[p] for p in ansatz.parameters], dtype=float)

	if initial_point is not None and optimal_point is not None:
		try:
			update_norm = float(
				np.linalg.norm(np.asarray(optimal_point) - np.asarray(initial_point))
			)
			print(f"VQE parameter update norm: {update_norm}")
		except Exception:
			print("VQE parameter update norm: n/a")
	else:
		print("VQE parameter update norm: n/a")

	_summarize_energy_trace(energy_trace)
	try:
		result.energy_trace = list(energy_trace)
		result.initial_energy = initial_energy
	except Exception:
		pass
	return energy, result


def compute_ground_state(
		qubit_op,
		*,
		method="exact",
		vqe_options=None,
		sector=None,
):
	method_norm = method.strip().lower()
	if method_norm == "exact":
		return exact_ground_energy(qubit_op, sector=sector)

	if method_norm != "vqe":
		raise ValueError(f"Unknown method: {method}")

	if isinstance(qubit_op, PauliPolynomial):
		try:
			from pydephasing.quantum.qiskit_bridge import (
				pauli_polynomial_to_sparse_pauli_op,
			)
			qubit_op = pauli_polynomial_to_sparse_pauli_op(qubit_op)
		except Exception as exc:
			raise ImportError("qiskit is required for VQE conversion") from exc

	if vqe_options is None:
		vqe_options = {}

	ansatz = vqe_options.get("ansatz")
	if ansatz is None:
		t = vqe_options.get("t")
		U = vqe_options.get("U")
		dv = vqe_options.get("dv")
		reps = vqe_options.get("reps", 1)
		initial_occupations = vqe_options.get("initial_occupations")
		ansatz_kind = vqe_options.get("ansatz_kind", "clustered")
		if t is None or U is None or dv is None:
			raise ValueError("vqe_options must include t, U, dv when ansatz is omitted")
		ansatz, _ = ansatz_factory(
			ansatz_kind,
			t=t,
			U=U,
			dv=dv,
			reps=reps,
			initial_occupations=initial_occupations,
		)

	optimizer = vqe_options.get("optimizer")
	estimator = vqe_options.get("estimator")
	transpiler = vqe_options.get("transpiler")
	initial_point = vqe_options.get("initial_point")

	energy, result = vqe_ground_energy(
		qubit_op,
		ansatz=ansatz,
		optimizer=optimizer,
		estimator=estimator,
		transpiler=transpiler,
		initial_point=initial_point,
	)
	return energy, result.optimal_parameters
