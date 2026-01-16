def build_hubbard_clustered_ansatz(*, t, U, dv, reps=1, initial_occupations=None):
	try:
		from qiskit import QuantumCircuit
		from qiskit.circuit import ParameterVector
		from qiskit.circuit.library import PauliEvolutionGate
	except Exception as exc:
		raise ImportError("qiskit is required for the clustered ansatz") from exc

	from pydephasing.quantum.pauli_polynomial_class import (
		fermion_minus_operator,
		fermion_plus_operator,
	)
	from pydephasing.quantum.qiskit_bridge import pauli_polynomial_to_sparse_pauli_op

	if initial_occupations is None:
		if dv >= 0:
			initial_occupations = [0, 2]
		else:
			initial_occupations = [1, 3]

	qc = QuantumCircuit(4)
	for q in initial_occupations:
		qc.x(q)

	theta_t = ParameterVector("theta_t", reps)
	theta_u = ParameterVector("theta_u", reps)
	theta_v = ParameterVector("theta_v", reps)
	theta_pt = ParameterVector("theta_pt", reps)
	theta_se = ParameterVector("theta_se", reps)

	nq = 4
	c_dag = [fermion_plus_operator("JW", nq, j) for j in range(nq)]
	c = [fermion_minus_operator("JW", nq, j) for j in range(nq)]

	# Double-excitation generators for singlet correlations.
	pair_transfer = c_dag[1] * c_dag[3] * c[0] * c[2]
	pair_transfer += c_dag[2] * c_dag[0] * c[3] * c[1]
	spin_exchange = c_dag[1] * c_dag[2] * c[0] * c[3]
	spin_exchange += c_dag[3] * c_dag[0] * c[2] * c[1]

	pair_transfer_op = pauli_polynomial_to_sparse_pauli_op(pair_transfer)
	spin_exchange_op = pauli_polynomial_to_sparse_pauli_op(spin_exchange)

	for r in range(reps):
		angle_t = 2 * theta_t[r]
		qc.rxx(angle_t, 0, 1)
		qc.ryy(angle_t, 0, 1)
		qc.rxx(angle_t, 2, 3)
		qc.ryy(angle_t, 2, 3)

		angle_u = 2 * theta_u[r]
		qc.rzz(angle_u, 0, 2)
		qc.rzz(angle_u, 1, 3)

		angle_v = 2 * theta_v[r]
		qc.rz(angle_v, 0)
		qc.rz(angle_v, 2)
		qc.rz(-angle_v, 1)
		qc.rz(-angle_v, 3)

		qc.append(PauliEvolutionGate(pair_transfer_op, time=theta_pt[r]), range(nq))
		qc.append(PauliEvolutionGate(spin_exchange_op, time=theta_se[r]), range(nq))

	return qc
