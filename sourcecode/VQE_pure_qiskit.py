import numpy as np
import argparse
import qiskit
from qiskit.algorithms import VQE
from qiskit import *
from qiskit import QuantumCircuit
from pennylane_qiskit import vqe_runner, upload_vqe_runner
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.primitives import Estimator
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit.circuit.library import EfficientSU2
#from qiskit_nature.second_q.algorithms import GroundStateEigensolver

def create_hamiltonian(source_path):
    """
    Creates lists of coefficients and Pauli strings operators from .txt file
    and uses them to establish the Hamiltonian observable.

    Args:
        source_path: Path to the Hamiltionian textual desciption.
    """
    coefs = []
    ops = []
    pauli_op = []
    #translating from file's default format to Pennylane notation
    with open(source_path, "r") as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break

            #parsing a single line
            coef_op = line.split("*")
            coef = float(coef_op[0].strip().replace(" ", ""))
            op = str(coef_op[1].strip())
            coef_op[0] = op
            coef_op[1] = coef
            #op = [o for o in coef_op[1].strip()]
            print(coef_op)
            print(coef)
            print(op)

            formatted_op = 0

            # #creating a pennylane format operator
            # op_penny =  qml.Identity(wires=0)
            # for i, pauli in enumerate(op):
            #     if pauli == 'I':
            #         op_penny = op_penny @ qml.Identity(wires=i) 
            #     elif pauli == 'X':
            #         op_penny = op_penny @ qml.PauliX(wires=i) 
            #     elif pauli == 'Y':
            #         op_penny = op_penny @ qml.PauliY(wires=i) 
            #     elif pauli == 'Z':
            #         op_penny = op_penny @ qml.PauliZ(wires=i)    

            pauli_op.append(coef_op)
    
    H = PauliSumOp.from_list([op for op in pauli_op])
    return H

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=str, default="qiskit.aer")
    parser.add_argument('--source_path', type=str, default='part0.txt')
    parser.add_argument('--positive_energy_flag', action='store_true')
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--skip_final_rotation_layer', action='store_true')
    args = parser.parse_args()

    H = create_hamiltonian(args.source_path)
    num_of_qubits = H.num_qubits
    #print(num_of_qubits)
    #print(H)

    np.random.seed(42)

    # Define the device
    #dev = qml.device(args.device, wires=qubits)
    program_id = upload_vqe_runner(hub="ibm-q", group="open", project="main")
    # Define the qnode
    #@qml.qnode(dev) 
    
    def ansatz(params, num_of_qubits, reps, skip_final_rotation_layer):
        pind = 0
        quantum_circuit = QuantumCircuit(num_of_qubits, num_of_qubits)

        for _ in range(reps):
            for qubit in range(num_of_qubits):
                quantum_circuit.ry(params[pind], qubit)
                quantum_circuit.rz(params[pind+1], qubit)
                pind += 2
          
            for qubit in range(0, num_of_qubits-1):
                quantum_circuit.cx(qubit, qubit+1)
            
        
        if not skip_final_rotation_layer:
            for qubit in range(num_of_qubits):
                quantum_circuit.ry(params[pind], qubit)
                quantum_circuit.rz(params[pind+1], qubit)
                pind += 2
        
        quantum_circuit.measure_all()
        #return qml.expval(H)
        return quantum_circuit
    
    
    def cost_function(params, **arg):
        
        
        h_expval = ansatz(params, **arg)
        coef = 1
        if args.positive_energy_flag:
            coef = -1
        return coef * h_expval
    
    nr_params = (args.reps+1)*num_of_qubits*2
    if args.skip_final_rotation_layer:
        nr_params -= num_of_qubits*2

    # Define the initial values of the circuit parameters
    params = np.random.normal(0, np.pi, nr_params)
    print(params)

    lazy = QuantumCircuit(num_of_qubits, num_of_qubits)
    theta = Parameter('theta')
    for _ in range(1):
            for qubit in range(num_of_qubits):
                lazy.ry(theta, qubit)
                lazy.rz(theta, qubit)
             
          
            for qubit in range(0, num_of_qubits-1):
                lazy.cx(qubit, qubit+1)

    para_lazy = [lazy.bind_parameters({theta : n}) for n in params]

    # qi = QuantumInstance(backend,
    #                 coupling_map=coupling_map,
    #                 noise_model=NOISE_MODEL,
    #                 measurement_error_mitigation_cls=CompleteMeasFitter)

    backend = Aer.get_backend('qasm_simulator')

    ansatz_circuit = ansatz(params = params,
                            num_of_qubits = num_of_qubits,
                            reps = args.reps,
                            skip_final_rotation_layer = args.skip_final_rotation_layer)

    optimizer = SPSA(maxiter=100)
    var_form = EfficientSU2(H.num_qubits, entanglement="linear")
    b = VQE(var_form, optimizer=optimizer,quantum_instance = backend)



    # a = VQE(ansatz = para_lazy,
    #                         optimizer = None,
    #                         initial_point = params,
    #                         expectation = H,
    #                         quantum_instance = backend)


    
    

    res = b.get_energy_evaluation(H)
    res
    print(res)
    #print(b.energy_evaluation(params))
    #print(b.optimal_value)
    #qc = circuit(params=params, wires=wires, reps = args.reps, skip_final_rotation_layer= args.skip_final_rotation_layer)
    # Define the optimizer
    #optimizer = qml.AdamOptimizer(stepsize=0.1)

    # job = vqe_runner(
    #         program_id=program_id,
    #         backend="ibmq_qasm_simulator",
    #         hamiltonian=H,
    #         ansatz=circuit,
    #         x0=params,
    #         shots=1024,
    #         optimizer="SPSA",
    #         optimizer_config={"maxiter": 40},
    #         kwargs={"hub": "ibm-q", "group": "open", "project": "main"})

    # Optimize the circuit parameters and compute the energy
  
    
    # prev_energy = 0
    # for n in range(1000):
    #     params, energy = optimizer.step_and_cost(cost_function, params,
    #                                             wires=range(qubits), reps=args.reps, 
    #                                             skip_final_rotation_layer=args.skip_final_rotation_layer)

        
    #     if args.positive_energy_flag:
    #         energy *= -1
        
    #     print("step = {:},  E = {:.8f}".format(n, energy))
    #     if abs(energy - prev_energy) < 0.0000000005: # depending on precision
    #         break
    #     prev_energy = energy

