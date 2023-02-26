import pennylane as qml
from pennylane import numpy as np
import argparse
import qiskit
from qiskit import *
from pennylane_qiskit import vqe_runner, upload_vqe_runner

def create_hamiltonian(source_path):
    """
    Creates lists of coefficients and Pauli strings operators from .txt file
    and uses them to establish the Hamiltonian observable.

    Args:
        source_path: Path to the Hamiltionian textual desciption.
    """
    coefs = []
    ops = []

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
            op = [o for o in coef_op[1].strip()]

            #creating a pennylane format operator
            op_penny =  qml.Identity(wires=0)
            for i, pauli in enumerate(op):
                if pauli == 'I':
                    op_penny = op_penny @ qml.Identity(wires=i) 
                elif pauli == 'X':
                    op_penny = op_penny @ qml.PauliX(wires=i) 
                elif pauli == 'Y':
                    op_penny = op_penny @ qml.PauliY(wires=i) 
                elif pauli == 'Z':
                    op_penny = op_penny @ qml.PauliZ(wires=i)    

            coefs.append(coef)
            ops.append(op_penny)

    H = qml.Hamiltonian(coefs, ops)
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
    wires = list(H.wires)
    qubits=len(wires)

    np.random.seed(42)

    # Define the device
    #dev = qml.device(args.device, wires=qubits)
    program_id = upload_vqe_runner(hub="ibm-q", group="open", project="main")
    # Define the qnode
    #@qml.qnode(dev) 
    def circuit(params, wires, reps, skip_final_rotation_layer):
        pind = 0
        for _ in range(reps):
            for wire in wires:
                qml.RY(params[pind], wires=wire)
                qml.RZ(params[pind+1], wires=wire)
                pind += 2
            qml.Barrier(only_visual=True)
            for wire in range(0, len(wires)-1):
                qml.CNOT(wires=[wire, wire+1])
            qml.Barrier(only_visual=True)
        
        if not skip_final_rotation_layer:
            for wire in wires:
                qml.RY(params[pind], wires=wire)
                qml.RZ(params[pind+1], wires=wire)
                pind += 2

        return qml.expval(H)
    
    def cost_function(params, **arg):
        h_expval = circuit(params, **arg)
        coef = 1
        if args.positive_energy_flag:
            coef = -1
        return coef * h_expval
    
    nr_params = (args.reps+1)*len(wires)*2
    if args.skip_final_rotation_layer:
        nr_params -= len(wires)*2

    # Define the initial values of the circuit parameters
    params = np.random.normal(0, np.pi, nr_params)
    print(params)
    print(params[0])
    qc = circuit(params=params, wires=wires, reps = args.reps, skip_final_rotation_layer= args.skip_final_rotation_layer)
    # Define the optimizer
    optimizer = qml.AdamOptimizer(stepsize=0.1)

    job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=H,
            ansatz=circuit,
            x0=params,
            shots=1024,
            optimizer="SPSA",
            optimizer_config={"maxiter": 40},
            kwargs={"hub": "ibm-q", "group": "open", "project": "main"})

    # Optimize the circuit parameters and compute the energy
    print(job)
    
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

