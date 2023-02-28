import pennylane as qml
from pennylane import numpy as np
import argparse
from timeit import default_timer as timer
import json
from hamiltonian import create_hamiltonian

start = timer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=str, default="default.qubit")
    parser.add_argument('--molecule', type=str, default="beh2")
    parser.add_argument('--part', type=int, default= 0)
    #parser.add_argument('--source_path', type=str, default=f'./hamiltonian_data/{args.molecule}/{args.part}.txt')
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--positive_energy_flag', action='store_true')
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--skip_final_rotation_layer', action='store_true')
    args = parser.parse_args()
    
    

    if args.source_path is not None:
        source = args.source_path
    else:
        source = f'./hamiltonian_data/{args.molecule}/part{args.part}.txt'

    print("source_path:", source)
    print("molecule:", args.molecule)
    print("Hamiltonian Part:", args.part)

    H = create_hamiltonian(source)
    wires = list(H.wires)
    qubits=len(wires)
    np.random.seed(42)

    file_manager = {"default.qubit": ["pennylane", "default-qubit"],
                        "lightning.qubit": ["pennylane", "lightning-qubit"],
                        "ionqdevice": ["braket", "ionqdevice"],
                        "lucy" : ["braket", "lucy"],
                        "sv1": ["braket", "sv1"],
                        "aspen.m2" : ["braket", "aspen-m2"],
                        "aspen.m3" : ["braket", "aspen-m3"],
                        "qiskit.ibmq": ["ibm", "qasm_simulator"] }
  
    # Define the device
    if args.device == "default.qubit":
        device = qml.device("default.qubit",
                            wires=qubits)
    
    elif args.device == "lightning.qubit":
        device = qml.device("lightning.qubit",
                            wires=qubits)

    elif args.device == "ionqdevice":
        device =  qml.device("braket.aws.qubit", 
                            device_arn = 'arn:aws:braket:::device/qpu/ionq/ionQdevice',
                            wires = qubits)

    elif args.device == "lucy":
        device = qml.device('braket.aws.qubit', 
                            device_arn = 'arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy',
                            wires = qubits) 

    elif args.device == "sv1":
        device = qml.device('braket.aws.qubit', 
                            device_arn = 'arn:aws:braket:::device/quantum-simulator/amazon/sv1',
                            wires = qubits) 

    elif args.device == "aspen-m2":
        device = qml.device('braket.aws.qubit', 
                            device_arn = 'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2',
                            wires = qubits) 
    
    elif args.device == "aspen-m3":
        device = qml.device('braket.aws.qubit', 
                            device_arn = 'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3',
                            wires = qubits) 
        
    elif args.device== "qiskit.ibmq":
        device = qml.device('qiskit.ibmq', wires = wires, backend = 'ibmq_qasm_simulator')
    
    else:
        print("not valid device, resorting to default.qubit")
        device = qml.device("default.qubit",
                            wires=qubits)
        

    
    
    print(device)
    # Define the qnode
    if args.device == "lightning.qubit":
        @qml.qnode(device, diff_method="adjoint")
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
    else: 
        @qml.qnode(device)
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

    # Define the optimizer
    optimizer = qml.AdamOptimizer(stepsize=0.1)

    # Optimize the circuit parameters and compute the energy
    prev_energy = 0
    n_list = []
    energy_list = []

    for n in range(100):
        params, energy = optimizer.step_and_cost(cost_function, params,
                                                wires=range(qubits), reps=args.reps, 
                                                skip_final_rotation_layer=args.skip_final_rotation_layer)

        
        if args.positive_energy_flag:
            energy *= -1
        
        n_list.append(n)
        energy_list.append(energy)

        print("step = {:},  E = {:.8f}".format(n, energy))
        if abs(energy - prev_energy) < 0.0000000005: # depending on precision
            break
        prev_energy = energy
    
    energy_list = [float(energy_list[i]) for i in range(len(energy_list))]

    end = timer()
    results_dict = {"n_iterations": n_list,
                     "energy_list": energy_list,
                     "run_time": end-start}

    results_path = f'../results/{args.molecule}/part{args.part}/{file_manager[args.device][0]}-{file_manager[args.device][1]}.txt'
    print("Run time:", end - start, "seconds")
    print("Results stored in", results_path)
    with open(results_path, 'w+') as results_file:
        results_file.write(json.dumps(results_dict))
