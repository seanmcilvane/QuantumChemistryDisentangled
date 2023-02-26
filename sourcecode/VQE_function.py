
import pennylane as qml
from pennylane import numpy as np
from VQE_braket import create_hamiltonian


def run_vqe(device:str, source_path:str, positive_energy_flag:str, reps:int, skip_final_rotation_layer:str):

    H = create_hamiltonian(source_path)

    wires = list(H.wires)
    qubits=len(wires)

    np.random.seed(42)

    # Define the device
    if device == "braket.aws.qubit":
        dev = qml.device(device, 
                        device_arn='arn:aws:braket:::device/quantum-simulator/amazon/sv1',
                        wires=qubits)

    else: 
        dev = qml.device(device, wires=qubits)

    # Define the qnode
    @qml.qnode(dev) 
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
    
    def cost_function(params, wires, reps, skip_final_rotation_layer):
        h_expval = circuit(params = params,
                         wires = wires,
                         reps = reps,
                        skip_final_rotation_layer= skip_final_rotation_layer)
        coef = 1
        if positive_energy_flag == "True":
            coef = -1
        return coef * h_expval
    
    nr_params = (reps+1)*len(wires)*2
    if skip_final_rotation_layer == "True":
        nr_params -= len(wires)*2

    # Define the initial values of the circuit parameters
    params = np.random.normal(0, np.pi, nr_params)

    # Define the optimizer
    optimizer = qml.AdamOptimizer(stepsize=0.1)

    # Optimize the circuit parameters and compute the energy
    prev_energy = 0


    n_list = []
    energy_list = []

    for n in range(1000):
        params, energy = optimizer.step_and_cost(cost_function, params,
                                                wires=range(qubits), reps=reps, 
                                                skip_final_rotation_layer=skip_final_rotation_layer)

        
        if positive_energy_flag == "True":
            energy *= -1

        n_list.append(n)
        energy_list.append(energy)
        
        #print("step = {:},  E = {:.8f}".format(n, energy))
        if abs(energy - prev_energy) < 0.0000000005: # depending on precision
           break
        prev_energy = energy

    return n_list, energy_list
