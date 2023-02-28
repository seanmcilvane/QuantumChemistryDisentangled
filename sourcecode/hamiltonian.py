import pennylane as qml
from pennylane import numpy as np

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
            #op_penny =  qml.Identity(wires=0)
            for i, pauli in enumerate(op):

                if i == 0:
                  
                    if pauli == 'I':
                        op_penny = qml.Identity(wires=i) 
                    elif pauli == 'X':
                        op_penny = qml.PauliX(wires=i) 
                    elif pauli == 'Y':
                        op_penny = qml.PauliY(wires=i) 
                    elif pauli == 'Z':
                        op_penny = qml.PauliZ(wires=i)   
                    continue
                
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
