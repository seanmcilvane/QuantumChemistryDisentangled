# QuantumChemistryDisentangled





## Usage
```
python VQE.py --part 0 --positive_energy_flag
```


Using on a different device and on different part (Note that the lightning.qubit device requires Linux) 
```
python VQE.py --device lightning.qubit --part 3 --postive_energy_flag
```
## List of compatible devices

default.qubit : Pennylane's default.qubit simulator \
lightning.qubit : Pennylane's lightning.qubit simulator that supports adjoint differentiation, requires Linux \
ionqdevice : Use AWS Braket to run on IonQ's ionQdevice \
lucy: Use AWS braket to run on OQC's Lucy \
sv1: Use AWS braket to run on AWS's stave vector simulator \
aspen.m2 : Use AWS Braket to run on Rigetti's Aspen-M-2 \
aspen.m3 : Use AWS Braket to run on Rigetti's Aspen-M-3 \
qiskit.ibmq: Use IBMQ to access the ibmq qasm simulator \

If a not compatible device is specified, Pennylane's default.qubit will be used. 

## Using AWS Braket

Install the AWS CLI and run:

```
aws configure
```
Enter your access key and secret access key corresponding to your AWS account. 

Run using desired device
```
python VQE.py --device aspen.m2 --part 2 --positive_energy_flag 
```

## Displaying Resuts

Open the Example.ipynb notebook where functions plot_results and print_results are defined to help you visualize the results. 

Output for plot_results




Output for print_results
