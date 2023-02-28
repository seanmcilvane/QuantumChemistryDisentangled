# QuantumChemistryDisentangled





Using the 
'''
python VQE.py --part 0 --positive_energy_flag
'''


Using on a different device and on different part (Note that the lightning.qubit device requires Linux) 
'''
python VQE.py --device lightning.qubit --part 3 --postive_energy_flag


## Using AWS Braket

Install the AWS CLI and run:

'''
aws configure
'''
Enter your access key and secret access key corresponding to your AWS account. 

Run using desired device
'''
python VQE.py --device aspen.m2 --part 2 --positive_energy_flag 
'''

