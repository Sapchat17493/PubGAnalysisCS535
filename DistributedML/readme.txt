Run 'analysis.py' to perform network training

To run, log into machine which is set as MASTER_ADDR => Run using 'python3' bash command; 4 command line arguments are also necessary: {machine number}, {total number of machines}, {fpp/tpp} and {train/test}
machine number should start from 0, all the way up to (total number of machines - 1); number 0 should be run only MASTER_ADDR, all other numbers can be run on any other machine
For fpp, use 1, for tpp, use 2
For training use 0, for testing use 1


'mlutils.py' contains code to partition dataset into training and testing sets

'neuralnetworks.py' contains code for neural network creation, training and testing

'neuralnetworks_distributed.py' contains code for distributed training; an extension of neuralnetworks