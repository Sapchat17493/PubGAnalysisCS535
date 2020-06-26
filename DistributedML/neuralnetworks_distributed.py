import torch
import torch.distributed as dist
from torch.autograd import Variable
import time

import neuralnetworks as nn

class NN_distributed(nn.NN):
    
    def average_gradients(self):
        size = float(dist.get_world_size())
        for param in self.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
    
    def train_pytorch(self, X, T, n_iterations, batch_size, learning_rate=10**-3, use_SGD=False, verbose=False):
        start_time = time.time()
        X, T = self.process(X, T)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) if not use_SGD else torch.optim.SGD(self.parameters(), lr=learning_rate)
        loss_func = torch.nn.MSELoss()
        errors = []
        n_examples = X.shape[0]
        for i in range(n_iterations):
            num_batches = n_examples//batch_size
            for j in range(num_batches):
                start, end = j*batch_size, (j+1)*batch_size
                X_batch, T_batch = Variable(X[start:end, ...], requires_grad=False), Variable(T[start:end, ...], requires_grad=False)
                # Forward pass
                outputs = self(X_batch)
                loss = loss_func(outputs, T_batch)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                self.average_gradients()
                optimizer.step()
            errors.append(torch.sqrt(loss)) # Error at end of iteration
            if verbose:
                print(f'Iteration {i+1} training completed. Error rate: {errors[-1]}')
        self.time = time.time()-start_time
        return self, errors
