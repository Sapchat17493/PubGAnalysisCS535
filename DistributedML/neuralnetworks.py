import torch
from torch.autograd import Variable
import time

class NN(torch.nn.Module):
    
    def __init__(self, n_inputs, network, n_outputs, relu=False, gpu=True):
        super(NN, self).__init__()
        network_layers = [torch.nn.Linear(n_inputs, network[0])]
        if len(network) > 1:
            network_layers.append(torch.nn.Tanh() if not relu else torch.nn.ReLU())
            for i in range(len(network)-1):
                network_layers.append(torch.nn.Linear(network[i], network[i+1]))
                network_layers.append(torch.nn.Tanh() if not relu else torch.nn.ReLU())
        network_layers.append(torch.nn.Linear(network[-1], n_outputs))
        self.model = torch.nn.Sequential(*network_layers)
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).double()
        self.processed = False
    
    def tensor(self, np_array):
        return torch.from_numpy(np_array.astype('double')).cuda() if torch.cuda.is_available() else torch.from_numpy(np_array.astype('double')) # Return tensor for Torch
    
    def standardise(self, data, mean, sd):
        return (data-mean)/sd
    
    def process(self, X, T):
        X, T = self.tensor(X), self.tensor(T)
        if not self.processed:
            self.processed = True
            self.Xmeans, self.Xstds, self.Tmeans, self.Tstds = X.mean(dim=0), X.std(dim=0), T.mean(dim=0), T.std(dim=0)
        return self.standardise(X, self.Xmeans, self.Xstds), self.standardise(T, self.Tmeans, self.Tstds) # Return standardised inputs
    
    def forward(self, X):
        return self.model(X) # Output of forward pass is passing data through the model
    
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
                optimizer.step()
            errors.append(torch.sqrt(loss)) # Error at end of iteration
            if verbose:
                print(f'Iteration {i+1} training completed. Error rate: {errors[-1]}')
        self.time = time.time()-start_time
        return self, errors
    
    def use_pytorch(self, X):
        X = self.tensor(X)
        with torch.no_grad():
            return self(X).cpu().numpy() if torch.cuda.is_available() else self(X).numpy() # Return Y
