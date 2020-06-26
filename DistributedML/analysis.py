import os
import sys
import traceback
import datetime
import socket

import numpy as np
import pandas as pd
from random import Random
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from hdfs import InsecureClient

import neuralnetworks_distributed as nn
import mlutils as ml


class Partition(object):
    
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
    

class DataPartitioner(object):
    
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
            
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
    

def partition_dataset(data):
    size = dist.get_world_size()
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(data, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = np.array(list(partition))
    return train_set

def run_train(rank, size, mode):
    
    client = InsecureClient('http://juneau:46731', user='sapchat') # HDFS Web UI port!!
    with client.read("/pubg/aggregate/agg_match_stats_0.csv") as f:
        df = pd.read_csv(f, usecols=[1, 3, 4, 9, 12]).replace(to_replace={'tpp': 2, 'fpp': 1}, value=None)
    with client.read("/pubg/aggregate/agg_match_stats_1.csv") as f:
        temp =  pd.read_csv(f, usecols=[1, 3, 4, 9, 12]).replace(to_replace={'tpp': 2, 'fpp': 1}, value=None)
        df = df.append(temp, ignore_index=True)
    with client.read("/pubg/aggregate/agg_match_stats_2.csv") as f:
        temp =  pd.read_csv(f, usecols=[1, 3, 4, 9, 12]).replace(to_replace={'tpp': 2, 'fpp': 1}, value=None)
        df = df.append(temp, ignore_index=True)
    with client.read("/pubg/aggregate/agg_match_stats_3.csv") as f:
        temp =  pd.read_csv(f, usecols=[1, 3, 4, 9, 12]).replace(to_replace={'tpp': 2, 'fpp': 1}, value=None)
        df = df.append(temp, ignore_index=True)
    with client.read("/pubg/aggregate/agg_match_stats_4.csv") as f:
        temp =  pd.read_csv(f, usecols=[1, 3, 4, 9, 12]).replace(to_replace={'tpp': 2, 'fpp': 1}, value=None)
        df = df.append(temp, ignore_index=True)
    
    #df = pd.read_csv('agg_match_stats_0.csv', usecols=[1, 3, 4, 9, 12], nrows=50000).replace(to_replace={'tpp': 2, 'fpp': 1}, value=None) # local read instead of through HDFS
    print(f'Shape of data read: {df.shape}')
    
    df = df[df['player_survive_time'] < 2500] # removing outlier survival times
    if mode == 1:
        X = df[df['match_mode'] == 1].drop(columns=['match_mode']).values.astype('double')
        T = df[df['match_mode'] == 1].iloc[:, 4:].values.astype('double').reshape(-1, 1)
    if mode == 2:
        X = df[df['match_mode'] == 2].drop(columns=['match_mode']).values.astype('double')
        T = df[df['match_mode'] == 2].iloc[:, 4:].values.astype('double').reshape(-1, 1)
    #print(f'X.shape: {X.shape}, T.shape: {T.shape}')
    
    frac = 0.8
    X_train, X_test, T_train, T_test = ml.partition(X, T, frac)
    train = partition_dataset(np.concatenate((X_train, T_train), axis=1))
    X_train, T_train = train[:, :4], train[:, 4:]
    
    network = [5]
    relu = True
    n_iterations = 500
    batch_size = 67000
    learn_rate = 10**-5
    
    Qnet = nn.NN_distributed(X_train.shape[1], network, T_train.shape[1], relu)
    net, err = Qnet.train_pytorch(X_train, T_train, n_iterations, batch_size, learn_rate, verbose=True)
    
    print(f'Final Train RMSE error: {err[-1].detach().cpu().numpy()}, training time: {net.time}')
    Y_test = net.use_pytorch(X_test) # predictions
    RMSE_net = np.sqrt(np.mean((Y_test-T_test)**2)) # errors = predictions - targets
    print(f'Test RMSE: {RMSE_net}')
    print(f'Sample Target: {T_test[0][0]}, Predicted Value: {net.use_pytorch(X_test[0])[0]}') # sample prediction
    
    model = nn.NN_distributed(X_train.shape[1], network, T_train.shape[1], relu)
    if mode == 1:
        model.load_state_dict(torch.load('Best network (FPP).pth'))
    if mode == 2:
        model.load_state_dict(torch.load('Best network (TPP).pth'))
    Y_test_best = model.use_pytorch(X_test)
    RMSE_model = np.sqrt(np.mean((Y_test_best-T_test)**2))
    print(f'Best network Test RMSE: {RMSE_model}')
    if RMSE_net < RMSE_model/2:
        n_epochs = len(err)
        fig = plt.figure(figsize=(12, 12))
        plt.plot(list(range(1, n_epochs+1)), err)
        plt.xlim(1-0.05*n_epochs, n_epochs*1.05)
        plt.xlabel('Epochs')
        plt.ylabel('Train RMSE')
        if mode == 1:
            torch.save(net.state_dict(), 'Best network (FPP).pth')
            plt.savefig('Error rate - best network (FPP).png')
        if mode == 2:
            torch.save(net.state_dict(), 'Best network (TPP).pth')
            plt.savefig('Error rate - best network (TPP).png')
        print('Saving as new best network')

def run_test(mode):
    client = InsecureClient('http://juneau:46731', user='sapchat') # HDFS Web UI port!!
    with client.read("/pubg/aggregate/agg_match_stats_0.csv") as f:
        df = pd.read_csv(f, usecols=[1, 3, 4, 9, 12], nrows=50000).replace(to_replace={'tpp': 2, 'fpp': 1}, value=None)
    #df = pd.read_csv('agg_match_stats_0.csv', usecols=[1, 3, 4, 9, 12], nrows=50000).replace(to_replace={'tpp': 2, 'fpp': 1}, value=None)
    if mode == 1:
        X = df[df['match_mode'] == 1].drop(columns=['match_mode']).values.astype('double')
        T = df[df['match_mode'] == 1].iloc[:, 4:].values.astype('double').reshape(-1, 1)
    if mode == 2:
        X = df[df['match_mode'] == 2].drop(columns=['match_mode']).values.astype('double')
        T = df[df['match_mode'] == 2].iloc[:, 4:].values.astype('double').reshape(-1, 1)
    
    network = [5]
    relu = True
    model = nn.NN_distributed(X.shape[1], network, T.shape[1], relu)
    if mode == 1:
        model.load_state_dict(torch.load('Best network (FPP).pth'))
    if mode == 2:
        model.load_state_dict(torch.load('Best network (TPP).pth'))
    Y = model.use_pytorch(X)
    RMSE_model = np.sqrt(np.mean((Y-T)**2))
    print(f'Best Network Test RMSE: {RMSE_model}')
    for i in range(1000, 5000, 500):
        print(f'Sample Target {i}: {T[i][0]}, Predicted Value: {model.use_pytorch(X[i])[0]}')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'anchovy' # change to run on different machine
    os.environ['MASTER_PORT'] = '35234'
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size), init_method='tcp://anchovy:17161', timeout=datetime.timedelta(weeks=120)) # ensure tcp://{MASTER_ADDR}
    torch.manual_seed(32) #??

if __name__ == '__main__':
    try:
        setup(sys.argv[1], sys.argv[2])
        print(f'{socket.gethostname()}: Setup completed!')
        if int(sys.argv[4]) == 0:
            run_train(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        if int(sys.argv[4]) == 1:
            run_test(int(sys.argv[3]))
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
