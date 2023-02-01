# Septmber 2022
# module for multi-variable input: LSTM+GCN+MLP 
# implemented for multi-sequence

import numpy as np
import torch
torch.manual_seed(1230)
from torch import nn
np.random.seed(1230)
torch.set_default_dtype(torch.float64)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
if is_cuda: device = torch.device("cuda:0")
else: device = torch.device("cpu")


class Model(nn.Module):
    def __init__(self, batch_size, input_size, output_size, hidden_dim, n_layers, n):
        super(Model, self).__init__()

        # Defining some parameters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.no_of_nodes = n
        adj = torch.load('A_pred.pt').to(device)  # can be used to resume training if accidentally stopped
        self.A = nn.Parameter(adj, requires_grad = True)  # Adjacency matrix as parameter to update during training
        self.gcn_lin_dim = hidden_dim

        #Defining the layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, self.hidden_dim, num_layers=n_layers, batch_first=True)  

        #GCN
        self.gcn_lin1 = nn.Sequential(
          nn.Linear(self.hidden_dim, self.gcn_lin_dim, bias=False))

        self.gcn_lin2 = nn.Sequential(
          nn.Linear(self.hidden_dim, self.gcn_lin_dim, bias=False))

        self.gcn_lin3 = nn.Sequential(
          nn.Linear(self.gcn_lin_dim, self.gcn_lin_dim, bias=False))
        
        self.relu = torch.nn.ReLU(inplace=False)  # element-wise

        # MLP layers
        self.fc = nn.Sequential(
          nn.Linear(self.gcn_lin_dim, output_size, bias=False),
          nn.Sigmoid())

        print("\nInitial A:\n")
        print(self.A.data)

    
    def forward(self, x):         
        batch_size = x.size(0)
        # LSTM
        h_LSTM = torch.empty((batch_size, self.hidden_dim, self.no_of_nodes)).to(device)
        self.hidden = self.init_hidden(batch_size)
        for j in range(self.no_of_nodes):
            #print('x input to LSTM: ', x[:,j,:,:].shape)
            lstm_out, (h_n,c_n)= self.lstm(x[:,j,:,:], self.hidden)
            #print('h_n shape inside forward: ', h_n.shape)
            h_LSTM[:, :, j] = torch.squeeze(h_n)[-1]  #shape = [batch, hidden_dim, no_of_nodes]
        #print('after reshaping: ', h_LSTM.shape)

        # GAT
        self.A_prime = torch.sigmoid(self.A)
        H_times_A= torch.einsum('ikj,jl->ikl', h_LSTM, self.A_prime)   #shape = [batch, hidden_dim, no_of_nodes]
        
        #GCN
        alpha = 0.9   #parameter for balance of self-information

        h_A = H_times_A.permute(0, 2, 1)  #shape = [batch, no_of_nodes, hidden_dim]
        W1_h_A = self.gcn_lin1(h_A) #shape = [batch, no_of_nodes, self.gcn_lin_dim]
        W1_h_A = W1_h_A.permute(0, 2, 1)   #shape = [batch, self.gcn_lin_dim, no_of_nodes]   
        ################### updated GCN with alpha
        W2_h_A = self.gcn_lin2(h_A) #shape = [batch, no_of_nodes, self.gcn_lin_dim]
        Relu_W2_h_A = self.relu(W2_h_A)
        W3_relu_HAW1 = self.gcn_lin3(Relu_W2_h_A) #shape = [batch, no_of_nodes, self.gcn_lin_dim]
        W3_relu_HAW1 = W3_relu_HAW1.permute(0, 2, 1)  #shape = [batch, self.gcn_lin_dim, no_of_nodes]
        AT_times_W3h= torch.einsum('ikj,jl->ikl', W3_relu_HAW1, self.A_prime)       #shape = [batch, self.gcn_lin_dim, no_of_nodes]  

        h_out = (1-alpha) * AT_times_W3h + alpha * W1_h_A   #shape = [batch, no_of_nodes, self.gcn_lin_dim]  # weighter summation version  

        # MLP
        out_MLP = torch.empty((batch_size, self.no_of_nodes, self.output_size)).to(device)
        for k in range(self.no_of_nodes):
            ma = h_out[:, :, k]
            out_MLP[:,k,:] = self.fc(ma)
        return (out_MLP)

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(device),
                            torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(device))
        return hidden

