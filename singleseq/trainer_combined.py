# September 2022
# trainer code for simultaneous training of LSTM-GCN-MLP

import numpy as np
import torch
import os
from torch import nn
torch.cuda.empty_cache()
from model_with_GCN import Model
torch.manual_seed(1230)
np.random.seed(1230)
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
#import wandb
torch.autograd.set_detect_anomaly(True)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
if is_cuda: device = torch.device("cuda:0")
else: device = torch.device("cpu")

# Preprocess data to form a matrix
def preprocess(data, n, window_size):
    #Normalize data column-wise
    sc = MinMaxScaler(feature_range=(0, 1))
    training_data = sc.fit_transform(data)
    # structuring the data 
    target = [] 
    in_seq = []
    for i in range(n-window_size-1):
      list1 = []
      for j in range(i,i+window_size):
           list1.append(training_data[j])
      in_seq.append(list1)
      target.append(training_data[j+1])
    #print(np.array(in_seq).shape)
    #print(np.array(target).shape)
    return np.array(in_seq), np.array(target)

class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float64).to(device)
        self.y = torch.tensor(y,dtype=torch.float64).to(device)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

def init_weights(m):
    if isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight)


### Training the 3 models ###
def train(path, is_saved, regularization_param, epochs):
    print('\nTraining started...\n')

    # Input-data
    x = pd.read_csv(path,header=None,sep=',').to_numpy()
    #x = pd.read_csv(path,sep=',', index_col=0).to_numpy()  # for different data format
    data = np.array(x)
    print(data.shape)

    #data = np.loadtxt(path)
    d = np.shape(data)[1]  # number of variables
    n = np.shape(data)[0]  # number of time-steps
    
    window_size = 5
    input_size = 1 
    hidden_dim = 128
    n_layers = 5
    lr=1e-5
    epochs = epochs
    l1_lambda = regularization_param
    batch_size=128
    x_train, y_train = preprocess(data, n, window_size)
    dataset = timeseries(x_train,y_train)
    train_loader = DataLoader(dataset,shuffle=False,batch_size=batch_size, drop_last=True)

    # define the model
    model = Model(input_size=input_size, output_size=input_size, hidden_dim=hidden_dim, \
        n_layers=n_layers, n=d, batch_size=batch_size).to(device)

    filename1 = 'modelLSTM_GCN_MLP' 
    PATH1 = os.path.join(r'./saved_models/', filename1)
    if is_saved == 1: model.load_state_dict(torch.load(PATH1))  # can be used to start training from the saved model
    else: model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # keep log using wandb
    #wandb.init(project='New Simultaneous Training')
    #wandb.watch(model, log='all')

    criterion = torch.nn.MSELoss()
    L = []

    # train to update A
    for epoch in range(epochs):
        total_loss = 0
        e = 0
        model.train()
        for _, (in_seq, target) in enumerate(train_loader):
            optimizer.zero_grad()
            model.hidden = (torch.zeros(n_layers,batch_size,hidden_dim).to(device),
                            torch.zeros(n_layers,batch_size,hidden_dim).to(device))
            out = model(in_seq) # returns the output from final MLP layer

            # Calculate error
            mse = criterion(out, target)
            l1_norm = torch.norm(model.A_prime)
            loss2 = l1_lambda * l1_norm
            l3 =  l1_lambda * torch.sum(torch.square(model.A_prime-torch.mean(model.A_prime,0)))
            loss = mse + loss2 - l3
            loss.backward()    # Does backpropagation and calculates gradients
            
            # Updates the weights accordingly
            optimizer.step() 

            # to keep a record of total loss per epoch
            total_loss += loss.item()
            e += 1

        avg_loss = total_loss/e
        #scheduler3.step()
        
        if epoch%1 == 0:
            print('MSE: ', mse)
            print('l2_norm_groupwise: ', l1_norm)
            print('l3: ', l3)
            print('Epoch: {} .............'.format(epoch), end=' ')
            print("Loss: {:.4f}".format(avg_loss))
        L.append(avg_loss)
        #wandb.log({'Training Loss': avg_loss})

    plt.plot(L)
    #plt.yscale("log")
    plt.ylabel('Loss')
    #plt.show()

    torch.save(model.state_dict(), os.path.join(r'./saved_models/', 'modelLSTM_GCN_MLP'))   # save the trained model
    A_pred = model.A_prime

    print('\nFinished training.\n')
    print(A_pred)
    torch.save(model.A.data,'A_pred.pt')
    return A_pred.data


    