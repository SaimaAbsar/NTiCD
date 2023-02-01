# September 2022
# Idea: Simultaneous model to train LSTM-GCN-MLP

import numpy as np
import torch
import argparse
torch.cuda.empty_cache()
import trainer_combined as trainer
torch.manual_seed(10)
np.random.seed(10)
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda:0")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

st = time.time() #start time
if __name__ == '__main__':    
    # Need to provide the input data directory as an argument
    parser = argparse.ArgumentParser("input info")
    parser.add_argument("--datapath", type=str, help="path to input directory")
    parser.add_argument("--d", type=int, help="Number of variables")
    args = parser.parse_args()
    data_dir = args.datapath
    d = args.d
    A_true = np.array([[1, 1, 0, 1,0,0], [0,0,0,0,0,0], [0, 1, 1, 0,0,1], [0,1,0,0,0,0], [0,0,1,0,1,1], [1,0,0,1,0,0]])  # 6_VAR_noise0.1.txt
    #A_true = np.load('./Data/A_true_50VAR4.npy')   # load the corrsponding g.t. file
    print(A_true)
    A_pred = torch.Tensor(1*np.zeros([d,d], dtype=np.double) + np.random.uniform(low=0, high=0.1, size=(d,d)))
    torch.save(A_pred,'./A_pred.pt')  #A_pred initialized to very small values, so that sigmid(A_pred)=0.5 initially; will be updated during training

    acc = 0
    mse = []
    reg_param = 2e-3    # regularization parameter
    epochs = 3000
    # training
    A_pred = trainer.train(path=data_dir, is_saved=0, regularization_param=reg_param, \
        epochs=epochs) 
    # the 'is_saved' parameter can resume training from the saved model if set to 1

    A_pred = A_pred.cpu().numpy()
    graph_thres = np.mean(A_pred,0) # columnwise mean to calculate different threshold for each variable
    print("Graph Threshold: ", graph_thres)

    # Evaluation
    A_pred[np.abs(A_pred) < graph_thres] = 0    
    A_pred[np.abs(A_pred) >= graph_thres] = 1
    acc = np.sum(A_pred == A_true)/(len(A_true)*len(A_true)) * 100 
    shd = np.count_nonzero(A_true!=A_pred)
    precision = precision_score(A_true.flatten(),A_pred.flatten())
    recall = recall_score(A_true.flatten(),A_pred.flatten())
    f1 = f1_score(A_true.flatten(),A_pred.flatten())
    print('A true: \n', A_true)
    print('A predicted: \n', A_pred)
    print('\ncurrent lambda value: ', reg_param)
    print('\nPercentage accuracy: ', acc)
    print('Current shd: ', shd)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1)

    np.savetxt('A_pred.txt', A_pred, delimiter=',') # the predicted adjacency matrix is saved in the current directory

# Get the end time
et = time.time()
# Get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print('Results of GCN sum version')