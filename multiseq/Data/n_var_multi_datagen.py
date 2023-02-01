# July 2022
# function to generate n-variable data with random coeffients

import numpy as np
np.random.seed(1230)
import random

def generate_cosine(n, d, f, noise_scale):
    # n = no. of time steps, d = no. of variables, f = no. of sequences/features
    A = np.random.randint(2, size=(d,d)) # ground truth A
    W = np.random.rand(d,d) # randonly generated VAR coeff

    np.save('./A_true_15multi1', A)
    print('A:\n', A)
    #print('W:\n', W)
    #print('Constant: ', Constant)
    T = np.empty(shape=(d,n,f), dtype = 'float')
    T[:, 0:5, :] = np.random.rand(d,5,f) # generate with 5 sequences/features
    mu, sigma = 0, 1 # mean and standard deviation of noise

    for j in range(d):
        coeff = W[:,j] * A[:,j] #VAR coeff for each variable
        #print(j,'\n', coeff)
        E = []  # Generate f sequences of noise for each var
        for _ in range(f):
            E.append(np.random.normal(mu, sigma, n))    #shape=[f,n]
        E = np.array(E)
        #print(E)
        for i in range(5,n):
            for q in range(f):
                y = sum(coeff[k]*(3*np.cos(T[k,i-5,q]+1)+5*np.cos(T[k,i-4,q]+1)+2*np.cos(T[k,i-3,q]+1)+np.cos(T[k,i-2,q]+1)\
                    +8*np.cos(T[k,i-1,q]+1)) for k in range(d))   # all calculations are done in radian
                #print('y: ', y)
                t = y + noise_scale*E[q,i]*sum(A[:,j])
                #print('E: ', E[q,i])
                #print('t: ', t)
                T[j,i,q] = t
    #print(T.shape)
    return np.array(T)


if __name__ == '__main__': 
    n=3000
    f=5
    d=15
    T = generate_cosine(n,d,f, 0.05)
    print(T.shape)
    # reshaping the array from 3D
    # matrice to 2D matrice.
    T_reshaped = T.reshape(T.shape[0], -1)
    print(T_reshaped.shape)
    
    np.savetxt('15_multi1_noise0.05.txt', T_reshaped, delimiter=',')
    
# Notes:
# 15_multi1_noise0.05.txt => multi version of 15_VAR1_noise0.05.txt
# files will be saved in the ./Data directory
# generated data is of shape: (n x d*f), where n = #time-steps, d = #Variables, f = #features
# with no index and header, comma-seperated file
# will be reshaped automatically before training to convert to (n x d x f)

