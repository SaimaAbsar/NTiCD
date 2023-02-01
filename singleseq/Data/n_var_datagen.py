# Oct 2022
# function to generate n-variable data with random coeffients

import numpy as np
np.random.seed(9000)

def generate_cosine(n, d, noise_scale):

    A = np.random.randint(2, size=(d,d)) 
    W = np.random.rand(d,d) 
    np.save('./A_true_50VAR', A)    # save the ground truth A with appropriate name
    print('A:\n', A)
    T = np.empty(shape=(n,d), dtype = 'float')
    T[0:5, :] = np.random.rand(5,d) 

    for j in range(d):
        # Generate noise for each variable
        mu, sigma = 0, 1 # mean and standard deviation
        e = np.random.normal(mu, sigma, n)
        coeff = W[:,j] * A[:,j]
        #print(coeff)
        for i in range(5,n):
            #print(T[i-5:i,:].shape)
            y = sum(coeff[k]*(5*np.cos(T[i-5,k]+1)+2*np.cos(T[i-4,k]+1)+3*np.cos(T[i-3,k]+1)+8*np.cos(T[i-2,k]+1)\
                +np.cos(T[i-1,k]+1)) for k in range(d))
            t = y + noise_scale*sum(A[:,j])*e[i]  
            T[i,j] = t
    return np.array(T)


if __name__ == '__main__': 
    n = 3000
    d = 50
    noise = 0.05
    T = generate_cosine(n,d,noise)
    print(T.shape)
    np.savetxt('50_VAR_noise0.05.txt', T, delimiter=',')

# Note: this file generates non-linear data with it ground truth
# files will be saved in the ./Data directory
# generated data is of shape: (n x d), where n = # time-steps, d = # Variables; 
# with no index and header, comma-seperated file


