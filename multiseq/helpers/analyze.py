import numpy as np

def convert_logits_to_sigmoid(W, tau=1.0):
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    d = W.shape[0]
    W = np.copy(W)
    sigmoid_W = sigmoid(W/tau)
    #sigmoid_W[np.arange(d), np.arange(d)] = 0    # Mask diagonal to 0
    return sigmoid_W

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


# function to transform A by gumbel sigmoid
def sample_gumbel(shape, eps=1e-20):
    U = np.random.uniform(0, 1, shape)
    return -np.log(-np.log(U + eps) + eps)
def gumbel_sigmoid(logits, temperature):
    gumbel_softmax_sample = logits + sample_gumbel(np.shape(logits)) - sample_gumbel(np.shape(logits))
    y = sigmoid(gumbel_softmax_sample / temperature)
    return y
