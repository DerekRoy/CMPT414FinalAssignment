import numpy as np

# Activation (Rectified Linear Unit): if value < 0: value = 0   
def relu(x):
    return np.maximum(0,x)

# Derivative of relu in regards to output
def relu_backprop(out):
    out[out > 0] = 1
    out[out < 0] = 0
    return out

# Activation (Leaky Rectified Linear Unit): if value < .01*value: value = .01*value   
def leaky_relu(x):
    return np.maximum(.01*x,x)

# Derivative of leaky relu in regards to output
def leaky_relu_backprop(out):
    out[out > 0] = 1
    out[out < 0] = .01
    return out

# Sigmoid activation
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid in regards to output
def sigmoid_backprop(out):
    return out*(1-out)

# Soft max to transform last fully connected layer in network 
def softmax(logits):
    exps = np.exp(logits - logits.max())
    return exps / np.sum(exps)

# Softmax derivative for back propogation
def softmax_backprop(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)
