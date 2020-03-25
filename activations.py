import numpy as np

# Activation (Rectified Linear Unit): if value < 0: value = 0   
def relu(x):
    return np.maximum(0,x)

# Activation (Leaky Rectified Linear Unit): if value < .01*value: value = .01*value   
def leaky_relu(x):
    return np.maximum(.01*x,x)

# Sigmoid activation
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Soft max to transform last fully connected layer in network 
def softmax(logits):
    return np.exp(logits)/np.sum(np.exp(logits), axis=0)
