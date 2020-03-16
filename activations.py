import numpy as np

# Activation (Rectified Linear Unit): if value < 0: value = 0   
def relu(x):
    return np.maximum(0,x)

# Soft max to transform last fully connected layer in network 
def softmax(logits):
    return np.exp(logits)/np.sum(np.exp(logits), axis=0)
