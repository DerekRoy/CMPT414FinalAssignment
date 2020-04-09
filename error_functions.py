import numpy as np

# Cross entropy error function 
def cross_entropy(t,y):
    return np.sum(t*np.log2(y))*-1

# Derivative of Cross entropy for backprop
def backprop_cross_entropy(t,y):
    return -t/y

# Root mean square error function 
def rmse(t,y):
    return np.sqrt(sum_sqaured_error(y,t)/len(t))

# Sum squared error function
def sum_sqaured_error(t,y):
    return .5*np.sum(np.power((t-y),2))
