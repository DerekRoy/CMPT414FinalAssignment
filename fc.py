import numpy as np
from activations import sigmoid_backprop, sigmoid, softmax

R = 0.1

class fully_connected_layer:
    def __init__(self, inpt, nodes, dropout=0):
        self.inpt = inpt
        self.nodes = nodes
        self.weights = np.random.normal(size=(nodes,inpt),scale=(2/(nodes*inpt)))
        self.biases = np.random.normal(size=(nodes, 1),scale=(2/(nodes*inpt)))

        self.cache_x = None
        self.cache_output = None

        self.drop_out = dropout

    # Return the weights matrix
    def get_weights(self):
        return self.weights

    # Set the values for the weights matrix
    def set_weights(self,w):
        self.weights = w

    # Return the bias array
    def get_bias(self):
        return self.biases

    # Set the values for the biases array
    def set_bias(self,b):
        self.biases = b

    # Return the output size
    def out(self):
        return self.nodes

    def feed_forward(self, x):
        output = np.dot(x,self.weights) + self.biases

        self.cache_x = x
        self.cache_output = output

        return softmax(output)

    def back_prop(self, desired_output, actual_output):
        desired_output = desired_output.reshape(actual_output.shape)
        dX = -(np.divide(desired_output, actual_output) - np.divide(1 - desired_output, 1 - actual_output))

        delta_output = sigmoid_backprop(dX, self.cache_output)
        delta_weights = np.dot(delta_output, self.cache_x.T) / self.cache_x.shape[1]
        delta_biases = np.sum(delta_output, axis=1, keepdims=True) / self.cache_x.shape[1]
        grad = np.dot(delta_weights.T, delta_output)

        self.weights -= R * delta_weights
        self.biases -= R * delta_biases

        return grad
