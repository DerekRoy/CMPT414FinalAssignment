# Test run code:
#     from data import get_data
#     image = get_data()[0][0]
#     d = discriminator(image)
#     d.run(image)

from conv import convolution_layer 
from activations import leaky_relu,softmax,leaky_relu_backprop
from flatten import flatten
from fc import fully_connected_layer
from error_functions import backprop_cross_entropy
import numpy as np
import pickle

class CNN:
    def __init__(self,image):
        # Create place holder variables
        self.image = []
        self.c1_out = []
        self.c2_out = []
        self.output = []
        
        # Set initialized convolutional layers
        self.c1 = convolution_layer(image.shape,64,5,2,True)
        self.c2 = convolution_layer(self.c1.out_dim,128,5,2,True)
        self.flat = flatten(self.c2.out_dim)
        self.dense = fully_connected_layer(self.flat.output_shape[0],10)
    
    def feed_forward(self,image):
        # Image input
        self.image = image
        
        # Convolutional layer 1
        self.c1_out = self.c1.conv(image)
        self.c1_activation = leaky_relu(self.c1_out)
        
        # Convolutional layer 2
        self.c2_out = self.c2.conv(self.c1_activation)
        self.c2_activation = leaky_relu(self.c2_out)
        
        # Flatten out 
        self.flattened = self.flat.flatten(self.c2_activation)
        
        # Out Layer 
        self.logits = self.dense.feed_forward(flattened)
        self.output = softmax(self.logits)
        
        return self.output
    
    # Run prediction 
    def predict(self,image):
        return self.feed_forward(image)
    
    # Save the model weights to a pickle file
    def save_model(self, name="weights"):
        # Put weights into dictionary to be saved
        weights = {"c1_weights":self.c1.get_filters(),"c2_weights":self.c2.get_filters(),"dense_weights":self.dense.get_weights()}
        
        # Open file and save with pickle
        with open(name, 'wb') as pickle_file:
            pickle.dump(weights, pickle_file)
        print("Model Saved")
   
    # Load a model from a pickle file
    def load_model(self,name="weights"):
        with open(name, 'rb') as f:
            weights = pickle.load(f)
        
        # Set all the weights from the pickle file
        self.c1.set_filters(weights['c1_weights'])
        self.c2.set_filters(weights['c2_weights'])
        self.dense.set_weights(weights['dense_weights'])
        print("Model Loaded")
