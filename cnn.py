from data import get_data
import numpy as np
from conv import convolution_layer 
from activations import relu,softmax
from max_pool import max_pool
from fc import fully_connected_layer

class CNN:
    def __init(self,x):
        c1 = convolution_layer(x.shape,32,5)
        max1 = max_pool(c1.out(),2,2)
        c2 = convolution_layer(max1.out(),64,3)
        max2 = max_pool(c2.out(),2,2)
        flat = flatten(max1.out())
        fc1 = fully_connected_layer(flat.out(),1024,.25)
        out = fully_connected_layer(fc1.out(),2)

    def back_prop(self,outputs):
        print("Back propogation for CNN under construction")

    def feed_forward(self,x,train=False):
        # Convolutional layer one with relu activation into maxpooling 
        out_c1 = relu(self.c1.conv(x))
        max_pool_c1 = self.max1.max_pool(out_c1)

        # Convolutional layer two with relu activation into maxpooling 
        out_c2 = relu(self.c2.conv(max_pool_c1))
        max_pool_c2 = self.max2.max_pool(out_c2)

        # Fully connected layer one from flatten 
        fc1_out = relu(self.fc1.feed_forward(flat.flatten(max_pool_c2),train))

        # Final output layer 
        return softmax(self.out.feed_forward(fc1_out))
