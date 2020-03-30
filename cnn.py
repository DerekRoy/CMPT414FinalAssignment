# Test run code:
#     from data import get_data
#     image = get_data()[0][0]
#     d = discriminator(image)
#     d.run(image)

from conv import convolution_layer 
from activations import leaky_relu,softmax
from flatten import flatten
from fc import fully_connected_layer

class discriminator:
    def __init__(self,image):
        c1 = convolution_layer(image.shape,64,5,2,True)
        c2 = convolution_layer(c1.out_dim,128,5,2,True)
        flat = flatten(c2.out_dim)
        dense = fully_connected_layer(flat.output_shape[0],10)
    
    def run(self,image):
        c1_out = c1.conv(image)
        c1_relu = leaky_relu(c1_out)
        c2_out = c2.conv(c1_relu)
        c2_relu = leaky_relu(c2_out)
        flattened = flat.flatten(c2_relu)
        return softmax(dense.feed_forward(flattened))
