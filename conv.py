# Sample usage 
# c1 = convolution_layer(image.shape,2,3,True)
# feature_maps = c1.conv(image)
import numpy as np

class convolution_layer:
    def __init__(self, inpt, num_filters, kernel_size, testing):
        self.in_dim = inpt
        self.out_dim = None

        # Output shape
        if len(self.in_dim) > 2:
            self.out_dim = (self.in_dim[0]-kernel_size+1,self.in_dim[1]-kernel_size+1,self.in_dim[2]*num_filters)
        else:
            self.out_dim = (self.in_dim[0]-kernel_size+1,self.in_dim[1]-kernel_size+1,num_filters)
        
        # Weights and biases
        self.filters = np.random.normal(size=(num_filters,kernel_size,kernel_size),scale=(2/(num_filters*kernel_size**2))) # Glorot initiaized, size num_filters of size (kernel_size x kernel_size)
        self.bias = np.zeros((self.out_dim[2])) # initializ bias to 0
        
        if testing: 
            #      Section for testing 0 and 90 sobel filters      # 3x3
            if kernel_size == 3:
                self.filters[0] = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                self.filters[1] = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            ########################################################

            #      Section for testing 0 and 90 sobel filters      # 5x5
            if kernel_size == 5:
                self.filters[0] = np.array([[-1,0,0,0,1],[-1,0,0,0,1],[-1,0,0,0,1],[-1,0,0,0,1],[-1,0,0,0,1]])
                self.filters[1] = np.array([[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[-1,-1,-1,-1,-1]])
            ########################################################

        self.num_filters = num_filters
        
    # Return filters in layer 
    def get_filters(self):
        return self.filters
    
    # Set the filters in layer to new filters given
    def set_filters(self, new_filters):
        self.filters = new_filters
        
    # Return the bias value    
    def get_bias(self):
        return self.bias
    
    # Set the bias value equal to b
    def set_bias(self, b):
        self.bias = b
    
    # Return the output dimensions of the layer 
    def out(self):
        return self.out_dim
    
    # Take image and a filter shape and return a feature map after convolving the image
    def convolve(self, img, fltr):
        displacement = int(fltr.shape[0]/2) # calculation of difference in image and kernel size
        
        # Get image dimensions and set up a matrix to hold the feature map
        rows,columns = img.shape
        feature_map = np.zeros((rows-2*displacement,columns-2*displacement))
        
        # Convolve over image
        for y in range(displacement,rows-displacement):
            for x in range(displacement,columns-displacement):
                feature_map[y-displacement,x-displacement] = img[y-displacement:y+displacement+1,x-displacement:x+displacement+1].ravel().dot(fltr.ravel())
        
        return feature_map
    
    # Takes in an image and then returns the feature maps created from convolving
    # filters over that image   
    def conv(self, img):
        # Initializes feature maps matrix to hold new feature maps from convolution
        feature_maps = np.zeros(self.out_dim)
        
        # Iterates through class filters and applies convolution to create a feature map
        # and assigns the feature map to the feature 
        # maps matrix
        if len(self.in_dim) > 2:
            for j in range(self.in_dim[2]):
                im = img[:,:,j]
                for i in range(self.num_filters):
                    feature_map = self.convolve(im,self.filters[i]) + self.bias[i]
                    feature_maps[:,:,i] = feature_map
        else:      
            for i in range(self.num_filters):
                feature_map = self.convolve(img,self.filters[i]) + self.bias[i]
                feature_maps[:,:,i] = feature_map
        
        return feature_maps  # feature maps in the shape of [number of filters, rows, columns]  
