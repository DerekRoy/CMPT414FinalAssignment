# Sample usage 
# c1 = convolution_layer(2,3,True)
# feature_maps = c1.conv(image)

class convolution_layer:
    def __init__(self, num_filters, kernel_size, testing):
        # Gaussian -1 to 1 initiaized filters
        self.filters = np.random.normal(size=(num_filters,kernel_size,kernel_size)) 
        self.bias = 0 # initializa bias to 0
        
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
    def set_filters(new_filters):
        self.filters = new_filters
        
    # Return the bias value    
    def get_bias(self):
        return self.bias
    
    # Set the bias value equal to b
    def set_bias(b):
        self.bias = b
    
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
        # Get the row and column values for the feature maps
        feature_map_row = img.shape[0]-self.filters.shape[1]+1
        feature_map_col = img.shape[1]-self.filters.shape[1]+1
        
        # Initializes feature maps matrix to hold new feature maps from convolution
        feature_maps = np.zeros((self.num_filters,feature_map_row,feature_map_col))
        
        # Iterates through class filters and applies convolution to create a feature map,
        # applies relu to the feature map, and assigns the feature map to the feature 
        # maps matrix
        for i in range(self.num_filters):
            feature_map = self.convolve(img,self.filters[i]) + self.bias
            feature_map = self.relu(feature_map)
            feature_maps[i,:,:] = feature_map
        
        return feature_maps  # feature maps in the shape of [number of filters, rows, columns]  
    
    # Activation for the convolutional layer (Rectified Linear Unit) takes a feature map 
    # and returns modified feature map: if pixel value < 0: pixel value = 0   
    def relu(self, feature_map):
        return np.maximum(0,feature_map)
