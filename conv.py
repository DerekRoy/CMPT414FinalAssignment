# Sample usage 
# c1 = convolution_layer(image.shape,2,3,True)
# feature_maps = c1.conv(image)
import numpy as np

R = 0.1

class convolution_layer:
    def __init__(self, inpt, num_filters, kernel_size, stride, padding, testing=False):
        self.in_dim = inpt
        self.out_dim = None
        self.offset = kernel_size - 1
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.cache_img = None
        self.cache_feature_maps = None


        # If padding True i.e: same set offset to 0
        if padding:
            self.offset = 0
            
        # Output shape
        if len(self.in_dim) > 2:
            self.out_dim = (int((self.in_dim[0]-self.offset)/self.stride),int((self.in_dim[1]-self.offset)/self.stride),self.in_dim[2]*num_filters)
        else:
            self.out_dim = (int((self.in_dim[0]-self.offset)/self.stride),int((self.in_dim[1]-self.offset)/self.stride),num_filters)
        
        # Weights
        self.filters = np.random.normal(size=(num_filters,kernel_size,kernel_size),scale=(2/(num_filters*kernel_size**2))) # Glorot initiaized, size num_filters of size (kernel_size x kernel_size)

        # Biases
        self.biases = np.zeros((num_filters, 1))
        # self.biases = np.random.normal(size=(num_filters, 1),scale=(2/(num_filters*kernel_size**2))) # Glorot initiaized, size num_filters of size (kernel_size x kernel_size)

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
    
    # Take image and a filter shape and return a feature map after convolving the image
    def convolve(self, img, fltr):
        displacement = int(fltr.shape[0]/2) # calculation of difference in image and kernel size
        
        # Get image dimensions and set up a matrix to hold the feature map
        rows,columns = img.shape
        
        if self.padding:
            feature_map = np.zeros((int(rows/self.stride),int(columns/self.stride)))
            img = np.pad(img,((displacement, displacement), (displacement, displacement)),'constant', constant_values=(0))
            rows,columns = img.shape
        else:
            feature_map = np.zeros((int((rows-2*displacement)/self.stride),int((columns-2*displacement)/self.stride)))
        
        # Convolve over image
        for y in range(displacement,rows-displacement,self.stride):
            for x in range(displacement,columns-displacement,self.stride):
                feature_map[int((y-displacement)/self.stride),int((x-displacement)/self.stride)] = img[y-displacement:y+displacement+1,x-displacement:x+displacement+1].ravel().dot(fltr.ravel())
        
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
                    feature_map = self.convolve(im,self.filters[i])
                    feature_maps[:, :, j * self.num_filters + i] = feature_map
        else:
            for i in range(self.num_filters):
                feature_map = self.convolve(img,self.filters[i])
                feature_maps[:,:,i] = feature_map

        self.cache_img = img
        self.cache_feature_maps = feature_maps

        return feature_maps  # feature maps in the shape of [number of filters, rows, columns]

    def back_prop(self, in_grad):
        delta_filters = np.zeros(self.filters.shape)
        delta_biases = np.zeros(self.biases.shape)

        displacement = int(self.filters[0].shape[0] / 2) # calculation of difference in image and kernel size

        rows, columns = self.cache_img.shape[0], self.cache_img.shape[1]

        out_grad = np.zeros(self.cache_img.shape)

        if len(self.in_dim) > 2:
            for j in range(self.in_dim[2]):
                im = self.cache_img[:,:,j]
                for i in range(self.num_filters):
                    for y in range(displacement, rows-displacement, self.stride):
                        for x in range(displacement, columns-displacement, self.stride):
                            image_part = im[y-displacement:y+displacement+1,x-displacement:x+displacement+1]
                            left_side = in_grad[int((y-displacement)/self.stride), int((x-displacement)/self.stride), j * self.num_filters + i]
                            delta_filters[i] = image_part * left_side

                            out_grad[y-displacement:y+displacement+1, x-displacement:x+displacement+1, j] = left_side * self.filters[i]

                    delta_biases += in_grad[:, :, j * self.num_filters + i].sum()
        else:
            for i in range(self.num_filters):
                for y in range(displacement, rows-displacement, self.stride):
                    for x in range(displacement, columns-displacement, self.stride):
                        image_part = self.cache_img[y-displacement:y+displacement+1,x-displacement:x+displacement+1]
                        left_side = in_grad[int((y-displacement)/self.stride), int((x-displacement)/self.stride), i]
                        delta_filters[i] = image_part * left_side

                        out_grad[y-displacement:y+displacement+1, x-displacement:x+displacement+1] = left_side * self.filters[i]

                delta_biases += in_grad[:, :, i].sum()

        self.filters -= R * delta_filters
        self.biases -= R * delta_biases

        return out_grad