class convolution_layer:
    def init(num_filters, strides, padding):
        self.filters = np.random.normal(size=(num_filters,3,3))
        self.strides = strides


    def convolve(img, fltr):
        displacement = int(fltr.shape[0]/2)
        padded_img = np.pad(img, (displacement, displacement), 'constant', constant_values=(0))
        rows,columns = padded_img.shape
        new_img = np.zeros(img.shape)

        for y in range(displacement,rows-displacement):
            for x in range(displacement,columns-displacement):
                new_img[y-displacement,x-displacement] = padded_img[y-displacement:y+displacement+1,x-displacement:x+displacement+1].ravel().dot(fltr.ravel())
        return new_img
