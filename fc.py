class fully_connected_layer:
    def __init__(self, inpt, nodes, dropout=0):
        self.inpt = inpt
        self.nodes = nodes
        self.weights = np.random.normal(size=(nodes,inpt),scale=(2/(nodes*inpt)))
        self.bias = np.zeros(nodes)
        self.drop_out = dropout
    
    # Return the weights matrix
    def get_weights(self):
        return self.weights
    
    # Set the values for the weights matrix
    def set_weights(self,w):
        self.weights = w
    
    # Get the bias array
    def get_bias(self):
        return self.bias
    
    # Set the bias terms
    def set_bias(self,b):
        self.bias = b
    
    # Return the output size
    def out(self):
        return self.nodes
        
    # Take in array x and output new array after x*weights + bias -> relu activation
    def feed_forward(self, x, train=False):
        # Initialize outputs
        output = np.zeros((self.nodes))
        
        if self.drop_out and train:
            rng = np.random.default_rng()
            drop = rng.choice(self.inpt, size=int(self.inpt*self.drop_out), replace=False)
            np.put(x,drop,0)
        
        # Calculate new node activation per value
        for i,w in enumerate(self.weights):
            output[i] = np.sum(x*w)+self.bias[i]
        
        return output
