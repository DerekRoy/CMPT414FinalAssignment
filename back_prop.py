# Cross entropy error function 
def cross_entropy(labels,predictions):
    return np.sum(labels*np.log2(predictions))*-1
