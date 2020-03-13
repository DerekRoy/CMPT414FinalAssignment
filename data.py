from scipy.io import loadmat
import matplotlib.pyplot as plt

def get_data():
    mnist = loadmat("./mnist-original.mat")
    inputData, labels =  mnist["data"].T, mnist["label"][0]

    # Restructure rows into 28 x 28 images 
    X = np.array([digit.reshape(28,28) for digit in inputData])

    # One hot encode y labels
    y = []
    for i in labels:
        temp_y = np.zeros((10))
        temp_y[int(i)] = 1
        y.append(temp_y)
    y = np.array(y)

    return X,y
