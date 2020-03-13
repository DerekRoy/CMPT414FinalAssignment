from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def get_data():
    # Load the data and split 
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
    
    # Split into train test split and shuffle data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
