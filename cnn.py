# Test run code:
#     from data import get_data
#     image = get_data()[0][0]
#     d = discriminator(image)
#     d.run(image)

from conv import convolution_layer
from activations import leaky_relu,softmax,leaky_relu_backprop
from flatten import flatten
from fc import fully_connected_layer
from error_functions import backprop_cross_entropy
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from error_functions import cross_entropy
import time

class CNN:
    def __init__(self,image):
        self.c1 = convolution_layer(image.shape,8,5,2,True)
        self.c2 = convolution_layer(self.c1.out_dim, 8, 5, 2, True)
        self.flat = flatten(self.c2.out_dim)
        self.dense = fully_connected_layer(self.flat.output_shape[0],10)

    def feed_forward(self,image):
        # Convolutional layer 1
        self.c1_out = self.c1.conv(image)
        self.c1_activation = leaky_relu(self.c1_out)

        self.c2_out = self.c2.conv(self.c1_activation)
        self.c2_activation = leaky_relu(self.c2_out)

        # Flatten out
        self.flattened = self.flat.flatten(self.c2_activation)

        # Out Layer
        self.output = self.dense.feed_forward(self.flattened)

        return self.output

    # Train the model with training data
    def train(self, X_train, y_train, epochs=10, images_limit=100_000):
        X_train = X_train[:images_limit]
        y_train = y_train[:images_limit]

        start_time = time.time()

        for epoch in range(epochs):
            for i in range(len(X_train)):
                percentage_done = (epoch * len(X_train) + i) / (epochs * len(X_train))
                time_taken = time.time() - start_time
                estimated_time_left = time_taken / percentage_done - time_taken if percentage_done != 0 else float('inf')

                print('%.1f' % (percentage_done * 100) + '% done. ' + 'Time left: %.0f sec' % estimated_time_left)

                image = X_train[i]
                label = y_train[i]
                self.feed_forward(image)
                self.back_prop(label)

    # Test the accuracy of model with test set
    def test(self,X_test,y_test):
        # Loss and accuracy place holders
        loss = []
        accuracy = []

        # Images to test
        images = X_test.shape[0]

        # Go through test data and run predictions
        for i,image in enumerate(X_test):
            print("Testing: {}/{} images".format(i+1,images))
            # Predict Number
            prediction = self.feed_forward(image)

            # Add loss to epoch loss
            loss.append(cross_entropy(y_test[i],prediction))

            # Get number of correct predictions and wrong predictions
            if np.argmax(y_test[i]) == np.argmax(prediction):
                accuracy.append(1)
            else:
                accuracy.append(0)

        # Calculate average loss and accuracy
        loss = np.mean(loss)
        accuracy = np.mean(accuracy)

        print("Loss: {} Accuracy: {}".format(loss,accuracy))


    def save_model(self, name="weights"):
        weights = {
            "c1_weights":self.c1.get_filters(),
            "c2_weights":self.c2.get_filters(),
            "dense_weights":self.dense.get_weights(),
            "dense_bias":self.dense.get_bias()}
        with open(name, 'wb') as pickle_file:
            pickle.dump(weights, pickle_file)
        print("Model Saved")

    def load_model(self,name="weights"):
        with open(name, 'rb') as f:
            weights = pickle.load(f)
        self.c1.set_filters(weights['c1_weights'])
        self.c2.set_filters(weights['c2_weights'])
        self.dense.set_weights(weights['dense_weights'])
        self.dense.set_bias(weights['dense_bias'])
        print("Model Loaded")

    def back_prop(self, desired_output):
        fc_grad = self.dense.back_prop(desired_output, self.output)

        fc_grad = fc_grad.reshape(self.c2.out_dim)

        fc_grad = leaky_relu_backprop(fc_grad)

        c2_grad = self.c2.back_prop(fc_grad)

        c2_grad = leaky_relu_backprop(c2_grad)

        self.c1.back_prop(c2_grad)
