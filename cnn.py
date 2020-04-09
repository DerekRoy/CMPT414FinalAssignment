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

class CNN:
    def __init__(self,image):
        self.c1 = convolution_layer(image.shape,64,5,2,True)
        self.flat = flatten(self.c1.out_dim)
        self.dense = fully_connected_layer(self.flat.output_shape[0],10)

    def feed_forward(self,image):
        # Convolutional layer 1
        self.c1_out = self.c1.conv(image)
        self.c1_activation = leaky_relu(self.c1_out)

        # Flatten out
        self.flattened = self.flat.flatten(self.c1_activation)

        # Out Layer
        self.logits = self.dense.feed_forward(self.flattened)
        self.output = softmax(self.logits)

        return self.output

    # Train the model with training data
    def train(self,X_train,y_train,epochs=10):
        # Run through each epoch
        for e in range(epochs):
            print("Epoch {}/{}".format(e+1,epochs))

            # Random Shuffle on data and split to create train and validation set
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

            # Place holders for loss, accuracy, and validation loss and accuracy per epoch
            loss = []
            accuracy = []
            val_loss = []
            val_accuracy = []

            # Metrics to update progress of model
            num_images = X_train.shape[0]
            last_percentage = 0
            print("\tPercent finished in training {}%".format(last_percentage), end = ' ')

            # Train the model
            for i,image in enumerate(X_train):
                print("\t {}/{}".format(i,num_images))
                # Get percent done with epoch
#                 percent_done = int(i/num_images*100)
#                 if percent_done%10 == 0 and last_percentage != percent_done:
#                     print("{}%".format(percent_done), end = ' ')
#                     last_percentage = percent_done

                # Predict number
                prediction = self.feed_forward(image)

                # Add loss to epoch loss
                loss.append(cross_entropy(y_train[i],prediction))

                # Get number of correct predictions and wrong predictions
                if np.argmax(y_train[i]) == np.argmax(prediction):
                    accuracy.append(1)
                else:
                    accuracy.append(0)

                # Do Back Propogation
            #     self.backprop(y_train[i],prediction)

            last_percentage = 0
            print("\n\tPercent finished in validation {}%".format(last_percentage), end = ' ')

            # Get validation
            for i,image in enumerate(X_val):
                print("\t {}/{}".format(i,num_images))
#                 # Get percent done with epoch
#                 percent_done = int(i/num_images*100)
#                 if percent_done%10 == 0 and last_percentage != percent_done:
#                     print("{}%".format(percent_done), end = ' ')
#                     last_percentage = percent_done

                # Predict Number
                prediction = self.feed_forward(image)

                # Add loss to val loss
                val_loss.append(cross_entropy(y_val[i],prediction))

                # Get number of correct predictions and wrong predictions
                if np.argmax(y_val[i]) == np.argmax(prediction):
                    val_accuracy.append(1)
                else:
                    val_accuracy.append(0)

            # Get averages over epochs
            loss = np.mean(loss)
            accuracy = np.mean(accuracy)
            val_loss = np.mean(val_loss)
            val_accuracy = np.mean(val_accuracy)

            # Print out preformance
            print("\ttrain loss: {} train accuracy: {} \n\tvalidation loss: {} validation accuracy: {}\n".format(loss,accuracy,val_loss,val_accuracy))

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
        weights = {"c1_weights":self.c1.get_filters(),"dense_weights":self.dense.get_weights(),"dense_bias":self.dense.get_bias()}
        with open(name, 'wb') as pickle_file:
            pickle.dump(weights, pickle_file)
        print("Model Saved")

    def load_model(self,name="weights"):
        with open(name, 'rb') as f:
            weights = pickle.load(f)
        self.c1.set_filters(weights['c1_weights'])
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
