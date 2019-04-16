import numpy as np
import random


def l2_loss(predictions,Y):
    '''
        Computes L2 loss (sum squared loss) between true values, Y, and predictions.

        :param Y A 1D Numpy array with real values (float64)
        :param predictions A 1D Numpy array of the same size of Y
        :return L2 loss using predictions for Y.
    '''
    # TODO
    return np.sum((Y - predictions)**2)

def sigmoid(x):
    '''
        Sigmoid function f(x) =  1/(1 + exp(-x))
        :param x A scalar or Numpy array
        :return Sigmoid function evaluated at x (applied element-wise if it is an array)
    '''
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def sigmoid_derivative(x):
    '''
        First derivative of the sigmoid function with respect to x.
        :param x A scalar or Numpy array
        :return Derivative of sigmoid evaluated at x (applied element-wise if it is an array)
    '''
    return sigmoid(x) * (1 - sigmoid(x))

class OneLayerNN:
    '''
        One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the neural network model.
        '''
        self.weights = None
        pass

    def train(self, X, Y, learning_rate=0.001, epochs=25, print_loss=True):
        '''
        Trains the OneLayerNN model using SGD.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        self.weights = np.zeros(len(X[0]))
        for e in range(epochs):
            error = 0.0
            for i in range(len(X)):
                row = X[i]
                y = np.dot(row, self.weights)
                L_y = 2 * (y - Y[i])
                L_w = row * L_y
                self.weights -= (learning_rate * L_w)
            if print_loss:
                print(self.loss(X,Y))

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        return sigmoid(np.dot(X, self.weights))

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]


class TwoLayerNN:

    def __init__(self, hidden_size=10, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            output_neurons: The number of outputs of the network
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size

        # In this assignment, we will only use output_neurons = 1.
        self.output_neurons = 1
        self.W = None
        self.v = None
        self.b1 = 0
        self.b2 = 0

    def train(self, X, Y, learning_rate=0.01, epochs=25, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        self.W = np.random.randn(len(X[0]), self.hidden_size)
        self.v = np.random.randn(self.hidden_size, self.output_neurons)

        for e in range(epochs):
            for i in range(len(X)):
                #FORWARD
                x_i = X[i]
                x_i = x_i[:,None].T
                z = np.dot(x_i, self.W)
                layer1 = self.activation(z + self.b1)
                z2 = np.dot(layer1,self.v)
                layer2 = self.activation(z2 + self.b2)

                # BACK
                layer2_error = Y[i] - layer2 # d_t
                layer2_delta = layer2_error * self.activation_derivative(layer2)

                layer1_error = layer2_delta.dot(self.v.T)
                layer1_delta = layer1_error * self.activation_derivative(layer1)

                self.W += x_i.T.dot(layer1_delta)
                self.v += layer1.T.dot(layer2_delta)

            if print_loss:
                print(self.loss(X,Y))
        pass

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        layer1 = self.activation(np.dot(X, self.W) + self.b1)
        return self.activation(np.dot(layer1, self.v) + self.b2)

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
