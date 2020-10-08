from three_layer_neural_network import NeuralNetwork
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def add_type(fn, type):
    def wrapper(*args, **kwargs):
        return fn(type=type, *args, **kwargs)

    return wrapper


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, n_hidden, input_dim, hidden_dim, output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.seed = seed

        # initialize the layers
        self.layers = [Layer(self.input_dim, self.hidden_dim, add_type(self.actFun, actFun_type),
                             add_type(self.diff_actFun, actFun_type))]
        for _ in range(self.n_hidden):
            self.layers += [Layer(hidden_dim, hidden_dim, add_type(self.actFun, actFun_type),
                                  add_type(self.diff_actFun, actFun_type))]

        # initialize output layer weight and bias
        self.W_out = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.b_out = np.zeros((1, self.output_dim))

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.feedforward(X)

        self.z_out = X.dot(self.W_out) + self.b_out

        def softmax(z):
            return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)

        self.probs = softmax(self.z_out)

    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        # calculate the dw for the last layer
        y_onehot = np.stack((1 - y, y), -1)
        self.dW_out = self.layers[-1].a.T.dot(self.probs - y_onehot)
        da = (self.probs - y_onehot).dot(self.W_out.T)

        for layer in reversed(self.layers):
            layer.backprop(da)
            da = layer.dX

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss
        y_onehot = np.stack((1 - y, y), -1)
        data_loss = -(y_onehot * np.log(self.probs)).sum()

        data_loss += self.reg_lambda / 2 * (
            np.sum(np.square(np.concatenate([layer.W.ravel() for layer in self.layers]))))
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        # Gradient descent.
        for i in range(0, num_passes):
            self.feedforward(X)
            self.backprop(X, y)
            self.dW_out += self.reg_lambda * self.W_out
            for layer in self.layers:
                layer.dW += self.reg_lambda * layer.W

            self.W_out += -epsilon * self.dW_out
            for layer in self.layers:
                layer.W += -epsilon * layer.dW
                layer.b += -epsilon * layer.db

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


class Layer():
    def __init__(self, input_dim, output_dim, actFun, diff_actFun, seed=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun = actFun
        self.diff_actFun = diff_actFun

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self.output_dim))

    def feedforward(self, X):
        self.X = X
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z)
        return self.a

    def backprop(self, da):
        num_examples = len(self.X)
        self.dW = self.X.T.dot(da * (self.diff_actFun(self.z)))
        self.db = np.ones(num_examples).dot(da * (self.diff_actFun(self.z)))
        self.dX = (da * self.diff_actFun(self.z)).dot(self.W.T)


if __name__ == "__main__":
    X, y = generate_data()
    model = DeepNeuralNetwork(n_hidden=5, input_dim=2, hidden_dim=8, output_dim=2, actFun_type="Tanh")
    model.fit_model(X, y, epsilon=0.001, print_loss=False)
    model.visualize_decision_boundary(X, y)
