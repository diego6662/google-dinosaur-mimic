import numpy as np


class Perceptron:
    def __init__(self, units: int):
        self.units = units
        self.activation = relu
        self.weights = None
        self.bias = None
        self.input_shape = None

    def set_weights(self):
        if self.weights is None:
            self.weights = 2 * np.random.rand(self.units, self.input_shape) - 1

    def set_bias(self):
        if self.bias is None:
            self.bias = 2 * np.random.rand(self.units) - 1

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape[0]

    def compute(self, X):
        value = np.dot(self.weights, X) + self.bias
        return self.activation(value)


@np.vectorize
def relu(X):
    return max(0, X)
