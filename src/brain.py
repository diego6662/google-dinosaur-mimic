import numpy as np

from layers import Perceptron
from models import Sequential


class NN:
    def __init__(
        self,
    ):
        self.model = Sequential(layers=[Perceptron(16), Perceptron(2)])

    def run(self, X):
        return self.model.run(X)

    def mutate(
        self,
    ):
        for l in self.model.layers:
            for i, c in enumerate(l.weights):
                if np.random.rand() <= 0.2:
                    l.weights[i, :] = c * (
                        np.random.randint(0, 3) * np.random.randn(*c.shape)
                    )

                if np.random.rand() <= 0.2:
                    l.weights[i, :] = c + (
                        np.random.randint(0, 3) * np.random.randn(*c.shape)
                    )

                if np.random.rand() <= 0.2:
                    l.weights[i, :] = c - (
                        np.random.randint(0, 3) * np.random.randn(*c.shape)
                    )

            if np.random.rand() <= 0.2:
                l.bias = l.bias * (
                    np.random.randint(0, 3) * np.random.randn(*l.bias.shape)
                )

            if np.random.rand() <= 0.2:
                l.bias = l.bias + (
                    np.random.randint(0, 3) * np.random.randn(*l.bias.shape)
                )

            if np.random.rand() <= 0.2:
                l.bias = l.bias - (
                    np.random.randint(0, 3) * np.random.randn(*l.bias.shape)
                )

            if np.random.rand() <= 0.1:
                np.random.shuffle(l.weights)

    def cross(self, s1):
        for i, l in enumerate(self.model.layers):
            for j, c in enumerate(l.weights):
                if np.random.rand() <= 0.2:
                    l.weights[j, :] = c * s1.brain.model.layers[i].weights[j]

                if np.random.rand() <= 0.2:
                    l.weights[j, :] = c + s1.brain.model.layers[i].weights[j]

                if np.random.rand() <= 0.2:
                    l.weights[j, :] = c - s1.brain.model.layers[i].weights[j]

                if np.random.rand() <= 0.2:
                    l.weights[j, :], s1.brain.model.layers[i].weights[j] = (
                        s1.brain.model.layers[i].weights[j],
                        c,
                    )

            if np.random.rand() <= 0.2:
                l.bias = l.bias * s1.brain.model.layers[i].bias

            if np.random.rand() <= 0.2:
                l.bias = l.bias + s1.brain.model.layers[i].bias

            if np.random.rand() <= 0.2:
                l.bias = l.bias - s1.brain.model.layers[i].bias
