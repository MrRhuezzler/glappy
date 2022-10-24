import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


class FCLayer:
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        # self.biases = np.random.rand(1, output_size) - 0.5

    def propagate(self, input):
        # output = np.dot(input, self.weights) + self.biases
        output = np.dot(input, self.weights)
        return self.activation(output)
    
    def set(self, weights):
        self.weights = weights
        # self.biases = biases

    def flattenw(self):
        return list(self.weights.flatten())

    def flattenb(self):
        return list(self.biases.flatten())


class Network:
    def __init__(self):
        self.layers = []

    def propagate(self, input):
        output = input
        for layer in self.layers:
            output = layer.propagate(output)
        return output

    def from_genome(self, genome):
        self.layers = genome.network
