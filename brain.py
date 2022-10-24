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

    def propagate(self, input):
        output = np.dot(input, self.weights)
        return self.activation(output)

    def set(self, weights):
        self.weights = weights

    def flatten(self):
        return list(self.weights.flatten())

from genome import Genome

class Network:
    def __init__(self, genome: Genome):
        self.layers = genome.network

    def propagate(self, input):
        output = input
        for layer in self.layers:
            output = layer.propagate(output)
        return output
