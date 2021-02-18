import numpy as np
import sys
sys.path.append("./layers")
from sigmoid import Sigmoid

class NeuralNetwork:
    def __init__(self, x, W, b):
        self.x = x
        self.W = W
        self.b = b

    def get_hidden_layer(self):
        return np.dot(self.x, self.W) + self.b

    def get_output_layer(self, h):
        sigmoid = Sigmoid()
        a = sigmoid.forward(h)
        return np.dot(a, self.W) + self.b
