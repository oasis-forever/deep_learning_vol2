import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/layers")
from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        x  = np.random.randn(10, 2)
        W1 = np.random.randn(2, 4)
        b1 = np.random.randn(4)
        W2 = np.random.randn(4, 3)
        b2 = np.random.randn(3)
        self.neural_network_1 = NeuralNetwork(x, W1, b1)
        self.neural_network_2 = NeuralNetwork(x, W2, b2)

    def test_get_hidden_layer_dim(self):
        h = self.neural_network_1.get_hidden_layer()
        self.assertEqual((10, 4), h.shape)

    def test_get_output_layer(self):
        h = self.neural_network_1.get_hidden_layer()
        dx = self.neural_network_2.get_output_layer(h)
        self.assertEqual((10, 3), dx.shape)

if __name__ == "__main__":
    unittest.main()
