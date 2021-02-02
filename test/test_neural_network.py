import unittest
import numpy as np
import sys
sys.path.append("../lib")
from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.x  = np.random.randn(10, 2)
        self.W1 = np.random.randn(2, 4)
        self.b1 = np.random.randn(4)
        self.W2 = np.random.randn(4, 3)
        self.b2 = np.random.randn(3)

    def test_get_hidden_layer_dim(self):
        neural_network_1 = NeuralNetwork(self.x, self.W1, self.b1)
        self.assertEqual((10, 4), neural_network_1.get_hidden_layer().shape)

    def test_get_output_layer(self):
        neural_network_1 = NeuralNetwork(self.x, self.W1, self.b1)
        h = neural_network_1.get_hidden_layer()
        neural_network_2 = NeuralNetwork(self.x, self.W2, self.b2)
        self.assertEqual((10, 3), neural_network_2.get_output_layer(h).shape)

if __name__ == "__main__":
    unittest.main()
