import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
sys.path.append("../lib/models")
from two_layer_net import TwoLayerNet

class TestTwoLayerNet(unittest.TestCase):
    def setUp(self):
        self.two_layer_net = TwoLayerNet(2, 4, 3)
        self.x = np.random.randn(4, 2)
        self.t = np.random.randn(4, 3)

    def test_predict(self):
        self.assertEqual((4, 3), self.two_layer_net._predict(self.x).shape)

    def test_forward(self):
        self.assertEqual(1, int(self.two_layer_net.forward(self.x, self.t)))

    def test_backward(self):
        self.two_layer_net.forward(self.x, self.t)
        self.assertEqual((4, 2), self.two_layer_net.backward().shape)

if __name__ == "__main__":
    unittest.main()
