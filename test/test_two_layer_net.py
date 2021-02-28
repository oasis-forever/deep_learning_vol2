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
        x = self.two_layer_net._predict(self.x)
        self.assertEqual((4, 3), x.shape)

    def test_forward(self):
        loss = self.two_layer_net.forward(self.x, self.t)
        self.assertEqual(1, int(loss))

    def test_backward(self):
        self.two_layer_net.forward(self.x, self.t)
        dout = self.two_layer_net.backward()
        self.assertEqual((4, 2), dout.shape)

if __name__ == "__main__":
    unittest.main()
