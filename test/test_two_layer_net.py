import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from two_layer_net import TwoLayerNet

class TestTwoLayerNet(unittest.TestCase):
    def setUp(self):
        self.two_layer_net = TwoLayerNet(2, 4, 3)

    def test_predict(self):
        x = np.random.randn(10, 2)
        self.assertEqual((10, 3), self.two_layer_net.predict(x).shape)

if __name__ == "__main__":
    unittest.main()
