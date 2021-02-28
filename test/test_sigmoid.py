import unittest
import numpy as np
import sys
sys.path.append("../lib/layers")
from sigmoid import Sigmoid

class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid = Sigmoid()
        self.x = np.random.randn(10, 4)

    def test_forward(self):
        out = self.sigmoid.forward(self.x)
        self.assertEqual((10, 4), out.shape)

    def test_backward(self):
        self.sigmoid.forward(self.x)
        dout = np.random.randn(10, 4)
        dx = self.sigmoid.backward(dout)
        self.assertEqual((10, 4), dx.shape)

if __name__ == "__main__":
    unittest.main()
