import unittest
import numpy as np
import sys
sys.path.append("../lib/concerns")
from sigmoid import Sigmoid

class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid = Sigmoid()

    def test_forward(self):
        x = np.random.randn(10, 4)
        self.assertEqual((10, 4), self.sigmoid.forward(x).shape)

    def test_backward(self):
        x = np.random.randn(10, 4)
        self.sigmoid.forward(x)
        dout = np.random.randn(10, 4)
        self.assertEqual((10, 4), self.sigmoid.backward(dout).shape)

if __name__ == "__main__":
    unittest.main()
