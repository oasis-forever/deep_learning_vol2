import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/layers")
from attention import Attention

class TestAttemtion(unittest.TestCase):
    def setUp(self):
        self.attention = Attention()
        self.hs = np.random.randn(10, 5, 4)
        self.h  = np.random.randn(10, 4)

    def test_forward(self):
        out = self.attention.forward(self.hs, self.h)
        self.assertEqual((10, 4), out.shape)

    def test_backward(self):
        dout = self.attention.forward(self.hs, self.h)
        dhs, dh = self.attention.backward(dout)
        self.assertEqual((10, 5, 4), dhs.shape)
        self.assertEqual((10, 5), dh.shape)

if __name__ == "__main__":
    unittest.main()
