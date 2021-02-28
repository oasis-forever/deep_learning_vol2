import unittest
import numpy as np
import sys
sys.path.append("../lib/layers")
from affine import Affine

class TestAffine(unittest.TestCase):
    def setUp(self):
        W = np.random.randn(2, 4)
        b = np.random.randn(4)
        self.affine = Affine(W, b)

    def test_forward(self):
        out = self.affine.forward(np.random.randn(4, 2))
        self.assertEqual((4, 4), out.shape)

    def test_backward(self):
        self.affine.forward(np.random.randn(4, 2))
        dx = self.affine.backward(np.random.randn(4, 4))
        self.assertEqual((4, 2), dx.shape)

if __name__ == "__main__":
    unittest.main()
