import unittest
import numpy as np
import sys
sys.path.append("../lib/layers")
from time_affine import TimeAffine

class TestTimeAffine(unittest.TestCase):
    def setUp(self):
        W = np.random.randn(3, 3)
        b = np.random.randn(3,)
        self.time_affine = TimeAffine(W, b)
        self.x = np.random.randn(3, 3, 3)

    def test_forward(self):
        out = self.time_affine.forward(self.x)
        self.assertEqual((3, 3, 3), out.shape)

    def test_backward(self):
        dout = self.time_affine.forward(self.x)
        dx = self.time_affine.backward(dout)
        self.assertEqual((3, 3, 3), dout.shape)

if __name__ == "__main__":
    unittest.main()
