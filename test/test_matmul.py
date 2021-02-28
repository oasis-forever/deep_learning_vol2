import unittest
import numpy as np
import sys
sys.path.append("../lib/layers")
from matmul import MatMul

class TestMatMul(unittest.TestCase):
    def setUp(self):
        W = np.random.rand(4, 2)
        self.matmul = MatMul(W)
        self.x = np.random.rand(2, 4)

    def test_forward(self):
        out = self.matmul.forward(self.x)
        self.assertEqual((2, 2), out.shape)

    def test_backward(self):
        dout = self.matmul.forward(self.x)
        dx = self.matmul.backward(dout)
        self.assertEqual((2, 4), dx.shape)

if __name__ == "__main__":
    unittest.main()
