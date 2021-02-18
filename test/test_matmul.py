import unittest
import numpy as np
import sys
sys.path.append("../lib/layers")
from matmul import MatMul

class TestMatMul(unittest.TestCase):
    def setUp(self):
        W = np.random.rand(4, 2)
        self.matmul = MatMul(W)

    def test_forward(self):
        x = np.random.rand(2, 4)
        self.assertEqual((2, 2), self.matmul.forward(x).shape)

    def test_backward(self):
        x = np.random.rand(2, 4)
        dout = self.matmul.forward(x)
        self.assertEqual((2, 4), self.matmul.backward(dout).shape)

if __name__ == "__main__":
    unittest.main()
