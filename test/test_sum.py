import unittest
import numpy as np
import sys
sys.path.append("../lib/concerns")
from sum import Sum

class TestSum(unittest.TestCase):
    def setUp(self):
        self.sum = Sum(8, 7)

    def test_forward(self):
        x = np.random.randn(self.sum.N, self.sum.D)
        self.assertEqual((1, 8), self.sum.forward(x).shape)

    def test_backward(self):
        dy = np.random.randn(1, self.sum.D)
        self.assertEqual((7, 8), self.sum.backward(dy).shape)

if __name__ == "__main__":
    unittest.main()
