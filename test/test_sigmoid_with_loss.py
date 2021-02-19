import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib/layers")
from sigmoid_with_loss import SigmoidWithLoss

class TestNegativeSamplingLoss(unittest.TestCase):
    def setUp(self):
        self.sigmoid_with_loss = SigmoidWithLoss()

    def test_forward(self):
        x = None
        t = None
        loss = self.sigmoid_with_loss.forward(x, t)
        self.assertEqual(None, loss)

    def test_backward(self):
        dx = self.sigmoid_with_loss.backward()
        self.assertEqual(None, dx)

if __name__ == "__main__":
    unittest.main()
