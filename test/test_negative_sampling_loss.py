import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
from negative_sampling_loss import NegativeSamplingLoss

class TestNegativeSamplingLoss(unittest.TestCase):
    def setUp(self):
        W = np.arange(21).reshape(7, 3)
        corpus = np.array([0, 1, 2, 3, 4, 1, 5, 2, 6])
        self.negative_sampling_loss = NegativeSamplingLoss(W, corpus)
        self.h = np.arange(3)
        self.target = np.array([1, 3, 0])

    def test_params(self):
        param, *_ = self.negative_sampling_loss.params
        assert_array_equal(np.array([
            [ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20]
        ]), param)

    def test_grads(self):
        grad, *_ = self.negative_sampling_loss.grads
        assert_array_equal(np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]), grad)

    def test_forward(self):
        loss = self.negative_sampling_loss.forward(self.h, self.target)
        self.assertEqual(True, 65 <= loss < 75)

    def test_backward(self):
        self.negative_sampling_loss.forward(self.h, self.target)
        dh = self.negative_sampling_loss.backward()
        assert_array_equal(np.array([
            [0. ,  1.61,  3.22],
            [5. ,  6.67,  8.33],
            [10., 11.67, 13.33]
        ]), np.round(dh, decimals=2))

if __name__ == "__main__":
    unittest.main()
