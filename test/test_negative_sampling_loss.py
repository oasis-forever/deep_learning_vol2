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
        self.nsl = NegativeSamplingLoss(W, corpus)
        self.h = np.arange(3)
        self.target = np.array([1, 3, 0])

    def test_params(self):
        param, *_ = self.nsl.params
        assert_array_equal(np.array([
            [ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20]
        ]), param)

    def test_initial_grads(self):
        grad, *_ = self.nsl.grads
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
        loss = self.nsl.forward(self.h, self.target)
        self.assertEqual(True, 65 <= loss < 75)

    def test_backward(self):
        self.nsl.forward(self.h, self.target)
        dh = self.nsl.backward()
        assert_array_equal(np.array([
            [14. , 15.7, 17.3],
            [12. , 13.7, 15.3],
            [17. , 18.7, 20.3]
        ]), np.round(dh, decimals=1))

if __name__ == "__main__":
    unittest.main()
