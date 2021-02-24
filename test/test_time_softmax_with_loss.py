import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
from time_softmax_with_loss import TimeSoftmaxWithLoss

class TestTimeSoftmaxWithLoss(unittest.TestCase):
    def setUp(self):
        self.time_softmax_with_loss = TimeSoftmaxWithLoss()
        self.xs = np.array([
            [[3, 1, 3], [1, 3, 0], [2, 1, 1]],
            [[2, 4, 3], [0, 1, 0], [0, 4, 2]],
            [[2, 1, 0], [3, 2, 3], [1, 0, 2]],
        ])
        self.ts = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])

    def test_forward(self):
        loss = self.time_softmax_with_loss.forward(self.xs, self.ts)
        self.assertEqual(1.473233716554577, loss)

    def test_backward(self):
        self.time_softmax_with_loss.forward(self.xs, self.ts)
        dx = self.time_softmax_with_loss.backward()
        assert_almost_equal(np.array([
            [
                [-0.05907661,  0.0070421 ,  0.0520345 ],
                [-0.09842276,  0.09375497,  0.00466779],
                [-0.04709812,  0.02354906,  0.02354906]
            ],
            [
                [-0.10110771,  0.07391566,  0.02719205],
                [ 0.02354906, -0.04709812,  0.02354906],
                [-0.10934708,  0.09631259,  0.01303449]
            ],
            [
                [-0.03719545,  0.02719205,  0.0100034 ],
                [-0.0641868 ,  0.01726249,  0.04692431],
                [-0.08391906,  0.0100034 ,  0.07391566]
            ]
        ]), dx)

if __name__ == "__main__":
    unittest.main()
