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
        self.ts = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ])

    def test_forward(self):
        loss = self.time_softmax_with_loss.forward(self.xs, self.ts)
        self.assertEqual(1.1399003832212435, loss)

    def test_backward(self):
        self.time_softmax_with_loss.forward(self.xs, self.ts)
        dx = self.time_softmax_with_loss.backward()
        assert_almost_equal(np.array([
            [
                [-0.05907661,  0.0070421 ,  0.0520345 ],
                [ 0.01268836, -0.01735614,  0.00466779],
                [ 0.06401299,  0.02354906, -0.08756205]
            ],
            [
                [-0.10110771,  0.07391566,  0.02719205],
                [ 0.02354906, -0.04709812,  0.02354906],
                [ 0.00176403,  0.09631259, -0.09807662]
            ],
            [
                [-0.03719545,  0.02719205,  0.0100034 ],
                [ 0.04692431, -0.09384862,  0.04692431],
                [ 0.02719205,  0.0100034 , -0.03719545]
            ]
        ]), dx)

if __name__ == "__main__":
    unittest.main()
