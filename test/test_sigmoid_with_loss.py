import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
from sigmoid_with_loss import SigmoidWithLoss

class TestNegativeSamplingLoss(unittest.TestCase):
    def setUp(self):
        self.sigmoid_with_loss = SigmoidWithLoss()
        self.x = np.array([
            [-0.27291637,  3.0623984 ,  1.08772839,  1.21167545],
            [ 0.77815361,  1.20011612, -0.37179735,  1.93945452],
            [-1.02360881, -0.23723418, -1.42713268, -0.6484095 ],
            [-0.6631865 ,  0.01433258, -2.450729  , -2.02298841]
        ])
        self.t = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ])

    def test_forward(self):
        loss = self.sigmoid_with_loss.forward(self.x, self.t)
        self.assertEqual(2.74623389782839, loss)

    def test_backward(self):
        self.sigmoid_with_loss.forward(self.x, self.t)
        dx = self.sigmoid_with_loss.backward()
        assert_almost_equal(np.array([
            [ 0.10804782, -0.0111713 ,  0.18698843,  0.19264882],
            [ 0.17132051,  0.19213636,  0.1020267 , -0.03142695],
            [ 0.06608126, -0.13975799,  0.04838646,  0.08583701],
            [ 0.08500604, -0.12410423,  0.01984631,  0.02920258]
        ]), dx)

if __name__ == "__main__":
    unittest.main()
