import unittest
import numpy as np
import sys
from numpy.testing import assert_almost_equal
sys.path.append("../lib/concerns")
from softmax_with_loss import SoftMaxWithLoss

class TestSoftMaxWithLoss(unittest.TestCase):
    def setUp(self):
        self.softmax_with_loss = SoftMaxWithLoss()
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
        self.assertEqual(0.5879459780961449, self.softmax_with_loss.forward(self.x, self.t))

    def test_backward(self):
        self.softmax_with_loss.forward(self.x, self.t)
        assert_almost_equal(np.array([
            [ 0.00668465, -0.06224663,  0.0260615 ,  0.02950047],
            [ 0.04142029,  0.06316393,  0.01311583, -0.11770004],
            [ 0.04700427, -0.14680577,  0.03139707,  0.06840443],
            [ 0.0736796 , -0.10492584,  0.01233183,  0.01891441]
        ]), self.softmax_with_loss.backward())

if __name__ == "__main__":
    unittest.main()
