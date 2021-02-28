import unittest
import numpy as np
import sys
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
from lstm import LSTM

class TestLSTM(unittest.TestCase):
    def setUp(self):
        Wx = np.random.randn(3, 12)
        Wh = np.random.randn(3, 12)
        b  = np.random.randn(12,)
        self.lstm = LSTM(Wx, Wh, b)
        self.x = np.random.randn(12, 3)
        self.h_prev = np.random.randn(12, 3)
        self.c_prev = np.random.randn(12, 3)

    def test_forward(self):
        c_next, h_next = self.lstm.forward(self.x, self.h_prev, self.c_prev)
        self.assertEqual((12, 3), c_next.shape)
        self.assertEqual((12, 3), h_next.shape)

    def test_backward(self):
        dc_next, dh_next = self.lstm.forward(self.x, self.h_prev, self.c_prev)
        dx, dh_prev, dc_prev = self.lstm.backward(dc_next, dh_next)
        self.assertEqual((12, 3), dx.shape)
        self.assertEqual((12, 3), dh_prev.shape)
        self.assertEqual((12, 3), dc_prev.shape)

if __name__ == "__main__":
    unittest.main()
