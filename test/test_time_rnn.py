import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib/layers")
from time_rnn import TimeRNN

class TestTimeRNN(unittest.TestCase):
    def setUp(self):
        Wx = np.random.randn(3, 3)
        Wh = np.random.randn(3, 3)
        b  = np.random.randn(3,)
        self.time_rnn = TimeRNN(Wx, Wh, b)
        self.xs = np.random.randn(3, 3, 3)

    def test_state(self):
        h = np.random.randn(7, 7)
        self.time_rnn.set_state(h)
        assert_array_equal(h, self.time_rnn.h)
        self.time_rnn.reset_state()
        self.assertEqual(None, self.time_rnn.h)

    def test_forward(self):
        hs = self.time_rnn.forward(self.xs)
        self.assertEqual((3, 3, 3), hs.shape)

    def test_backward(self):
        hs = self.time_rnn.forward(self.xs)
        dxs = self.time_rnn.backward(hs)
        self.assertEqual((3, 3, 3), dxs.shape)

if __name__ == "__main__":
    unittest.main()
