import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/layers")
from time_attention import TimeAttention

class TestTimeAttention(unittest.TestCase):
    def setUp(self):
        self.time_attention = TimeAttention()
        self.hs_enc = np.random.randn(10, 4, 4)
        self.hs_dec = np.random.randn(10, 5, 4)

    def test_forward(self):
        out = self.time_attention.forward(self.hs_enc, self.hs_dec)
        self.assertEqual((10, 5, 4), out.shape)

    def test_backward(self):
        dout = self.time_attention.forward(self.hs_enc, self.hs_dec)
        dhs_enc, dhs_dec = self.time_attention.backward(dout)
        self.assertEqual((10, 4, 4), dhs_enc.shape)
        self.assertEqual((10, 5, 4), dhs_dec.shape)

if __name__ == "__main__":
    unittest.main()
