import unittest
import numpy as np
import sys
sys.path.append("../lib/layers")
from attention_weight import AttentionWeight

class TestAttentionWeight(unittest.TestCase):
    def setUp(self):
        self.attention_weight = AttentionWeight()
        self.hs = np.random.randn(10, 5, 4)
        self.h  = np.random.randn(10, 4)

    def test_forward(self):
        a = self.attention_weight.forward(self.hs, self.h)
        self.assertEqual((10, 5), a.shape)

    def test_backward(self):
        da = self.attention_weight.forward(self.hs, self.h)
        dhs, da = self.attention_weight.backward(da)
        self.assertEqual((10, 5, 4), dhs.shape)
        self.assertEqual((10, 5), da.shape)

if __name__ == "__main__":
    unittest.main()
