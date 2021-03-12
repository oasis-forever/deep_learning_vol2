import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/layers")
from attention_encoder import AttentionEncoder

class TestAttentionEncoder(unittest.TestCase):
    def setUp(self):
        vocab_size   = 13
        wordvec_size = 100
        hidden_size  = 100
        self.attention_encoder = AttentionEncoder(vocab_size, wordvec_size, hidden_size)
        self.xs = np.random.randint(0, 13, (7, 3))

    def test_forward(self):
        hs = self.attention_encoder.forward(self.xs)
        self.assertEqual((7, 3, 100), hs.shape)

    def test_backward(self):
        dhs = self.attention_encoder.forward(self.xs)
        dout = self.attention_encoder.backward(dhs)
        self.assertEqual(None, dout)

if __name__ == "__main__":
    unittest.main()
