import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/layers")
from peeky_decoder import PeekyDecoder

class TestPeekyDecoder(unittest.TestCase):
    def setUp(self):
        vocab_size   = 13
        wordvec_size = 16
        hidden_size  = 128
        self.peeky_decoder = PeekyDecoder(vocab_size, wordvec_size, hidden_size)
        self.xs = np.random.randint(0, 13, (13, 16))
        self.h = np.random.randn(13, 128)

    def test_forward(self):
        score = self.peeky_decoder.forward(self.xs, self.h)
        self.assertEqual((13, 16, 13), score.shape)

    def test_backward(self):
        dscore = self.peeky_decoder.forward(self.xs, self.h)
        dh = self.peeky_decoder.backward(dscore)
        self.assertEqual((13, 128), dh.shape)

    def test_generate(self):
        h = np.random.randn(1, 128)
        start_id = 0
        sample_size = 10
        sampled = self.peeky_decoder.generate(h, start_id, sample_size)
        self.assertEqual(10, len(sampled))

if __name__ == "__main__":
    unittest.main()
