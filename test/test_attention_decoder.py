import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/layers")
from decoder import Decoder

class TestDecoder(unittest.TestCase):
    def setUp(self):
        vocab_size   = 13
        wordvec_size = 100
        hidden_size  = 100
        self.decoder = Decoder(vocab_size, wordvec_size, hidden_size)
        self.xs = np.random.randint(0, 13, (13, 100))
        self.enc_hs = np.random.randn(13, 100)

    def test_forward(self):
        score = self.decoder.forward(self.xs, self.enc_hs)
        self.assertEqual((13, 100, 13), score.shape)

    def test_backward(self):
        dscore = self.decoder.forward(self.xs, self.enc_hs)
        dh = self.decoder.backward(dscore)
        self.assertEqual((13, 100), dh.shape)

    def test_generate(self):
        enc_hs = np.random.randn(1, 100)
        start_id = 0
        sample_size = 10
        sampled = self.decoder.generate(enc_hs, start_id, sample_size)
        self.assertEqual(10, len(sampled))

if __name__ == "__main__":
    unittest.main()
