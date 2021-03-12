import unittest
import numpy as np
from os import path
import sys
sys.path.append("../lib/")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
from better_rnnlm import BetterRNNLM
from count_based_methods import CountBasedMethod

class TestBetterRNNLM(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye and I said hello."
        cbm = CountBasedMethod()
        word_list = cbm.text_to_word_list(text)
        word_to_id, *_ = cbm.preprocess(word_list)
        vocab_size = len(word_to_id)
        wordvec_size = 100
        hidden_size  = 100
        self.better_rnnlm = BetterRNNLM(vocab_size, wordvec_size, hidden_size)
        self.xs = np.array([
            [0, 4, 4, 1],
            [4, 0, 2, 1]
        ])
        self.ts = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        self.file_path = "../pkl/better_rnnlm.pkl"

    def test_predict(self):
        score = self.better_rnnlm._predict(self.xs)
        self.assertEqual((2, 4, 7), score.shape)

    def test_forward(self):
        loss = self.better_rnnlm.forward(self.xs, self.ts)
        self.assertEqual(1.95, round(loss, 2))

    def test_backward(self):
        self.better_rnnlm.forward(self.xs, self.ts)
        dout = self.better_rnnlm.backward()
        self.assertEqual(None, dout)

    def test_reset_state(self):
        self.better_rnnlm.forward(self.xs, self.ts)
        self.better_rnnlm.backward()
        self.assertEqual((2, 100), self.better_rnnlm.lstm_layers[0].h.shape)
        self.better_rnnlm.reset_state()
        self.assertEqual(None, self.better_rnnlm.lstm_layers[0].h)

    def test_save_params(self):
        self.better_rnnlm.forward(self.xs, self.ts)
        self.better_rnnlm.backward()
        self.better_rnnlm.save_params(self.file_path)
        self.assertEqual(True, path.exists(self.file_path))

    def test_load_params(self):
        self.better_rnnlm.load_params(self.file_path)
        a, b, c, d, e, f, g, h, i = self.better_rnnlm.params
        self.assertEqual((7, 100), a.shape)
        self.assertEqual((100, 400), b.shape)
        self.assertEqual((100, 400), c.shape)
        self.assertEqual((400,), d.shape)
        self.assertEqual((100, 400), e.shape)
        self.assertEqual((100, 400), f.shape)
        self.assertEqual((400,), g.shape)
        self.assertEqual((100, 7), h.shape)
        self.assertEqual((7,), i.shape)

if __name__ == "__main__":
    unittest.main()
