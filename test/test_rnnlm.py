import unittest
import numpy as np
import os.path
from os import path
import sys
sys.path.append("../lib/")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
from rnnlm import RNNLM
from count_based_methods import CountBasedMethod

class TestRNNLM(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye and I said hello."
        cbm = CountBasedMethod()
        word_list = cbm.text_to_word_list(text)
        word_to_id, *_ = cbm.preprocess(word_list)
        vocab_size = len(word_to_id)
        wordvec_size = 100
        hidden_size  = 100
        self.rnnlm = RNNLM(vocab_size, wordvec_size, hidden_size)
        self.xs = np.array([
            [0, 4, 4, 1],
            [4, 0, 2, 1]
        ])
        self.ts = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

    def test_predict(self):
        score = self.rnnlm._predict(self.xs)
        self.assertEqual((2, 4, 7), score.shape)

    def test_forward(self):
        loss = self.rnnlm.forward(self.xs, self.ts)
        self.assertEqual(1.95, round(loss, 2))

    def test_backward(self):
        self.rnnlm.forward(self.xs, self.ts)
        dout = self.rnnlm.backward()
        self.assertEqual(None, dout)

    def test_reset_state(self):
        self.rnnlm.forward(self.xs, self.ts)
        self.rnnlm.backward()
        self.assertEqual((2, 100), self.rnnlm.lstm_layer.h.shape)
        self.rnnlm.reset_state()
        self.assertEqual(None, self.rnnlm.lstm_layer.h)

    def test_save_params(self):
        self.rnnlm.forward(self.xs, self.ts)
        self.rnnlm.backward()
        self.rnnlm.save_params()
        self.assertEqual(True, path.exists("../pkl/rnnlm.pkl"))

    def test_load_params(self):
        self.rnnlm.load_params()
        a, b, c, d, e, f = self.rnnlm.params
        self.assertEqual((7, 100), a.shape)
        self.assertEqual((100, 400), b.shape)
        self.assertEqual((100, 400), c.shape)
        self.assertEqual((400,), d.shape)
        self.assertEqual((100, 7), e.shape)
        self.assertEqual((7,), f.shape)

if __name__ == "__main__":
    unittest.main()
