import unittest
import numpy as np
import sys
sys.path.append("../lib/")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
from simple_rnnlm import SimpleRNNLM
from count_based_methods import CountBasedMethod

class TestSimpleRNNLM(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye and I said hello."
        cbm = CountBasedMethod()
        word_list = cbm.text_to_word_list(text)
        word_to_id, *_ = cbm.preprocess(word_list)
        vocab_size = len(word_to_id)
        wordvec_size = 100
        hidden_size  = 100
        self.simple_rnnlm = SimpleRNNLM(vocab_size, wordvec_size, hidden_size)
        self.xs = np.array([
            [0, 4, 4, 1],
            [4, 0, 2, 1]
        ])
        self.ts = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

    def test_predict(self):
        score = self.simple_rnnlm._predict(self.xs)
        self.assertEqual((2, 4, 7), score.shape)

    def test_forward(self):
        loss = self.simple_rnnlm.forward(self.xs, self.ts)
        self.assertEqual(1.95, round(loss, 2))

    def test_backward(self):
        self.simple_rnnlm.forward(self.xs, self.ts)
        dout = self.simple_rnnlm.backward()
        self.assertEqual(None, dout)

    def test_reset_state(self):
        self.simple_rnnlm.forward(self.xs, self.ts)
        self.simple_rnnlm.backward()
        self.assertEqual((2, 100), self.simple_rnnlm.rnn_layer.h.shape)
        self.simple_rnnlm.reset_state()
        self.assertEqual(None, self.simple_rnnlm.rnn_layer.h)

if __name__ == "__main__":
    unittest.main()
