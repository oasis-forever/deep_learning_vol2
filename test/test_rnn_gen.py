import unittest
import numpy as np
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from ptb import *
from rnnlm_gen import RNNLMGen

class TestRNNLMGen(unittest.TestCase):
    def setUp(self):
        corpus, self.word_to_id, self.id_to_word = load_data("train")
        vocab_size     = len(self.word_to_id)
        wordvec_size   = 100
        hidden_size    = 100
        self.rnnlm_gen = RNNLMGen(vocab_size, wordvec_size, hidden_size)
        start_word     = "you"
        self.start_id  = self.word_to_id[start_word]
        skip_words     = ["N", "<unk>", "$"]
        self.skip_ids  = [self.word_to_id[w] for w in skip_words]

    def test_word_ids_list(self):
        word_ids = self.rnnlm_gen.word_ids_list(self.start_id, self.skip_ids)
        self.assertEqual(100, len(word_ids))

    def test_generate_text(self):
        word_ids = self.rnnlm_gen.word_ids_list(self.start_id, self.skip_ids)
        text = self.rnnlm_gen.generate_text(self.id_to_word, word_ids)
        self.assertEqual(True, 750 < len(text) < 870)

if __name__ == "__main__":
    unittest.main()
