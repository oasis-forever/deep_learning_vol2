import unittest
import numpy as np
from numpy.testing import assert_array_equal
import os.path
from os import path
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/optimisers")
from rnnlm_trainer import RNNLMTrainer
from simple_rnnlm import SimpleRNNLM
from ptb import *
from sgd import SGD

class TestRNNLMTrainer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        wordvec_size    = 100
        hidden_size     = 100
        self.time_size  = 5
        learning_rate   = 0.1
        self.max_epoch  = 100
        corpus, word_to_id, id_to_word = load_data("train")
        corpus_size = 1000
        corpus = corpus[:corpus_size]
        vocab_size = int(max(corpus) + 1)
        self.xs = corpus[:-1]
        self.ts = corpus[1:]
        model = SimpleRNNLM(vocab_size, wordvec_size, hidden_size)
        optimiser = SGD(learning_rate)
        self.rnnlm_trainer = RNNLMTrainer(model, optimiser)

    def test_get_batch(self):
        batch_x, batch_t = self.rnnlm_trainer._get_batch(self.xs, self.ts, self.batch_size, self.time_size)
        assert_array_equal(np.array([
            [  0,   1,   2,   3,   4],
            [ 42,  76,  77,  64,  78],
            [ 26,  26,  98,  56,  40],
            [ 24,  32,  26, 175,  98],
            [208, 209,  80, 197,  32],
            [ 26,  79,  26,  80,  32],
            [274, 275, 276,  42,  61],
            [ 88, 303,  26, 304,  26],
            [ 42,  35,  72, 350,  64],
            [339, 359, 181, 328, 386]
        ]), batch_x)
        assert_array_equal(np.array([
            [  1,   2,   3,   4,   5],
            [ 76,  77,  64,  78,  79],
            [ 26,  98,  56,  40, 128],
            [ 32,  26, 175,  98,  61],
            [209,  80, 197,  32,  82],
            [ 79,  26,  80,  32, 241],
            [275, 276,  42,  61,  24],
            [303,  26, 304,  26,  32],
            [ 35,  72, 350,  64,  27],
            [359, 181, 328, 386, 387]
        ]), batch_t)

    def test_fit(self):
        training_process = self.rnnlm_trainer.fit(self.xs, self.ts, self.max_epoch, self.batch_size, self.time_size)
        self.assertEqual(100, len(training_process))

    def test_save_plot_image(self):
        self.rnnlm_trainer.fit(self.xs, self.ts, self.max_epoch, self.batch_size, self.time_size)
        self.rnnlm_trainer.save_plot_image("../img/rnnlm_trainer.png")
        self.assertEqual(True, path.exists("../img/rnnlm_trainer.png"))

if __name__ == "__main__":
    unittest.main()
