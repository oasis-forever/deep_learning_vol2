import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib")
from count_based_methods import CountBasedMethod
from simple_word2vec import SimpleWord2Vec

class TestWord2Vec(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye and I said hello."
        cbm = CountBasedMethod()
        word_list = cbm.text_to_word_list(text)
        self.word_to_id, _, self.corpus = cbm.preprocess(word_list)
        self.simple_word2vec = SimpleWord2Vec()

    def test_corpus(self):
        assert_array_equal(np.array([0, 1, 2, 3, 4, 1, 5, 6]), self.corpus)

    def test_create_contexts_target(self):
        contexts_array, target_array = self.simple_word2vec.create_contexts_target(self.corpus)
        assert_array_equal(np.array([
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 1],
            [4, 5],
            [1, 6]
        ]), contexts_array)
        assert_array_equal(np.array([1, 2, 3, 4, 1, 5]), target_array)

