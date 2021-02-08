import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from count_based_methods import CountBasedMethod
from list_handler import uniq_list

class TestCountBasedMethod(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye, and I said hello."
        self.cbm = CountBasedMethod(text)
        self.cbm.preprocess()
        vocab_size = len(uniq_list(self.cbm.corpus))
        self.co_matrix = self.cbm.create_co_matrix(vocab_size)

    def test_words(self):
        self.assertEqual(["you", "said", "good-bye", ",", "and", "i", "said", "hello", "."], self.cbm.words)

    def test_preprocess(self):
        assert_array_equal(np.array([0, 1, 2, 3, 4, 5, 1, 6, 7]), self.cbm.corpus)
        self.assertEqual({
            "you": 0,
            "said": 1,
            "good-bye": 2,
            ",": 3,
            "and": 4,
            "i": 5,
            "hello": 6,
            ".": 7
        }, self.cbm.word_to_id)
        self.assertEqual({
            0: "you",
            1: "said",
            2: "good-bye",
            3: ",",
            4: "and",
            5: "i",
            6: "hello",
            7: "."
        }, self.cbm.id_to_word)

    def test_create_co_matrix(self):
        assert_array_equal(np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ]), self.co_matrix)

    def test_cos_similarity(self):
        x = self.co_matrix[self.cbm.word_to_id["you"]]
        y = self.co_matrix[self.cbm.word_to_id["i"]]
        self.assertEqual(0.7071067691154799, self.cbm.cos_similarity(x, y))

if __name__ == "__main__":
    unittest.main()
