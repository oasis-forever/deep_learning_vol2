import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib")
from count_based_methods import CountBasedMethod

class TestCountBasedMethod(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye, and I said hello."
        self.cbm = CountBasedMethod(text)

    def test_words(self):
        self.assertEqual(["you", "said", "good-bye", ",", "and", "i", "said", "hello", "."], self.cbm.words)

    def test_preprocess(self):
        corpus, word_to_id, id_to_word = self.cbm.preprocess()
        assert_array_equal(np.array([0, 1, 2, 3, 4, 5, 1, 6, 7]), corpus)
        self.assertEqual({
            "you": 0,
            "said": 1,
            "good-bye": 2,
            ",": 3,
            "and": 4,
            "i": 5,
            "hello": 6,
            ".": 7
        }, word_to_id)
        self.assertEqual({
            0: "you",
            1: "said",
            2: "good-bye",
            3: ",",
            4: "and",
            5: "i",
            6: "hello",
            7: "."
        }, id_to_word)

if __name__ == "__main__":
    unittest.main()
