import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
sys.path.append("../lib")
from count_based_methods import CountBasedMethod

class TestCountBasedMethod(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye, and I said hello."
        self.cbm = CountBasedMethod(text)
        self.cbm.preprocess()
        vocab_size = len(self.word_to_id)
        self.cbm.create_co_matrix(vocab_size)
        self.query = "you"

    def test_words(self):
        self.assertEqual(["you", "said", "good-bye", ",", "and", "i", "said", "hello", "."], self.cbm.words)

    def test_take_out_query(self):
        query_info, query_vec = self.cbm._take_out_query(self.query)
        self.assertEqual({"query": "you"}, query_info)
        assert_array_equal(np.array([0, 1, 0, 0, 0, 0, 0, 0]), query_vec)

    def test_cos_similarity(self):
        x = self.cbm.co_matrix[self.cbm.word_to_id["you"]]
        y = self.cbm.co_matrix[self.cbm.word_to_id["i"]]
        self.assertEqual(0.7071067691154799, self.cbm._cos_similarity(x, y))

    def test_calc_cos_similarity(self):
        *_, query_vec = self.cbm._take_out_query(self.query)
        assert_almost_equal(np.array([
            1., 0., 0.7071068, 0., 0., 0.7071068, 0.7071068, 0.
        ]), self.cbm._calc_cos_similarity(query_vec))

    def test_output_result_asc(self):
        *_, query_vec = self.cbm._take_out_query(self.query)
        similarity = self.cbm._calc_cos_similarity(query_vec)
        self.assertEqual({
            "good-bye": 0.7071067691154799,
            "i": 0.7071067691154799,
            "hello": 0.7071067691154799,
            "said": 0.0,
            ",": 0.0
        }, self.cbm._output_result_asc(similarity, self.query))

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
        ]), self.cbm.co_matrix)

    def test_rank_similarity(self):
        self.assertEqual({
            'query': 'you',
            'good-bye': 0.7071067691154799,
            'i': 0.7071067691154799,
            'hello': 0.7071067691154799,
            'said': 0.0,
            ',': 0.0
        }, self.cbm.rank_similarities(self.query))

if __name__ == "__main__":
    unittest.main()
