import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
sys.path.append("../lib")
import os.path
from os import path
from count_based_methods import CountBasedMethod

class TestCountBasedMethod(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye, and I said hello."
        self.cbm = CountBasedMethod(text)
        self.cbm.preprocess()
        vocab_size = len(self.cbm.word_to_id)
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

    def test_ppmi(self):
        self.cbm.ppmi()
        assert_array_equal(np.array([
            [0., 2., 0., 0., 0., 0., 0., 0.],
            [2., 0., 1., 0., 0., 1., 1., 0.],
            [0., 1., 0., 2., 0., 0., 0., 0.],
            [0., 0., 2., 0., 2., 0., 0., 0.],
            [0., 0., 0., 2., 0., 2., 0., 0.],
            [0., 1., 0., 0., 2., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 3.],
            [0., 0., 0., 0., 0., 0., 3., 0.]
        ]), self.cbm.M)

    def test_svd(self):
        self.cbm.ppmi()
        self.cbm.singular_value_deconposition()
        assert_almost_equal(np.array([
            [2.2358029e-01,  2.5730875e-01, -1.8918733e-15,  7.0071183e-02,  4.1559568e-01, -6.1818111e-01, -1.5125957e-16,  5.6923324e-01],
            [4.0655869e-01, -4.3377417e-01,  1.8911931e-15,  1.0841788e-01, -4.9208382e-01, -5.7702208e-01,  4.3715032e-16, -2.4432483e-01],
            [3.4078276e-01,  1.6512336e-01, -3.7174803e-01, -1.9356677e-01,  3.3881739e-01, -1.8085355e-02, -6.0150093e-01, -4.5167357e-01],
            [4.1640049e-01, -6.1479867e-02,  6.0150093e-01, -3.5370579e-01, -1.5513299e-01,  2.7162981e-01, -3.7174803e-01,  3.1602857e-01],
            [4.1640049e-01, -6.1479867e-02, -6.0150093e-01, -3.5370579e-01, -1.5513299e-01,  2.7162981e-01,  3.7174803e-01,  3.1602857e-01],
            [3.4078276e-01,  1.6512336e-01,  3.7174803e-01, -1.9356677e-01,  3.3881739e-01, -1.8085355e-02,  6.0150093e-01, -4.5167357e-01],
            [3.4984776e-01,  6.1765921e-01, -1.3564715e-15,  5.8249110e-01, -3.4352767e-01,  1.9532609e-01, -9.8723344e-17, -2.5382388e-02],
            [2.8858954e-01, -5.4958016e-01,  2.8579891e-15,  5.6470162e-01,  4.3519601e-01,  3.1388810e-01, -9.3341079e-17,  8.8704653e-02]
        ]), self.cbm.U)

    def test_save_svd_plot_image(self):
        self.cbm.ppmi()
        self.cbm.singular_value_deconposition()
        self.cbm.save_svd_plot_image("../img/svd_plot.png")
        self.assertEqual(True, path.exists("../img/svd_plot.png"))

if __name__ == "__main__":
    unittest.main()
