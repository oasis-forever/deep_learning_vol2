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
        text = "You said good-bye and I said hello."
        self.cbm = CountBasedMethod(text)
        self.cbm.preprocess()
        vocab_size = len(self.cbm.word_to_id)
        self.cbm.create_co_matrix(vocab_size)
        self.query = "you"

    def test_words(self):
        self.assertEqual(["you", "said", "good-bye", "and", "i", "said", "hello", "."], self.cbm.words)

    def test_take_out_query(self):
        query_info, query_vec = self.cbm._take_out_query(self.query)
        self.assertEqual({"query": "you"}, query_info)
        assert_array_equal(np.array([0, 1, 0, 0, 0, 0, 0]), query_vec)

    def test_cos_similarity(self):
        x = self.cbm.co_matrix[self.cbm.word_to_id["you"]]
        y = self.cbm.co_matrix[self.cbm.word_to_id["i"]]
        self.assertEqual(0.7071067691154799, self.cbm._cos_similarity(x, y))

    def test_calc_cos_similarity(self):
        *_, query_vec = self.cbm._take_out_query(self.query)
        assert_almost_equal(np.array([
            1., 0., 0.7071068, 0., 0.7071068, 0.7071068, 0.
        ]), self.cbm._calc_cos_similarity(query_vec))

    def test_output_result_asc(self):
        *_, query_vec = self.cbm._take_out_query(self.query)
        similarity = self.cbm._calc_cos_similarity(query_vec)
        self.assertEqual({
            "good-bye": 0.7071067691154799,
            "i": 0.7071067691154799,
            "hello": 0.7071067691154799,
            "said": 0.0,
            "and": 0.0
        }, self.cbm._output_result_asc(similarity, self.query))

    def test_preprocess(self):
        assert_array_equal(np.array([0, 1, 2, 3, 4, 1, 5, 6]), self.cbm.corpus)
        self.assertEqual({
            "you": 0,
            "said": 1,
            "good-bye": 2,
            "and": 3,
            "i": 4,
            "hello": 5,
            ".": 6
        }, self.cbm.word_to_id)
        self.assertEqual({
            0: "you",
            1: "said",
            2: "good-bye",
            3: "and",
            4: "i",
            5: "hello",
            6: "."
        }, self.cbm.id_to_word)

    def test_create_co_matrix(self):
        assert_array_equal(np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0]
        ]), self.cbm.co_matrix)

    def test_rank_similarity(self):
        self.assertEqual({
            'query': 'you',
            'good-bye': 0.7071067691154799,
            'i': 0.7071067691154799,
            'hello': 0.7071067691154799,
            'said': 0.0,
            'and': 0.0
        }, self.cbm.rank_similarities(self.query))

    def test_ppmi(self):
        self.cbm.ppmi()
        assert_almost_equal(np.array([
            [0., 1.8073549, 0., 0., 0., 0., 0.],
            [1.8073549, 0., 0.8073549, 0., 0.8073549, 0.8073549, 0.],
            [0., 0.8073549, 0., 1.8073549, 0., 0., 0.],
            [0., 0., 1.8073549, 0., 1.8073549, 0., 0.],
            [0., 0.8073549, 0., 1.8073549, 0., 0., 0.],
            [0., 0.8073549, 0., 0., 0., 0., 2.807355],
            [0., 0., 0., 0., 0., 2.807355, 0.]
        ]), self.cbm.M)

    def test_svd(self):
        self.cbm.ppmi()
        self.cbm.singular_value_deconposition()
        assert_almost_equal(np.array([
            [ 3.40948761e-01,  0.00000000e+00, -1.20516241e-01, -3.88578059e-16, -9.32324946e-01, -1.11022302e-16, -2.42574685e-17],
            [ 0.00000000e+00, -5.97636402e-01,  0.00000000e+00,  1.80237904e-01,  0.00000000e+00, -7.81245828e-01,  0.00000000e+00],
            [ 4.36312199e-01, -5.55111512e-17, -5.08782864e-01, -2.22044605e-16,  2.25325629e-01, -1.38777878e-17, -7.07106769e-01],
            [ 1.11022302e-16, -4.97828126e-01,  2.77555756e-17,  6.80396318e-01, -1.11022302e-16,  5.37799239e-01,  7.46693292e-17],
            [ 4.36312199e-01, -3.12375064e-17, -5.08782864e-01, -1.59998290e-16,  2.25325629e-01, -1.30164976e-17,  7.07106769e-01],
            [ 7.09237099e-01, -3.12375064e-17,  6.83926761e-01, -1.59998290e-16,  1.70958877e-01, -1.30164976e-17,  2.31390806e-17],
            [-1.66533454e-16, -6.28488600e-01, -4.16333634e-17, -7.10334539e-01,  2.22044605e-16,  3.16902101e-01, -9.61431563e-17]
        ]), self.cbm.U)

    def test_save_svd_plot_image(self):
        self.cbm.ppmi()
        self.cbm.singular_value_deconposition()
        self.cbm.save_svd_plot_image("../img/svd_plot.png")
        self.assertEqual(True, path.exists("../img/svd_plot.png"))

if __name__ == "__main__":
    unittest.main()
