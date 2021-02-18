import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
sys.path.append("../lib/concerns")
import os.path
from os import path
from count_based_methods import CountBasedMethod

class TestCountBasedMethod(unittest.TestCase):
    def setUp(self):
        text = "You said good-bye and I said hello."
        self.cbm = CountBasedMethod()
        self.word_list = self.cbm.text_to_word_list(text)
        self.query = "you"
        self.word_to_id, self.id_to_word, self.corpus = self.cbm.preprocess(self.word_list)
        self.vocab_size = len(self.word_to_id)
        self.co_matrix = self.cbm.create_co_matrix(self.corpus, self.vocab_size)

    def test_words(self):
        self.assertEqual(["you", "said", "good-bye", "and", "i", "said", "hello", "."], self.word_list)

    def test_take_out_query(self):
        query_info, query_vec = self.cbm._take_out_query(self.query, self.word_to_id, self.co_matrix)
        self.assertEqual({"query": "you"}, query_info)
        assert_array_equal(np.array([0, 1, 0, 0, 0, 0, 0]), query_vec)

    def test_cos_similarity(self):
        x = self.co_matrix[self.word_to_id["you"]]
        y = self.co_matrix[self.word_to_id["i"]]
        cos_similarity = self.cbm._cos_similarity(x, y)
        self.assertEqual(0.7071067691154799, cos_similarity)

    def test_calc_cos_similarity(self):
        *_, query_vec = self.cbm._take_out_query(self.query, self.word_to_id, self.co_matrix)
        similarity = self.cbm._calc_cos_similarity(self.vocab_size, self.co_matrix, query_vec)
        assert_almost_equal(np.array([
            1., 0., 0.7071068, 0., 0.7071068, 0.7071068, 0.
        ]), similarity)

    def test_output_result_asc(self):
        *_, query_vec = self.cbm._take_out_query(self.query, self.word_to_id, self.co_matrix)
        similarity = self.cbm._calc_cos_similarity(self.vocab_size, self.co_matrix, query_vec)
        result = self.cbm._output_result_asc(similarity, self.query, self.id_to_word)
        self.assertEqual({
            "good-bye": 0.7071067691154799,
            "i": 0.7071067691154799,
            "hello": 0.7071067691154799,
            "said": 0.0,
            "and": 0.0
        }, result)

    def test_preprocess(self):
        self.assertEqual({
            "you": 0,
            "said": 1,
            "good-bye": 2,
            "and": 3,
            "i": 4,
            "hello": 5,
            ".": 6
        }, self.word_to_id)
        self.assertEqual({
            0: "you",
            1: "said",
            2: "good-bye",
            3: "and",
            4: "i",
            5: "hello",
            6: "."
        }, self.id_to_word)
        assert_array_equal(np.array([0, 1, 2, 3, 4, 1, 5, 6]), self.corpus)

    def test_create_co_matrix(self):
        assert_array_equal(np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0]
        ]), self.co_matrix)

    def test_rank_similarity(self):
        top_five_similarities = self.cbm.rank_similarities(self.query, self.word_to_id, self.co_matrix, self.vocab_size, self.id_to_word)
        self.assertEqual({
            'query': 'you',
            'good-bye': 0.7071067691154799,
            'i': 0.7071067691154799,
            'hello': 0.7071067691154799,
            'said': 0.0,
            'and': 0.0
        }, top_five_similarities)

    def test_ppmi(self):
        M = self.cbm.ppmi(self.co_matrix)
        assert_almost_equal(np.array([
            [0., 1.8073549, 0., 0., 0., 0., 0.],
            [1.8073549, 0., 0.8073549, 0., 0.8073549, 0.8073549, 0.],
            [0., 0.8073549, 0., 1.8073549, 0., 0., 0.],
            [0., 0., 1.8073549, 0., 1.8073549, 0., 0.],
            [0., 0.8073549, 0., 1.8073549, 0., 0., 0.],
            [0., 0.8073549, 0., 0., 0., 0., 2.807355],
            [0., 0., 0., 0., 0., 2.807355, 0.]
        ]), M)

    def test_svd(self):
        M = self.cbm.ppmi(self.co_matrix)
        U = self.cbm.singular_value_deconposition(M)
        assert_almost_equal(np.array([
            [ 3.40948761e-01,  0.00000000e+00, -1.20516241e-01, -3.88578059e-16, -9.32324946e-01, -1.11022302e-16, -2.42574685e-17],
            [ 0.00000000e+00, -5.97636402e-01,  0.00000000e+00,  1.80237904e-01,  0.00000000e+00, -7.81245828e-01,  0.00000000e+00],
            [ 4.36312199e-01, -5.55111512e-17, -5.08782864e-01, -2.22044605e-16,  2.25325629e-01, -1.38777878e-17, -7.07106769e-01],
            [ 1.11022302e-16, -4.97828126e-01,  2.77555756e-17,  6.80396318e-01, -1.11022302e-16,  5.37799239e-01,  7.46693292e-17],
            [ 4.36312199e-01, -3.12375064e-17, -5.08782864e-01, -1.59998290e-16,  2.25325629e-01, -1.30164976e-17,  7.07106769e-01],
            [ 7.09237099e-01, -3.12375064e-17,  6.83926761e-01, -1.59998290e-16,  1.70958877e-01, -1.30164976e-17,  2.31390806e-17],
            [-1.66533454e-16, -6.28488600e-01, -4.16333634e-17, -7.10334539e-01,  2.22044605e-16,  3.16902101e-01, -9.61431563e-17]
        ]), U)

    def test_save_svd_plot_image(self):
        M = self.cbm.ppmi(self.co_matrix)
        U = self.cbm.singular_value_deconposition(M)
        filepath = "../img/svd_plot.png"
        self.cbm.save_svd_plot_image(self.word_to_id, U, filepath)
        self.assertEqual(True, path.exists(filepath))

if __name__ == "__main__":
    unittest.main()
