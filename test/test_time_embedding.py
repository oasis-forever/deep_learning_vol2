import unittest
import numpy as np
from numpy.testing import assert_array_equal
import copy
import sys
sys.path.append("../lib/layers")
from time_embedding import TimeEmbedding

class TestTimeEmbedding(unittest.TestCase):
    def setUp(self):
        W = np.array([
            [ 0.3280114 ,  0.51719588,  2.56566087],
            [ 0.53182669,  0.85125353, -0.05985388],
            [-1.03053182,  0.89397269,  1.24217588],
            [-0.7373072 ,  0.57723267, -1.13001464],
            [-0.90229024,  0.97809838, -0.09150674],
            [ 0.71714065, -1.28034957,  0.693378  ],
            [-2.13395059,  1.04338757,  1.14265363]
        ])
        self.time_embedding = TimeEmbedding(W)
        self.xs = np.array([
            [0, 4, 4, 1],
            [4, 0, 2, 1]
        ])

    def test_forward(self):
        out = self.time_embedding.forward(self.xs)
        self.assertEqual((2, 4, 3), out.shape)

    def test_params(self):
        _params, = self.time_embedding.params
        before_params = copy.copy(_params)
        dout = self.time_embedding.forward(self.xs)
        self.time_embedding.backward(dout)
        after_params, = self.time_embedding.params
        params = before_params == after_params
        assert_array_equal(np.array([
            [ True,  True,  True],
            [ True,  True,  True],
            [ True,  True,  True],
            [ True,  True,  True],
            [ True,  True,  True],
            [ True,  True,  True],
            [ True,  True,  True]
        ]), params)

    def test_grads(self):
        _grads, = self.time_embedding.grads
        before_grads = copy.copy(_grads)
        dout = self.time_embedding.forward(self.xs)
        self.time_embedding.backward(dout)
        after_grads, = self.time_embedding.grads
        grads = before_grads == after_grads
        assert_array_equal(np.array([
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [ True,  True,  True],
            [False, False, False],
            [ True,  True,  True],
            [ True,  True,  True]
        ]), grads)

if __name__ == "__main__":
    unittest.main()
