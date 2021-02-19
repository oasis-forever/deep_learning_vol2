import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib/layers")
from embedding import Embedding

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        W = np.arange(21).reshape(7, 3)
        self.embedding = Embedding(W)

    def test_params(self):
        params, = self.embedding.params
        assert_array_equal(np.array([
            [ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20]
       ]), params)

    def test_grads(self):
        grads, = self.embedding.grads
        assert_array_equal(np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
       ]), grads)

    def test_forward(self):
        index = np.array([0, 2, 0, 4])
        out = self.embedding.forward(index)
        assert_array_equal(np.array([
            [ 0,  1,  2],
            [ 6,  7,  8],
            [ 0,  1,  2],
            [12, 13, 14]
       ]), out)

    def test_backward(self):
        index = np.array([0, 2, 0, 4])
        dout = self.embedding.forward(index)
        self.embedding.backward(dout)
        grads, = self.embedding.grads
        assert_array_equal(np.array([
            [ 0,  2,  4],
            [ 0,  0,  0],
            [ 6,  7,  8],
            [ 0,  0,  0],
            [12, 13, 14],
            [ 0,  0,  0],
            [ 0,  0,  0]
        ]), grads)

if __name__ == "__main__":
    unittest.main()
