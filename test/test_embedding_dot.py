import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib/layers")
from embedding_dot import EmbeddingDot

class TestEmbeddingDot(unittest.TestCase):
    def setUp(self):
        W = np.arange(21).reshape(7, 3)
        self.embedding_dot = EmbeddingDot(W)
        self.index = np.array([0, 3, 1])
        self.h = np.arange(9).reshape(3, 3)

    def test_params(self):
        params, = self.embedding_dot.params
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
        grads, = self.embedding_dot.grads
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
        out = self.embedding_dot.forward(self.index, self.h)
        assert_array_equal(np.array([5, 122, 86]), out)

    def test_h(self):
        self.embedding_dot.forward(self.index, self.h)
        h, *_ = self.embedding_dot.cache
        assert_array_equal(np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
       ]), h)

    def test_target_W(self):
        self.embedding_dot.forward(self.index, self.h)
        *_, target_W = self.embedding_dot.cache
        assert_array_equal(np.array([
            [0,  1,  2],
            [9, 10, 11],
            [3,  4,  5]
        ]), target_W)

    def test_backward(self):
        dout = self.embedding_dot.forward(self.index, self.h)
        dh = self.embedding_dot.backward(dout)
        assert_array_equal(np.array([
            [   0,    5,   10],
            [1098, 1220, 1342],
            [ 258,  344,  430]
        ]), dh)

    def test_dtarget_W(self):
        dout = self.embedding_dot.forward(self.index, self.h)
        self.embedding_dot.backward(dout)
        assert_array_equal(np.array([
            [  0,   5,  10],
            [366, 488, 610],
            [516, 602, 688]
        ]), self.embedding_dot.cache)

if __name__ == "__main__":
    unittest.main()
