import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
from negative_sampling_loss import NegativeSamplingLoss

class TestNegativeSamplingLoss(unittest.TestCase):
    def setUp(self):
        W = np.arange(21).reshape(7, 3)
        corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
        self.negative_sampling_loss = NegativeSamplingLoss(W, corpus)

    def test_loss_layers(self):
        loss_layers = self.negative_sampling_loss.loss_layers
        self.assertEqual([], loss_layers)

    def test_embedding_layers(self):
        embedding_layers = self.negative_sampling_loss.embedding_layers
        self.assertEqual([], embedding_layers)

    def test_params(self):
        params = self.negative_sampling_loss.params
        assert_array_equal(np.array([
            None
        ]), params)

    def test_grads(self):
        grads = self.negative_sampling_loss.grads
        assert_array_equal(np.array([
            None
        ]), grads)

    def test_forward(self):
        h = None
        target = None
        loss = self.negative_sampling_loss.forward(h, target)
        self.assertEqual(None, loss)

    def test_backward(self):
        h = None
        target = None
        self.negative_sampling_loss.forward(h, target)
        dh = self.negative_sampling_loss.backward()
        self.assertEqual(None, dh)

if __name__ == "__main__":
    unittest.main()
