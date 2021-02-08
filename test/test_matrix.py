import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib")
from matrix import Matrix

class TestVector(unittest.TestCase):
    def setUp(self):
        self.matrix = Matrix(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_get_class_name(self):
        self.assertEqual("<class 'numpy.ndarray'>", str(self.matrix.W.__class__))

    def test_get_shape(self):
        self.assertEqual((2, 3), self.matrix.W.shape)

    def test_get_dim(self):
        self.assertEqual(2, self.matrix.W.ndim)

    def test_calc_sum(self):
        assert_array_equal(np.array([
            [1, 3, 5],
            [7, 9, 11]
        ]), self.matrix.calc_sum(np.array([[0, 1 ,2], [3, 4, 5]])))

    def test_calc_product(self):
        assert_array_equal(np.array([
            [0, 2, 6],
            [12, 20, 30]
        ]), self.matrix.calc_product(np.array([[0, 1 ,2], [3, 4, 5]])))

    def test_calc_scala_broadcast(self):
        assert_array_equal(np.array([
            [10, 20, 30],
            [40, 50, 60]
        ]), self.matrix.calc_scala_broadcast(10))

    def test_calc_array_broadcast(self):
        assert_array_equal(np.array([
            [10, 40, 90],
            [40, 100, 180]
        ]), self.matrix.calc_array_broadcast(np.array([10, 20, 30])))

    def test_calc_inner_product(self):
        assert_array_equal(np.array([
            [58, 64],
            [139, 154]
        ]), self.matrix.calc_inner_product(np.array([[7, 8], [9, 10], [11, 12]])))

if __name__ == "__main__":
    unittest.main()
