import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib")
from matrix import Matrix

class TestVector(unittest.TestCase):
    def setUp(self):
        self.matrix = Matrix([[1, 2, 3], [4, 5, 6]])

    def test_get_class_name(self):
        self.assertEqual("<class 'numpy.ndarray'>", self.matrix.get_class_name())

    def test_get_shape(self):
        self.assertEqual((2, 3), self.matrix.get_shape())

    def test_get_dim(self):
        self.assertEqual(2, self.matrix.get_dim())

    def test_calc_sum(self):
        assert_array_equal(np.array([
            [1, 3, 5],
            [7, 9, 11]
        ]), self.matrix.calc_sum([[0, 1 ,2], [3, 4, 5]]))

    def test_calc_product(self):
        assert_array_equal(np.array([
            [0, 2, 6],
            [12, 20, 30]
        ]), self.matrix.calc_product([[0, 1 ,2], [3, 4, 5]]))

    def test_calc_scala_broadcast(self):
        assert_array_equal(np.array([
            [10, 20, 30],
            [40, 50, 60]
        ]), self.matrix.calc_scala_broadcast(10))

    def test_calc_array_broadcast(self):
        assert_array_equal(np.array([
            [10, 40, 90],
            [40, 100, 180]
        ]), self.matrix.calc_array_broadcast([10, 20, 30]))

if __name__ == "__main__":
    unittest.main()
