import unittest
import sys
sys.path.append("../lib")
from vector import Vector

class TestVector(unittest.TestCase):
    def setUp(self):
        self.vector = Vector([1, 2, 3])

    def test_get_class_name(self):
        self.assertEqual("<class 'numpy.ndarray'>", self.vector.get_class_name())

    def test_get_shape(self):
        self.assertEqual((3,), self.vector.get_shape())

    def test_get_dim(self):
        self.assertEqual(1, self.vector.get_dim())

    def test_calc_inner_product(self):
        self.assertEqual(32, self.vector.calc_inner_product([4, 5, 6]))

if __name__ == "__main__":
    unittest.main()
