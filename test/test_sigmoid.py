import unittest
import numpy as np
import sys
sys.path.append("../lib")
from sigmoid import Sigmoid

class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid = Sigmoid()

    def test_forward(self):
        self.assertEqual((10, 4), self.sigmoid.forward(np.random.randn(10, 4)).shape)

if __name__ == "__main__":
    unittest.main()
