import unittest
import numpy as np
import sys
sys.path.append("../lib/concerns")
from cross_entropy_error import *

class TestCrossEntropyError(unittest.TestCase):
    def test_cross_entropy_error(self):
        y = np.array([
            [0.02673862, 0.75101348, 0.10424601, 0.11800189],
            [0.16568116, 0.2526557 , 0.05246332, 0.52919982],
            [0.18801706, 0.41277693, 0.12558827, 0.27361774],
            [0.29471841, 0.58029664, 0.04932731, 0.07565764]
        ])
        t = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ])
        self.assertEqual(0.5879459847664801, cross_entropy_error(y, t))

if __name__ == "__main__":
    unittest.main()
