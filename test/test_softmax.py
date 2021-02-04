import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
sys.path.append("../lib/concerns")
from softmax import *

class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        x = np.array([
            [-0.27291637,  3.0623984 ,  1.08772839,  1.21167545],
            [ 0.77815361,  1.20011612, -0.37179735,  1.93945452],
            [-1.02360881, -0.23723418, -1.42713268, -0.6484095 ],
            [-0.6631865 ,  0.01433258, -2.450729  , -2.02298841]
        ])
        assert_almost_equal(np.array([
            [0.02673862, 0.75101348, 0.10424601, 0.11800189],
            [0.16568116, 0.2526557 , 0.05246332, 0.52919982],
            [0.18801706, 0.41277693, 0.12558827, 0.27361774],
            [0.29471841, 0.58029664, 0.04932731, 0.07565764]
        ]), softmax(x))

if __name__ == "__main__":
    unittest.main()
