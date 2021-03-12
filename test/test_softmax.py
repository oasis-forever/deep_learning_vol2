import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
sys.path.append("../lib/layers")
from softmax import Softmax

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.softmax = Softmax()
        self.x = np.array([
            [-0.27291637,  3.0623984 ,  1.08772839,  1.21167545],
            [ 0.77815361,  1.20011612, -0.37179735,  1.93945452],
            [-1.02360881, -0.23723418, -1.42713268, -0.6484095 ],
            [-0.6631865 ,  0.01433258, -2.450729  , -2.02298841]
        ])

    def test_calcsoftmax(self):
        x = self.softmax.calc_softmax(self.x)
        assert_almost_equal(np.array([
            [0.02673862, 0.75101348, 0.10424601, 0.11800189],
            [0.16568116, 0.2526557 , 0.05246332, 0.52919982],
            [0.18801706, 0.41277693, 0.12558827, 0.27361774],
            [0.29471841, 0.58029664, 0.04932731, 0.07565764]
        ]), x)

    def test_forward(self):
        self.softmax.forward(self.x)
        assert_almost_equal(np.array([
            [0.02673862, 0.75101348, 0.10424601, 0.11800189],
            [0.16568116, 0.2526557 , 0.05246332, 0.52919982],
            [0.18801706, 0.41277693, 0.12558827, 0.27361774],
            [0.29471841, 0.58029664, 0.04932731, 0.07565764]
        ]), self.softmax.out)

    def test_backward(self):
        self.softmax.forward(self.x)
        dout = np.array([
            [ 0.11843554, -1.15122357,  1.47778478, -1.61246747],
            [ 1.42841483,  0.51888186,  0.18154817,  0.37469379],
            [-0.37009244,  0.21842416, -0.72251804, -0.20918206],
            [-1.47353003, -0.08212526,  0.90979081,  1.11006032]
        ])
        dx = self.softmax.backward(dout)
        assert_almost_equal(np.array([
            [0.0267386, 0.7510135, 0.104246 , 0.1180019],
            [0.1656812, 0.2526557, 0.0524633, 0.5291998],
            [0.1880171, 0.4127769, 0.1255883, 0.2736177],
            [0.2947184, 0.5802966, 0.0493273, 0.0756576]
        ]), self.softmax.out)

if __name__ == "__main__":
    unittest.main()
