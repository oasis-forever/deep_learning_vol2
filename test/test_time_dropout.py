import unittest
import numpy as np
import sys
sys.path.append("../lib/layers")
from time_dropout import TimeDropout

class TestTimeDropout(unittest.TestCase):
    def setUp(self):
        self.time_dropout = TimeDropout()
        self.xs = np.array([
            [-2.02263879,  1.79293276, -0.64214657],
            [-0.74505721, -0.81903631, -0.27078458],
            [-0.16396182, -0.19952967, -0.88905705],
            [ 0.70678772, -0.71987105,  2.28794441],
            [-0.79917515, -0.53706248,  0.15224767],
            [-0.73616716,  0.05082873, -0.54139266],
            [-1.12802119,  1.3867995 ,  0.42552788]
        ])

    def test_forward(self):
        out = self.time_dropout.forward(self.xs)
        self.assertEqual((7, 3), out.shape)

    def test_backward(self):
        dout = self.time_dropout.forward(self.xs)
        dx = self.time_dropout.backward(dout)
        self.assertEqual((7, 3), dx.shape)

if __name__ == "__main__":
    unittest.main()
