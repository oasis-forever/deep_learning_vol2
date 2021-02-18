import unittest
import os.path
from os import path
import sys
sys.path.append("../lib")
from spiral_dataset import SpiralDataset
sys.path.append("../lib/concerns")

class TestSpiralDataset(unittest.TestCase):
    def setUp(self):
        self.spiral_dataset = SpiralDataset()

    def test_x_shape(self):
        self.assertEqual((300, 2), self.spiral_dataset.x.shape)

    def test_t_shape(self):
        self.assertEqual((300, 3), self.spiral_dataset.t.shape)

    def test_save_plot_image(self):
        self.spiral_dataset.save_plot_image("../img/spiral_plot.png")
        self.assertEqual(True, path.exists("../img/spiral_plot.png"))

if __name__ == "__main__":
    unittest.main()
