import unittest
from os import path
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from spiral_dataset import SpiralDataset

class TestSpiralDataset(unittest.TestCase):
    def setUp(self):
        self.spiral_dataset = SpiralDataset()

    def test_x_shape(self):
        self.assertEqual((300, 2), self.spiral_dataset.x.shape)

    def test_t_shape(self):
        self.assertEqual((300, 3), self.spiral_dataset.t.shape)

    def test_save_plot_image(self):
        file_path = "../img/spiral_plot.png"
        self.spiral_dataset.save_plot_image(file_path)
        self.assertEqual(True, path.exists(file_path))

if __name__ == "__main__":
    unittest.main()
