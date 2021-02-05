import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
import os.path
from os import path
from train_custom_loop import TrainCustomLoop

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.train_custom_loop = TrainCustomLoop(max_epoch=300, batch_size=30, hidden_size=10)

    def test_shuffle_data(self):
        index, x, t = self.train_custom_loop._shuffle_data()
        self.assertEqual((300, 2), x.shape)
        self.assertEqual((300, 3), t.shape)

    def test_update_params_with_grads(self):
        batch_x = self.train_custom_loop.x[1 * self.train_custom_loop.batch_size: (1 + 1) * self.train_custom_loop.batch_size]
        batch_t = self.train_custom_loop.t[1 * self.train_custom_loop.batch_size: (1 + 1) * self.train_custom_loop.batch_size]
        self.train_custom_loop._update_params_with_grads(batch_x, batch_t)
        self.assertEqual(1.1074495352433567, self.train_custom_loop.total_loss)
        self.assertEqual(1, self.train_custom_loop.loss_count)

    def test_learning_process(self):
        batch_x = self.train_custom_loop.x[1 * self.train_custom_loop.batch_size: (1 + 1) * self.train_custom_loop.batch_size]
        batch_t = self.train_custom_loop.t[1 * self.train_custom_loop.batch_size: (1 + 1) * self.train_custom_loop.batch_size]
        self.train_custom_loop._update_params_with_grads(batch_x, batch_t)
        self.assertEqual("| epoch 10 | iter 10 / 10 | loss 1.10", self.train_custom_loop._learning_process(9, 9))

    def test_update(self):
        self.train_custom_loop.update()
        self.assertEqual(300, len(self.train_custom_loop.loss_list))

    def test_save_plot_image(self):
        self.train_custom_loop.update()
        self.train_custom_loop.save_plot_image("../img/train_plot.png")
        self.assertEqual(True, path.exists("../img/train_plot.png"))

    def test_save_dicision_boundary_image(self):
        self.train_custom_loop.update()
        self.train_custom_loop.save_dicision_boundary_image("../img/dicision_boundary.png")
        self.assertEqual(True, path.exists("../img/dicision_boundary.png"))

if __name__ == "__main__":
    unittest.main()
