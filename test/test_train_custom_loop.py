import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
import os.path
from os import path
from train_custom_loop import TrainCustomLoop
from spiral import *

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.train_custom_loop = TrainCustomLoop()
        self.x, self.t = load_data()
        self.max_epoch = 300
        self.batch_size = 30
        self.batch_x = self.x[1 * self.batch_size: (1 + 1) * self.batch_size]
        self.batch_t = self.t[1 * self.batch_size: (1 + 1) * self.batch_size]
        self.total_loss = 0
        self.loss_count = 0

    def test_shuffle_data(self):
        xx, tt = self.train_custom_loop._shuffle_data(self.x, self.t)
        self.assertEqual((300, 2), xx.shape)
        self.assertEqual((300, 3), tt.shape)

    def test_update_params_with_grads(self):
        loss = self.train_custom_loop._update_params_with_grads(self.batch_x, self.batch_t, self.total_loss, self.loss_count)
        self.assertEqual(1.1074495352433567, loss)

    def test_learning_process(self):
        loss = self.train_custom_loop._update_params_with_grads(self.batch_x, self.batch_t, self.total_loss, self.loss_count)
        self.total_loss += loss
        self.loss_count += 1
        epoch = 9
        iters = 9
        data_size = len(self.x)
        max_iters = data_size // self.batch_size
        *_, process = self.train_custom_loop._learning_process(self.total_loss, self.loss_count, epoch, iters, max_iters)
        self.assertEqual("| epoch 10 | iter 10 / 10 | loss 1.10", process)

    def test_update(self):
        loss_list = self.train_custom_loop.update(self.x, self.t, self.max_epoch, self.batch_size)
        self.assertEqual(300, len(loss_list))

    def test_save_plot_image(self):
        loss_list = self.train_custom_loop.update(self.x, self.t, self.max_epoch, self.batch_size)
        self.train_custom_loop.save_plot_image(loss_list, "../img/train_custom_loop_plot.png")
        self.assertEqual(True, path.exists("../img/train_custom_loop_plot.png"))

    def test_save_dicision_boundary_image(self):
        self.train_custom_loop.update(self.x, self.t, self.max_epoch, self.batch_size)
        self.train_custom_loop.save_dicision_boundary_image(self.x, self.t, "../img/dicision_boundary.png")
        self.assertEqual(True, path.exists("../img/dicision_boundary.png"))

if __name__ == "__main__":
    unittest.main()
