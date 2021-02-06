import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
import numpy as np
import time
import os.path
from os import path
from two_layer_net import TwoLayerNet
from sgd import SGD
from spiral import *
from trainer import Trainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.max_epoch  = 300
        self.batch_size = 30
        hidden_size     = 10
        learning_rate   = 1.0
        model           = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
        optimizer       = SGD(lr=learning_rate)
        self.trainer    = Trainer(model, optimizer)
        self.x, self.t = load_data()
        self.data_size = len(self.x)
        xx, tt = self.trainer._shuffle_data(self.data_size, self.x, self.t)
        self.batch_x = xx[1 * self.batch_size: (1 + 1) * self.batch_size]
        self.batch_t = tt[1 * self.batch_size: (1 + 1) * self.batch_size]
        self.loss = self.trainer._calculate_loss(self.batch_x, self.batch_t)
        self.params, self.grads = self.trainer._remove_duplicate()
        self.trainer._update_params_with_grads(self.params, self.grads, self.loss)
        self.training_process = self.trainer.fit(self.x, self.t, self.max_epoch, self.batch_size, eval_interval=10)

    def test_shuffle_data(self):
        self.assertEqual((300, 2), np.array(self.x).shape)
        self.assertEqual((300, 3), np.array(self.t).shape)

    def test_calculate_loss(self):
        self.assertEqual(1.1, float("{:.1f}".format(self.loss)))

    def test_remove_duplicate(self):
        param_1, param_2, param_3, param_4 = self.params
        grad_1, grad_2, grad_3, grad_4 = self.grads
        self.assertEqual((2, 10), param_1.shape)
        self.assertEqual((10,), param_2.shape)
        self.assertEqual((10, 3), param_3.shape)
        self.assertEqual((3,), param_4.shape)
        self.assertEqual((2, 10), grad_1.shape)
        self.assertEqual((10,), grad_2.shape)
        self.assertEqual((10, 3), grad_3.shape)
        self.assertEqual( (3,), grad_4.shape)

    def test_update_params_with_grads(self):
        self.assertEqual(1.0684538647607518, self.trainer.total_loss)
        self.assertEqual(9, self.trainer.loss_count)

    def test_evaluate(self):
        start_time = time.time()
        max_iters = self.data_size // self.batch_size
        self.assertEqual("| epoch 301 |  iter 2 / 10 | time 0[s] | loss 0.12", self.trainer._evaluate(start_time, 1, max_iters))

    def test_fit(self):
        self.assertEqual(300, len(self.training_process))

    def test_save_plot_image(self):
        self.trainer.save_plot_image("../img/training_plot.png")
        self.assertEqual(True, path.exists("../img/training_plot.png"))

if __name__ == "__main__":
    unittest.main()
