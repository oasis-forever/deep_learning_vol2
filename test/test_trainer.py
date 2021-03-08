import unittest
import numpy as np
from numpy.testing import assert_array_equal
import time
from os import path
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
sys.path.append("../lib/models")
sys.path.append("../lib/optimisers")
from trainer import Trainer
from spiral import *
from two_layer_net import TwoLayerNet
from sgd import SGD

class TestTrainer(unittest.TestCase):
    def setUp(self):
        model          = TwoLayerNet(input_size=2, hidden_size=10, output_size=3)
        optimizer      = SGD(lr=1.0)
        self.trainer   = Trainer(model, optimizer)
        self.x, self.t = load_data()
        self.data_size = len(self.x)

    def test_shuffle_data(self):
        xx, tt = self.trainer._shuffle_data(self.data_size, self.x, self.t)
        self.assertEqual((300, 2), np.array(xx).shape)
        self.assertEqual((300, 3), np.array(tt).shape)

    def test_calculate_loss(self):
        batch_size = 32
        xx, tt = self.trainer._shuffle_data(self.data_size, self.x, self.t)
        batch_x = xx[1 * batch_size: (1 + 1) * batch_size]
        batch_t = tt[1 * batch_size: (1 + 1) * batch_size]
        loss = self.trainer._calculate_loss(batch_x, batch_t)
        self.assertEqual(1.1, round(loss, 1))

    def test_remove_duplicate(self):
        batch_size = 32
        xx, tt = self.trainer._shuffle_data(self.data_size, self.x, self.t)
        batch_x = xx[1 * batch_size: (1 + 1) * batch_size]
        batch_t = tt[1 * batch_size: (1 + 1) * batch_size]
        loss = self.trainer._calculate_loss(batch_x, batch_t)
        params, grads = self.trainer._remove_duplicate()
        param_1, param_2, param_3, param_4 = params
        grad_1, grad_2, grad_3, grad_4 = grads
        self.assertEqual((2, 10), param_1.shape)
        self.assertEqual((10,), param_2.shape)
        self.assertEqual((10, 3), param_3.shape)
        self.assertEqual((3,), param_4.shape)
        self.assertEqual((2, 10), grad_1.shape)
        self.assertEqual((10,), grad_2.shape)
        self.assertEqual((10, 3), grad_3.shape)
        self.assertEqual( (3,), grad_4.shape)

    def test_evaluate(self):
        batch_size = 32
        xx, tt = self.trainer._shuffle_data(self.data_size, self.x, self.t)
        batch_x = xx[1 * batch_size: (1 + 1) * batch_size]
        batch_t = tt[1 * batch_size: (1 + 1) * batch_size]
        total_loss = 0
        loss_count = 0
        loss = self.trainer._calculate_loss(batch_x, batch_t)
        total_loss += loss
        loss_count += 1
        params, grads = self.trainer._remove_duplicate()
        start_time = time.time()
        current_epoch = 0
        max_iters = self.data_size // batch_size
        avarage_loss, training_status = self.trainer._evaluate(total_loss, loss_count, start_time, current_epoch, 1, max_iters)
        self.assertEqual(1.0982153338384055, avarage_loss)
        self.assertEqual("| epoch 1 |  iter 2 / 9 | time 0[s] | loss 1.10", training_status)

    def test_fit(self):
        loss_list, training_process = self.trainer.fit(self.x, self.t)
        assert_array_equal(np.array([
            1.095670688853155,
            1.166632183444487,
            1.1176750111873825,
            1.1790605366611793,
            1.1308071491715788,
            1.1796679902482832,
            1.1401573329955508,
            1.1263507954471639,
            1.1260232915246664,
            1.1018855740540143
        ]), np.array(loss_list))
        self.assertEqual([
            "| epoch 1 |  iter 1 / 9 | time 0[s] | loss 1.10",
            "| epoch 2 |  iter 1 / 9 | time 0[s] | loss 1.17",
            "| epoch 3 |  iter 1 / 9 | time 0[s] | loss 1.12",
            "| epoch 4 |  iter 1 / 9 | time 0[s] | loss 1.18",
            "| epoch 5 |  iter 1 / 9 | time 0[s] | loss 1.13",
            "| epoch 6 |  iter 1 / 9 | time 0[s] | loss 1.18",
            "| epoch 7 |  iter 1 / 9 | time 0[s] | loss 1.14",
            "| epoch 8 |  iter 1 / 9 | time 0[s] | loss 1.13",
            "| epoch 9 |  iter 1 / 9 | time 0[s] | loss 1.13",
            "| epoch 10 |  iter 1 / 9 | time 0[s] | loss 1.10"
        ], training_process)

    def test_save_plot_image(self):
        loss_list, *_ = self.trainer.fit(self.x, self.t, max_epoch=300, batch_size=30)
        file_path = "../img/training_plot.png"
        self.trainer.save_plot_image(loss_list, file_path)
        self.assertEqual(True, path.exists(file_path))

if __name__ == "__main__":
    unittest.main()
