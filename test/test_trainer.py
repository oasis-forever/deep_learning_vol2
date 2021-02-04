import unittest
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
import os.path
from os import path
from trainer import Trainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = Trainer(max_epoch=300, batch_size=30, hidden_size=10)

    def test_shuffle_data(self):
        index, x, t = self.trainer._shuffle_data()
        self.assertEqual((300,), index.shape)
        self.assertEqual((300, 2), x.shape)
        self.assertEqual((300, 3), t.shape)

    def test_update_params_with_grads(self):
        batch_x = self.trainer.x[1 * self.trainer.batch_size: (1 + 1) * self.trainer.batch_size]
        batch_t = self.trainer.t[1 * self.trainer.batch_size: (1 + 1) * self.trainer.batch_size]
        self.trainer._update_params_with_grads(batch_x, batch_t)
        self.assertEqual(1.1074495352433567, self.trainer.total_loss)
        self.assertEqual(1, self.trainer.loss_count)

    def test_learning_process(self):
        batch_x = self.trainer.x[1 * self.trainer.batch_size: (1 + 1) * self.trainer.batch_size]
        batch_t = self.trainer.t[1 * self.trainer.batch_size: (1 + 1) * self.trainer.batch_size]
        self.trainer._update_params_with_grads(batch_x, batch_t)
        self.assertEqual("| epoch 10 | iter 10 / 10 | loss 1.10", self.trainer._learning_process(9, 9))

    def test_update(self):
        self.trainer.update()
        self.assertEqual(300, len(self.trainer.loss_list))

    def test_save_plot_image(self):
        self.trainer.update()
        self.trainer.save_plot_image("../img/train_plot.png")
        self.assertEqual(True, path.exists("../img/train_plot.png"))

    def test_save_dicision_boundary_image(self):
        self.trainer.update()
        self.trainer.save_dicision_boundary_image("../img/dicision_boundary.png")
        self.assertEqual(True, path.exists("../img/dicision_boundary.png"))

if __name__ == "__main__":
    unittest.main()
