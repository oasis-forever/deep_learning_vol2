import numpy as np
import sys
sys.path.append("./concerns")
from sgd import SGD
from spiral import *
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

class TrainCustomLoop:
    def __init__(self, max_epoch, batch_size, hidden_size, learning_rate=1.0):
        #Hyper params
        self.max_epoch  = max_epoch
        self.batch_size = batch_size
        # Generate model, optimiser
        self.model      = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
        self.optimizer  = SGD(lr=learning_rate)
        # Load data
        self.x, self.t  = load_data()
        # Variables for training
        self.data_size  = len(self.x)
        self.max_iters  = self.data_size // batch_size
        self.total_loss = 0
        self.loss_count = 0
        self.loss_list  = []

    def _shuffle_data(self):
        index = np.random.permutation(self.data_size)
        x = self.x[index]
        t = self.t[index]
        return index, x, t

    def _update_params_with_grads(self, batch_x, batch_t):
        loss = self.model.forward(batch_x, batch_t)
        self.model.backward()
        self.optimizer.update(self.model.params, self.model.grads)
        self.total_loss += loss
        self.loss_count += 1

    def _learning_process(self, epoch, iters):
        avarage_loss = self.total_loss / self.loss_count
        self.loss_list.append(avarage_loss)
        self.total_loss = 0
        self.loss_count = 0
        return "| epoch %d | iter %d / %d | loss %.2f" % (epoch + 1, iters + 1, self.max_iters, avarage_loss)

    def update(self):
        for epoch in range(self.max_epoch):
            index, x, t = self._shuffle_data()
            for iters in range(self.max_iters):
                batch_x = x[iters * self.batch_size: (iters + 1) * self.batch_size]
                batch_t = t[iters * self.batch_size: (iters + 1) * self.batch_size]
                self._update_params_with_grads(batch_x, batch_t)
                if (iters + 1) % 10 == 0:
                    self._learning_process(epoch, iters)

    def save_plot_image(self, path):
        plt.figure()
        plt.plot(np.arange(len(self.loss_list)), self.loss_list, label='train')
        plt.xlabel("iterations (x10)")
        plt.ylabel("loss")
        plt.savefig(path)

    def save_dicision_boundary_image(self, path):
        # Plot boundary
        h = 0.001
        x_min, x_max = self.x[:, 0].min() - .1, self.x[:, 0].max() + .1
        y_min, y_max = self.x[:, 1].min() - .1, self.x[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        X = np.c_[xx.ravel(), yy.ravel()]
        score = self.model._predict(X)
        predict_cls = np.argmax(score, axis=1)
        Z = predict_cls.reshape(xx.shape)
        plt.contourf(xx, yy, Z)
        plt.axis('off')
        # Plot data points
        x, t = load_data()
        SAMPLE_NUMS_PER_CLASS = 100
        CLASS_NUNS = 3
        markers = ['o', 'x', '^']
        for i in range(CLASS_NUNS):
            plt.scatter(
                x[i * SAMPLE_NUMS_PER_CLASS:(i + 1) * SAMPLE_NUMS_PER_CLASS, 0],
                x[i * SAMPLE_NUMS_PER_CLASS:(i + 1) * SAMPLE_NUMS_PER_CLASS, 1],
                s=40,
                marker=markers[i]
            )
        plt.savefig(path)
