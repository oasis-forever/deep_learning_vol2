import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("./models")
sys.path.append("./optimisers")
from sgd import SGD
from two_layer_net import TwoLayerNet

class TrainCustomLoop:
    def __init__(self, input_size=2, hidden_size=10, output_size=3, learning_rate=1.0):
        # Generate model, optimiser
        self.model     = TwoLayerNet(input_size, hidden_size, output_size)
        self.optimizer = SGD(learning_rate)
        self.loss_list = []

    def _shuffle_data(self, x, t):
        data_size = len(x)
        index = np.random.permutation(data_size)
        xx = x[index]
        tt = t[index]
        return xx, tt

    def _update_params_with_grads(self, batch_x, batch_t, total_loss, loss_count):
        loss = self.model.forward(batch_x, batch_t)
        self.model.backward()
        self.optimizer.update(self.model.params, self.model.grads)
        return loss

    def _learning_process(self, total_loss, loss_count, epoch, iters, max_iters):
        avarage_loss = total_loss / loss_count
        process = "| epoch %d | iter %d / %d | loss %.2f" % (epoch + 1, iters + 1, max_iters, avarage_loss)
        return avarage_loss, process

    def update(self, x, t, max_epoch, batch_size):
        data_size = len(x)
        max_iters = data_size // batch_size
        total_loss = 0
        loss_count = 0
        for epoch in range(max_epoch):
            xx, tt = self._shuffle_data(x, t)
            for iters in range(max_iters):
                batch_x = xx[iters * batch_size: (iters + 1) * batch_size]
                batch_t = tt[iters * batch_size: (iters + 1) * batch_size]
                loss = self._update_params_with_grads(batch_x, batch_t, total_loss, loss_count)
                total_loss += loss
                loss_count += 1
                if (iters + 1) % 10 == 0:
                    avarage_loss, *_ = self._learning_process(total_loss, loss_count, epoch, iters, max_iters)
                    self.loss_list.append(avarage_loss)
                    total_loss = 0
                    loss_count = 0
        return self.loss_list

    def save_plot_image(self, file_path):
        plt.figure()
        plt.plot(np.arange(len(self.loss_list)), self.loss_list, label='train')
        plt.xlabel("iterations (x10)")
        plt.ylabel("loss")
        plt.savefig(file_path)

    def save_dicision_boundary_image(self, x, t, file_path, h=0.001):
        # Plot boundary
        x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
        y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        X = np.c_[xx.ravel(), yy.ravel()]
        score = self.model._predict(X)
        predict_cls = np.argmax(score, axis=1)
        Z = predict_cls.reshape(xx.shape)
        plt.contourf(xx, yy, Z)
        plt.axis('off')
        # Plot data points
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
        plt.savefig(file_path)
