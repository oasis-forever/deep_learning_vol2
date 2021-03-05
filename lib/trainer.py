import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("./concerns")
from clip_grads import *

class Trainer:
    def __init__(self, model, optimizer):
        self.model     = model
        self.optimizer = optimizer

    def _shuffle_data(self, data_size, x, t):
        index = np.random.permutation(data_size)
        xx = x[index]
        tt = t[index]
        return xx, tt

    def _calculate_loss(self, batch_x, batch_t):
        loss = self.model.forward(batch_x, batch_t)
        self.model.backward()
        return loss

    def _remove_duplicate(self):
        """
        Integrate duplicate weights into one and add gradients corresponding to the weights.
        """
        # Copy list
        params = self.model.params[:]
        grads = self.model.grads[:]
        while True:
            find_flag = False
            L = len(params)
            for i in range(0, L - 1):
                for j in range(i + 1, L):
                    # In case of sharing heaviness
                    if params[i] is params[j]:
                        # Add gradient
                        grads[i] += grads[j]
                        find_flag = True
                        params.pop(j)
                        grads.pop(j)
                    # In case of sharing heaviness as transpose list(weight typing)
                    elif params[i].ndim == 2 and params[j].ndim == 2 and \
                         params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                        grads[i] += grads[j].T
                        find_flag = True
                        params.pop(j)
                        grads.pop(j)
                    if find_flag: break
                if find_flag: break
            if not find_flag: break
        return params, grads

    def _evaluate(self, total_loss, loss_count, start_time, current_epoch, iters, max_iters):
        avarage_loss = (total_loss / loss_count)
        elapsed_time = time.time() - start_time
        training_status = "| epoch %d |  iter %d / %d | time %d[s] | loss %.2f" % (current_epoch + 1, iters + 1, max_iters, elapsed_time, avarage_loss)
        return avarage_loss, training_status

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        total_loss = 0
        loss_count = 0
        data_size = len(x)
        max_iters = data_size // batch_size
        start_time = time.time()
        loss_list = []
        training_process = []
        current_epoch = 0
        for epoch in range(max_epoch):
            xx, tt = self._shuffle_data(data_size, x, t)
            for iters in range(max_iters):
                batch_x = xx[iters * batch_size: (iters + 1) * batch_size]
                batch_t = tt[iters * batch_size: (iters + 1) * batch_size]
                loss = self._calculate_loss(batch_x, batch_t)
                params, grads = self._remove_duplicate()
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                self.optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avarage_loss, training_status = self._evaluate(total_loss, loss_count, start_time, current_epoch, iters, max_iters)
                    total_loss = 0
                    loss_count = 0
                    loss_list.append(avarage_loss)
                    training_process.append(training_status)
            current_epoch += 1
        return loss_list, training_process

    def save_plot_image(self, loss_list, file_path, eval_interval=20, ylim=None):
        plt.figure()
        x = np.arange(len(loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, loss_list, label="train")
        plt.xlabel("iterations (x{})".format(str(eval_interval)))
        plt.ylabel("loss")
        plt.savefig(file_path)
