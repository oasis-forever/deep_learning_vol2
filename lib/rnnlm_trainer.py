import numpy as np
import time
import matplotlib.pyplot as plt
from clip_grads import *

class RNNLMTrainer:
    def __init__(self, model, optimizer):
        self.model         = model
        self.optimizer     = optimizer
        self.time_index    = 0
        self.ppl_list      = []
        self.eval_interval = None
        self.current_epoch = 0

    def _remove_duplicate(self, params, grads):
        """
        Integrate duplicate weights into one and add gradients corresponding to the weights.
        """
        # Copy list
        params = self.model.params[:]
        grads  = self.model.grads[:]
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

    def _get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")
        data_size = len(x)
        jump = data_size // batch_size
        # Initial position to load each sample of mini batches
        offsets = [i * jump for i in range(batch_size)]
        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_index) % data_size]
                batch_t[i, time] = t[(offset + self.time_index) % data_size]
            self.time_index += 1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.eval_interval = eval_interval
        total_loss = 0
        loss_count = 0
        start_time = time.time()
        training_process = []
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self._get_batch(xs, ts, batch_size, time_size)
                # Calculate gradients and update parameters
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                params, grads = self._remove_duplicate(self.model.params, self.model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                self.optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
                # Eveluate perplexity
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    training_process.append("| Epoch %d |  Iter %d / %d | Time %d[s] | Perplexity %.2f" % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss = 0
                    loss_count = 0
            self.current_epoch += 1
        return training_process

    def save_plot_image(self, file_path, ylim=None):
        plt.figure()
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label="train")
        plt.xlabel("Iterations (x{})".format(str(self.eval_interval)))
        plt.ylabel("Perplexity")
        plt.savefig(file_path)
