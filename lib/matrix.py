import numpy as np

class Matrix:
    def __init__(self, arr):
        self.W = np.array(arr)

    def get_class_name(self):
        return str(self.W.__class__)

    def get_shape(self):
        return self.W.shape

    def get_dim(self):
        return self.W.ndim

    def calc_sum(self, X):
        return self.W + np.array(X)

    def calc_product(self, X):
        return self.W * np.array(X)

    def calc_scala_broadcast(self, num):
        return self.W * num

    def calc_array_broadcast(self, arr):
        return self.W * np.array(arr)
