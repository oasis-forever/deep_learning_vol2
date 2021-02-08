import numpy as np

class Matrix:
    def __init__(self, W):
        self.W = W

    def calc_sum(self, X):
        return self.W + X

    def calc_product(self, X):
        return self.W * X

    def calc_scala_broadcast(self, num):
        return self.W * num

    def calc_array_broadcast(self, X):
        return self.W * X

    def calc_inner_product(self, X):
        return np.dot(self.W, X)
