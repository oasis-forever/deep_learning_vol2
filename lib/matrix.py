import numpy as np

class Matrix:
    def __init__(self, W):
        self.W = W

    def calc_sum(self, X):
        sum = self.W + X
        return sum

    def calc_product(self, X):
        product = self.W * X
        return product

    def calc_scala_broadcast(self, num):
        scala_broadcast = self.W * num
        return scala_broadcast

    def calc_array_broadcast(self, X):
        array_broadcast = self.W * X
        return array_broadcast

    def calc_inner_product(self, X):
        inner_product = np.dot(self.W, X)
        return inner_product
