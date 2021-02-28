import numpy as np

class Vector:
    def __init__(self, x):
        self.x = x

    def calc_inner_product(self, y):
        inner_product = np.dot(self.x, y)
        return inner_product
