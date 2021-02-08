import numpy as np

class Vector:
    def __init__(self, x):
        self.x = x

    def calc_inner_product(self, y):
        return np.dot(self.x, y)
