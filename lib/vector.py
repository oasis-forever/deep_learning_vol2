import numpy as np

class Vector:
    def __init__(self, x):
        self.x = x

    def get_class_name(self):
        return str(self.x.__class__)

    def get_shape(self):
        return self.x.shape

    def get_dim(self):
        return self.x.ndim

    def calc_inner_product(self, y):
        return np.dot(self.x, y)
