import os
import pickle
import numpy as np

class BaseModel:
    def __init__(self):
        self.params = None
        self.grads  = None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_path=None):
        if file_path is None:
            file_path = "../../pkl/{}.pkl".format(self.__class__.__name__).lower()
        params = [p.astype(np.float16) for p in self.params]
        with open(file_path, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_path=None):
        if file_path is None:
            file_path = "../../pkl/{}.pkl".format(self.__class__.__name__).lower()
        if "/" in file_path:
            file_path = file_path.replace("/", os.sep)
        if not os.path.exists(file_path):
            raise IOError("No file: {}".format(file_path))
        with open(file_path, "rb") as f:
            params = pickle.load(f)
        params = [p.astype("f") for p in params]
        for i, param in enumerate(self.params):
            param[...] = params[i]
