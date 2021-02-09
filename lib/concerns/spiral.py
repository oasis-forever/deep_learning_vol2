import numpy as np

def load_data(seed=1984):
    np.random.seed(seed)
    SAMPLE_NUMS_PER_CLASS = 100
    DATA_ELEM_NUMS = 2
    CLASS_NUNS = 3
    x = np.zeros((SAMPLE_NUMS_PER_CLASS * CLASS_NUNS, DATA_ELEM_NUMS))
    t = np.zeros((SAMPLE_NUMS_PER_CLASS * CLASS_NUNS, CLASS_NUNS), dtype=np.int32)
    for j in range(CLASS_NUNS):
        for i in range(SAMPLE_NUMS_PER_CLASS):
            rate = i / SAMPLE_NUMS_PER_CLASS
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0  * rate + np.random.randn() * 0.2
            ix = SAMPLE_NUMS_PER_CLASS * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix, j] = 1
    return x, t
