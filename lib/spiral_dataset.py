import sys
sys.path.append("./concerns")
from spiral import *
import matplotlib.pyplot as plt

class SpiralDataset:
    def __init__(self):
        self.x, self.t = load_data()

    def save_plot_image(self, file_path):
        SAMPLE_NUMS_PER_CLASS = 100
        CLASS_NUNS = 3
        markers = ['o', 'x', '^']
        plt.figure()
        for i in range(CLASS_NUNS):
            plt.scatter(
                self.x[i*SAMPLE_NUMS_PER_CLASS:(i+1)*SAMPLE_NUMS_PER_CLASS, 0],
                self.x[i*SAMPLE_NUMS_PER_CLASS:(i+1)*SAMPLE_NUMS_PER_CLASS, 1],
                s=40,
                marker=markers[i]
            )
        plt.savefig(file_path)
