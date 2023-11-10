import numpy as np

class Sigmoid:
    def __call__(self, z) -> float:
        return 1 / 1 + np.exp(-z)
    