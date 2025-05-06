import numpy as np

class ZScoreModel:
    def __init__(self):
        pass

    def run(self, X_train, X_test):
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        z_test = np.mean((X_test - mean) / std, axis=1)
        return None, z_test
