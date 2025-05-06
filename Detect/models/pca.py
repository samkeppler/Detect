import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

class PCAModel:
    def __init__(self, n_components=0.85):
        self.pca = PCA(n_components=n_components)
        self.mean_ = None
        self.inv_cov_ = None

    def fit(self, X):
        X_pca = self.pca.fit_transform(X)
        self.mean_ = np.mean(X_pca, axis=0)
        cov = np.cov(X_pca, rowvar=False)
        self.inv_cov_ = np.linalg.pinv(cov)

    def transform(self, X):
        X_pca = self.pca.transform(X)
        scores = [mahalanobis(x, self.mean_, self.inv_cov_) for x in X_pca]
        return np.array(scores)
