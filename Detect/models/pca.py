from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

def save(model):
    pass

def covar_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    diff = data - mean_distr
    md = np.zeros(len(diff))
    for i in range(len(diff)):
        md[i] = np.sqrt(diff[i].dot(inv_cov_matrix).dot(diff[i]))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = [i for i, d in enumerate(dist) if d >= threshold]
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    return np.mean(dist) * k

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    return False

def run(model):
    X_train = model.get_train()
    X_test = model.get_test()
    pca = PCA(0.85, svd_solver='full')

    X_train_PCA = pca.fit_transform(X_train)
    X_test_PCA = pca.transform(X_test)

    st.write("Explained variance:", pca.explained_variance_ratio_)

    data_train = X_train_PCA
    data_test = X_test_PCA

    cov_matrix, inv_cov_matrix = covar_matrix(data_train)
    mean_distr = np.mean(data_train, axis=0)

    Mob_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test)
    M_test = np.squeeze(Mob_test)

    Mob_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train)
    M_train = np.squeeze(Mob_train)

    return M_train, M_test

# Wrapper class for use in inspector.py
class PCAModel:
    def __init__(self, n_components=0.85):
        self.n_components = n_components
        self.train_scores = None
        self.test_scores = None

    def fit(self, X_train):
        self.X_train = X_train

    def transform(self, X_test):
        class DummyModel:
            def __init__(self, train, test):
                self.X_train = train
                self.X_test = test

            def get_train(self):
                return self.X_train

            def get_test(self):
                return self.X_test

        dummy = DummyModel(self.X_train, X_test)
        M_train, M_test = run(dummy)
        self.train_scores = M_train
        self.test_scores = M_test
        return np.array(M_test)

    def inverse_transform(self, X):
        # Not used with Mahalanobis-based PCA anomaly detection
        return X
