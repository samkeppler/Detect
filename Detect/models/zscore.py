from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns

def save(model):
    pass

def run(model):
    X_train = model.get_train()
    X_test = model.get_test()

    meanProfile = np.mean(X_train)
    stdProfile = np.std(X_train)

    zscoreTrain = np.mean((X_train - meanProfile)/stdProfile, axis=1)
    zscoreTest = np.mean((X_test - meanProfile)/stdProfile, axis=1)

    return zscoreTrain, zscoreTest

class ZScoreModel:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X_train):
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)

    def run(self, X_train, X_test):
        if self.mean is None or self.std is None:
            self.fit(X_train)

        z_test = np.mean((X_test - self.mean) / self.std, axis=1)
        return None, z_test