from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from models import autoencoder, model_prep
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.model_prep import Model
from utils import reporter
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score

def getSubject(HC, y_HC, X, subject, original, insert=False):
    X_train_split = HC.loc[HC['ID'] != subject]
    y_train_split = y_HC.loc[y_HC['ID'] != subject]

    if insert:
        X_train_split = pd.concat([X_train_split, X.loc[X['ID'] == original]])
        y_train_split = pd.concat([y_train_split, X_train_split[['Group', 'ID']]])

    X_test_split = X.loc[X['ID'] == subject]
    y_test_split = X_test_split[['Group', 'ID']]

    X_train_split = X_train_split.drop(['Group', 'ID'], axis=1)
    X_test_split = X_test_split.drop(['Group', 'ID'], axis=1)

    return X_train_split, y_train_split, X_test_split, y_test_split

def run(subject, df_data, df_demog, regress, tracts, hemi, metric, model_type='AutoEncoder'):
    if model_type == "AutoEncoder":
        from models.autoencoder import AutoEncoderModel as Model
    elif model_type == "PCA":
        from models.pca import PCAModel as Model
    elif model_type == "ZScore":
        from models.zscore import ZScoreModel as Model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    st.warning("Computing permutations ... estimated time: " + str(np.round(len(df_demog)*2/60, 2)) + " minutes.")

    # 1. Select features
    X = df_data.loc[:, df_data.columns.str.startswith('Group') |
                    df_data.columns.str.startswith('ID') |
                    df_data.columns.str.contains('|'.join(tracts))]

    # 2. Separate healthy controls
    HC = X[X['Group'] == 0]
    y_HC = HC[['Group', 'ID']]
    X_train, y_train, X_test, y_test = getSubject(HC, y_HC, X, subject, False)

    # 3. Normalize
    scaler, X_train, X_test = model_prep.normalize_features(X_train, X_test, "void")

    # 4. Regress confounds
    if regress:
        if 'sex' in df_demog and 'age' in df_demog:
            X_train, X_test = model_prep.regress_confound(X_train, X_test, df_demog)
        else:
            st.error("No age or sex information found. Skipping regression step.")

    # 5. Model initialization and execution
    if model_type == "ZScore":
        model = Model()
        z_train, z_test = model.run(X_train, X_test)
        return None, None, z_test, None, None, None

    elif model_type == "PCA":
        model = Model(n_components=2)
        model.fit(X_train)
        X_test_transformed = model.transform(X_test)
        return X_test_transformed, None, None, None, None, None

    elif model_type == "AutoEncoder":
        model = Model(X_train, X_test, "Autoencoder")
        x_hat = model.run_once()

        x_hat_inv = scaler.inverse_transform(x_hat)
        x_inv = scaler.inverse_transform(X_test)
        mae = np.mean(np.abs(x_hat_inv - x_inv), axis=1)
        sub_orig = x_hat_inv - x_inv

        p = np.zeros(len(sub_orig[0]))
        count = 0

        for s in y_HC['ID'].values:
            st.write("Computing permutations (LOOCV) with", s)
            X_train, y_train, X_test, y_test = getSubject(HC, y_HC, X, s, subject, True)
            scaler, X_train, X_test = model_prep.normalize_features(X_train, X_test, "void")

            if regress:
                if 'sex' in df_demog and 'age' in df_demog:
                    X_train, X_test = model_prep.regress_confound(X_train, X_test, df_demog)

            model = Model(X_train, X_test, "Autoencoder")
            k_hat = model.run_once()

            k_hat_inv = scaler.inverse_transform(k_hat)
            k_inv = scaler.inverse_transform(X_test)
            k_mae = np.mean(np.abs(k_hat_inv - k_inv), axis=1)
            sub = k_hat_inv - k_inv

            for e in range(len(sub_orig[0])):
                if sub_orig[0][e] > 0:
                    if sub[0][e] >= sub_orig[0][e]:
                        p[e] += 1
                else:
                    if sub[0][e] < sub_orig[0][e]:
                        p[e] += 1

            if np.mean(k_mae) > np.mean(mae):
                count += 1

        p_div = len(X_train) + len(X_test)
        overall_p = count / p_div
        p_crit = p / p_div
        p_along = np.zeros(len(p_crit))

        max_p = max(0.01, 1 / p_div)
        for i in range(len(p_crit)):
            if p_crit[i] <= max_p:
                p_along[i] = 1

        return x_inv, x_hat_inv, mae, p_along, overall_p, p_div
