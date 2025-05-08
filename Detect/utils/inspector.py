from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import streamlit as st
from sklearn.preprocessing import StandardScaler
from models import model_prep
from models.model_prep import Model
from models.pca import PCAModel
from utils import reporter

def getSubject(HC, y_HC, X, subject, original, insert=False):
    X_train_split = HC.loc[HC['ID'] != subject]
    y_train_split = y_HC.loc[y_HC['ID'] != subject]
    
    if insert and not (X_train_split['ID'] == original).any():
        original_row = X.loc[X['ID'] == original]
        X_train_split = pd.concat([X_train_split, original_row])
        y_train_split = pd.concat([y_train_split, original_row[['Group', 'ID']]])
    
    X_test_split = X.loc[X['ID'] == subject]
    y_test_split = X_test_split[['Group', 'ID']]
    
    X_train_split = X_train_split.drop(['Group', 'ID'], axis=1)
    X_test_split = X_test_split.drop(['Group', 'ID'], axis=1)
    
    return X_train_split, y_train_split, X_test_split, y_test_split

def run(subject, df_data, df_demog, regress, tracts, hemi, metric):
    st.warning("Computing permutations ... estimated time: " + str(np.round(len(df_demog)*2/60,2)) + " minutes.")

    #1 Select features
    X = df_data.loc[:, df_data.columns.str.startswith('Group') | 
                    df_data.columns.str.startswith('ID') |
                    df_data.columns.str.contains('|'.join(tracts))]

    #Separate HC from PATIENTS
    HC = X[X['Group'] == 0]
    y_HC = HC[['Group', 'ID']]
    
    X_train_split, y_train_split, X_test_split, y_test_split = getSubject(HC, y_HC, X, subject, False)
    scaler, X_train_split, X_test_split = model_prep.normalize_features(X_train_split, X_test_split, "void")

    #3 Linear regression of confound
    if regress:
        if 'sex' in df_demog and 'age' in df_demog:
            X_train, X_test = model_prep.regress_confound(X_train_split, X_test_split, df_demog)
        else:
            st.error("No age or sex information found. Skipping regression step.")
    
    #6 Run 
    model = Model(X_train, X_test, "PCAModel")
    z = model.run_once()
    mae = np.abs(z).flatten()
    x_hat = z
    x_hat_inv = np.zeros_like(X_test)
    x_inv = X_test
    sub_orig = z.T

    #To accumulate error Distances
    p = np.zeros(len(sub_orig[0]))
    count = 0

    # ✅ INSERT SUBJECT ONCE
    HC_augmented = pd.concat([HC, X.loc[X['ID'] == subject]])
    y_HC_augmented = pd.concat([y_HC, X.loc[X['ID'] == subject]])[['Group', 'ID']]

    for s in y_HC['ID'].values:
        st.write("Computing permutations (LOOCV) with", s)

        X_train_split, y_train_split, X_test_split, y_test_split = getSubject(HC_augmented, y_HC_augmented, X, s, subject, insert=False)
        scaler, X_train_split, X_test_split = model_prep.normalize_features(X_train_split, X_test_split, "void")

        if regress:
            if 'sex' in df_demog and 'age' in df_demog:
                X_train, X_test = model_prep.regress_confound(X_train_split, X_test_split, df_demog)

            model = Model(X_train, X_test, "PCAModel")
            k_hat = model.run_once()
            k_hat_inv = np.zeros_like(X_test)
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

    p_div = len(X_train_split) + len(X_test_split)
    overall_p = count / p_div
    p_crit = p / p_div
    p_along = (p_crit <= max(0.01, 1 / p_div)).astype(int)

    return x_inv, x_hat_inv, mae, p_along, overall_p, p_div
