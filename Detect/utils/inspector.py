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
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
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
    if(regress):
        if'sex' in df_demog and 'age' in df_demog:
            X_train, X_test = model_prep.regress_confound(X_train_split, X_test_split, df_demog)
        else:
            st.error("No age or sex information found. Skipping regression step.")
        
    

    #6 Run 
    #Run once to get Kreal whch is x_hat - x. 
    model = Model(X_train, X_test, "Z-score")
    x_hat = model.run_once()
    
    #unnormalize
    x_hat_inv = x_hat  # No inverse_transform for Z-score model
    x_inv = None
    mae = np.abs(x_hat_inv)  # use absolute z-score as "error"
    sub_orig = x_hat_inv  # for plotting

    # Convert Series to NumPy arrays
    sub_orig = sub_orig.to_numpy()
    sub = sub_orig.copy()  # assuming you're comparing to a permuted version later

    #To accumulate error Distances
    p = np.zeros(len(sub_orig))  # sub_orig is now 1D
    #Then, swap patient with HC, save in a vector a new K.
    #repeat for all HC, save in a matrix.
    count = 0
    for s in y_HC['ID'].values:
        st.write("Computing permutations (LOOCV) with", s)
        X_train_split, y_train_split, X_test_split, y_test_split = getSubject(HC, y_HC, X, s, subject, True)
        
        scaler, X_train_split, X_test_split = model_prep.normalize_features(X_train_split, X_test_split, "void")
        
        if(regress):
            if'sex' in df_demog and 'age' in df_demog:
                X_train, X_test = model_prep.regress_confound(X_train_split, X_test_split, df_demog)
        
        

        #6 Run 
        model = Model(X_train, X_test, "Z-score")
        k_hat = model.run_once()

        # Ensure NumPy arrays
        k_hat = k_hat.to_numpy()
        sub = k_hat
        k_mae = np.abs(k_hat)

        for e in range(len(sub_orig)):
            if sub_orig[e] > 0:
                if sub[e] >= sub_orig[e]:
                    p[e] += 1
            else:
                if sub[e] < sub_orig[e]:
                    p[e] += 1
                    
        if (np.mean(k_mae) > np.mean(mae)):
            count = count + 1
            
    #Then its, p < 1/HC+Dis for each tract section.
    p_div = len(X_train_split)+len(X_test_split)
    overall_p = count/p_div
    p_crit = p/p_div
    p_along = np.zeros(len(p_crit))

    max_p =  max(0.01,1/p_div)
    for i in range(len(p_crit)):
        if p_crit[i] <= max_p:
            p_along[i] = 1
                
    #K could be Zscore.
    return x_inv, x_hat_inv, mae, p_along, overall_p, p_div