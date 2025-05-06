
import numpy as np
import pandas as pd
import streamlit as st
from models import model_prep
from utils.reporter import write_pval

def getSubject(HC, y_HC, X, subject, original, insert=False):
    X_train = HC.loc[HC['ID'] != subject]
    y_train = y_HC.loc[y_HC['ID'] != subject]

    if insert:
        X_train = pd.concat([X_train, X.loc[X['ID'] == original]])
        y_train = pd.concat([y_train, X_train[['Group', 'ID']]])

    X_test = X.loc[X['ID'] == subject]
    y_test = X_test[['Group', 'ID']]

    return (
        X_train.drop(['Group', 'ID'], axis=1),
        y_train,
        X_test.drop(['Group', 'ID'], axis=1),
        y_test
    )

def run(subject, df_data, df_demog, regress, tracts, metric, model_type='AutoEncoder'):
    if model_type == "AutoEncoder":
        from models.autoencoder import AutoEncoderModel as Model
    elif model_type == "PCA":
        from models.pca import PCAModel as Model
    elif model_type == "ZScore":
        from models.zscore import ZScoreModel as Model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    X = df_data.loc[:, df_data.columns.str.startswith('Group') |
                    df_data.columns.str.startswith('ID') |
                    df_data.columns.str.contains('|'.join(tracts))]

    HC = X[X['Group'] == 0]
    y_HC = HC[['Group', 'ID']]
    X_train, y_train, X_test, y_test = getSubject(HC, y_HC, X, subject, subject)

    if X_train.empty or X_test.empty:
        st.error("No data for analysis — check tract selection.")
        return None, None, None, None, None, None

    # Normalize
    _, X_train, X_test = model_prep.normalize_features(X_train, X_test, "void")

    if regress and 'age' in df_demog.columns and 'sex' in df_demog.columns:
        X_train, X_test = model_prep.regress_confound(X_train, X_test, df_demog)

    if model_type == "AutoEncoder":
        model = Model(X_train, X_test, "Autoencoder")
        x_hat = model.run_once()
        mae = np.mean(np.abs(X_test - x_hat), axis=1)
        sub_diff = x_hat - X_test
        bin_vector = (np.abs(sub_diff) > np.mean(mae)).astype(int).iloc[0]
        global_score = np.mean(mae)
        return X_test, x_hat, bin_vector, global_score, subject, y_test

    elif model_type == "PCA":
        model = Model(n_components=2)
        model.fit(X_train)
        scores = model.transform(X_test)
        bin_vector = (scores > np.percentile(scores, 95)).astype(int)
        return X_test, None, bin_vector, np.mean(scores), subject, y_test

    elif model_type == "ZScore":
        model = Model()
        _, z_test = model.run(X_train, X_test)
        bin_vector = (np.abs(z_test) > 2).astype(int)
        return X_test, None, bin_vector, np.mean(z_test), subject, y_test
