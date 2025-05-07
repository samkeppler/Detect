import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.pca import PCAModel
from sklearn.preprocessing import StandardScaler
import random


def inspector(X_train, X_test, y_test, out_path, model_type="pca", n_perm=1000, threshold=2.0):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "pca":
        model = PCAModel()
        model.fit(X_train_scaled)
        x_hat = model.transform(X_test_scaled)
        x_inv = model.inverse_transform(x_hat)

        k_mae = np.abs(X_test_scaled - x_inv)
        k_bin = (k_mae > threshold).astype(int)
        k_hat = np.mean(k_mae, axis=1)

    else:
        raise ValueError("Only 'pca' model is supported in this version.")

    # Write reconstructed feature MAEs and binary anomalies
    pd.DataFrame(k_mae).to_excel(f"{out_path}/recon.xlsx", index=False)
    pd.DataFrame(k_bin).to_excel(f"{out_path}/binary.xlsx", index=False)

    # Permutation test for global anomaly score
    sub_orig = k_hat[y_test == 1]
    sub = k_hat[y_test == 0]

    sub_orig = np.array(sub_orig)
    sub = np.array(sub)

    p = np.zeros(len(sub_orig))
    for e in range(len(sub_orig)):
        orig_val = sub_orig[e]
        concat = np.concatenate((sub, [orig_val]))
        gt = concat[-1]
        count = 0
        for _ in range(n_perm):
            random.shuffle(concat)
            if np.where(concat == gt)[0][0] >= len(concat) - 1:
                count += 1
        p[e] = count / n_perm

    pd.DataFrame(p, columns=["pval"]).to_excel(f"{out_path}/global.xlsx", index=False)

    # Plot the global anomaly scores
    plt.figure(figsize=(10, 5))
    plt.plot(p, marker='o')
    plt.axhline(0.05, color='red', linestyle='--')
    plt.title("Global Anomaly Scores (p-values)")
    plt.xlabel("Patient Index")
    plt.ylabel("p-value")
    plt.savefig(f"{out_path}/global_plot.png")
    plt.close()
