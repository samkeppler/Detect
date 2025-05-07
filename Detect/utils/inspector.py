import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.pca import PCAModel
from sklearn.preprocessing import StandardScaler
import random
import os


def plot_pca_anomaly(X_original, X_reconstructed, binary_mask, subject_id, save_path):
    # Ensure 2D row-wise shape: (1, n_features)
    X_original = np.atleast_2d(X_original)
    X_reconstructed = np.atleast_2d(X_reconstructed)
    binary_mask = np.atleast_2d(binary_mask)

    # Flatten each to 1D: shape (n_features,)
    original_line = X_original[0].flatten()
    reconstructed_line = X_reconstructed[0].flatten()
    anomaly_mask = binary_mask[0].flatten()

    x_vals = np.arange(original_line.shape[0])
    plt.figure(figsize=(12, 4))
    plt.plot(x_vals, original_line, label="Original", color="orangered")
    plt.plot(x_vals, reconstructed_line, label="Reconstructed", linestyle="--", color="purple")

    for i in range(len(anomaly_mask)):
        if anomaly_mask[i] == 1:
            plt.axvline(x=i, color="orchid", linewidth=1, alpha=0.5)

    plt.legend()
    plt.ylabel("FA")
    plt.title(subject_id)
    plt.tight_layout()
    plt.savefig(f"{save_path}/reconstruction_plot_{subject_id}.png")
    plt.close()


def run(X_train, X_test, y_test, out_path, model_type="pca", n_perm=1000, threshold=2.0, subject_id=None):
    subject_id = subject_id or "unknown"
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

        # Save plot matching AE-style reconstruction
        plot_pca_anomaly(
            X_original=X_test_scaled,
            X_reconstructed=x_inv,
            binary_mask=k_bin,
            subject_id=subject_id,
            save_path=out_path
        )


    else:
        raise ValueError("Only 'pca' model is supported in this version.")

    pd.DataFrame(k_mae).to_excel(f"{out_path}/recon.xlsx", index=False)
    pd.DataFrame(k_bin).to_excel(f"{out_path}/binary.xlsx", index=False)

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

    plt.figure(figsize=(10, 5))
    plt.plot(p, marker='o')
    plt.axhline(0.05, color='red', linestyle='--')
    plt.title("Global Anomaly Scores (p-values)")
    plt.xlabel("Patient Index")
    plt.ylabel("p-value")
    plt.savefig(f"{out_path}/global_plot.png")
    plt.close()
