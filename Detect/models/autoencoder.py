from __future__ import division, print_function, absolute_import

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras import regularizers, backend
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import datetime
import seaborn as sns

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

def save(model):
    pass

def plot_loss(model_history):
    train_loss = [value for key, value in model_history.items() if 'loss' in key.lower()][0]
    valid_loss = [value for key, value in model_history.items() if 'loss' in key.lower()][1]

    with _lock:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Epoch', size=42)
        ax1.set_ylabel('Loss', color="black", size=42)
        ax1.plot(train_loss, '--', color="black", label='Train Loss', linewidth=4)
        ax1.plot(valid_loss, color='xkcd:purply', label='Test Loss', linewidth=4)
        ax1.tick_params(axis='y', labelcolor="black")
        plt.legend(loc='upper right', fontsize=28)
        plt.title('Model Loss', size=48)
        ax1.tick_params(labelsize=32)
        fig.tight_layout()
        fig.savefig('figures/AE_loss.png', dpi=200)
        st.write(fig)
        plt.close(fig)

def fit(autoencoder, X_train, epochs, size):
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=size,
        shuffle=True,
        validation_split=0.1,
        verbose=0
    )
    return pd.DataFrame(history.history)

def create_model(input_dim, lr=0.001, acts='tanh'):
    backend.clear_session()
    encoding_dim = input_dim / 2

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(int(encoding_dim), activation='relu')(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
    decoder = Dense(int(encoding_dim), activation='relu')(encoder)
    decoder = Dense(input_dim, activation=acts)(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    optimizer = keras.optimizers.Adam(lr=lr)
    autoencoder.compile(optimizer=optimizer, loss='mse',
                        metrics=["mean_squared_error", "mean_absolute_error"])
    return autoencoder

#  CLASS-BASED INTERFACE
class AutoEncoderModel:
    def __init__(self, X_train, X_test, model_name="Autoencoder"):
        self.X_train = X_train
        self.X_test = X_test
        self.model_name = model_name
        self.autoencoder = None

    def run_once(self):
        seed(10)
        tf.random.set_seed(10)

        input_dim = self.X_train.shape[1]
        self.autoencoder = create_model(input_dim)
        fit(self.autoencoder, self.X_train, epochs=12, size=24)

        x_hat = self.autoencoder.predict(np.array(self.X_test))
        x_hat = pd.DataFrame(x_hat, columns=self.X_test.columns, index=self.X_test.index)

        return x_hat

    def run(self):
        tf.random.set_seed(10)
        lr = [0.001]
        n_epochs = [25]
        n_batch = [24]
        acts = ['tanh']

        for i in lr:
            for j in acts:
                for k in n_epochs:
                    for l in n_batch:
                        st.write("Training config - lr:", i, "activation:", j, "epochs:", k, "batch size:", l)
                        self.autoencoder = create_model(self.X_train.shape[1], i, j)
                        fit(self.autoencoder, self.X_train, k, l)

        X_pred_train = self.autoencoder.predict(np.array(self.X_train))
        X_pred_test = self.autoencoder.predict(np.array(self.X_test))

        X_pred_train = pd.DataFrame(X_pred_train, columns=self.X_train.columns, index=self.X_train.index)
        X_pred_test = pd.DataFrame(X_pred_test, columns=self.X_test.columns, index=self.X_test.index)

        MAE_train = np.mean(np.abs(X_pred_train - self.X_train), axis=1)
        MAE_test = np.mean(np.abs(X_pred_test - self.X_test), axis=1)

        return MAE_train, MAE_test
