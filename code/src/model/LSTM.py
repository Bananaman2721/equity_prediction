# 2. Import libaries
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from keras.models import *
from keras.layers import *
import pickle


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# 5. Modeling
def LSTM_auto_encoder(train_X):
    inputs_ae = Input(shape=(train_X.shape[1], train_X.shape[2]))
    encoded_ae = LSTM(128, return_sequences=True, dropout=0.3)(inputs_ae, training=True)
    decoded_ae = LSTM(32, return_sequences=True, dropout=0.3)(encoded_ae, training=True)
    out_ae = TimeDistributed(Dense(train_X.shape[2]))(decoded_ae)

    sequence_autoencoder = Model(inputs_ae, out_ae)
    sequence_autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
    sequence_autoencoder.summary()

    sequence_autoencoder.fit(train_X, train_X, batch_size=1024, epochs=50, verbose=2, shuffle=True)

    encoder = Model(inputs_ae, encoded_ae)
    return encoder


def LSTM_forecaster(train_X, train_y, target_window=15):
    input_fc = Input(shape=(train_X.shape[1], train_X.shape[2]))
    lstm_fc = LSTM(128, return_sequences=True, dropout=0.3)(input_fc, training=True)
    lstm_fc = LSTM(32, return_sequences=False, dropout=0.3)(lstm_fc, training=True)
    dense_fc = Dense(50)(lstm_fc)
    out_fc = Dense(target_window)(dense_fc)

    model_fc = Model(input_fc, out_fc)

    model_fc.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model_fc.fit(train_X, train_y, epochs=50, batch_size=1024, verbose=2, shuffle=True)
    return model_fc

def train(train_X, train_y, target_window=15):
    encoder = LSTM_auto_encoder(train_X)
    train_X = encoder.predict(train_X)
    forecaster = LSTM_forecaster(train_X, train_y, target_window)
    return encoder, forecaster


def predict(model, df):
    y_pred = model.predict(df)
    return y_pred


def evaluate(y_pred, y_true):
    y_diff = y_pred - y_true
    y_diff_square = y_diff * y_diff
    y_rmse = [np.sqrt(np.mean(y_diff[:, i])) for i in range(y_true.shape[1])]
    y_mape = np.mean(abs(y_diff))
    print('Evaluation result - rmse: ', y_rmse, ' mape: ', y_mape)

