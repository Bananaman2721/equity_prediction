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

# 4. Feature Engineering
# Get data in the Kera's format
def series_to_supervised(data, n_in=1, n_out=1, lead_time=0, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(lead_time, lead_time + n_out):
        cols.append(df.iloc[:, 1].shift(-i))
        if i == 0:
            names += [('var%d(t)' % (n_vars))]
        else:
            names += [('var%d(t+%d)' % (n_vars, i))]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def feature_engineering(df, stage='TRAIN', model=None, feature_window=60, target_window=1, lead_time_window=0):
    # get time series data
    values = df.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    if stage == 'TRAIN':
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(values)
    else:
        scaler = model['scaler']
    scaled = scaler.transform(values)
    print('scaled: ', scaled.shape)

    # frame as supervised learning
    if stage == 'TRAIN':
        df_reframed = series_to_supervised(scaled, feature_window, target_window, lead_time_window, True)
    else:
        df_reframed = series_to_supervised(scaled, feature_window, 0, 0, True)

    print(df_reframed.head(10))
    print(len(df_reframed))
    return scaler, df_reframed

def split_data(df_reframed, train_ratio=1.0, target_window=1):
    # split into train and test sets
    dataset = df_reframed.values
    train_size = int(len(dataset) * train_ratio)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # split into input and outputs
    train_X, train_y = train[:, :-target_window], train[:, -target_window:]
    test_X, test_y = test[:, :-target_window], test[:, -target_window:]
    return train_X, train_y, test_X, test_y


def reshape(X, feature_window=60):
    # reshape input to be 3D [samples, timesteps, features]
    num_features = int(X.shape[1] / feature_window)
    X = X.reshape((X.shape[0], feature_window, num_features))
    return X

