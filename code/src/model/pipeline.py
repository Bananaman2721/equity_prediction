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

from data.data_preprocessing import *
from model.LSTM import *
from model.feature_engineering import *

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# 6. Workflow
def workflow_train(data_freq='60s', train_ratio=1.0, feature_window=60, target_window=1, lead_time_window=0, result_path=None):
    from numpy.random import seed
    seed(1)
    import tensorflow
    tensorflow.random.set_seed(2)

    df = read_data()
    scaler, df_reframed = feature_engineering(df, stage='TRAIN', model=None, feature_window=feature_window, target_window=target_window, lead_time_window=lead_time_window)
    train_X, train_y, test_X, test_y = split_data(df_reframed, train_ratio=train_ratio, target_window=target_window)
    train_X = reshape(train_X, feature_window)
    encoder, forecaster = train(train_X, train_y, target_window=target_window)
    if train_ratio < 1.0:
        test_X = reshape(test_X, feature_window)
        y_pred_encoder = predict(encoder, test_X)
        y_pred_forecaster = predict(forecaster, y_pred_encoder)
        evaluate(y_pred_forecaster, test_y)
    model = {'scaler': scaler, 'encoder': encoder, 'forecaster': forecaster}
    pickle.dump(model, open(os.path.join(result_path, 'model.pkl'), 'wb'))

    return model


def workflow_predict(df, model):
    from numpy.random import seed
    seed(1)
    import tensorflow
    tensorflow.random.set_seed(2)

    _, test_X = feature_engineering(df, stage='PREDICT', model=model)
    test_X = reshape(test_X.values, feature_window=60)
    print("after reshape: ", test_X)
    y_pred_encoder = predict(model['encoder'], test_X)
    print("After encoder: ", y_pred_encoder)
    y_pred_forecaster = predict(model['forecaster'], y_pred_encoder)
    print("After forecaster: ", y_pred_forecaster)
    return y_pred_forecaster