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


# 3. Read data
def read_data():
    # SP500 + DOW30 + Nasdaq
    df_spy = pd.read_csv('../../data/sp500_max.csv')
    df_dowjones = pd.read_csv('../../data/dowjones_max.csv')
    df_nasdaq = pd.read_csv('../../data/nasdaq_max.csv')
    df_spy = df_spy.set_index('Date').add_suffix('_spy')
    df_dowjones = df_dowjones.set_index('Date').add_suffix('_dowjones')
    df_nasdaq = df_nasdaq.set_index('Date').add_suffix('_nasdaq')
    df = df_spy.merge(df_dowjones, left_index=True, right_index=True, how='inner')
    df = df.merge(df_nasdaq, left_index=True, right_index=True, how='inner')
    df = df.reset_index()
    df = df[['Open_spy', 'Close_spy', 'Volume_spy', 'Low_spy', 'High_spy', \
             'Open_dowjones', 'Close_dowjones', 'Volume_dowjones', 'Low_dowjones', 'High_dowjones', \
             'Open_nasdaq', 'Close_nasdaq', 'Volume_nasdaq', 'Low_nasdaq', 'High_nasdaq']]
    return df


def TA_analysis(df):
    # TA Analysis #1: finta
    from finta import TA
    for method in TA.__dict__.keys():
        if method.startswith('__'):
            continue
        else:
            try:
                df[method] = getattr(TA, method)(df)
            except:
                print(method, 'cannot be called')
    import pandas_ta as ta
    df.ta.indicators()
    # df.ta.log_return(cumulative=True, append=True)
    # df.ta.percent_return(cumulative=True, append=True)

    return df