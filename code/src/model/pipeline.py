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
from model.neural_forecast import *

def workflow(stage,
             stock,
             start_date,
             end_date,
             metric_column,
             horizon=5):

    # Parameters
    date_column = 'date'
    id_column = 'stock'
    threshold = 0.9
    
    frequency = 'D'
    input_size = 48
    hidden_size = 20
    loss = DistributionLoss(distribution='StudentT', level=[80, 90]) # 'Poisson', 'Normal'
    learning_rate = 0.05
    stat_exog_list = []
    hist_exog_list = ['open', 'high', 'low', 'volume', 'holiday', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4'] #, 'weekend']
    futr_exog_list = ['holiday', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4'] #, 'weekend']
    max_steps = 500
    val_check_steps = 10
    early_stop_patience_steps = 10
    scaler_type = 'robust'
    windows_batch_size = 16
    enable_progress_bar = True
    encoder_hidden_size = 64
    decoder_hidden_size = 64
    n_freq_downsample = [2, 1, 1]
    llm = 'gpt2'
    prompt_prefix = "The dataset contains data on daily stock price. There is a weekly, monthly and yearly seasonality."
    batch_size = 16
    valid_batch_size = 16
    result_path = '../../result/' + stock + '/' + metric_column + '/' + str(horizon) + '/'

    # Query data
    df = query_data(stock,
                    start_date,
                    end_date)

    # Get TA indicators
    # df = TA_analysis(df)
    
    # Get correlated stocks
    correlated_stocks = get_correlated_stocks(df, 
                                              date_column=date_column,
                                              metric_column=metric_column,
                                              id_column=id_column,
                                              threshold=threshold)

    # Build models: forecasting model and stack model
    iterations = 1
    if stage == 'train': # Train the forecasting and stack models
        iterations = int(90/horizon)

    df_result = None
    df_local = df.copy()
    for i in range(iterations):
        df_train, df_test, hist_exog_list_extra = prepare_data(df_local,
                                                               horizon,
                                                               stock,
                                                               correlated_stocks[stock],
                                                               stage=stage,
                                                               date_column=date_column,
                                                               metric_column=metric_column,
                                                               id_column=id_column)
        forecast_model = train(df_train,
                               horizon=horizon,
                               frequency=frequency,
                               input_size=input_size,
                               hidden_size=hidden_size,
                               loss=loss,
                               learning_rate=learning_rate,
                               stat_exog_list=stat_exog_list,
                               futr_exog_list=futr_exog_list,
                               hist_exog_list=hist_exog_list + hist_exog_list_extra,
                               max_steps=max_steps,
                               val_check_steps=val_check_steps,
                               early_stop_patience_steps=early_stop_patience_steps,
                               scaler_type=scaler_type,
                               windows_batch_size=windows_batch_size,
                               enable_progress_bar=enable_progress_bar,
                               encoder_hidden_size=encoder_hidden_size,
                               decoder_hidden_size=decoder_hidden_size,
                               n_freq_downsample=n_freq_downsample,
                               llm=llm,
                               prompt_prefix=prompt_prefix,
                               batch_size=batch_size,
                               valid_batch_size=valid_batch_size,
                               result_path=result_path)
    
        df_result = stack_data(forecast_model,
                               df_train,
                               df_test,
                               df_result)

        df_local = df_local[df_local['date'] < df_local['date'].values[-horizon]]

    result = stack(df_result,
                   metric_column,
                   stage=stage,
                   result_path=result_path)
    
    return result
