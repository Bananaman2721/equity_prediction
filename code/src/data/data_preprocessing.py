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
import yfinance as yf

import holidays
from datetime import datetime
from dateutil.relativedelta import relativedelta


# def read_data():
#     # SP500 + DOW30 + Nasdaq
#     df_spy = pd.read_csv('../../data/sp500_max.csv')
#     df_dowjones = pd.read_csv('../../data/dowjones_max.csv')
#     df_nasdaq = pd.read_csv('../../data/nasdaq_max.csv')
#     df_spy = df_spy.set_index('Date').add_suffix('_spy')
#     df_dowjones = df_dowjones.set_index('Date').add_suffix('_dowjones')
#     df_nasdaq = df_nasdaq.set_index('Date').add_suffix('_nasdaq')
#     df = df_spy.merge(df_dowjones, left_index=True, right_index=True, how='inner')
#     df = df.merge(df_nasdaq, left_index=True, right_index=True, how='inner')
#     df = df.reset_index()
#     df = df[['Open_spy', 'Close_spy', 'Volume_spy', 'Low_spy', 'High_spy', \
#              'Open_dowjones', 'Close_dowjones', 'Volume_dowjones', 'Low_dowjones', 'High_dowjones', \
#              'Open_nasdaq', 'Close_nasdaq', 'Volume_nasdaq', 'Low_nasdaq', 'High_nasdaq']]
#     return df


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
    # import pandas_ta as ta
    # df.ta.indicators()
    # df.ta.log_return(cumulative=True, append=True)
    # df.ta.percent_return(cumulative=True, append=True)

    return df

def query_data(stock,
               start_date,
               end_date):
    df = yf.download(stock, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.columns = df.columns.droplevel(level=1)
    df.columns = df.columns.str.lower()

    # df['date'] = pd.to_datetime(df['date'])
    # full_date_range = pd.date_range(start=start_date, end=end_date)
    # df = df.set_index('date').reindex(full_date_range).reset_index()
    # df.rename(columns={'index': 'date'}, inplace=True)
    # df.fillna(0.0, inplace=True)
    # df.dropna(inplace=True)

    df['change'] = df['close'].diff()
    df.dropna(inplace=True)
    df['direction'] = 0
    df.loc[df['change'] >= 0, 'direction'] = 1

    df['stock'] = stock
    
    return df
    

def get_correlated_stocks(df, 
                          date_column='date',
                          metric_column='close',
                          id_column='stock',
                          threshold=0.8):
    df_pivot = pd.pivot_table(df, 
                              index=[date_column],
                              columns=[id_column],
                              values=metric_column,
                              aggfunc="sum").reset_index()
    df_corr = df_pivot.corr()
    correlated_stocks = {}
    for stock in df_pivot.drop(columns=[date_column]).columns:
        correlated_stocks[stock] = list(set(df_corr[df_corr[stock] >= threshold].index) - set([stock]))
   
    return correlated_stocks

def prepare_data(df,
                 horizon,
                 stock,
                 correlated_stocks,
                 stage='train',
                 date_column='date',
                 metric_column='close',
                 id_column='stock'):
    df_local = df[df[id_column] == stock].sort_values(by=date_column).reset_index(drop=True)
    if stage == 'forecast':
        df_local = prepare_forecast_data(df_local,
                                         horizon=horizon,
                                         date_column=date_column)
        
    hist_exog_list_extra = []
    for correlated_stock in correlated_stocks:
        correlated_metric_column = correlated_stock + '_' + metric_column
        df_correlated_metric = df[df[id_column] == correlated_stock][[date_column, metric_column]].rename(columns={metric_column: correlated_metric_column})
        df_local = df_local.merge(df_correlated_metric, on=date_column, how='left')
        df_local[correlated_metric_column] = df_local[correlated_metric_column].fillna(0.0)
        hist_exog_list_extra.append(correlated_metric_column)
    df_local = df_local.rename(columns={date_column: 'ds', id_column: 'unique_id', metric_column: 'y'})
    df_local = df_local.sort_values(by=['unique_id', 'ds'])
    df_local.dropna(inplace=True)
    
    # Daily
    # # Weekend
    # weekend_mask = df_local['ds'].dt.day_name().isin(['Saturday', 'Sunday'])
    # df_local.loc[weekend_mask, 'weekend'] = 1
    # df_local.loc[~weekend_mask, 'weekend'] = 0

    # Weekday
    df_local['weekday'] = df_local['ds'].dt.weekday
    df_local_weekday = pd.get_dummies(df_local['weekday'], drop_first=True, prefix='weekday')
    df_local = pd.concat([df_local, df_local_weekday], axis=1)
    
    # Month
    df_local['month'] = df_local['ds'].dt.month
    df_local_month = pd.get_dummies(df_local['month'], drop_first=True, prefix='month')
    df_local = pd.concat([df_local, df_local_month], axis=1)
    
    # Holiday
    us_holidays = []
    for year in range(2016, 2026):
        us_holidays.extend(holidays.US(years=year))
    holiday_mask = df_local['ds'].isin(us_holidays) 
    df_local.loc[~holiday_mask, 'holiday'] = 0
    df_local.loc[holiday_mask, 'holiday'] = 1
    # df_local.loc[df_local['month']>=11, 'holiday'] = 2

    df_local = df_local.sort_values(by=['ds'])
    # Data split
    df_train = df_local[df_local.ds<df_local['ds'].values[-horizon]]
    df_test = df_local[df_local.ds>=df_local['ds'].values[-horizon]].reset_index(drop=True)
    # df_train = df_train.set_index(['ds', 'unique_id', 'holiday', 'weekend']).ewm(alpha=0.5, ignore_na=True).mean().reset_index()
    
    # Weekly
    # df_local_weekly = df_local.drop(columns=['unique_id']).resample('W', label='right', closed = 'right', on='ds').sum().reset_index().sort_values(by='ds')
    # df_local_weekly['unique_id'] = 'zazzle_shirt'
    # Y_train_df = df_local_weekly.iloc[:-horizon]
    # Y_test_df = df_local_weekly[-horizon:]
    
    # Monthly
    # df_local_monthly = df_local.drop(columns=['unique_id']).resample('M', label='right', closed = 'right', on='ds').sum().reset_index().sort_values(by='ds')
    # df_local_monthly['unique_id'] = 'zazzle_shirt'
    # Y_train_df = df_local_monthly[df_local_monthly.ds<df_local_monthly['ds'].values[-horizon]] # 132 train
    # Y_test_df = df_local_monthly[df_local_monthly.ds>=df_local_monthly['ds'].values[-horizon]].reset_index(drop=True) # 12 test

    return df_train, df_test, hist_exog_list_extra

def prepare_forecast_data(df,
                          horizon=5,
                          date_column='date'):
    """
    Prepare features for future time periods for the forecasting purpose

    Args:
        df: the Pandas dataframe
        horizon: the forecast period in days
        date_column: the date column name
    Returns:
      a Pandas dataframe that includes the result feature data (historical + future)                                                                                                                                                                                                                                                                                                                                                                                             
    """
    # df_holiday = pd.read_csv('./data/market_holidays.csv')
    # holiday_list = [pd.to_datetime(d) for d in df_holiday['date'].tolist()]
    holiday_list = ['2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19', '2025-07-03', '2025-07-04', '2025-09-01', '2025-11-27', '2025-11-28', '2025-12-24', '2025-12-25', '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27', '2024-06-19', '2024-07-03', '2024-07-04', '2024-09-02', '2024-11-28', '2024-11-29', '2024-12-24', '2024-12-25']
    i = 0
    current_date_local = pd.to_datetime(df['date'].values[-1])
    while i < horizon:
        current_date_local = current_date_local + pd.tseries.offsets.DateOffset(days=1)
        if current_date_local not in holiday_list and current_date_local.weekday() < 5:
            df = pd.concat([df, df.tail(1)], ignore_index=True)
            df.loc[df.index[-1], date_column] = current_date_local
            i = i + 1

    df = df.sort_values(by=[date_column])
    df[date_column] = pd.to_datetime(df[date_column])
    df.to_csv('forecast.csv', index=False)
    # df.ffill(inplace=True)
    
    return df