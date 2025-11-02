import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pickle

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT, LSTM, NHITS, TimeLLM
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss, GMM, PMM
from neuralforecast.auto import AutoNHITS, AutoLSTM, AutoTFT

from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from pathlib import Path
import plotly.graph_objs as go

def train(df,
          horizon=5,
          frequency='D',
          input_size=48,
          hidden_size=20,
          loss=DistributionLoss(distribution='StudentT', level=[80, 90]), # 'Poisson', 'Normal'
          learning_rate=0.005,
          stat_exog_list=[],
          futr_exog_list=[],
          hist_exog_list=[],
          max_steps=500,
          val_check_steps=10,
          early_stop_patience_steps=10,
          scaler_type='robust',
          windows_batch_size=16,
          enable_progress_bar=True,
          encoder_hidden_size=64,
          decoder_hidden_size=64,
          n_freq_downsample=[2, 1, 1],
          llm='gpt2',
          prompt_prefix = "The dataset contains data on daily stock price. There is a weekly, monthly and yearly seasonality.",
          batch_size=16,
          valid_batch_size=16,
          result_path=None):
    forecast_model = NeuralForecast(
        models=[TFT(h=horizon,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    loss=loss,
                    learning_rate=learning_rate,
                    stat_exog_list=stat_exog_list,
                    futr_exog_list=futr_exog_list,
                    hist_exog_list=hist_exog_list,
                    max_steps=max_steps,
                    val_check_steps=val_check_steps,
                    early_stop_patience_steps=early_stop_patience_steps,
                    scaler_type=scaler_type,
                    windows_batch_size=windows_batch_size,
                    enable_progress_bar=enable_progress_bar),
               LSTM(h=horizon,
                    max_steps=max_steps,
                    scaler_type=scaler_type,
                    encoder_hidden_size=encoder_hidden_size,
                    decoder_hidden_size=decoder_hidden_size),
              NHITS(h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    n_freq_downsample=n_freq_downsample),
            TimeLLM(h=horizon,
                    input_size=input_size,
                    llm=llm,
                    prompt_prefix=prompt_prefix,
                    batch_size=batch_size,
                    valid_batch_size=valid_batch_size,
                    windows_batch_size=windows_batch_size)
        ],
        freq=frequency
    )
    forecast_model.fit(df=df,
                       val_size=horizon)
    output_dir = Path(result_path + '/forecast')
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_model.save(path=str(output_dir),
                        model_index=None, 
                        overwrite=True,
                        save_dataset=True)
            
    return forecast_model
    
def predict(forecast_model, df):
    df_future = forecast_model.make_future_dataframe()
    df_ds = df['ds'].copy()
    df['ds'] = df_future['ds']
    
    df_pred = forecast_model.predict(futr_df=df)
    
    df['ds'] = df_ds

    return df_pred

def stack_data(forecast_model,
               df_train,
               df_test,
               df_result_prev=None):

    df_pred = predict(forecast_model, df_test).reset_index().drop(columns=['unique_id','ds'])

    # start_date_last = df_test['ds'].min() + pd.offsets.DateOffset(years=-1)
    # end_date_last = df_test['ds'].max() + pd.offsets.DateOffset(years=-1)
    # df_last = df_train[(df_train['ds'] >= start_date_last) & (df_train['ds'] <= end_date_last)].rename(columns={'y': 'y_last'})[['y_last']]
    # start_date_last2 = df_test['ds'].min() + pd.offsets.DateOffset(years=-2)
    # end_date_last2 = df_test['ds'].max() + pd.offsets.DateOffset(years=-2)
    # df_last2 = df_train[(df_train['ds'] >= start_date_last2) & (df_train['ds'] <= end_date_last2)].rename(columns={'y': 'y_last2'})[['y_last2']]
    df_result = pd.concat([df_test.reset_index(drop=True),
                           df_pred.reset_index(drop=True)], axis=1)
#                           df_last.reset_index(drop=True),
#                           df_last2.reset_index(drop=True)], axis=1)
#    df_result['y_last2'].fillna(df_result['y_last'], inplace=True)
    features = ['TFT', 'LSTM'] #, 'NHITS', 'TimeLLM']
    df_result['ensemble'] = df_result[features].mean(axis=1)
    if df_result_prev is not None:
        df_result = pd.concat([df_result_prev, df_result], axis=0)
    df_result = df_result.sort_values(by='ds').reset_index(drop=True)

    return df_result
    
def stack(df_result,
          metric_column,
          stage='train',
          result_path=None):

    features = ['TFT', 'LSTM', 'NHITS', 'TimeLLM'] #, 'y_last', 'y_last2', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekend']
    target = 'y'
    X = df_result[features]
    y = df_result[target]

    output_dir = Path(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('output_dir: ', output_dir)
    if stage == 'train':
        # Model (with hyperparameter tuning)
        def objective(n_estimators,
                      max_depth,
                      min_samples_split,
                      max_features):
            model = RandomForestRegressor(n_estimators=int(n_estimators),
                                          max_depth=int(max_depth),
                                          min_samples_split=int(min_samples_split),
                                          max_features=min(max_features, 0.999),
                                          random_state=42)
            tscv = TimeSeriesSplit(n_splits=3, test_size=10)
            return 1.0 * cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_percentage_error').mean()
        
        # Bounds for hyperparameters
        param_bounds = {
            'n_estimators': (10, 50),
            'max_depth': (5, 10),
            'min_samples_split': (2, 25),
            'max_features': (0.1, 0.999),
        }
        
        optimizer = BayesianOptimization(f=objective,
                                         pbounds=param_bounds,
                                         random_state=42)
        optimizer.maximize(init_points=5, n_iter=100)
        
        best_params = optimizer.max['params']
        stack_model = RandomForestRegressor(n_estimators=int(best_params['n_estimators']),
                                             max_depth=int(best_params['max_depth']),
                                             min_samples_split=int(best_params['min_samples_split']),
                                             max_features=best_params['max_features'],
                                             random_state=42)
        stack_model.fit(X, y)
        pickle.dump(stack_model, open(str(output_dir) + '/stack_model.pkl', 'wb'))
        return stack_model
    elif stage == 'predict':
        stack_model = pickle.load(open(str(output_dir) + '/stack_model.pkl', 'rb'))
        y_pred = stack_model.predict(X)
        df_pred = pd.DataFrame(y_pred, columns=['y_pred'])
        plot_df = pd.concat([df_result, df_pred], axis=1)
        plot_df.rename(columns={'ds': 'date', \
                                'unique_id': 'stock', \
                                'y': metric_column + '_actual', \
                                'y_pred': metric_column + '_forecast', \
                                'ensemble': metric_column + '_ensemble'}, \
                       inplace=True)
        # plot_df = plot_df[(plot_df['weekend'] == 0) & (plot_df['holiday'] == 0)]
        plot_df['error_stack'] = plot_df[metric_column + '_forecast'] - plot_df[metric_column + '_actual']
        plot_df['error_percent_stack'] = plot_df['error_stack'] / (plot_df[metric_column + '_actual'] + 0.01)
        mape_stack = round(plot_df['error_percent_stack'].abs().mean(), 2)
        error_cumulative_stack = round(plot_df['error_stack'].sum() / plot_df[metric_column + '_actual'].sum(), 2)

        plot_df['error_ensemble'] = plot_df[metric_column + '_ensemble'] - plot_df[metric_column + '_actual']
        plot_df['error_percent_ensemble'] = plot_df['error_ensemble'] / (plot_df[metric_column + '_actual'] + 0.01)
        mape_ensemble = round(plot_df['error_percent_ensemble'].abs().mean(), 2)
        error_cumulative_ensemble = round(plot_df['error_ensemble'].sum() / plot_df[metric_column + '_actual'].sum(), 2)

        plot_df.to_csv(output_dir/'predict.csv', index=False)
        plot_df['date'] = plot_df['date'].dt.strftime('%Y-%m-%d')
        plot_df.plot.bar(x = 'date', \
                         y = [metric_column + '_actual', metric_column + '_forecast', metric_column + '_ensemble'], \
                         title = metric_column + ': stack error=' + str(error_cumulative_stack) \
                                               + ': ensemble error=' + str(error_cumulative_ensemble), \
                         figsize = (16, 12)) \
               .get_figure().savefig(output_dir/'predict.png')

        # Plot the results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df[metric_column + '_actual'], mode='lines+markers', name=metric_column + '_actual'))
        fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df[metric_column + '_forecast'], mode='lines+markers', name=metric_column + '_forecast'))
        fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df[metric_column + '_ensemble'], mode='lines+markers', name=metric_column + '_ensemble'))
        fig.update_layout(title="Predicted Close Price of next " + str(len(df_result)) + " days",
                          xaxis_title="Date",
                          yaxis_title="Close Price",
                          xaxis=dict(type = "category"))
        return fig
        
    elif stage == 'forecast':
        stack_model = pickle.load(open(str(output_dir) + '/stack_model.pkl', 'rb'))
        y_pred = stack_model.predict(X)
        df_pred = pd.DataFrame(y_pred, columns=['y_pred'])
        plot_df = pd.concat([df_result, df_pred], axis=1)
        plot_df.rename(columns={'ds': 'date', \
                                'unique_id': 'stock', \
                                'y_pred': metric_column + '_forecast', \
                                'ensemble': metric_column + '_ensemble'}, \
                       inplace=True)
        # plot_df = plot_df[(plot_df['weekend'] == 0) & (plot_df['holiday'] == 0)]
        plot_df.to_csv(output_dir/'forecast.csv', index=False)
        plot_df['date'] = plot_df['date'].dt.strftime('%Y-%m-%d')
        plot_df.plot.bar(x = 'date', \
                         y = [metric_column + '_forecast', metric_column + '_ensemble'], \
                         title = metric_column, \
                         figsize = (16, 12)) \
               .get_figure().savefig(output_dir/'forecast.png')
        
        # Plot the results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df[metric_column + '_forecast'], mode='lines+markers', name=metric_column + '_forecast'))
        fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df[metric_column + '_ensemble'], mode='lines+markers', name=metric_column + '_ensemble'))
        fig.update_layout(title="Forecasted Close Price of Next " + str(len(df_result)) + " Trading Days",
                          xaxis_title="Date",
                          yaxis_title="Close Price",
                          xaxis=dict(type = "category"))
        return fig   
    else:
        return None
