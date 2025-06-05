import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
from datetime import timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.graph_objs import *
from curl_cffi import requests

# model
from forecast_model import forecast_model

session = requests.Session(impersonate="chrome")

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "Equity Price Analysis and Forecasting"
server = app.server

stocks = ['^GSPC', '^IXIC', '^DJI', 'GOOG', 'AAPL', 'TSLA', 'MSFT', 'NVDA', 'AMD', 'META', 'AMZN', 'NFLX', 'BTC']
TA_indicators = ['ADL', 'ADX', 'AO', 'ATR', 'BBWIDTH', 'BOP', 'CCI', 'CFI', 'CHAIKIN', 'CMO', 'COPP', 'DEMA', 'DYMI', 'EFI', 'EMA', 'EMV', 'ER', 'FISH', 'FRAMA', 'FVE', 'HMA', 'IFT_RSI', 'MFI', 'MI', 'MOM', 'MSD', 'OBV', 'PERCENT_B', 'PZO', 'QSTICK', 'ROC', 'RSI', 'SAR', 'SMA', 'SMM', 'SMMA', 'SQZMI', 'SSMA', 'STC', 'STOCH', 'STOCHD', 'STOCHRSI', 'TEMA', 'TP', 'TR', 'TRIMA', 'TRIX', 'UO', 'VAMA', 'VBM', 'VFI', 'VPT', 'VWAP', 'VZO', 'WILLIAMS', 'WMA', 'WOBV', 'ZLEMA']
horizons = [1, 5, 10, 15, 20, 25, 30]
metrics = ['close', 'change', 'direction']

# Layout of Dash App
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.A(
                            html.Img(
                                className="logo",
                                src=app.get_asset_url("dash-logo.png"),
                            ),
                            # href="https://plotly.com/dash/",
                        ),
                        html.H2(app.title),
                        
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                html.Label(['Show Stock Daily Price'],
                                           style={'font-weight': 'bold',
                                                  'text-align': 'left',
                                                  'color': 'brown'}),
                                
                                dcc.Dropdown(stocks,
                                             'GOOG',
                                             id='stock-code',
                                             className="div-for-dropdown"),

                                dcc.DatePickerSingle(
                                    id="start_date_picker",
                                    min_date_allowed=dt(1980, 1, 1),
                                    max_date_allowed=dt.now(),
                                    date=dt(2000, 1, 1).date(),
                                    display_format="MMMM D, YYYY",
                                    style={"border": "0px solid black"},
                                    className="div-for-dropdown"
                                ),

                                dcc.DatePickerSingle(
                                    id="end_date_picker",
                                    min_date_allowed=dt(1980, 1, 1),
                                    max_date_allowed=dt.now(),
                                    date=dt.now().date(),
                                    display_format="MMMM D, YYYY",
                                    style={"border": "0px solid black"},
                                    className="div-for-dropdown"
                                ),
                                
                                                                
                                html.Button('Get Stock Price', id='stock-price-button'),
                            ],
                        ),

                        # TA indicators
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                html.Label(['Show Stock Technical Indicators'],
                                           style={'font-weight': 'bold',
                                                  'text-align': 'left',
                                                  'color': 'brown'}),
                               
                                # Indicators
                                dcc.Dropdown(TA_indicators,
                                             'EMA',
                                             id='ta_indicator',
                                             className="div-for-dropdown"),
                                
                                # Indicators button
                                html.Button('Get Indicators', id='indicators-button'),
                            ],
                        ),
                
                        # Forecast
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                html.Label(['Show Stock Daily Forecasting Price'],
                                           style={'font-weight': 'bold',
                                                  'text-align': 'left',
                                                  'color': 'brown'}),
                                
                                # Number of days of forecast input
                                dcc.Dropdown(horizons,
                                             5,
                                             id='forecast-days',
                                             className="div-for-dropdown"),
                    
                                # Metric
                                dcc.Dropdown(metrics,
                                             'close',
                                             id='metric_forecast',
                                             className="div-for-dropdown"),
                    
                                # Forecast button
                                html.Button('Get Forecast', id='forecast-button')                                
                            ],
                        ),
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        html.Div([], id="graphs-content"),
                        html.Div([], id="main-content"),
                        html.Div([], id="forecast-content")
                    ],
                ),                
            ],
        )
    ]
)

# Callbacks
# Callback for displaying stock price graphs
@app.callback(
    [Output("graphs-content", "children")],
    [
        Input("stock-price-button", "n_clicks"),
        Input('start_date_picker', 'date'),
        Input('end_date_picker', 'date')
    ],
    [State("stock-code", "value")]
)
def stock_price(n, start_date, end_date, stock):
    if n is None:
        return [""]
    if stock is None:
        raise PreventUpdate
    else:
        if start_date is not None:
            df = yf.download(stock, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    df.reset_index(inplace=True)
    df.columns = df.columns.droplevel(level=1) 
    fig = px.line(df, x="Date", y=["Close"], title="Daily Close Price for " + stock)
    return [dcc.Graph(figure=fig, style={'width': '100%', 'height': '330px'})]


# Callback for displaying indicators
@app.callback(
    [Output("main-content", "children")],
    [
        Input("indicators-button", "n_clicks"),
        Input('start_date_picker', 'date'),
        Input('end_date_picker', 'date')
    ],
    [State("stock-code", "value"),
     State("ta_indicator", "value"),]
)
def indicators(n,
               start_date,
               end_date,
               stock,
               indicator):
    if n is None:
        return [""]
    if stock is None:
        return [""]
    if start_date is None:
        df_more = yf.download(stock)
    else:
        df_more = yf.download(stock, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    df_more.columns = df_more.columns.droplevel(level=1) 
    fig = get_more(df_more, stock, indicator)
    
    return [dcc.Graph(figure=fig, style={'width': '100%', 'height': '330px'})]


def get_more(df, stock, indicator):
    from finta import TA
    try:
        df[indicator] = getattr(TA, indicator)(df)
    except:
        print(method, 'cannot be called')
 
    fig = px.scatter(df, x="Date", y=indicator, title=indicator + ' for ' + stock)
    fig.update_traces(mode='lines+markers')
    return fig


# Callback for displaying forecast
@app.callback(
    [Output("forecast-content", "children")],
    [Input("forecast-button", "n_clicks"),
     Input('start_date_picker', 'date'),
     Input('end_date_picker', 'date')],
    [State("stock-code", "value"),
     State("metric_forecast", "value"),
     State("forecast-days", "value")]
)
def forecast(n,
             start_date,
             end_date,
             stock,
             metric,
             horizon):
    
    if n is None:
        return [""]
        
    if stock is None:
        raise PreventUpdate

    print('metric: ', metric)
    fig = forecast_model(start_date,
                         end_date,
                         stock,
                         metric,
                         int(horizon))
    
    return [dcc.Graph(figure=fig, style={'width': '100%', 'height': '330px'})]


if __name__ == '__main__':
    app.run_server(debug=True)

