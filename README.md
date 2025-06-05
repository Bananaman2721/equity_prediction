# Financial Market Forecasting with Deep Learning Time Series Models

This project demonstrates how to apply cutting-edge deep learning time series forecasting models to financial market data. It focuses on predicting asset prices across various markets, including:

- Stocks  
- Indices  
- Foreign exchange (Forex)  
- Commodities  

## Models Included

We leverage several state-of-the-art deep learning architectures for time series forecasting, including:

- **Temporal Fusion Transformer (TFT)**  
- **Long Short-Term Memory (LSTM) Networks**  
- **Neural Hierarchical Interpolation for Time Series (NHITS)**  
- **Time-LLM (Large Language Models for Time Series)**  

All model implementations are sourced from this library: [NeuralForecast library by Nixtla](https://github.com/Nixtla/neuralforecast). Other models from this library can be easily adapted for this work.

Besides, a stack model and an ensemble model were used to merge the forecasting results from multiple models into the final forecasting result.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

Make sure you have Python 3.8 or later installed, then run:

```bash
pip install -r requirements.txt
```

This will install all necessary libraries to run the models and experiments. It is recommended to use a GPU-enabled environment to run this library, as it involves training and forecasting with deep learning models.

### 3. Run the codes: model training and inference
There are two ways to run the stock forecasting
 - Jupyter notebook: code/notebook/stock_prediction.ipynb
 - Python code: python code/src/main.py

### 4. Stock forecasting through App UI
Run python code/src/app/app.py from commandline, a webserver with the stock forecasting application will be started. Then you can access the web
interface to get stock data, technical indicators and perform stock forecasting. By default, here is the web address: http://127.0.0.1:8050/

## Goal

Our aim is to bridge the gap between financial expertise and modern AI/ML techniques. This project provides a practical framework for:

- Financial analysts interested in deep learning forecasting tools  
- AI/ML researchers looking to explore real-world financial time series applications  

## Contributing

We welcome contributions from the community! Whether you're a financial specialist, data scientist, or researcher, feel free to review, experiment with, and contribute to this project.

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
You may use, modify, and distribute this software in accordance with the terms of the license.
