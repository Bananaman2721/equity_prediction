# Imports
import sys
import time
from datetime import date
from neuralforecast.losses.pytorch import DistributionLoss
from model.pipeline import workflow

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Parameters
# data
stage = 'train'
start_date = '2000-01-01'
end_date = date.today()
horizon = 5
stocks = ['^GSPC', '^IXIC', '^DJI', 'GOOG', 'AAPL', 'TSLA', 'MSFT', 'NVDA', 'AMD', 'META', 'AMZN', 'NFLX', 'BTC-USD']
metrics = ['close', 'change', 'direction']

def main():
    start = time.time()
    for stock in stocks:
        for metric_column in metrics:
            for stage in ['train', 'predict', 'forecast']:
                workflow(stage,
                         stock,
                         start_date,
                         end_date,
                         metric_column,
                         horizon=horizon)
    end = time.time()
    print(
        f"Total time elapsed (in minutes): "
        + "{:.2f}".format((end - start) / 60.0)
    )
    
if __name__ == "__main__":
    main()