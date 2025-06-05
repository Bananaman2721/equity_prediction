import sys
sys.path.append('../') 

from model.pipeline import workflow

def forecast_model(start_date,
                   end_date,
                   stock,
                   metric,
                   horizon):
    try:
        result = workflow('forecast',
                          stock,
                          start_date,
                          end_date,
                          metric,
                          horizon)
    except:
        for stage in ['train', 'predict', 'forecast']:
            result = workflow(stage,
                              stock,
                              start_date,
                              end_date,
                              metric,
                              horizon)        
    
    return result