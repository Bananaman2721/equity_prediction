from model import pipeline

# 7. Parameters
train_ratio=0.99
feature_window=60
target_window=1
lead_time_window=0
result_path='/Users/zhang_family_mac/Yongqiang/stock_prediction/result'

# 8. Main function
def main():
    pipeline.workflow_train(train_ratio=train_ratio,
                   feature_window=feature_window,
                   target_window=target_window,
                   lead_time_window=lead_time_window,
                   result_path=result_path)

if __name__ == '__main__':
    main()