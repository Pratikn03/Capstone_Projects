import pandas as pd
from gridpulse.data_pipeline.split_time_series import time_split

def test_time_split_order():
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100, freq="h"), "x": range(100)})
    train, val, test = time_split(df, 0.7, 0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train["timestamp"].max() < val["timestamp"].min()
