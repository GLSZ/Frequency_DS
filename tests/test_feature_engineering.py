import pandas as pd
from feature_engineering import add_temporal_features

def test_add_temporal_features():
    df = pd.DataFrame({"timestamp": ["2022-01-01 08:00", "2022-01-02 15:30"]})
    df = add_temporal_features(df, "timestamp")
    assert "month" in df.columns
    assert "day_of_week" in df.columns
    assert "hour" in df.columns
