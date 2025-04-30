import pandas as pd
from data_preprocessing import drop_missing_values, encode_labels

def test_drop_missing_values():
    df = pd.DataFrame({"a": [1, None, 3]})
    clean_df = drop_missing_values(df)
    assert clean_df.isnull().sum().sum() == 0
    assert clean_df.shape[0] == 2

def test_encode_labels():
    df = pd.DataFrame({"label": [0.0, 1.0]})
    df = encode_labels(df, "label")
    assert df["label"].dtype == int
