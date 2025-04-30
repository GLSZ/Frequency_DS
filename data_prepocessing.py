import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def encode_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df[label_col] = df[label_col].astype(int)
    return df
