import pandas as pd

def add_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.month
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["hour"] = df[date_col].dt.hour
    return df

def drop_unused_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    return df.drop(columns=columns_to_drop, errors='ignore')
