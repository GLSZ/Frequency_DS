import pandas as pd

def describe_data(df: pd.DataFrame):
    print(df.shape)
    print(df.dtypes)
    print(df.head())
