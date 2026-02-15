import pandas as pd


def flatten_column_names(df: pd.DataFrame):
    df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
    return df
