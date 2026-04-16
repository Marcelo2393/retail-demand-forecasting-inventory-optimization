import pandas as pd
import numpy as np

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def create_lag_features(df: pd.DataFrame, lags=[1,7,14,28]) -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["sales"].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rolling_mean_7"] = df.groupby(["store_id", "item_id"])["sales"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df.groupby(["store_id", "item_id"])["sales"].shift(1).rolling(14).mean()
    df["rolling_std_7"] = df.groupby(["store_id", "item_id"])["sales"].shift(1).rolling(7).std()
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = df.dropna()
    return df