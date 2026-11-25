from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job",
]

NUMERICAL_FEATURES = [
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_weekday",
    "customer_age",
    "distance_km",
]

DROP_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "dob",
    "trans_num",
    "unix_time",
    "Unnamed: 0",
]


def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(value))


def build_features(transactions):
    """Create the model's training/inference columns from raw transaction fields."""
    features = transactions.copy()
    transaction_time = pd.to_datetime(features["trans_date_trans_time"])
    birth_date = pd.to_datetime(features["dob"])

    features["transaction_hour"] = transaction_time.dt.hour
    features["transaction_day"] = transaction_time.dt.day
    features["transaction_month"] = transaction_time.dt.month
    features["transaction_weekday"] = transaction_time.dt.dayofweek
    features["customer_age"] = (
        (transaction_time - birth_date).dt.days / 365.25
    ).round(1)

    lat1 = np.radians(features["lat"].to_numpy(dtype=float))
    lon1 = np.radians(features["long"].to_numpy(dtype=float))
    lat2 = np.radians(features["merch_lat"].to_numpy(dtype=float))
    lon2 = np.radians(features["merch_long"].to_numpy(dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    features["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(value))

    features = features.drop(columns=DROP_COLUMNS, errors="ignore")
    return features
