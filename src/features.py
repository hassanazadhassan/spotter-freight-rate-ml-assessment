from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
CAT_COLUMNS = ["pickup", "delivery", "equipment"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "weight" in df:
        df["weight"] = df["weight"].abs()
        df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["month"] = d.month
    df["day_of_week"] = d.dayofweek
    df["day_of_year"] = d.dayofyear
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["is_month_edge"] = ((d.day <= 3) | (d.day >= 28)).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    df["distance_per_weight"] = df["distance"] / df["weight"].clip(lower=1)
    df["weight_per_distance"] = df["weight"] / df["distance"].clip(lower=1)
    df["log_distance"] = np.log1p(df["distance"])
    return df


class LaneEncoder:

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.global_rpm_ = None
        self.pickup_stats_ = None
        self.delivery_stats_ = None
        self.equipment_stats_ = None
        self.pickup_freq_ = None
        self.delivery_freq_ = None

    def fit(self, df: pd.DataFrame, rate_per_mile: pd.Series) -> "LaneEncoder":
        tmp = df.assign(rpm=rate_per_mile.values)
        self.global_rpm_ = tmp["rpm"].mean()
        self.pickup_stats_ = self._smoothed_mean(tmp, "pickup")
        self.delivery_stats_ = self._smoothed_mean(tmp, "delivery")
        self.equipment_stats_ = self._smoothed_mean(tmp, "equipment")
        self.pickup_freq_ = df["pickup"].value_counts(normalize=True)
        self.delivery_freq_ = df["delivery"].value_counts(normalize=True)
        return self

    def _smoothed_mean(self, tmp: pd.DataFrame, col: str) -> pd.Series:
        agg = tmp.groupby(col)["rpm"].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + self.global_rpm_ * self.smoothing) / (
            agg["count"] + self.smoothing
        )
        return smoothed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["pickup_hist_rpm"] = df["pickup"].map(self.pickup_stats_).fillna(self.global_rpm_)
        df["delivery_hist_rpm"] = df["delivery"].map(self.delivery_stats_).fillna(self.global_rpm_)
        df["equipment_hist_rpm"] = df["equipment"].map(self.equipment_stats_).fillna(self.global_rpm_)
        df["pickup_freq"] = df["pickup"].map(self.pickup_freq_).fillna(0.0)
        df["delivery_freq"] = df["delivery"].map(self.delivery_freq_).fillna(0.0)
        return df


NUMERIC_FEATURES = [
    "distance",
    "weight",
    "log_distance",
    "distance_per_weight",
    "weight_per_distance",
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "is_month_edge",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "pickup_hist_rpm",
    "delivery_hist_rpm",
    "equipment_hist_rpm",
    "pickup_freq",
    "delivery_freq",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CAT_COLUMNS


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df)
    df = add_calendar_features(df)
    df = add_logistics_features(df)
    return df


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_COLUMNS].copy()
    for c in CAT_COLUMNS:
        out[c] = out[c].astype("category")
    return out
