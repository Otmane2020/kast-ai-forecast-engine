"""
Parse the JSON payload into pandas structures.
Handles grouping by granularity and feature engineering for XGBoost.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.models.schemas import ForecastRequest, ColumnInfo

logger = logging.getLogger(__name__)

FREQ_ALIASES = {"D": "D", "W": "W", "M": "MS", "Q": "QS", "Y": "YS"}


# ─────────────────────────────────────────────
# DataFrame builder
# ─────────────────────────────────────────────

def build_dataframe(payload: ForecastRequest) -> pd.DataFrame:
    df = pd.DataFrame(payload.data)

    date_col = payload.mapping.dateCol
    value_col = payload.mapping.valueCol

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data.")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in data.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Drop rows with invalid dates or values
    before = len(df)
    df = df.dropna(subset=[date_col, value_col])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with missing date/value.")

    df = df.sort_values(date_col).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Frequency detection
# ─────────────────────────────────────────────

def detect_frequency(series: pd.Series) -> str:
    """Infer time series frequency from DatetimeIndex."""
    try:
        inferred = pd.infer_freq(series.index)
        if inferred:
            if inferred.startswith("M") or inferred.startswith("MS"):
                return "M"
            if inferred.startswith("W"):
                return "W"
            if inferred.startswith("D"):
                return "D"
            if inferred.startswith("Q"):
                return "Q"
            if inferred.startswith("Y") or inferred.startswith("A"):
                return "Y"
    except Exception:
        pass

    # Fallback: median delta
    if len(series) < 2:
        return "M"
    deltas = pd.Series(series.index).diff().dropna()
    median_days = deltas.median().days
    if median_days <= 1:
        return "D"
    if median_days <= 8:
        return "W"
    if median_days <= 35:
        return "M"
    if median_days <= 100:
        return "Q"
    return "Y"


def seasonal_period(freq: str) -> int:
    return {"D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}.get(freq, 12)


# ─────────────────────────────────────────────
# Grouping
# ─────────────────────────────────────────────

def build_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str,
) -> pd.Series:
    """Aggregate DataFrame to a clean time series."""
    s = df.copy()
    s = s.groupby(date_col)[value_col].sum()
    s.index = pd.DatetimeIndex(s.index)
    resample_freq = FREQ_ALIASES.get(freq, "MS")
    s = s.resample(resample_freq).sum()
    s = s.fillna(0)
    # Drop leading zeros
    first_nonzero = s[s > 0].index.min()
    if first_nonzero is not None:
        s = s[s.index >= first_nonzero]
    return s


def get_groups(
    df: pd.DataFrame,
    payload: ForecastRequest,
    freq: str,
) -> Dict[str, pd.Series]:
    """Return dict of group_key → time series based on granularity."""
    date_col = payload.mapping.dateCol
    value_col = payload.mapping.valueCol
    product_col = payload.mapping.productCol
    category_col = payload.mapping.categoryCol
    granularity = payload.granularity

    groups: Dict[str, pd.Series] = {}

    if granularity == "global" or (not product_col and not category_col):
        groups["global"] = build_series(df, date_col, value_col, freq)
        return groups

    if granularity == "sku" and product_col and product_col in df.columns:
        for prod, grp in df.groupby(product_col):
            key = str(prod)
            series = build_series(grp, date_col, value_col, freq)
            if len(series) >= 4:
                groups[key] = series

    elif granularity == "family" and category_col and category_col in df.columns:
        for cat, grp in df.groupby(category_col):
            key = str(cat)
            series = build_series(grp, date_col, value_col, freq)
            if len(series) >= 4:
                groups[key] = series

    elif granularity == "subfamily":
        cols = [c for c in [category_col, product_col] if c and c in df.columns]
        if cols:
            for keys, grp in df.groupby(cols):
                key = "__".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
                series = build_series(grp, date_col, value_col, freq)
                if len(series) >= 4:
                    groups[key] = series

    if not groups:
        groups["global"] = build_series(df, date_col, value_col, freq)

    return groups


# ─────────────────────────────────────────────
# XGBoost feature engineering
# ─────────────────────────────────────────────

def make_time_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=dates)
    df["month"] = dates.month
    df["quarter"] = dates.quarter
    df["year"] = dates.year
    df["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    df["is_q4"] = (dates.quarter == 4).astype(int)
    return df


def make_lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    df = pd.DataFrame(index=series.index)
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    # Rolling statistics
    df["rolling_mean_3"] = series.shift(1).rolling(3, min_periods=1).mean()
    df["rolling_std_3"] = series.shift(1).rolling(3, min_periods=1).std().fillna(0)
    return df


def build_xgb_features(
    series: pd.Series,
    df_full: pd.DataFrame,
    feature_cols: List[ColumnInfo],
    freq: str,
    n_lags: Optional[int] = None,
) -> pd.DataFrame:
    """Build full feature matrix for XGBoost (in-sample)."""
    m = seasonal_period(freq)
    if n_lags is None:
        n_lags = min(m, max(3, len(series) // 4))

    lags = list(range(1, min(n_lags + 1, len(series) // 2 + 1)))
    lags += [m] if m not in lags and m < len(series) else []

    time_feats = make_time_features(series.index)
    lag_feats = make_lag_features(series, lags)

    X = pd.concat([time_feats, lag_feats], axis=1)

    # Extra feature columns (price, region, etc.)
    for col_info in feature_cols:
        col = col_info.name
        if col not in df_full.columns:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df_full[col]):
                # Use mean per date as a numeric feature
                col_series = df_full.groupby(df_full.index)[col].mean().reindex(series.index)
                X[col] = col_series.ffill().bfill().fillna(0).values
            else:
                # Categorical: encode and use mode per date
                le = LabelEncoder()
                encoded = le.fit_transform(df_full[col].astype(str).fillna("unknown"))
                col_enc = pd.Series(encoded, index=df_full.index)
                col_mode = col_enc.groupby(df_full.index).first().reindex(series.index)
                X[col] = col_mode.ffill().bfill().fillna(0).values
        except Exception as e:
            logger.debug(f"Skipping feature '{col}': {e}")

    return X.fillna(0)


def build_xgb_future_features(
    series: pd.Series,
    future_index: pd.DatetimeIndex,
    df_full: pd.DataFrame,
    feature_cols: List[ColumnInfo],
    freq: str,
    predicted_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Build feature matrix for future prediction steps."""
    m = seasonal_period(freq)
    n_lags = min(m, max(3, len(series) // 4))
    lags = list(range(1, min(n_lags + 1, len(series) // 2 + 1)))
    lags += [m] if m not in lags and m < len(series) else []

    # Extend series with predicted values
    extended = series.copy()
    if predicted_values:
        ext_series = pd.Series(
            predicted_values,
            index=future_index[:len(predicted_values)],
        )
        extended = pd.concat([extended, ext_series])

    time_feats = make_time_features(future_index)
    lag_feats = make_lag_features(extended, lags).reindex(future_index).fillna(0)

    X = pd.concat([time_feats, lag_feats], axis=1)

    # Extra features: use last known value for each
    for col_info in feature_cols:
        col = col_info.name
        if col not in df_full.columns:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df_full[col]):
                last_val = pd.to_numeric(df_full[col], errors="coerce").dropna().iloc[-1] if len(df_full) else 0
                X[col] = float(last_val)
            else:
                le = LabelEncoder()
                le.fit(df_full[col].astype(str).fillna("unknown"))
                last_cat = str(df_full[col].iloc[-1]) if len(df_full) else "unknown"
                try:
                    X[col] = le.transform([last_cat])[0]
                except Exception:
                    X[col] = 0
        except Exception:
            X[col] = 0

    return X.fillna(0)
