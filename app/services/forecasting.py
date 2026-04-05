"""
All 6 forecasting models.
Each function signature:
    fit_predict(train, n_pred, freq, **kwargs) -> np.ndarray  (length = n_pred)
    forecast_future(series, horizon, freq, **kwargs) -> np.ndarray  (length = horizon)
"""
from __future__ import annotations
import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _safe_clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0, None)


def _future_index(series: pd.Series, horizon: int, freq: str) -> pd.DatetimeIndex:
    from app.services.data_processor import FREQ_ALIASES
    f = FREQ_ALIASES.get(freq, "MS")
    return pd.date_range(start=series.index[-1], periods=horizon + 1, freq=f)[1:]


# ─────────────────────────────────────────────────────────────────────
# 1. ARIMA  (AIC grid search)
# ─────────────────────────────────────────────────────────────────────

def _auto_arima_order(series: pd.Series) -> tuple:
    """Grid search best (p,d,q) by AIC."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    try:
        pvalue = adfuller(series.dropna())[1]
        d = 0 if pvalue < 0.05 else 1
    except Exception:
        d = 1

    best_aic, best_order = np.inf, (1, d, 1)
    for p in range(3):
        for q in range(3):
            try:
                m = ARIMA(series, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic, best_order = m.aic, (p, d, q)
            except Exception:
                pass
    return best_order


def arima_fit_predict(
    train: pd.Series,
    n_pred: int,
    freq: str,
    order: Optional[tuple] = None,
) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    order = order or _auto_arima_order(train)
    m = ARIMA(train, order=order).fit()
    return _safe_clip(np.array(m.forecast(n_pred)))


def arima_forecast(series: pd.Series, horizon: int, freq: str) -> np.ndarray:
    return arima_fit_predict(series, horizon, freq)


# ─────────────────────────────────────────────────────────────────────
# 2. Prophet
# ─────────────────────────────────────────────────────────────────────

def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)


def prophet_fit_predict(
    train: pd.Series,
    n_pred: int,
    freq: str,
) -> np.ndarray:
    from prophet import Prophet
    df = _to_prophet_df(train)
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(freq in ("D", "W")),
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    m.fit(df)
    future = m.make_future_dataframe(periods=n_pred, freq=freq if freq != "M" else "MS")
    fc = m.predict(future)
    return _safe_clip(fc["yhat"].values[-n_pred:])


def prophet_forecast(series: pd.Series, horizon: int, freq: str) -> np.ndarray:
    return prophet_fit_predict(series, horizon, freq)


# ─────────────────────────────────────────────────────────────────────
# 3. XGBoost  (lag + time + extra features)
# ─────────────────────────────────────────────────────────────────────

def xgboost_fit_predict(
    train: pd.Series,
    n_pred: int,
    freq: str,
    X_train: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    import xgboost as xgb
    from app.services.data_processor import (
        make_time_features, make_lag_features, seasonal_period
    )

    m = seasonal_period(freq)
    lags = list(range(1, min(m + 1, len(train) // 2 + 1)))

    if X_train is None:
        tf = make_time_features(train.index)
        lf = make_lag_features(train, lags)
        X_train = pd.concat([tf, lf], axis=1).fillna(0)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, train.values)

    # Recursive prediction
    history = train.values.tolist()
    future_idx = _future_index(train, n_pred, freq)
    preds: list[float] = []
    for i, date in enumerate(future_idx[:n_pred]):
        if X_test is not None and i < len(X_test):
            x = X_test.iloc[[i]].values
        else:
            tf_row = make_time_features(pd.DatetimeIndex([date]))
            s_ext = pd.Series(history, index=pd.date_range(end=train.index[-1], periods=len(history), freq=train.index.freq or "MS"))
            lf_row = make_lag_features(s_ext, lags).iloc[[-1]].fillna(0)
            x = pd.concat([tf_row.reset_index(drop=True), lf_row.reset_index(drop=True)], axis=1).fillna(0).values

        p = float(model.predict(x)[0])
        preds.append(max(0.0, p))
        history.append(p)

    return _safe_clip(np.array(preds))


def xgboost_forecast(
    series: pd.Series,
    horizon: int,
    freq: str,
    X_full: Optional[pd.DataFrame] = None,
    X_future: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    return xgboost_fit_predict(series, horizon, freq, X_full, X_future)


# ─────────────────────────────────────────────────────────────────────
# 4. LSTM  (1-layer, sequence-based)
# ─────────────────────────────────────────────────────────────────────

def lstm_fit_predict(
    train: pd.Series,
    n_pred: int,
    freq: str,
    look_back: Optional[int] = None,
    epochs: int = 30,
) -> np.ndarray:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")

    values = train.values.astype(float)
    lb = look_back or min(12, max(3, len(values) // 4))

    # Normalize
    vmin, vmax = values.min(), values.max()
    scale = vmax - vmin if vmax != vmin else 1.0
    norm = (values - vmin) / scale

    X, y = [], []
    for i in range(lb, len(norm)):
        X.append(norm[i - lb: i])
        y.append(norm[i])
    X, y = np.array(X).reshape(-1, lb, 1), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(lb, 1)),
        Dropout(0.15),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=16,
        validation_split=0.1 if len(X) > 10 else 0.0,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=0,
    )

    # Recursive predict
    window = norm[-lb:].tolist()
    preds_norm = []
    for _ in range(n_pred):
        x = np.array(window[-lb:]).reshape(1, lb, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        preds_norm.append(p)
        window.append(p)

    return _safe_clip(np.array(preds_norm) * scale + vmin)


def lstm_forecast(series: pd.Series, horizon: int, freq: str) -> np.ndarray:
    return lstm_fit_predict(series, horizon, freq)


# ─────────────────────────────────────────────────────────────────────
# 5. Holt-Winters  (try add + mul, pick best AIC)
# ─────────────────────────────────────────────────────────────────────

def holtwinters_fit_predict(
    train: pd.Series,
    n_pred: int,
    freq: str,
) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from app.services.data_processor import seasonal_period

    m = seasonal_period(freq)
    m = max(2, m if len(train) >= 2 * m else max(2, len(train) // 4))
    has_zeros = (train <= 0).any()

    best_fit = None
    best_aic = np.inf

    variants = [
        {"trend": "add", "seasonal": "add"},
        {"trend": "add", "seasonal": None},
    ]
    if not has_zeros:
        variants.append({"trend": "add", "seasonal": "mul"})

    for kw in variants:
        try:
            sp = m if kw["seasonal"] else None
            fit = ExponentialSmoothing(
                train,
                trend=kw["trend"],
                seasonal=kw["seasonal"],
                seasonal_periods=sp,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=True)
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
        except Exception:
            pass

    if best_fit is None:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        best_fit = SimpleExpSmoothing(train, initialization_method="estimated").fit()

    return _safe_clip(np.array(best_fit.forecast(n_pred)))


def holtwinters_forecast(series: pd.Series, horizon: int, freq: str) -> np.ndarray:
    return holtwinters_fit_predict(series, horizon, freq)


# ─────────────────────────────────────────────────────────────────────
# 6. ETS  (Error-Trend-Seasonality)
# ─────────────────────────────────────────────────────────────────────

def ets_fit_predict(
    train: pd.Series,
    n_pred: int,
    freq: str,
) -> np.ndarray:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from app.services.data_processor import seasonal_period

    m = seasonal_period(freq)
    m = max(2, m if len(train) >= 2 * m else max(2, len(train) // 4))
    has_zeros = (train <= 0).any()

    candidates = [
        {"error": "add", "trend": "add", "seasonal": "add", "seasonal_periods": m},
        {"error": "add", "trend": "add", "seasonal": None, "seasonal_periods": None},
    ]
    if not has_zeros:
        candidates.append({"error": "mul", "trend": "add", "seasonal": "mul", "seasonal_periods": m})

    best_fit = None
    best_aic = np.inf

    for kw in candidates:
        try:
            fit = ETSModel(
                train,
                initialization_method="estimated",
                **kw,
            ).fit(disp=False)
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
        except Exception:
            pass

    if best_fit is None:
        return holtwinters_fit_predict(train, n_pred, freq)

    return _safe_clip(np.array(best_fit.forecast(n_pred)))


def ets_forecast(series: pd.Series, horizon: int, freq: str) -> np.ndarray:
    return ets_fit_predict(series, horizon, freq)
