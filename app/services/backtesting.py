"""
Train/test split, model evaluation, and metrics computation.
Each model is run in a thread pool for parallelism.
"""
from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.models.schemas import ForecastRequest, ModelResult
from app.services.data_processor import (
    seasonal_period,
    build_xgb_features,
    build_xgb_future_features,
    _future_index as _fidx,
)

logger = logging.getLogger(__name__)

MIN_LEN_ARIMA = 12
MIN_LEN_PROPHET = 12
MIN_LEN_LSTM = 15
MIN_LEN_ETS = 8
MIN_LEN_HW = 8
MIN_LEN_XGB = 8


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    n = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]

    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 999.0
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    bias = float(np.mean(predicted - actual))

    return {
        "mape": round(min(mape, 999.0), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "bias": round(bias, 4),
    }


# ─────────────────────────────────────────────────────────────────────
# Single model evaluation
# ─────────────────────────────────────────────────────────────────────

def _eval_model(
    name: str,
    train: pd.Series,
    test: pd.Series,
    series: pd.Series,
    horizon: int,
    freq: str,
    df_full: pd.DataFrame,
    payload: ForecastRequest,
) -> Optional[ModelResult]:
    """Fit on train → eval on test → refit on full → future forecast."""
    from app.services import forecasting as F

    n = len(series)
    test_len = len(test)

    t0 = time.perf_counter()
    try:
        # ── BACKTEST predictions ──
        if name == "ARIMA":
            if n < MIN_LEN_ARIMA:
                return None
            bt_preds = F.arima_fit_predict(train, test_len, freq)
            future_preds = F.arima_forecast(series, horizon, freq)

        elif name == "Prophet":
            if n < MIN_LEN_PROPHET:
                return None
            bt_preds = F.prophet_fit_predict(train, test_len, freq)
            future_preds = F.prophet_forecast(series, horizon, freq)

        elif name == "XGBoost":
            if n < MIN_LEN_XGB:
                return None
            feat_cols = payload.feature_columns()
            X_full_train = build_xgb_features(train, df_full.loc[df_full.index <= train.index[-1]], feat_cols, freq)
            X_full = build_xgb_features(series, df_full, feat_cols, freq)
            X_future = build_xgb_future_features(series, _fidx(series, horizon, freq), df_full, feat_cols, freq)
            bt_preds = F.xgboost_fit_predict(train, test_len, freq, X_full_train)
            future_preds = F.xgboost_forecast(series, horizon, freq, X_full, X_future)

        elif name == "LSTM":
            if n < MIN_LEN_LSTM:
                return None
            bt_preds = F.lstm_fit_predict(train, test_len, freq)
            future_preds = F.lstm_forecast(series, horizon, freq)

        elif name == "HoltWinters":
            if n < MIN_LEN_HW:
                return None
            bt_preds = F.holtwinters_fit_predict(train, test_len, freq)
            future_preds = F.holtwinters_forecast(series, horizon, freq)

        elif name == "ETS":
            if n < MIN_LEN_ETS:
                return None
            bt_preds = F.ets_fit_predict(train, test_len, freq)
            future_preds = F.ets_forecast(series, horizon, freq)

        else:
            return None

        metrics = compute_metrics(test.values, bt_preds)
        duration_ms = int((time.perf_counter() - t0) * 1000)

        logger.info(
            f"  [{name}] MAPE={metrics['mape']:.2f}% RMSE={metrics['rmse']:.0f} "
            f"duration={duration_ms}ms"
        )

        return ModelResult(
            name=name,
            mape=metrics["mape"],
            bias=metrics["bias"],
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            predictions=[round(float(v), 2) for v in future_preds[:horizon]],
            durationMs=duration_ms,
        )

    except Exception as e:
        logger.warning(f"  [{name}] failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Evaluate all models for one series
# ─────────────────────────────────────────────────────────────────────

ALL_MODELS = ["ARIMA", "Prophet", "XGBoost", "LSTM", "HoltWinters", "ETS"]
# LSTM is heavy — run sequentially; others in thread pool
PARALLEL_MODELS = ["ARIMA", "Prophet", "XGBoost", "HoltWinters", "ETS"]
SEQUENTIAL_MODELS = ["LSTM"]


def evaluate_all_models(
    series: pd.Series,
    horizon: int,
    freq: str,
    df_full: pd.DataFrame,
    payload: ForecastRequest,
    test_ratio: float = 0.20,
) -> List[ModelResult]:
    """Run all models, backtest, return sorted results (best MAPE first)."""
    n = len(series)
    test_len = max(1, int(np.ceil(n * test_ratio)))
    train_len = n - test_len

    if train_len < 4:
        logger.warning(f"Series too short ({n}) for backtest — skipping.")
        return []

    train = series.iloc[:train_len]
    test = series.iloc[train_len:]

    logger.info(
        f"  Backtesting {n} pts | train={train_len} test={test_len} | "
        f"freq={freq} horizon={horizon}"
    )

    results: List[ModelResult] = []

    # Parallel (light models)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_eval_model, name, train, test, series, horizon, freq, df_full, payload): name
            for name in PARALLEL_MODELS
        }
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    # Sequential (LSTM)
    for name in SEQUENTIAL_MODELS:
        r = _eval_model(name, train, test, series, horizon, freq, df_full, payload)
        if r:
            results.append(r)

    results.sort(key=lambda x: x.mape)
    return results
