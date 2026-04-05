"""
Anomaly detection:
  1. Z-score on historical series
  2. Forecast deviation (actual vs predicted in backtest)
"""
from __future__ import annotations
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from app.models.schemas import AlertItem

logger = logging.getLogger(__name__)


def detect_zscore_anomalies(
    series: pd.Series,
    group_key: str,
    z_threshold: float = 2.5,
) -> List[AlertItem]:
    alerts: List[AlertItem] = []
    if len(series) < 4:
        return alerts

    mean = series.mean()
    std = series.std()
    if std == 0:
        return alerts

    z_scores = (series - mean) / std

    for date, z in z_scores.items():
        if abs(z) < z_threshold:
            continue
        val = float(series[date])
        direction = "spike" if z > 0 else "drop"
        severity: str
        if abs(z) >= 3.5:
            severity = "critical"
        elif abs(z) >= 3.0:
            severity = "warning"
        else:
            severity = "info"

        alerts.append(AlertItem(
            type="anomaly",
            severity=severity,
            message=(
                f"{group_key}: {direction} detected on {date.date()} "
                f"(value={val:.0f}, z={z:.2f}, mean={mean:.0f})"
            ),
            groupKey=group_key,
        ))
    return alerts


def detect_forecast_deviation(
    actual: pd.Series,
    predictions: List[float],
    group_key: str,
    threshold_pct: float = 0.20,
) -> List[AlertItem]:
    """Flag periods where actual diverges > threshold% from forecast."""
    alerts: List[AlertItem] = []
    if not predictions or len(actual) == 0:
        return alerts

    n = min(len(actual), len(predictions))
    for i in range(n):
        act = float(actual.iloc[i])
        pred = float(predictions[i])
        if act == 0 and pred == 0:
            continue
        denom = act if act != 0 else (pred if pred != 0 else 1)
        dev = (pred - act) / denom

        if abs(dev) >= threshold_pct:
            pct = dev * 100
            direction = "over-forecast" if dev > 0 else "under-forecast"
            severity = "critical" if abs(dev) >= 0.35 else "warning"
            date_str = str(actual.index[i].date()) if hasattr(actual.index[i], "date") else str(actual.index[i])
            alerts.append(AlertItem(
                type="deviation",
                severity=severity,
                message=(
                    f"{group_key}: {direction} {abs(pct):.1f}% on {date_str} "
                    f"(actual={act:.0f}, forecast={pred:.0f})"
                ),
                groupKey=group_key,
            ))
    return alerts


def detect_trend_change(
    series: pd.Series,
    group_key: str,
    window: int = 3,
) -> List[AlertItem]:
    """Detect sudden trend reversal in last `window` points."""
    alerts: List[AlertItem] = []
    if len(series) < window * 2 + 1:
        return alerts

    recent = series.iloc[-window:]
    previous = series.iloc[-(window * 2): -window]

    avg_recent = recent.mean()
    avg_prev = previous.mean()
    if avg_prev == 0:
        return alerts

    pct_change = (avg_recent - avg_prev) / avg_prev * 100
    if abs(pct_change) >= 25:
        direction = "upward" if pct_change > 0 else "downward"
        severity = "warning" if abs(pct_change) < 40 else "critical"
        alerts.append(AlertItem(
            type="trend",
            severity=severity,
            message=(
                f"{group_key}: {direction} trend shift of {abs(pct_change):.1f}% "
                f"over last {window} periods"
            ),
            groupKey=group_key,
        ))
    return alerts


def run_all_anomaly_checks(
    series: pd.Series,
    group_key: str,
    backtest_predictions: Optional[List[float]] = None,
    backtest_actual: Optional[pd.Series] = None,
) -> List[AlertItem]:
    alerts: List[AlertItem] = []
    alerts += detect_zscore_anomalies(series, group_key)
    alerts += detect_trend_change(series, group_key)
    if backtest_predictions and backtest_actual is not None:
        alerts += detect_forecast_deviation(backtest_actual, backtest_predictions, group_key)
    return alerts
