"""
Thread-safe in-memory store for KPIs and alerts.
Optionally backed by PostgreSQL when DATABASE_URL is set.
"""
from __future__ import annotations
import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_forecast_runs: List[Dict[str, Any]] = []
_alert_log: List[Dict[str, Any]] = []

MAX_RUNS = 500   # cap to avoid unbounded memory growth


async def record_forecast_run(
    best_model: str,
    mape: float,
    horizon: int,
    group_key: str = "global",
) -> None:
    async with _lock:
        _forecast_runs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "best_model": best_model,
            "mape": mape,
            "horizon": horizon,
            "group_key": group_key,
        })
        if len(_forecast_runs) > MAX_RUNS:
            _forecast_runs.pop(0)


async def record_alerts(alerts: list) -> None:
    async with _lock:
        for a in alerts:
            _alert_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "type": a.type,
                "severity": a.severity,
                "message": a.message,
                "groupKey": a.groupKey,
            })
        if len(_alert_log) > MAX_RUNS:
            _alert_log[:] = _alert_log[-MAX_RUNS:]


async def get_kpi_data() -> Dict[str, Any]:
    async with _lock:
        if not _forecast_runs:
            return {
                "totalForecasts": 0,
                "avgMape": 0.0,
                "medianMape": 0.0,
                "bestModelFrequency": {},
                "avgHorizon": 0.0,
                "lastUpdated": "-",
                "recentRuns": [],
            }

        import numpy as np
        mapes = [r["mape"] for r in _forecast_runs]
        horizons = [r["horizon"] for r in _forecast_runs]
        model_freq: Dict[str, int] = {}
        for r in _forecast_runs:
            model_freq[r["best_model"]] = model_freq.get(r["best_model"], 0) + 1

        return {
            "totalForecasts": len(_forecast_runs),
            "avgMape": round(float(np.mean(mapes)), 4),
            "medianMape": round(float(np.median(mapes)), 4),
            "bestModelFrequency": model_freq,
            "avgHorizon": round(float(np.mean(horizons)), 1),
            "lastUpdated": _forecast_runs[-1]["timestamp"],
            "recentRuns": list(reversed(_forecast_runs[-10:])),
        }


async def get_alert_data() -> List[Dict[str, Any]]:
    async with _lock:
        return list(reversed(_alert_log[-100:]))
