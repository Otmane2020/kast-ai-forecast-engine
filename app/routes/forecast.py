"""
POST /api/forecast — main forecasting endpoint.
Timeout: 120 seconds.
"""
from __future__ import annotations
import asyncio
import logging
from functools import partial
from typing import List

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ForecastRequest,
    ForecastResponse,
    GlobalForecast,
    GroupForecast,
    ModelResult,
    TimePoint,
    AlertItem,
)
from app.services.data_processor import (
    build_dataframe,
    detect_frequency,
    build_series,
    get_groups,
)
from app.services.backtesting import evaluate_all_models
from app.services.anomaly import run_all_anomaly_checks
from app.services.model_selector import select_best_model, mark_best, generate_business_insights
from app.services.store import record_forecast_run, record_alerts

logger = logging.getLogger(__name__)
router = APIRouter()

FORECAST_TIMEOUT = 120.0  # seconds


# ─────────────────────────────────────────────────────────────────────
# Sync processing (runs in thread pool via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────

def _process_forecast(payload: ForecastRequest) -> ForecastResponse:
    # ── 1. Build DataFrame ──
    df = build_dataframe(payload)
    date_col = payload.mapping.dateCol
    value_col = payload.mapping.valueCol

    # ── 2. Detect frequency ──
    global_series = build_series(df, date_col, value_col, "M")
    freq = detect_frequency(global_series)
    global_series = build_series(df, date_col, value_col, freq)

    logger.info(
        f"[Forecast] freq={freq} total_pts={len(global_series)} "
        f"granularity={payload.granularity} horizon={payload.horizon}"
    )

    # ── 3. Global forecast (all data) ──
    global_models = evaluate_all_models(
        global_series, payload.horizon, freq, df, payload
    )
    global_models = mark_best(global_models)
    best_global = select_best_model(global_models)

    global_forecast = GlobalForecast(
        bestModel=best_global.name if best_global else "N/A",
        models=global_models,
        horizon=payload.horizon,
        historicalLength=len(global_series),
    )

    # ── 4. Group forecasts ──
    groups = get_groups(df, payload, freq)
    group_forecasts: List[GroupForecast] = []
    all_alerts: List[AlertItem] = []

    # Anomalies on global series
    all_alerts += run_all_anomaly_checks(global_series, "global")

    for group_key, series in groups.items():
        logger.info(f"[Forecast] Group: {group_key} ({len(series)} pts)")

        models = evaluate_all_models(series, payload.horizon, freq, df, payload)
        models = mark_best(models)
        best = select_best_model(models)

        predictions = best.predictions if best else []
        timeseries = [
            TimePoint(date=str(idx.date()), value=round(float(v), 2))
            for idx, v in series.items()
        ]

        group_forecasts.append(GroupForecast(
            groupKey=group_key,
            bestModel=best.name if best else "N/A",
            models=models,
            timeSeries=timeseries,
            predictions=predictions,
        ))

        # Anomalies per group
        if group_key != "global":
            all_alerts += run_all_anomaly_checks(series, group_key)

    # ── 5. Business insights ──
    insights = generate_business_insights(
        global_models, group_forecasts, payload, len(global_series)
    )

    return ForecastResponse(
        status="success",
        globalForecast=global_forecast,
        groupForecasts=group_forecasts,
        alerts=all_alerts[:50],   # cap at 50
        businessInsights=insights,
    )


# ─────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────

@router.post("/api/forecast", response_model=ForecastResponse)
async def forecast(payload: ForecastRequest):
    """
    Run 6 forecasting models per group with 80/20 backtesting.
    Auto-selects best model by MAPE. Timeout: 120s.
    """
    try:
        result: ForecastResponse = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, partial(_process_forecast, payload)
            ),
            timeout=FORECAST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Forecast timed out after 120s. Reduce data size or horizon.",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected forecast error")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

    # Persist KPI records (async, non-blocking)
    gf = result.globalForecast
    if gf.models:
        asyncio.create_task(
            record_forecast_run(gf.bestModel, min(m.mape for m in gf.models), gf.horizon)
        )
    if result.alerts:
        asyncio.create_task(record_alerts(result.alerts))

    return result
