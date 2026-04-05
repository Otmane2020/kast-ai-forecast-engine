"""
Best model selection and business insights generation.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np

from app.models.schemas import ForecastRequest, ModelResult, GroupForecast

logger = logging.getLogger(__name__)


def select_best_model(results: List[ModelResult]) -> Optional[ModelResult]:
    if not results:
        return None
    return min(results, key=lambda r: r.mape)


def mark_best(results: List[ModelResult]) -> List[ModelResult]:
    best = select_best_model(results)
    for r in results:
        r.isBest = r.name == (best.name if best else "")
    return results


# ─────────────────────────────────────────────────────────────────────
# Business insights
# ─────────────────────────────────────────────────────────────────────

def generate_business_insights(
    global_models: List[ModelResult],
    group_forecasts: List[GroupForecast],
    payload: ForecastRequest,
    global_series_len: int = 0,
) -> str:
    insights: List[str] = []

    # ── Best model on global ──
    if global_models:
        best = select_best_model(global_models)
        if best:
            quality = (
                "excellent" if best.mape < 5 else
                "good" if best.mape < 10 else
                "acceptable" if best.mape < 20 else
                "poor"
            )
            insights.append(
                f"Global best model: **{best.name}** with MAPE={best.mape:.1f}% ({quality} accuracy)."
            )

    # ── Model frequency across groups ──
    if group_forecasts:
        freq_count: Dict[str, int] = {}
        for gf in group_forecasts:
            freq_count[gf.bestModel] = freq_count.get(gf.bestModel, 0) + 1
        top_model = max(freq_count, key=freq_count.get)
        top_count = freq_count[top_model]
        pct = round(top_count / len(group_forecasts) * 100)
        insights.append(
            f"**{top_model}** wins for {top_count}/{len(group_forecasts)} groups ({pct}%)."
        )

    # ── XGBoost note on features ──
    feat_cols = payload.feature_columns()
    if feat_cols and any(gf.bestModel == "XGBoost" for gf in group_forecasts):
        feat_names = ", ".join(c.name for c in feat_cols[:4])
        insights.append(
            f"XGBoost leverages extra features ({feat_names}) — these improve accuracy "
            "when price or regional variations are significant."
        )

    # ── Seasonality hint ──
    if global_series_len >= 24:
        insights.append("Sufficient history (≥24 periods) for reliable seasonal pattern detection.")
    elif global_series_len >= 12:
        insights.append("Moderate history (12–23 periods) — seasonal models may have limited accuracy.")
    else:
        insights.append(
            "Short history (<12 periods) — statistical models disabled; ML/smoothing methods used."
        )

    # ── Business context echo ──
    if payload.businessContext:
        insights.append(f"Context: {payload.businessContext[:120]}.")

    # ── Horizon note ──
    if payload.horizon > 12:
        insights.append(
            f"Long horizon ({payload.horizon} periods) — forecast uncertainty increases; "
            "consider re-running monthly with fresh data."
        )

    # ── Granularity note ──
    if payload.granularity == "sku" and len(group_forecasts) > 20:
        high_mape = [gf for gf in group_forecasts if select_best_model(gf.models) and select_best_model(gf.models).mape > 25]
        if high_mape:
            keys = ", ".join(g.groupKey for g in high_mape[:3])
            insights.append(
                f"{len(high_mape)} SKU(s) have MAPE > 25% (e.g., {keys}). "
                "Consider reviewing data quality or grouping them at family level."
            )

    return " ".join(insights) if insights else "Forecast completed successfully."
