from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal, Union


# ─────────────────────────────────────────────
# REQUEST
# ─────────────────────────────────────────────

class ColumnMapping(BaseModel):
    dateCol: str
    valueCol: str
    productCol: Optional[str] = None
    categoryCol: Optional[str] = None


class ColumnInfo(BaseModel):
    name: str
    role: str          # date | value | product | category | price | region | channel | …
    description: Optional[str] = None


class DateRange(BaseModel):
    min: str
    max: str


class ForecastRequest(BaseModel):
    fileName: str = ""
    mapping: ColumnMapping
    allColumns: List[ColumnInfo] = []
    businessContext: Optional[str] = None
    granularity: Literal["global", "sku", "family", "subfamily"] = "global"
    data: List[Dict[str, Any]]
    totalRows: int = 0
    uniqueProducts: Optional[List[str]] = None
    uniqueCategories: Optional[List[str]] = None
    dateRange: Optional[DateRange] = None
    horizon: int = Field(default=6, ge=1, le=36)

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, v):
        if not v:
            raise ValueError("data must not be empty")
        return v

    def feature_columns(self) -> List[ColumnInfo]:
        """Columns that are not date/value/product/category — usable as XGBoost features."""
        skip = {"date", "value", "product", "category"}
        return [c for c in self.allColumns if c.role.lower() not in skip]


# ─────────────────────────────────────────────
# RESPONSE
# ─────────────────────────────────────────────

class ModelResult(BaseModel):
    name: str
    mape: float
    bias: float
    mae: float
    rmse: float
    predictions: List[float]
    isBest: bool = False
    durationMs: int = 0


class GlobalForecast(BaseModel):
    bestModel: str
    models: List[ModelResult]
    horizon: int
    historicalLength: int


class TimePoint(BaseModel):
    date: str
    value: float


class GroupForecast(BaseModel):
    groupKey: str
    bestModel: str
    models: List[ModelResult]
    timeSeries: List[TimePoint]
    predictions: List[float]


class AlertItem(BaseModel):
    type: str                                  # anomaly | deviation | trend
    severity: Literal["info", "warning", "critical"]
    message: str
    groupKey: Optional[str] = None


class ForecastResponse(BaseModel):
    status: str
    globalForecast: GlobalForecast
    groupForecasts: List[GroupForecast]
    alerts: List[AlertItem]
    businessInsights: str


# ─────────────────────────────────────────────
# KPI / ALERTS endpoints
# ─────────────────────────────────────────────

class KPIResponse(BaseModel):
    status: str
    totalForecasts: int
    avgMape: float
    medianMape: float
    bestModelFrequency: Dict[str, int]
    avgHorizon: float
    lastUpdated: str
    recentRuns: List[Dict[str, Any]]


class AlertsResponse(BaseModel):
    status: str
    total: int
    critical: int
    warning: int
    info: int
    alerts: List[AlertItem]
