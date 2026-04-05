from fastapi import APIRouter
from app.models.schemas import AlertsResponse, AlertItem
from app.services.store import get_alert_data

router = APIRouter()


@router.get("/api/alerts", response_model=AlertsResponse)
async def get_alerts():
    """Return all detected anomalies and forecast deviations."""
    raw = await get_alert_data()
    alerts = [
        AlertItem(
            type=a["type"],
            severity=a["severity"],
            message=a["message"],
            groupKey=a.get("groupKey"),
        )
        for a in raw
    ]
    return AlertsResponse(
        status="success",
        total=len(alerts),
        critical=sum(1 for a in alerts if a.severity == "critical"),
        warning=sum(1 for a in alerts if a.severity == "warning"),
        info=sum(1 for a in alerts if a.severity == "info"),
        alerts=alerts,
    )
