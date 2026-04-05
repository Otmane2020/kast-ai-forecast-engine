from fastapi import APIRouter
from app.models.schemas import KPIResponse
from app.services.store import get_kpi_data

router = APIRouter()


@router.get("/api/kpi", response_model=KPIResponse)
async def get_kpi():
    """Return aggregated model performance KPIs across all forecast runs."""
    data = await get_kpi_data()
    return KPIResponse(status="success", **data)
