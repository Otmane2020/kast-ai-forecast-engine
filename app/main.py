"""
Kast AI — Forecasting Engine
FastAPI entrypoint with CORS, optional API-key auth, and health check.
"""
from __future__ import annotations
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.routes.forecast import router as forecast_router
from app.routes.kpi import router as kpi_router
from app.routes.alerts import router as alerts_router

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
KAST_API_KEY: str | None = os.getenv("KAST_API_KEY")
CORS_ORIGINS_RAW: str = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = (
    [o.strip() for o in CORS_ORIGINS_RAW.split(",")]
    if CORS_ORIGINS_RAW != "*"
    else ["*"]
)
VERSION = "1.0.0"
AVAILABLE_MODELS = ["ARIMA", "Prophet", "XGBoost", "LSTM", "HoltWinters", "ETS"]


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
def _warmup_imports() -> None:
    """Pre-import heavy libraries so the first /api/forecast request is fast
    and the healthcheck doesn't race against cold-import time."""
    import time
    libs = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("statsmodels", "statsmodels.tsa.arima.model"),
        ("scikit-learn", "sklearn.ensemble"),
        ("xgboost", "xgboost"),
        ("prophet", "prophet"),
        ("tensorflow", "tensorflow"),
    ]
    for label, module in libs:
        t = time.perf_counter()
        try:
            import importlib
            importlib.import_module(module)
            logger.info(f"  ✓ {label} loaded in {(time.perf_counter()-t)*1000:.0f}ms")
        except ImportError as e:
            logger.warning(f"  ✗ {label} not available: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Kast AI Forecast Engine v{VERSION} starting…")
    if KAST_API_KEY:
        logger.info("API key authentication ENABLED.")
    else:
        logger.info("API key authentication DISABLED (set KAST_API_KEY to enable).")

    # Pre-warm heavy imports in a thread so the event loop stays free
    import asyncio
    logger.info("Pre-warming ML libraries…")
    await asyncio.get_event_loop().run_in_executor(None, _warmup_imports)
    logger.info("All libraries ready. Server accepting requests.")

    yield
    logger.info("Kast AI Forecast Engine shutting down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Kast AI — Forecast Engine",
    description="6-model forecasting engine with automatic backtesting and model selection.",
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Optional API-key auth dependency
# ─────────────────────────────────────────────
async def verify_api_key(request: Request):
    if not KAST_API_KEY:
        return  # Auth disabled
    auth_header = request.headers.get("Authorization", "")
    x_key = request.headers.get("X-API-Key", "")
    token = x_key or (auth_header.removeprefix("Bearer ").strip() if auth_header else "")
    if token != KAST_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────────────────────────────────────
# Exception handlers
# ─────────────────────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "code": exc.status_code, "detail": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exc_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "code": 422,
            "detail": "Validation error",
            "errors": [
                {"field": " → ".join(str(l) for l in e["loc"]), "msg": e["msg"]}
                for e in exc.errors()
            ],
        },
    )


@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "code": 500, "detail": "Internal server error."},
    )


# ─────────────────────────────────────────────
# Routers  (all protected by optional API-key)
# ─────────────────────────────────────────────
_auth = [Depends(verify_api_key)]

app.include_router(forecast_router, dependencies=_auth)
app.include_router(kpi_router, dependencies=_auth)
app.include_router(alerts_router, dependencies=_auth)


# ─────────────────────────────────────────────
# Health / root
# ─────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "version": VERSION,
        "models": AVAILABLE_MODELS,
    }


@app.get("/", tags=["system"])
async def root():
    return {
        "service": "Kast AI Forecast Engine",
        "version": VERSION,
        "docs": "/docs",
        "endpoints": {
            "forecast": "POST /api/forecast",
            "kpi": "GET /api/kpi",
            "alerts": "GET /api/alerts",
            "health": "GET /health",
        },
    }


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "prod") == "dev",
        log_level="info",
        workers=int(os.getenv("WORKERS", 1)),
    )
