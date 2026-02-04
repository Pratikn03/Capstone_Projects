"""FastAPI application entrypoint."""
from fastapi import FastAPI

from gridpulse.utils.logging import setup_logging
from services.api.health import readiness_check
from services.api.routers import forecast, anomaly, optimize, monitor

setup_logging()
app = FastAPI(title="GridPulse API", version="0.1.0")

app.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
app.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"])
app.include_router(optimize.router, prefix="/optimize", tags=["optimize"])
app.include_router(monitor.router, prefix="/monitor", tags=["monitor"])

@app.get("/health")
def health():
    # Key: FastAPI application setup
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return readiness_check()
