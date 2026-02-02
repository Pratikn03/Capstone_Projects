"""FastAPI application entrypoint."""
from fastapi import FastAPI
from services.api.routers import forecast, anomaly, optimize, monitor

app = FastAPI(title="GridPulse API", version="0.1.0")

app.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
app.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"])
app.include_router(optimize.router, prefix="/optimize", tags=["optimize"])
app.include_router(monitor.router, prefix="/monitor", tags=["monitor"])

@app.get("/health")
def health():
    # Key: FastAPI application setup
    return {"status": "ok"}
