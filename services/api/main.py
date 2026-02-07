"""FastAPI application entrypoint."""
from fastapi import FastAPI, HTTPException, Security
from pydantic import BaseModel

from gridpulse.safety.bms import SafetyLayer, SafetyViolation
from gridpulse.safety.watchdog import SystemWatchdog
from gridpulse.utils.logging import setup_logging
from services.api.config import get_bms_config, get_watchdog_timeout
from services.api.health import readiness_check
from services.api.routers import forecast, anomaly, optimize, monitor
from services.api.routers.forecast_intervals import router as intervals_router
from services.api.security import get_api_key, verify_scope

setup_logging()
app = FastAPI(title="GridPulse API", version="0.1.0")

# Initialize Systems
bms_cfg = get_bms_config()
bms = SafetyLayer(
    capacity_mwh=bms_cfg["capacity_mwh"],
    max_power_mw=bms_cfg["max_power_mw"],
    min_soc_pct=bms_cfg["min_soc_pct"],
    max_soc_pct=bms_cfg["max_soc_pct"],
)
watchdog = SystemWatchdog(timeout_seconds=get_watchdog_timeout())


@app.on_event("startup")
async def startup_event():
    watchdog.start()


@app.on_event("shutdown")
async def shutdown_event():
    watchdog.stop()


app.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
app.include_router(intervals_router, prefix="/forecast", tags=["forecast"])
app.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"])
app.include_router(optimize.router, prefix="/optimize", tags=["optimize"])
app.include_router(monitor.router, prefix="/monitor", tags=["monitor"])


class DispatchRequest(BaseModel):
    charge_mw: float
    discharge_mw: float
    current_soc_mwh: float


@app.get("/health")
def health():
    # Key: FastAPI application setup
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return readiness_check()


@app.get("/system/health")
def health_check(api_key: str = Security(get_api_key)):
    """Public health check - View Only."""
    verify_scope("read", api_key)
    return {
        "status": "online",
        "mode": watchdog.get_status(),
        "safety_layer": "active",
    }


@app.post("/system/heartbeat")
def send_heartbeat(api_key: str = Security(get_api_key)):
    """Watchdog reset. Must be called every 30s."""
    verify_scope("write", api_key)
    watchdog.beat()
    return {"status": "heartbeat_received"}


@app.post("/control/dispatch")
def set_dispatch(command: DispatchRequest, api_key: str = Security(get_api_key)):
    """
    Send a dispatch command to the battery.
    Protected by:
    1. RBAC (Scope Check)
    2. Watchdog (Island Mode Check)
    3. BMS (Physics Check)
    """
    # 1. Security Check
    verify_scope("write", api_key)

    # 2. Resilience Check
    if watchdog.is_islanded:
        raise HTTPException(status_code=503, detail="System is ISLANDED. Remote control rejected.")

    # 3. Safety Check
    try:
        bms.validate_dispatch(
            current_soc=command.current_soc_mwh,
            charge_mw=command.charge_mw,
            discharge_mw=command.discharge_mw,
        )
    except SafetyViolation as exc:
        raise HTTPException(status_code=400, detail=f"SAFETY VIOLATION: {exc}") from exc

    # If we get here, the command is valid, safe, and authorized.
    return {"status": "accepted", "command": command}
