# ORIUS API Service

RESTful API for the ORIUS energy forecasting and optimization platform.

## Overview

This FastAPI-based service exposes endpoints for:
- **Forecasting**: Load, solar, wind, and price predictions
- **Anomaly Detection**: Real-time outlier identification
- **Optimization**: Battery dispatch scheduling
- **Monitoring**: Model drift and data quality metrics
- **Universal Framework**: Cross-domain ORIUS safety-step execution

## Architecture

```
services/api/
├── main.py            # FastAPI application entrypoint
├── config.py          # Configuration management
├── health.py          # Health check endpoints
├── security.py        # API key authentication
└── routers/
    ├── forecast.py           # /forecast/* endpoints
    ├── forecast_intervals.py # Prediction intervals
    ├── anomaly.py            # /anomaly/* endpoints
    ├── optimize.py           # /optimize/* endpoints
    ├── monitor.py            # /monitor/* endpoints
    └── universal.py         # /universal/* endpoints
```

## Running Locally

```bash
# Development server
uvicorn services.api.main:app --reload --port 8000

# Production
gunicorn services.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/forecast/predict` | POST | Generate forecasts |
| `/forecast/intervals` | POST | Prediction intervals |
| `/anomaly/detect` | POST | Detect anomalies |
| `/optimize` | POST | Optimize battery dispatch (`optimization_mode=robust|deterministic`, default robust) |
| `/monitor` | GET | Model drift metrics and retraining decision |
| `/monitor/research-metrics` | GET | Latest DE/US EVPI-VSS research summaries + frozen snapshot |
| `/universal/domains` | GET | List registered ORIUS framework domains, stages, and framework tables |
| `/universal/step` | POST | Run one universal ORIUS safety step for a selected domain |

## Authentication

Set the `ORIUS_API_KEY` environment variable for API key authentication.

## Configuration

See `configs/serving.yaml` for service configuration options.
