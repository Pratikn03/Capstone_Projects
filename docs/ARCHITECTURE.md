# Architecture

GridPulse is a decision-grade energy intelligence system: predictions feed an optimizer, and the impact is measured against baselines. The same pipeline runs on OPSD (Germany) and EIA-930 (US).

## System flow
1) **Data ingestion + validation**
   - OPSD Germany load/wind/solar (optional day-ahead price)
   - EIA-930 balancing-authority demand + generation (e.g., MISO)
   - Optional weather features (Open-Meteo)

2) **Time-aware feature store**
   - Hour/day/week seasonal features
   - Lags and rolling windows
   - Price and carbon proxies (if not provided)

3) **Forecasting engine**
   - GBM baseline (LightGBM)
   - LSTM and TCN deep models
   - 24-hour horizon with evaluation metrics

4) **Reliability layer**
   - Residual checks
   - Isolation Forest anomaly flags

5) **Decision layer (LP optimization)**
   - Battery SoC and power limits
   - Grid import caps
   - Objective: cost + carbon + peak penalty

6) **Impact evaluation**
   - Baselines: grid-only and naive battery
   - Metrics: cost savings, carbon reduction, peak shaving

7) **Product layer**
   - FastAPI endpoints
   - Next.js operator dashboard

## Key artifacts
- `configs/` for training and optimization
- `reports/` and `reports/eia930/` for figures and impact summaries
- `services/` for API
- `frontend/` for the Next.js dashboard
- `notebooks/` for runbooks and analysis
