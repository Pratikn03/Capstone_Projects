-- GridPulse Database Initialization
-- PostgreSQL schema for production deployment

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- =============================================================================
-- TABLES
-- =============================================================================

-- Telemetry data table
CREATE TABLE IF NOT EXISTS telemetry (
    id BIGSERIAL PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    region VARCHAR(10) NOT NULL DEFAULT 'DE',
    load_mw DOUBLE PRECISION,
    wind_mw DOUBLE PRECISION,
    solar_mw DOUBLE PRECISION,
    price_eur_mwh DOUBLE PRECISION,
    carbon_kg_per_mwh DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (timestamp_utc, region)
);

-- Forecasts table
CREATE TABLE IF NOT EXISTS forecasts (
    id BIGSERIAL PRIMARY KEY,
    forecast_timestamp TIMESTAMPTZ NOT NULL,
    target_timestamp TIMESTAMPTZ NOT NULL,
    region VARCHAR(10) NOT NULL DEFAULT 'DE',
    target VARCHAR(50) NOT NULL,
    model VARCHAR(50) NOT NULL,
    prediction DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    confidence_level DOUBLE PRECISION DEFAULT 0.90,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dispatch plans table
CREATE TABLE IF NOT EXISTS dispatch_plans (
    id BIGSERIAL PRIMARY KEY,
    plan_timestamp TIMESTAMPTZ NOT NULL,
    horizon_hours INTEGER NOT NULL DEFAULT 24,
    region VARCHAR(10) NOT NULL DEFAULT 'DE',
    optimization_mode VARCHAR(20) NOT NULL DEFAULT 'robust',
    battery_charge_schedule JSONB NOT NULL,
    battery_discharge_schedule JSONB NOT NULL,
    grid_import_schedule JSONB NOT NULL,
    total_cost DOUBLE PRECISION,
    baseline_cost DOUBLE PRECISION,
    cost_savings_pct DOUBLE PRECISION,
    carbon_reduction_kg DOUBLE PRECISION,
    feasible BOOLEAN DEFAULT TRUE,
    solver_status VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Anomalies table
CREATE TABLE IF NOT EXISTS anomalies (
    id BIGSERIAL PRIMARY KEY,
    detected_at TIMESTAMPTZ NOT NULL,
    region VARCHAR(10) NOT NULL DEFAULT 'DE',
    target VARCHAR(50) NOT NULL,
    detector VARCHAR(50) NOT NULL,
    anomaly_score DOUBLE PRECISION,
    value DOUBLE PRECISION,
    expected_value DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    severity VARCHAR(20) DEFAULT 'warning',
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    evaluation_timestamp TIMESTAMPTZ NOT NULL,
    region VARCHAR(10) NOT NULL DEFAULT 'DE',
    model VARCHAR(50) NOT NULL,
    target VARCHAR(50) NOT NULL,
    split VARCHAR(20) NOT NULL DEFAULT 'test',
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    mape DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    smape DOUBLE PRECISION,
    picp DOUBLE PRECISION,
    mpiw DOUBLE PRECISION,
    n_samples INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Drift monitoring table
CREATE TABLE IF NOT EXISTS drift_logs (
    id BIGSERIAL PRIMARY KEY,
    check_timestamp TIMESTAMPTZ NOT NULL,
    region VARCHAR(10) NOT NULL DEFAULT 'DE',
    feature VARCHAR(100) NOT NULL,
    ks_statistic DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    drift_detected BOOLEAN DEFAULT FALSE,
    reference_mean DOUBLE PRECISION,
    current_mean DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- System events table
CREATE TABLE IF NOT EXISTS system_events (
    id BIGSERIAL PRIMARY KEY,
    event_timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'info',
    source VARCHAR(100),
    message TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry (timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_telemetry_region_timestamp ON telemetry (region, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_forecasts_target_timestamp ON forecasts (target_timestamp);
CREATE INDEX IF NOT EXISTS idx_forecasts_region_target ON forecasts (region, target, model);

CREATE INDEX IF NOT EXISTS idx_dispatch_timestamp ON dispatch_plans (plan_timestamp);
CREATE INDEX IF NOT EXISTS idx_dispatch_region ON dispatch_plans (region);

CREATE INDEX IF NOT EXISTS idx_anomalies_detected ON anomalies (detected_at);
CREATE INDEX IF NOT EXISTS idx_anomalies_unacknowledged ON anomalies (acknowledged) WHERE acknowledged = FALSE;

CREATE INDEX IF NOT EXISTS idx_metrics_model_target ON model_metrics (model, target);
CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_logs (check_timestamp);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events (event_timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON system_events (event_type);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to get latest telemetry for a region
CREATE OR REPLACE FUNCTION get_latest_telemetry(p_region VARCHAR DEFAULT 'DE')
RETURNS TABLE (
    timestamp_utc TIMESTAMPTZ,
    load_mw DOUBLE PRECISION,
    wind_mw DOUBLE PRECISION,
    solar_mw DOUBLE PRECISION,
    price_eur_mwh DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT t.timestamp_utc, t.load_mw, t.wind_mw, t.solar_mw, t.price_eur_mwh
    FROM telemetry t
    WHERE t.region = p_region
    ORDER BY t.timestamp_utc DESC
    LIMIT 168;  -- Last week
END;
$$ LANGUAGE plpgsql;

-- Function to summarize cost savings
CREATE OR REPLACE FUNCTION summarize_cost_savings(
    p_region VARCHAR DEFAULT 'DE',
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    total_baseline_cost DOUBLE PRECISION,
    total_optimized_cost DOUBLE PRECISION,
    total_savings DOUBLE PRECISION,
    savings_pct DOUBLE PRECISION,
    num_plans INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        SUM(dp.baseline_cost) AS total_baseline_cost,
        SUM(dp.total_cost) AS total_optimized_cost,
        SUM(dp.baseline_cost - dp.total_cost) AS total_savings,
        (1 - SUM(dp.total_cost) / NULLIF(SUM(dp.baseline_cost), 0)) * 100 AS savings_pct,
        COUNT(*)::INTEGER AS num_plans
    FROM dispatch_plans dp
    WHERE dp.region = p_region
      AND dp.plan_timestamp >= NOW() - (p_days || ' days')::INTERVAL
      AND dp.feasible = TRUE;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS
-- =============================================================================

-- View for recent forecast accuracy
CREATE OR REPLACE VIEW v_forecast_accuracy AS
SELECT 
    f.region,
    f.target,
    f.model,
    DATE_TRUNC('day', f.target_timestamp) AS day,
    COUNT(*) AS n_forecasts,
    AVG(ABS(f.prediction - t.load_mw)) AS avg_absolute_error,
    AVG(ABS((f.prediction - t.load_mw) / NULLIF(t.load_mw, 0)) * 100) AS avg_pct_error
FROM forecasts f
JOIN telemetry t ON f.target_timestamp = t.timestamp_utc AND f.region = t.region
WHERE f.target = 'load_mw'
  AND f.target_timestamp >= NOW() - INTERVAL '7 days'
GROUP BY f.region, f.target, f.model, DATE_TRUNC('day', f.target_timestamp);

-- View for daily cost savings
CREATE OR REPLACE VIEW v_daily_cost_savings AS
SELECT 
    region,
    DATE_TRUNC('day', plan_timestamp) AS day,
    SUM(baseline_cost) AS baseline_cost,
    SUM(total_cost) AS optimized_cost,
    SUM(baseline_cost - total_cost) AS savings,
    AVG(cost_savings_pct) AS avg_savings_pct,
    COUNT(*) AS num_plans
FROM dispatch_plans
WHERE feasible = TRUE
GROUP BY region, DATE_TRUNC('day', plan_timestamp);

-- =============================================================================
-- GRANTS (for read-only dashboards)
-- =============================================================================

-- Create read-only user for dashboards
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'gridpulse_readonly') THEN
        CREATE ROLE gridpulse_readonly WITH LOGIN PASSWORD 'readonly123';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE gridpulse TO gridpulse_readonly;
GRANT USAGE ON SCHEMA public TO gridpulse_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO gridpulse_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO gridpulse_readonly;
