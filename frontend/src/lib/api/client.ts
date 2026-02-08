import { z } from 'zod';
import {
  type Anomaly,
  type ForecastMetrics,
  type ZoneSummary,
} from './schema';

/**
 * GridPulse API client.
 *
 * Routes aligned with FastAPI backend (services/api/main.py):
 *   GET  /forecast                → forecast router (targets, horizon params)
 *   GET  /forecast/with-intervals → conformal PI intervals
 *   POST /anomaly                 → anomaly detection (with payload)
 *   GET  /anomaly                 → anomaly detection (last 7 days)
 *   POST /optimize                → dispatch optimization
 *   GET  /monitor                 → monitoring / drift
 *   GET  /health                  → health check
 *   GET  /ready                   → readiness check
 *   GET  /system/health           → system health (API key required)
 *   POST /system/heartbeat        → watchdog heartbeat (API key required)
 *   POST /control/dispatch        → battery dispatch command (API key required)
 */

const API_BASE = process.env.FASTAPI_URL || 'http://localhost:8000';

async function apiFetchRaw(path: string, options?: RequestInit): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`, {
    cache: 'no-store',
    headers: {
      'X-GridPulse-Key': process.env.API_SECRET_KEY || '',
      'Content-Type': 'application/json',
    },
    ...options,
  });

  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${path}`);
  }

  return res.json();
}

// ─── Health ───
export async function checkHealth(): Promise<{ status: string }> {
  return apiFetchRaw('/health');
}

export async function checkReady(): Promise<any> {
  return apiFetchRaw('/ready');
}

// ─── Forecast (all targets at once) ───
export async function getForecast(horizonHours: number = 24, targets?: string[]): Promise<any> {
  const params = new URLSearchParams({ horizon: String(horizonHours) });
  if (targets) params.set('targets', targets.join(','));
  return apiFetchRaw(`/forecast?${params}`);
}

// ─── Forecast with Prediction Intervals ───
export async function getForecastWithIntervals(target: string = 'load_mw', horizon: number = 24): Promise<{
  yhat: number[];
  pi90_lower: number[] | null;
  pi90_upper: number[] | null;
}> {
  return apiFetchRaw(`/forecast/with-intervals?target=${target}&horizon=${horizon}`);
}

// ─── Anomaly Detection ───
export async function getAnomalies(): Promise<{
  residual_z: boolean[];
  iforest: boolean[];
  combined: boolean[];
  z_scores: number[];
}> {
  return apiFetchRaw('/anomaly');
}

export async function detectAnomalies(actual: number[], forecast: number[], features?: number[][]): Promise<{
  residual_z: boolean[];
  iforest: boolean[];
  combined: boolean[];
  z_scores: number[];
}> {
  return apiFetchRaw('/anomaly', {
    method: 'POST',
    body: JSON.stringify({ actual, forecast, features }),
  });
}

// ─── Dispatch Optimization ───
export async function optimizeDispatch(
  forecastLoadMw: number | number[],
  forecastRenewablesMw: number | number[],
  options?: {
    forecastPrice?: number | number[];
    forecastCarbon?: number | number[];
    loadInterval?: { lower?: number | number[]; upper?: number | number[] };
    renewablesInterval?: { lower?: number | number[]; upper?: number | number[] };
  }
): Promise<{
  dispatch_plan: Record<string, any>;
  expected_cost_usd: number | null;
  carbon_kg: number | null;
  carbon_cost_usd: number | null;
}> {
  return apiFetchRaw('/optimize', {
    method: 'POST',
    body: JSON.stringify({
      forecast_load_mw: forecastLoadMw,
      forecast_renewables_mw: forecastRenewablesMw,
      forecast_price_eur_mwh: options?.forecastPrice,
      forecast_carbon_kg_per_mwh: options?.forecastCarbon,
      load_interval: options?.loadInterval,
      renewables_interval: options?.renewablesInterval,
    }),
  });
}

// ─── Monitoring / Drift ───
export async function getMonitoringStatus(): Promise<{
  data_drift: Record<string, any>;
  model_drift: Record<string, any>;
  retraining: { retrain: boolean; reasons: string[]; last_trained_days_ago: number | null };
}> {
  return apiFetchRaw('/monitor');
}

// ─── System (API key required) ───
export async function getSystemHealth(): Promise<{ status: string; mode: string; safety_layer: string }> {
  return apiFetchRaw('/system/health');
}

export async function sendHeartbeat(): Promise<{ status: string }> {
  return apiFetchRaw('/system/heartbeat', { method: 'POST' });
}

export async function sendDispatchCommand(
  chargeMw: number,
  dischargeMw: number,
  currentSocMwh: number
): Promise<{ status: string; command: any }> {
  return apiFetchRaw('/control/dispatch', {
    method: 'POST',
    body: JSON.stringify({
      charge_mw: chargeMw,
      discharge_mw: dischargeMw,
      current_soc_mwh: currentSocMwh,
    }),
  });
}
