import 'server-only';

import { existsSync } from 'fs';
import fs from 'fs/promises';
import path from 'path';

/* ─────────────────────────────────────────────────────────
   Server-side loader for extracted dashboard data.
   Reads JSON files produced by scripts/extract_dashboard_data.py
   from data/dashboard/.
   ───────────────────────────────────────────────────────── */

// ─── Types ───

export type DatasetStats = {
  region: string;
  label: string;
  rows: number;
  columns: number;
  column_names: string[];
  date_range: { start: string | null; end: string | null };
  target_columns: string[];
  weather_features: number;
  lag_features: number;
  calendar_features: number;
  total_features: number;
  targets_summary: Record<string, TargetSummary>;
  targets: Record<string, TargetSummary>;
  missing_pct: Record<string, number>;
};

export type TargetSummary = {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  non_zero_pct: number;
};

export type TimeseriesPoint = {
  timestamp: string;
  load_mw?: number;
  wind_mw?: number;
  solar_mw?: number;
  price_eur_mwh?: number | null;
  carbon_kg_per_mwh?: number | null;
};

export type ForecastPoint = {
  timestamp: string;
  actual: number;
  predicted?: number;
  forecast: number;
  lower_90: number;
  upper_90: number;
  lower_50: number;
  upper_50: number;
};

export type DispatchPoint = {
  timestamp: string;
  load_mw: number;
  generation_solar: number;
  generation_wind: number;
  generation_gas: number;
  generation_coal?: number;
  generation_nuclear?: number;
  generation_hydro?: number;
  price_eur_mwh?: number | null;
};

export type HourlyProfile = {
  hour: number;
  mean: number;
  std: number;
  min: number;
  max: number;
};

export type ModelMetric = {
  target: string;
  model: string;
  rmse: number;
  mae: number;
  mape: number | null;
  smape: number | null;
  r2?: number;
  coverage_90?: number;
  n_features: number;
  residual_q10?: number;
  residual_q50?: number;
  residual_q90?: number;
  tuned_params?: Record<string, unknown>;
};

export type ModelRegistryEntry = {
  model: string;
  target: string;
  file: string;
  path?: string;
  size_bytes: number;
  size_mb: number;
  modified: string;
  region: string;
};

export type ImpactData = {
  region: string;
  baseline_cost_usd: number | null;
  gridpulse_cost_usd: number | null;
  cost_savings_pct: number | null;
  baseline_carbon_kg: number | null;
  gridpulse_carbon_kg: number | null;
  carbon_reduction_pct: number | null;
  baseline_peak_mw: number | null;
  gridpulse_peak_mw: number | null;
  peak_shaving_pct: number | null;
};

export type DashboardManifest = {
  generated_at: string;
  regions: Record<string, { id: string; label: string; stats: DatasetStats; timeseries_hours: number }>;
};

export type RegionDashboardData = {
  stats: DatasetStats | null;
  timeseries: TimeseriesPoint[];
  forecast: Record<string, ForecastPoint[]>;
  dispatch: DispatchPoint[];
  profiles: Record<string, HourlyProfile[]>;
  metrics: ModelMetric[];
  impact: ImpactData | null;
  registry: ModelRegistryEntry[];
};

// ─── Path resolution ───

function resolveDashboardDir(): string {
  const cwd = process.cwd();
  // When running from frontend/, go up one level
  const candidate1 = path.resolve(cwd, 'data', 'dashboard');
  if (existsSync(candidate1)) return candidate1;
  const candidate2 = path.resolve(cwd, '..', 'data', 'dashboard');
  if (existsSync(candidate2)) return candidate2;
  return candidate1; // fallback
}

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    if (!existsSync(filePath)) return null;
    const raw = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

// ─── Public API ───

export async function loadManifest(): Promise<DashboardManifest | null> {
  const dir = resolveDashboardDir();
  return readJsonFile<DashboardManifest>(path.join(dir, 'manifest.json'));
}

export async function loadRegionData(region: 'DE' | 'US'): Promise<RegionDashboardData> {
  const dir = resolveDashboardDir();
  const prefix = region.toLowerCase();

  const [stats, timeseries, forecast, dispatch, profiles, metrics, impact, registry] =
    await Promise.all([
      readJsonFile<DatasetStats>(path.join(dir, `${prefix}_stats.json`)),
      readJsonFile<TimeseriesPoint[]>(path.join(dir, `${prefix}_timeseries.json`)),
      readJsonFile<Record<string, ForecastPoint[]>>(path.join(dir, `${prefix}_forecast.json`)),
      readJsonFile<DispatchPoint[]>(path.join(dir, `${prefix}_dispatch.json`)),
      readJsonFile<Record<string, HourlyProfile[]>>(path.join(dir, `${prefix}_profiles.json`)),
      readJsonFile<ModelMetric[]>(path.join(dir, `${prefix}_metrics.json`)),
      readJsonFile<ImpactData>(path.join(dir, `${prefix}_impact.json`)),
      readJsonFile<ModelRegistryEntry[]>(path.join(dir, `${prefix}_registry.json`)),
    ]);

  return {
    stats: stats ?? null,
    timeseries: timeseries ?? [],
    forecast: forecast ?? {},
    dispatch: dispatch ?? [],
    profiles: profiles ?? {},
    metrics: metrics ?? [],
    impact: impact ?? null,
    registry: registry ?? [],
  };
}

export async function loadAllRegionData(): Promise<Record<string, RegionDashboardData>> {
  const [de, us] = await Promise.all([loadRegionData('DE'), loadRegionData('US')]);
  return { DE: de, US: us };
}
