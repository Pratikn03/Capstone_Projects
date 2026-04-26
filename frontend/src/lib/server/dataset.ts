import 'server-only';

import { existsSync } from 'fs';
import fs from 'fs/promises';
import path from 'path';
import { type DomainId, isBatteryDomain } from '@/lib/domain-options';

/* ─────────────────────────────────────────────────────────
   Legacy compatibility types for dataset/profile payloads.
   The frontend now treats backend-served tracked research artifacts
   as the canonical truth path; local dashboard caches are not an
   authority surface.
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
  primary_value?: number;
  secondary_value?: number;
  tertiary_value?: number;
  primary_label?: string;
  secondary_label?: string;
  tertiary_label?: string;
  source_index?: string | number;
};

export type ForecastPoint = {
  timestamp: string;
  actual: number;
  predicted?: number;
  forecast: number;
  lower_90?: number | null;
  upper_90?: number | null;
  lower_50?: number | null;
  upper_50?: number | null;
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
  orius_cost_usd: number | null;
  cost_savings_pct: number | null;
  baseline_carbon_kg: number | null;
  orius_carbon_kg: number | null;
  carbon_reduction_pct: number | null;
  baseline_peak_mw: number | null;
  orius_peak_mw: number | null;
  peak_shaving_pct: number | null;
};

export type DriftPoint = {
  date: string;
  ks_statistic: number;
  rolling_rmse: number;
  threshold: number;
  is_drift: boolean;
};

export type DriftedFeature = {
  column: string;
  ks_stat: number;
  p_value: number;
};

export type MonitoringData = {
  region: string;
  generated_at: string;
  summary: {
    data_drift_detected: boolean;
    model_drift_detected: boolean;
    retraining_needed: boolean;
    retraining_reasons: string[];
    last_trained_days_ago: number;
    current_rmse: number | null;
    current_mape: number | null;
  };
  drifted_features: DriftedFeature[];
  drift_timeline: DriftPoint[];
  total_features_with_drift: number;
  total_features_monitored: number;
};

export type Anomaly = {
  id: string;
  timestamp: string;
  type: 'load_spike' | 'load_drop' | 'solar_drop' | 'solar_surge' | 'wind_ramp' | 'wind_drop' | 'frequency_deviation' | 'battery_fault' | 'sensor_fault';
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'investigating' | 'resolved';
  zone_id: string;
  description: string;
  value?: number;
  threshold?: number;
};

export type AnomalyZScore = {
  timestamp: string;
  target: string;
  z_score: number;
  is_anomaly: boolean;
  residual_mw: number;
};

export type BatterySchedulePoint = {
  timestamp: string;
  soc_percent: number;
  power_mw: number;
  capacity_mwh: number;
  cycles_today: number;
};

export type BatterySchedule = {
  zone_id: string;
  schedule: BatterySchedulePoint[];
  metrics: {
    cost_savings_eur: number;
    carbon_reduction_kg: number;
    peak_shaving_pct: number;
    avg_efficiency: number;
  };
};

export type ParetoPoint = {
  carbon_weight: number;
  total_cost_eur: number;
  total_carbon_kg: number;
  cost_savings_pct: number;
  carbon_reduction_pct: number;
};

export type DashboardManifest = {
  generated_at: string;
  regions: Record<string, { id: string; label: string; stats: DatasetStats; timeseries_hours: number }>;
};

export type RegionDashboardData = {
  domain_id?: DomainId;
  domain_label?: string;
  source_artifacts?: string[];
  runtime_summary?: Record<string, unknown>[];
  stats: DatasetStats | null;
  timeseries: TimeseriesPoint[];
  forecast: Record<string, ForecastPoint[]>;
  dispatch: DispatchPoint[];
  profiles: Record<string, HourlyProfile[]>;
  metrics: ModelMetric[];
  impact: ImpactData | null;
  registry: ModelRegistryEntry[];
  monitoring: MonitoringData | null;
  anomalies: Anomaly[];
  zscores: AnomalyZScore[];
  battery: BatterySchedule | null;
  pareto: ParetoPoint[];
};

// ─── Path resolution ───

function resolveDashboardDir(): string {
  return resolveRepoPath('data', 'dashboard');
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

type CsvRow = Record<string, string>;

function resolveRepoRoot(): string {
  let current = process.cwd();
  for (let depth = 0; depth < 8; depth += 1) {
    if (existsSync(path.join(current, 'reports')) && existsSync(path.join(current, 'data'))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  const cwd = process.cwd();
  return path.basename(cwd) === 'frontend' ? path.resolve(cwd, '..') : cwd;
}

function resolveRepoPath(...parts: string[]): string {
  return path.resolve(resolveRepoRoot(), ...parts);
}

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = '';
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === ',' && !quoted) {
      values.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

async function readCsvRows(filePath: string): Promise<CsvRow[]> {
  if (!existsSync(filePath)) return [];
  const raw = await fs.readFile(filePath, 'utf-8');
  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]).map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row: CsvRow = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? '';
    });
    return row;
  });
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string' || !value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function sampleRows<T>(rows: T[], maxRows = 500): T[] {
  if (rows.length <= maxRows) return rows;
  const step = rows.length / maxRows;
  return Array.from({ length: maxRows }, (_, index) => rows[Math.floor(index * step)]).filter(Boolean);
}

function summarizeColumn(rows: CsvRow[], column: string): TargetSummary {
  const values = rows.map((row) => asNumber(row[column])).filter((value): value is number => value !== null);
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const variance = values.length
    ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
    : 0;
  const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
  return {
    mean,
    std: Math.sqrt(variance),
    min: sorted[0] ?? 0,
    max: sorted[sorted.length - 1] ?? 0,
    median,
    non_zero_pct: values.length ? (values.filter((value) => value !== 0).length / values.length) * 100 : 0,
  };
}

function buildStats(
  region: string,
  label: string,
  rows: CsvRow[],
  targetColumns: string[],
  dateRange: { start: string | null; end: string | null }
): DatasetStats {
  const columns = rows.length ? Object.keys(rows[0]) : targetColumns;
  const targets = Object.fromEntries(targetColumns.map((column) => [column, summarizeColumn(rows, column)]));
  return {
    region,
    label,
    rows: rows.length,
    columns: columns.length,
    column_names: columns,
    date_range: dateRange,
    target_columns: targetColumns,
    weather_features: 0,
    lag_features: 0,
    calendar_features: 0,
    total_features: columns.length,
    targets_summary: targets,
    targets,
    missing_pct: Object.fromEntries(columns.map((column) => [column, 0])),
  };
}

function runtimeMetrics(rows: CsvRow[], featureCount: number): ModelMetric[] {
  return rows.map((row) => ({
    target: 'runtime_tsvr',
    model: row.controller || 'runtime',
    rmse: asNumber(row.tsvr) ?? 0,
    mae: asNumber(row.oasg) ?? 0,
    mape: null,
    smape: null,
    r2: asNumber(row.gdq) ?? undefined,
    coverage_90: (asNumber(row.cva) ?? 0) * 100,
    n_features: featureCount,
  }));
}

function runtimeSummaryByController(rows: CsvRow[], controller: string): CsvRow | null {
  return rows.find((row) => row.controller === controller) ?? rows[0] ?? null;
}

function buildDriftTimeline(rows: CsvRow[], valueColumn: string, reliabilityColumn: string): DriftPoint[] {
  const sampled = sampleRows(rows, 30);
  return sampled.map((row, index) => {
    const reliability = asNumber(row[reliabilityColumn]) ?? 1;
    return {
      date: row.ts_utc || row.timestamp || row.step_index || String(index),
      ks_statistic: Math.max(0, 1 - reliability),
      rolling_rmse: Math.abs(asNumber(row[valueColumn]) ?? 0),
      threshold: 0.15,
      is_drift: reliability < 0.85,
    };
  });
}

function buildAnomalyRows(rows: CsvRow[], domain: 'AV' | 'HEALTHCARE'): Anomaly[] {
  const filtered = rows
    .filter((row) => row.intervened === 'True' || row.fallback_used === 'True' || row.true_constraint_violated === 'True')
    .slice(0, 20);
  return filtered.map((row, index) => ({
    id: `${domain.toLowerCase()}-${row.trace_id || row.patient_id || index}`,
    timestamp: row.ts_utc || row.timestamp || row.step_index || String(index),
    type: domain === 'AV' ? 'sensor_fault' : 'sensor_fault',
    severity: row.true_constraint_violated === 'True' ? 'high' : 'medium',
    status: 'resolved',
    zone_id: domain,
    description: row.intervention_reason || row.fallback_reason || row.fault_family || 'runtime intervention',
    value: asNumber(row.true_margin) ?? asNumber(row.true_spo2_pct) ?? undefined,
    threshold: domain === 'AV' ? 0 : 90,
  }));
}

function buildZScores(rows: CsvRow[], actualColumn: string, forecastColumn: string, target: string): AnomalyZScore[] {
  const sampled = sampleRows(rows, 200);
  const residuals = sampled.map((row) => (asNumber(row[actualColumn]) ?? 0) - (asNumber(row[forecastColumn]) ?? 0));
  const mean = residuals.length ? residuals.reduce((sum, value) => sum + value, 0) / residuals.length : 0;
  const variance = residuals.length ? residuals.reduce((sum, value) => sum + (value - mean) ** 2, 0) / residuals.length : 0;
  const std = Math.sqrt(variance) || 1;
  return sampled.map((row, index) => {
    const residual = residuals[index] ?? 0;
    const z = (residual - mean) / std;
    return {
      timestamp: row.ts_utc || row.timestamp || row.step_index || String(index),
      target,
      z_score: z,
      is_anomaly: Math.abs(z) >= 2.5,
      residual_mw: residual,
    };
  });
}

async function loadAvDomainData(): Promise<RegionDashboardData> {
  const tracePath = resolveRepoPath('reports', 'orius_av', 'nuplan_bounded', 'runtime_traces.csv');
  const summaryPath = resolveRepoPath('reports', 'orius_av', 'nuplan_bounded', 'runtime_summary.csv');
  const [traceRows, summaryRows] = await Promise.all([readCsvRows(tracePath), readCsvRows(summaryPath)]);
  const oriusRows = traceRows.filter((row) => row.controller === 'orius');
  const rows = oriusRows.length ? oriusRows : traceRows;
  const sampled = sampleRows(rows, 500);
  const targetColumns = ['true_margin', 'observed_margin', 'safe_acceleration_mps2', 'reliability_w'];

  return {
    domain_id: 'AV',
    domain_label: 'Autonomous Vehicles',
    source_artifacts: [
      'reports/orius_av/nuplan_bounded/runtime_summary.csv',
      'reports/orius_av/nuplan_bounded/runtime_traces.csv',
    ],
    runtime_summary: summaryRows,
    stats: buildStats('AV', 'Autonomous Vehicles (bounded nuPlan runtime traces)', rows, targetColumns, {
      start: rows[0]?.step_index ?? null,
      end: rows[rows.length - 1]?.step_index ?? null,
    }),
    timeseries: sampled.map((row) => ({
      timestamp: row.step_index || row.trace_id,
      load_mw: asNumber(row.true_margin) ?? undefined,
      wind_mw: asNumber(row.safe_acceleration_mps2) ?? undefined,
      solar_mw: asNumber(row.reliability_w) ?? undefined,
      primary_value: asNumber(row.true_margin) ?? undefined,
      secondary_value: asNumber(row.safe_acceleration_mps2) ?? undefined,
      tertiary_value: asNumber(row.reliability_w) ?? undefined,
      primary_label: 'True safety margin',
      secondary_label: 'Safe acceleration',
      tertiary_label: 'Reliability',
      source_index: row.trace_id,
    })),
    forecast: {
      true_margin: sampled.map((row) => ({
        timestamp: row.step_index || row.trace_id,
        actual: asNumber(row.true_margin) ?? 0,
        forecast: asNumber(row.observed_margin) ?? asNumber(row.true_margin) ?? 0,
      })),
    },
    dispatch: [],
    profiles: {},
    metrics: runtimeMetrics(summaryRows, targetColumns.length),
    impact: null,
    registry: [],
    monitoring: {
      region: 'AV',
      generated_at: new Date().toISOString(),
      summary: {
        data_drift_detected: false,
        model_drift_detected: false,
        retraining_needed: false,
        retraining_reasons: [],
        last_trained_days_ago: 0,
        current_rmse: asNumber(runtimeSummaryByController(summaryRows, 'orius')?.tsvr),
        current_mape: null,
      },
      drifted_features: [],
      drift_timeline: buildDriftTimeline(rows, 'true_margin', 'reliability_w'),
      total_features_with_drift: 0,
      total_features_monitored: targetColumns.length,
    },
    anomalies: buildAnomalyRows(rows, 'AV'),
    zscores: buildZScores(rows, 'true_margin', 'observed_margin', 'true_margin'),
    battery: null,
    pareto: [],
  };
}

async function loadHealthcareDomainData(): Promise<RegionDashboardData> {
  const densePath = resolveRepoPath('data', 'healthcare', 'processed', 'healthcare_bidmc_dense_orius.csv');
  const summaryPath = resolveRepoPath('reports', 'healthcare', 'runtime_summary.csv');
  const [denseRows, summaryRows] = await Promise.all([readCsvRows(densePath), readCsvRows(summaryPath)]);
  const sampled = sampleRows(denseRows, 500);
  const targetColumns = ['target', 'forecast', 'reliability'];

  return {
    domain_id: 'HEALTHCARE',
    domain_label: 'Healthcare Monitoring',
    source_artifacts: [
      'data/healthcare/processed/healthcare_bidmc_dense_orius.csv',
      'reports/healthcare/runtime_summary.csv',
    ],
    runtime_summary: summaryRows,
    stats: buildStats('HEALTHCARE', 'Healthcare Monitoring (BIDMC/MIMIC processed vitals)', denseRows, targetColumns, {
      start: denseRows[0]?.timestamp ?? null,
      end: denseRows[denseRows.length - 1]?.timestamp ?? null,
    }),
    timeseries: sampled.map((row) => ({
      timestamp: row.timestamp,
      load_mw: asNumber(row.target) ?? undefined,
      wind_mw: asNumber(row.forecast) ?? undefined,
      solar_mw: asNumber(row.reliability) ?? undefined,
      primary_value: asNumber(row.target) ?? undefined,
      secondary_value: asNumber(row.forecast) ?? undefined,
      tertiary_value: asNumber(row.reliability) ?? undefined,
      primary_label: 'SpO2 proxy',
      secondary_label: 'Forecast',
      tertiary_label: 'Reliability',
      source_index: row.sample_id,
    })),
    forecast: {
      spo2_proxy: sampled.map((row) => ({
        timestamp: row.timestamp,
        actual: asNumber(row.target) ?? 0,
        forecast: asNumber(row.forecast) ?? asNumber(row.target) ?? 0,
      })),
    },
    dispatch: [],
    profiles: {},
    metrics: runtimeMetrics(summaryRows, targetColumns.length),
    impact: null,
    registry: [],
    monitoring: {
      region: 'HEALTHCARE',
      generated_at: new Date().toISOString(),
      summary: {
        data_drift_detected: false,
        model_drift_detected: false,
        retraining_needed: false,
        retraining_reasons: [],
        last_trained_days_ago: 0,
        current_rmse: asNumber(runtimeSummaryByController(summaryRows, 'orius')?.tsvr),
        current_mape: null,
      },
      drifted_features: [],
      drift_timeline: buildDriftTimeline(denseRows, 'target', 'reliability'),
      total_features_with_drift: 0,
      total_features_monitored: targetColumns.length,
    },
    anomalies: buildAnomalyRows([], 'HEALTHCARE'),
    zscores: buildZScores(denseRows, 'target', 'forecast', 'spo2_proxy'),
    battery: null,
    pareto: [],
  };
}

// ─── Public API ───

export async function loadManifest(): Promise<DashboardManifest | null> {
  const dir = resolveDashboardDir();
  return readJsonFile<DashboardManifest>(path.join(dir, 'manifest.json'));
}

export async function loadRegionData(region: DomainId): Promise<RegionDashboardData> {
  if (!isBatteryDomain(region)) {
    return region === 'AV' ? loadAvDomainData() : loadHealthcareDomainData();
  }

  const dir = resolveDashboardDir();
  const prefix = region.toLowerCase();

  const [stats, timeseries, forecast, dispatch, profiles, metrics, impact, registry, monitoring, anomalies, zscores, battery, pareto] =
    await Promise.all([
      readJsonFile<DatasetStats>(path.join(dir, `${prefix}_stats.json`)),
      readJsonFile<TimeseriesPoint[]>(path.join(dir, `${prefix}_timeseries.json`)),
      readJsonFile<Record<string, ForecastPoint[]>>(path.join(dir, `${prefix}_forecast.json`)),
      readJsonFile<DispatchPoint[]>(path.join(dir, `${prefix}_dispatch.json`)),
      readJsonFile<Record<string, HourlyProfile[]>>(path.join(dir, `${prefix}_profiles.json`)),
      readJsonFile<ModelMetric[]>(path.join(dir, `${prefix}_metrics.json`)),
      readJsonFile<ImpactData>(path.join(dir, `${prefix}_impact.json`)),
      readJsonFile<ModelRegistryEntry[]>(path.join(dir, `${prefix}_registry.json`)),
      readJsonFile<MonitoringData>(path.join(dir, `${prefix}_monitoring.json`)),
      readJsonFile<Anomaly[]>(path.join(dir, `${prefix}_anomalies.json`)),
      readJsonFile<AnomalyZScore[]>(path.join(dir, `${prefix}_zscores.json`)),
      readJsonFile<BatterySchedule>(path.join(dir, `${prefix}_battery.json`)),
      readJsonFile<ParetoPoint[]>(path.join(dir, `${prefix}_pareto.json`)),
    ]);

  return {
    domain_id: region,
    domain_label: region === 'US' ? 'USA (EIA-930)' : 'Germany (OPSD)',
    source_artifacts: [`data/dashboard/${prefix}_*.json`],
    stats: stats ?? null,
    timeseries: timeseries ?? [],
    forecast: forecast ?? {},
    dispatch: dispatch ?? [],
    profiles: profiles ?? {},
    metrics: metrics ?? [],
    impact: impact ?? null,
    registry: registry ?? [],
    monitoring: monitoring ?? null,
    anomalies: anomalies ?? [],
    zscores: zscores ?? [],
    battery: battery ?? null,
    pareto: pareto ?? [],
  };
}

export async function loadAllRegionData(): Promise<Record<string, RegionDashboardData>> {
  const [de, us, av, healthcare] = await Promise.all([
    loadRegionData('DE'),
    loadRegionData('US'),
    loadRegionData('AV'),
    loadRegionData('HEALTHCARE'),
  ]);
  return { DE: de, US: us, AV: av, HEALTHCARE: healthcare };
}
