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
  artifact_warnings?: string[];
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
const DOMAIN_DATA_CACHE = new Map<DomainId, { signature: string; data: RegionDashboardData }>();

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
const CSV_CACHE = new Map<string, { mtimeMs: number; size: number; rows: CsvRow[] }>();

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

function resolveRepoRelPath(relPath: string): string {
  return resolveRepoPath(...relPath.split('/').filter(Boolean));
}

function resolveFirstExistingArtifact(candidates: string[]): {
  fullPath: string;
  relPath: string;
  warnings: string[];
} {
  const missing: string[] = [];
  for (const relPath of candidates) {
    const fullPath = resolveRepoRelPath(relPath);
    if (existsSync(fullPath)) {
      return {
        fullPath,
        relPath,
        warnings: missing.length
          ? [`Using ${relPath}; preferred artifact(s) not found: ${missing.join(', ')}`]
          : [],
      };
    }
    missing.push(relPath);
  }

  const fallback = candidates[0] ?? '';
  return {
    fullPath: resolveRepoRelPath(fallback),
    relPath: fallback,
    warnings: [`No dashboard artifact found. Checked: ${missing.join(', ')}`],
  };
}

async function artifactSignature(filePaths: string[]): Promise<string> {
  const parts = await Promise.all(
    filePaths.map(async (filePath) => {
      try {
        const stat = await fs.stat(filePath);
        return `${filePath}:${stat.mtimeMs}:${stat.size}`;
      } catch {
        return `${filePath}:missing`;
      }
    })
  );
  return parts.join('|');
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
  const stat = await fs.stat(filePath);
  const cached = CSV_CACHE.get(filePath);
  if (cached && cached.mtimeMs === stat.mtimeMs && cached.size === stat.size) {
    return cached.rows;
  }
  const raw = await fs.readFile(filePath, 'utf-8');
  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]).map((header) => header.trim());
  const rows = lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row: CsvRow = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? '';
    });
    return row;
  });
  CSV_CACHE.set(filePath, { mtimeMs: stat.mtimeMs, size: stat.size, rows });
  return rows;
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

function buildRuntimeStats(region: string, label: string, rows: CsvRow[], targetColumns: string[]): DatasetStats {
  const columns = rows.length ? Object.keys(rows[0]) : targetColumns;
  const nSteps = rows
    .map((row) => asNumber(row.n_steps))
    .filter((value): value is number => value !== null);
  return {
    region,
    label,
    rows: nSteps.length ? Math.max(...nSteps) : rows.length,
    columns: columns.length,
    column_names: columns,
    date_range: { start: null, end: null },
    target_columns: targetColumns,
    weather_features: 0,
    lag_features: 0,
    calendar_features: 0,
    total_features: columns.length,
    targets_summary: Object.fromEntries(targetColumns.map((column) => [column, summarizeColumn(rows, column)])),
    targets: Object.fromEntries(targetColumns.map((column) => [column, summarizeColumn(rows, column)])),
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

type ForecastTraceSpec = {
  target: string;
  model: string;
  actualColumn: string;
  forecastColumn: string;
};

function quantile(values: number[], p: number): number | null {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[index] ?? null;
}

function rowTimestamp(row: CsvRow, index: number): string {
  return row.timestamp || row.ts_utc || row.step_index || row.trace_id || row.sample_id || String(index);
}

function buildForecastTrace(rows: CsvRow[], spec: ForecastTraceSpec): ForecastPoint[] {
  const sampled = sampleRows(rows, 500);
  const base = sampled.map((row, index) => {
    const actual = asNumber(row[spec.actualColumn]);
    const forecast = asNumber(row[spec.forecastColumn]) ?? actual;
    if (actual === null || forecast === null) return null;
    return {
      timestamp: rowTimestamp(row, index),
      actual,
      forecast,
    };
  }).filter((row): row is Pick<ForecastPoint, 'timestamp' | 'actual' | 'forecast'> => row !== null);

  const absoluteResiduals = base.map((row) => Math.abs(row.actual - row.forecast));
  const values = base.flatMap((row) => [row.actual, row.forecast]);
  const spread = values.length ? Math.max(...values) - Math.min(...values) : 0;
  const meanMagnitude = values.length ? values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length : 0;
  const fallbackBand = Math.max(spread * 0.05, meanMagnitude * 0.01, 0.01);
  const band90 = quantile(absoluteResiduals, 0.9) || fallbackBand;
  const band50 = quantile(absoluteResiduals, 0.5) || band90 * 0.5;

  return base.map((row) => ({
    ...row,
    lower_90: row.forecast - band90,
    upper_90: row.forecast + band90,
    lower_50: row.forecast - band50,
    upper_50: row.forecast + band50,
  }));
}

function buildForecastMetric(spec: ForecastTraceSpec, series: ForecastPoint[], featureCount: number): ModelMetric | null {
  if (!series.length) return null;
  const residuals = series.map((point) => point.actual - point.forecast);
  const absoluteResiduals = residuals.map(Math.abs);
  const mse = residuals.reduce((sum, residual) => sum + residual ** 2, 0) / residuals.length;
  const mae = absoluteResiduals.reduce((sum, residual) => sum + residual, 0) / absoluteResiduals.length;
  const nonZeroActuals = series.filter((point) => Math.abs(point.actual) > 1e-9);
  const mape = nonZeroActuals.length
    ? (nonZeroActuals.reduce((sum, point) => sum + Math.abs((point.actual - point.forecast) / point.actual), 0) / nonZeroActuals.length) * 100
    : null;
  const smape = series.length
    ? (series.reduce((sum, point) => {
        const denominator = (Math.abs(point.actual) + Math.abs(point.forecast)) / 2;
        return denominator > 1e-9 ? sum + Math.abs(point.actual - point.forecast) / denominator : sum;
      }, 0) / series.length) * 100
    : null;
  const actualMean = series.reduce((sum, point) => sum + point.actual, 0) / series.length;
  const sst = series.reduce((sum, point) => sum + (point.actual - actualMean) ** 2, 0);
  const sse = residuals.reduce((sum, residual) => sum + residual ** 2, 0);
  const covered = series.filter(
    (point) =>
      typeof point.lower_90 === 'number' &&
      typeof point.upper_90 === 'number' &&
      point.actual >= point.lower_90 &&
      point.actual <= point.upper_90
  ).length;

  return {
    target: spec.target,
    model: spec.model,
    rmse: Math.sqrt(mse),
    mae,
    mape,
    smape,
    r2: sst > 0 ? 1 - sse / sst : 1,
    coverage_90: (covered / series.length) * 100,
    n_features: featureCount,
    residual_q10: quantile(residuals, 0.1) ?? undefined,
    residual_q50: quantile(residuals, 0.5) ?? undefined,
    residual_q90: quantile(residuals, 0.9) ?? undefined,
  };
}

function buildForecastMetrics(
  forecast: Record<string, ForecastPoint[]>,
  specs: ForecastTraceSpec[],
  featureCount: number
): ModelMetric[] {
  return specs
    .map((spec) => buildForecastMetric(spec, forecast[spec.target] ?? [], featureCount))
    .filter((metric): metric is ModelMetric => metric !== null);
}

async function loadAvDomainData(): Promise<RegionDashboardData> {
  const traceArtifact = resolveFirstExistingArtifact([
    'reports/orius_av/nuplan_bounded/runtime_traces.csv',
    'reports/orius_av/full_corpus/runtime_traces.csv',
  ]);
  const summaryArtifact = resolveFirstExistingArtifact([
    'reports/orius_av/nuplan_bounded/runtime_summary.csv',
    'reports/orius_av/full_corpus/runtime_summary.csv',
  ]);
  const signature = await artifactSignature([traceArtifact.fullPath, summaryArtifact.fullPath]);
  const cached = DOMAIN_DATA_CACHE.get('AV');
  if (cached?.signature === signature) return cached.data;

  const [traceRows, summaryRows] = await Promise.all([
    readCsvRows(traceArtifact.fullPath),
    readCsvRows(summaryArtifact.fullPath),
  ]);
  const oriusRows = traceRows.filter((row) => row.controller === 'orius');
  const rows = oriusRows.length ? oriusRows : traceRows;
  const sampled = sampleRows(rows, 500);
  const targetColumns = ['true_margin', 'observed_margin', 'safe_acceleration_mps2', 'reliability_w'];
  const warnings = [...traceArtifact.warnings, ...summaryArtifact.warnings];
  const forecastSpecs: ForecastTraceSpec[] = [
    {
      target: 'true_margin',
      model: 'Observed margin trace',
      actualColumn: 'true_margin',
      forecastColumn: 'observed_margin',
    },
    {
      target: 'safe_acceleration_mps2',
      model: 'Candidate acceleration trace',
      actualColumn: 'safe_acceleration_mps2',
      forecastColumn: 'candidate_acceleration_mps2',
    },
    {
      target: 'reliability_w',
      model: 'Runtime reliability trace',
      actualColumn: 'reliability_w',
      forecastColumn: 'reliability_w',
    },
  ];
  const forecast = Object.fromEntries(
    forecastSpecs.map((spec) => [spec.target, buildForecastTrace(rows, spec)])
  );

  const data: RegionDashboardData = {
    domain_id: 'AV',
    domain_label: 'Autonomous Vehicles',
    source_artifacts: [summaryArtifact.relPath, traceArtifact.relPath],
    artifact_warnings: warnings.length ? warnings : undefined,
    runtime_summary: summaryRows,
    stats: rows.length
      ? buildStats('AV', 'Autonomous Vehicles (runtime traces)', rows, targetColumns, {
          start: rows[0]?.step_index ?? null,
          end: rows[rows.length - 1]?.step_index ?? null,
        })
      : buildRuntimeStats('AV', 'Autonomous Vehicles (runtime summary)', summaryRows, ['tsvr', 'oasg', 'cva']),
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
    forecast,
    dispatch: [],
    profiles: {},
    metrics: [
      ...buildForecastMetrics(forecast, forecastSpecs, targetColumns.length),
      ...runtimeMetrics(summaryRows, targetColumns.length),
    ],
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
  DOMAIN_DATA_CACHE.set('AV', { signature, data });
  return data;
}

async function loadHealthcareDomainData(): Promise<RegionDashboardData> {
  const denseArtifact = resolveFirstExistingArtifact([
    'data/healthcare/processed/healthcare_bidmc_dense_orius.csv',
    'data/healthcare/processed/healthcare_max_input_orius.csv',
    'data/healthcare/processed/healthcare_orius.csv',
  ]);
  const summaryArtifact = resolveFirstExistingArtifact([
    'reports/healthcare/runtime_summary.csv',
    'reports/healthcare/runtime_comparator_summary.csv',
  ]);
  const signature = await artifactSignature([denseArtifact.fullPath, summaryArtifact.fullPath]);
  const cached = DOMAIN_DATA_CACHE.get('HEALTHCARE');
  if (cached?.signature === signature) return cached.data;

  const [denseRows, summaryRows] = await Promise.all([
    readCsvRows(denseArtifact.fullPath),
    readCsvRows(summaryArtifact.fullPath),
  ]);
  const sampled = sampleRows(denseRows, 500);
  const targetColumns = ['target', 'forecast', 'reliability'];
  const warnings = [...denseArtifact.warnings, ...summaryArtifact.warnings];
  const forecastSpecs: ForecastTraceSpec[] = [
    {
      target: 'spo2_proxy',
      model: 'SpO2 proxy forecast trace',
      actualColumn: 'target',
      forecastColumn: 'forecast',
    },
    {
      target: 'forecast',
      model: 'SpO2 prediction trace',
      actualColumn: 'target',
      forecastColumn: 'forecast',
    },
    {
      target: 'reliability',
      model: 'Runtime reliability trace',
      actualColumn: 'reliability',
      forecastColumn: 'reliability',
    },
  ];
  const forecast = Object.fromEntries(
    forecastSpecs.map((spec) => [spec.target, buildForecastTrace(denseRows, spec)])
  );

  const data: RegionDashboardData = {
    domain_id: 'HEALTHCARE',
    domain_label: 'Healthcare Monitoring',
    source_artifacts: [denseArtifact.relPath, summaryArtifact.relPath],
    artifact_warnings: warnings.length ? warnings : undefined,
    runtime_summary: summaryRows,
    stats: denseRows.length
      ? buildStats('HEALTHCARE', 'Healthcare Monitoring (BIDMC/MIMIC processed vitals)', denseRows, targetColumns, {
          start: denseRows[0]?.timestamp ?? null,
          end: denseRows[denseRows.length - 1]?.timestamp ?? null,
        })
      : buildRuntimeStats('HEALTHCARE', 'Healthcare Monitoring (runtime summary)', summaryRows, ['tsvr', 'oasg', 'cva']),
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
    forecast,
    dispatch: [],
    profiles: {},
    metrics: [
      ...buildForecastMetrics(forecast, forecastSpecs, targetColumns.length),
      ...runtimeMetrics(summaryRows, targetColumns.length),
    ],
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
  DOMAIN_DATA_CACHE.set('HEALTHCARE', { signature, data });
  return data;
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
