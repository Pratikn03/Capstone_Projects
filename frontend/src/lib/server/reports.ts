import 'server-only';

import { existsSync, type Dirent } from 'fs';
import fs from 'fs/promises';
import path from 'path';

import type { ForecastMetrics } from '@/lib/api/schema';
import type {
  ImpactSummary,
  ReportFile,
  ReportsApiResponse,
  RobustnessSummary,
  RegionReports,
  TrainingStatus,
} from '@/lib/api/report-types';

type MetricsSource = 'week2_metrics' | 'forecast_point_metrics' | 'publication_table' | 'missing';

const MODEL_LABELS: Record<string, string> = {
  gbm: 'GBM (LightGBM)',
  lstm: 'LSTM',
  tcn: 'TCN',
  persistence: 'Persistence',
};

const REPORT_LABELS: Record<string, { title: string; description: string }> = {
  'formal_evaluation_report.md': {
    title: 'Formal Evaluation Report',
    description: 'Comprehensive model evaluation with walk-forward validation.',
  },
  'walk_forward_report.json': {
    title: 'Walk-Forward Backtest',
    description: 'Rolling backtest metrics across targets and horizons.',
  },
  'impact_comparison.md': {
    title: 'Impact Comparison',
    description: 'Cost and carbon impact comparison against baselines.',
  },
  'impact_summary.csv': {
    title: 'Impact Summary',
    description: 'Headline cost, carbon, and peak shaving metrics.',
  },
  'data_quality_report.md': {
    title: 'Data Quality Report',
    description: 'Coverage, missing data, and validation checks.',
  },
  'monitoring_report.md': {
    title: 'Monitoring Report',
    description: 'Drift monitoring and retraining signals.',
  },
  'ml_vs_dl_comparison.md': {
    title: 'ML vs DL Comparison',
    description: 'Model family comparison across forecasting targets.',
  },
  'dispatch_validation.md': {
    title: 'Dispatch Validation',
    description: 'Dispatch optimization validation and constraints.',
  },
  'dispatch_validation.json': {
    title: 'Dispatch Validation (JSON)',
    description: 'Structured dispatch validation results.',
  },
};

const TYPE_MAP: Record<string, string> = {
  '.md': 'Markdown',
  '.json': 'JSON',
  '.csv': 'CSV',
  '.png': 'PNG',
  '.svg': 'SVG',
  '.pdf': 'PDF',
};

const ALLOWED_TARGETS = new Set(['load_mw', 'wind_mw', 'solar_mw']);
const ALLOWED_MODELS = new Set(['gbm', 'lstm', 'tcn']);

function resolveRepoRoot(): string {
  const cwd = process.cwd();
  if (existsSync(path.join(cwd, 'configs'))) {
    return cwd;
  }
  const parent = path.resolve(cwd, '..');
  if (existsSync(path.join(parent, 'configs'))) {
    return parent;
  }
  return cwd;
}

const REPO_ROOT = resolveRepoRoot();

export function resolveReportsDir(): string {
  const configured = process.env.REPORTS_DIR;
  if (configured) {
    return path.resolve(configured);
  }
  const cwd = process.cwd();
  const localReports = path.resolve(cwd, 'reports');
  if (existsSync(localReports)) {
    return localReports;
  }
  return path.resolve(cwd, '..', 'reports');
}

function toNumber(value: string | undefined): number | null {
  if (!value) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function normalizeModelKey(value: string | undefined): string | null {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  return normalized.length ? normalized : null;
}

function parseCsv(raw: string): Array<Record<string, string>> {
  const lines = raw.trim().split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    const row: Record<string, string> = {};
    headers.forEach((header, idx) => {
      row[header] = (values[idx] ?? '').trim();
    });
    return row;
  });
}

function toTitle(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    const stat = await fs.stat(filePath);
    return stat.isFile();
  } catch {
    return false;
  }
}

async function collectReportFiles(
  dir: string,
  root: string,
  ignoreDirs: string[] = []
): Promise<Array<{ fullPath: string; relPath: string }>> {
  const entries: Dirent[] = await fs.readdir(dir, { withFileTypes: true });
  const files: Array<{ fullPath: string; relPath: string }> = [];

  for (const entry of entries) {
    if (entry.name.startsWith('.')) {
      continue;
    }
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (ignoreDirs.includes(entry.name)) {
        continue;
      }
      const nested = await collectReportFiles(fullPath, root, ignoreDirs);
      files.push(...nested);
    } else if (entry.isFile()) {
      const rel = path.relative(root, fullPath).split(path.sep).join('/');
      files.push({ fullPath, relPath: rel });
    }
  }

  return files;
}

export async function loadReportList(
  reportsDir = resolveReportsDir(),
  options?: { rootDir?: string; ignoreDirs?: string[] }
): Promise<ReportFile[]> {
  let entries: Array<{ fullPath: string; relPath: string }> = [];
  try {
    const rootDir = options?.rootDir ?? reportsDir;
    entries = await collectReportFiles(reportsDir, rootDir, options?.ignoreDirs ?? []);
  } catch {
    return [];
  }

  const results = await Promise.all(
    entries.map(async (entry) => {
      const stat = await fs.stat(entry.fullPath);
      const ext = path.extname(entry.relPath).toLowerCase();
      const base = path.basename(entry.relPath, ext);
      const fileName = path.basename(entry.relPath);
      const label = REPORT_LABELS[fileName];
      return {
        name: fileName,
        title: label?.title ?? toTitle(base),
        description: label?.description ?? 'Generated report artifact from pipeline runs.',
        type: TYPE_MAP[ext] ?? ext.replace('.', '').toUpperCase(),
        date: new Date(stat.mtime).toISOString().slice(0, 10),
        path: entry.relPath,
        size_bytes: stat.size,
        _mtimeMs: stat.mtimeMs,
      };
    })
  );

  return results
    .sort((a, b) => b._mtimeMs - a._mtimeMs)
    .map(({ _mtimeMs, ...rest }) => rest);
}

function mergeReportLists(primary: ReportFile[], extra: ReportFile[]): ReportFile[] {
  const map = new Map(primary.map((report) => [report.path, report]));
  for (const report of extra) {
    if (!map.has(report.path)) {
      map.set(report.path, report);
    }
  }
  return Array.from(map.values()).sort((a, b) => {
    const dateDiff = b.date.localeCompare(a.date);
    if (dateDiff !== 0) return dateDiff;
    return a.path.localeCompare(b.path);
  });
}

function latestDate(dates: Array<string | undefined>): string | undefined {
  const valid = dates.filter((date): date is string => Boolean(date));
  if (!valid.length) return undefined;
  return valid.sort().at(-1);
}

export async function loadForecastMetrics(reportsDir = resolveReportsDir()): Promise<ForecastMetrics[]> {
  const metricsPath = path.join(reportsDir, 'metrics', 'forecast_point_metrics.csv');
  if (!(await fileExists(metricsPath))) {
    return [];
  }

  const metricsRaw = await fs.readFile(metricsPath, 'utf-8');
  const rows = parseCsv(metricsRaw);

  const intervalsPath = path.join(reportsDir, 'metrics', 'forecast_intervals.csv');
  const coverageMap = new Map<string, number>();
  if (await fileExists(intervalsPath)) {
    const intervalRaw = await fs.readFile(intervalsPath, 'utf-8');
    const intervalRows = parseCsv(intervalRaw);
    intervalRows.forEach((row) => {
      if (row.alpha !== '0.1') return;
      if (!row.target) return;
      const picp = toNumber(row.picp);
      if (picp !== null) {
        coverageMap.set(row.target, picp * 100);
      }
    });
  }

  const metrics: ForecastMetrics[] = [];
  for (const row of rows) {
    if (!ALLOWED_TARGETS.has(row.target)) continue;
    const modelKey = (row.model || '').toLowerCase();
    if (!ALLOWED_MODELS.has(modelKey)) continue;

    const rmse = toNumber(row.rmse);
    const mae = toNumber(row.mae);
    if (rmse === null || mae === null) continue;

    const rawMape = row.target === 'solar_mw' && row.daylight_mape ? row.daylight_mape : row.mape;
    const mapeValue = toNumber(rawMape);
    const mape = mapeValue !== null ? mapeValue * 100 : undefined;

    metrics.push({
      target: row.target,
      model: MODEL_LABELS[modelKey] ?? row.model,
      rmse,
      mae,
      mape,
      coverage_90: coverageMap.get(row.target),
    });
  }

  const targetOrder = ['load_mw', 'wind_mw', 'solar_mw'];
  const modelOrder = ['GBM (LightGBM)', 'LSTM', 'TCN'];
  metrics.sort((a, b) => {
    const targetDiff = targetOrder.indexOf(a.target) - targetOrder.indexOf(b.target);
    if (targetDiff !== 0) return targetDiff;
    return modelOrder.indexOf(a.model) - modelOrder.indexOf(b.model);
  });

  return metrics;
}

export async function loadTrainingMetrics(reportsDir = resolveReportsDir()): Promise<ForecastMetrics[]> {
  const trainingPath = path.join(reportsDir, 'week2_metrics.json');
  if (!(await fileExists(trainingPath))) {
    return [];
  }

  const raw = await fs.readFile(trainingPath, 'utf-8');
  const parsed = JSON.parse(raw) as { targets?: Record<string, any> };
  const targets = parsed.targets ?? {};

  const intervalsPath = path.join(reportsDir, 'metrics', 'forecast_intervals.csv');
  const coverageMap = new Map<string, number>();
  if (await fileExists(intervalsPath)) {
    const intervalRaw = await fs.readFile(intervalsPath, 'utf-8');
    const intervalRows = parseCsv(intervalRaw);
    intervalRows.forEach((row) => {
      if (row.alpha !== '0.1') return;
      if (!row.target) return;
      const picp = toNumber(row.picp);
      if (picp !== null) {
        coverageMap.set(row.target, picp * 100);
      }
    });
  }

  const metrics: ForecastMetrics[] = [];
  for (const target of Object.keys(targets)) {
    if (!ALLOWED_TARGETS.has(target)) continue;
    for (const modelKey of ['gbm', 'lstm', 'tcn']) {
      const model = targets[target]?.[modelKey];
      if (!model) continue;
      const rmse = toNumber(model.rmse);
      const mae = toNumber(model.mae);
      if (rmse === null || mae === null) continue;
      const rawMape =
        target === 'solar_mw' && model.daylight_mape !== undefined ? model.daylight_mape : model.mape;
      const mapeValue = toNumber(rawMape);
      metrics.push({
        target,
        model: MODEL_LABELS[modelKey] ?? modelKey,
        rmse,
        mae,
        mape: mapeValue !== null ? mapeValue * 100 : undefined,
        coverage_90: coverageMap.get(target),
      });
    }
  }

  const targetOrder = ['load_mw', 'wind_mw', 'solar_mw'];
  const modelOrder = ['GBM (LightGBM)', 'LSTM', 'TCN'];
  metrics.sort((a, b) => {
    const targetDiff = targetOrder.indexOf(a.target) - targetOrder.indexOf(b.target);
    if (targetDiff !== 0) return targetDiff;
    return modelOrder.indexOf(a.model) - modelOrder.indexOf(b.model);
  });

  return metrics;
}

export async function loadPublicationMetrics(metricsPath: string): Promise<ForecastMetrics[]> {
  if (!(await fileExists(metricsPath))) {
    return [];
  }

  const raw = await fs.readFile(metricsPath, 'utf-8');
  const rows = parseCsv(raw);
  const metrics: ForecastMetrics[] = [];

  for (const row of rows) {
    if (!ALLOWED_TARGETS.has(row.target)) continue;
    const modelKey = normalizeModelKey(row.model);
    if (!modelKey) continue;
    const rmse = toNumber(row.rmse);
    const mae = toNumber(row.mae);
    if (rmse === null || mae === null) continue;
    const rawMape =
      row.target === 'solar_mw' && row.daylight_mape ? row.daylight_mape : row.mape;
    const mapeValue = toNumber(rawMape);
    metrics.push({
      target: row.target,
      model: MODEL_LABELS[modelKey] ?? row.model,
      rmse,
      mae,
      mape: mapeValue !== null ? mapeValue * 100 : undefined,
    });
  }

  const targetOrder = ['load_mw', 'wind_mw', 'solar_mw'];
  const modelOrder = ['GBM (LightGBM)', 'LSTM', 'TCN', 'Persistence'];
  metrics.sort((a, b) => {
    const targetDiff = targetOrder.indexOf(a.target) - targetOrder.indexOf(b.target);
    if (targetDiff !== 0) return targetDiff;
    const modelDiff = modelOrder.indexOf(a.model) - modelOrder.indexOf(b.model);
    if (modelDiff !== 0) return modelDiff;
    return a.model.localeCompare(b.model);
  });

  return metrics;
}

function parseTargetsFromConfig(raw: string): string[] {
  const match = raw.match(/targets:\s*\[([^\]]+)\]/);
  if (!match) return [];
  return match[1]
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);
}

function parseConfigValue(raw: string, key: string): string | null {
  const match = raw.match(new RegExp(`${key}:\\s*([^\\n#]+)`));
  return match ? match[1].trim() : null;
}

function parseSectionOutDir(raw: string, section: string): string | null {
  const match = raw.match(new RegExp(`${section}:([\\s\\S]*?)\\n\\S`, 'm'));
  if (!match) return null;
  const outMatch = match[1].match(/out_dir:\s*([^\\n#]+)/);
  return outMatch ? outMatch[1].trim() : null;
}

async function loadTrainingStatus(
  configPath: string,
  reportsDir: string,
  trainingMetrics: ForecastMetrics[]
): Promise<TrainingStatus> {
  const configFullPath = path.isAbsolute(configPath)
    ? configPath
    : path.resolve(REPO_ROOT, configPath);
  let raw = '';
  try {
    raw = await fs.readFile(configFullPath, 'utf-8');
  } catch {
    return {
      features_path: null,
      features_exists: false,
      targets_expected: [],
      targets_trained: Array.from(new Set(trainingMetrics.map((m) => m.target))),
      targets_missing: [],
      models_dir: null,
      missing_models: [],
    };
  }

  const targetsExpected = parseTargetsFromConfig(raw);
  const featuresPathRaw = parseConfigValue(raw, 'processed_path');
  const modelsDirRaw = parseSectionOutDir(raw, 'artifacts');

  const featuresPath = featuresPathRaw
    ? path.resolve(REPO_ROOT, featuresPathRaw)
    : null;
  const modelsDir = modelsDirRaw
    ? path.resolve(REPO_ROOT, modelsDirRaw)
    : null;

  let targetsTrained = Array.from(new Set(trainingMetrics.map((m) => m.target)));
  const trainingPath = path.join(reportsDir, 'week2_metrics.json');
  if (await fileExists(trainingPath)) {
    try {
      const rawMetrics = JSON.parse(await fs.readFile(trainingPath, 'utf-8')) as { targets?: Record<string, unknown> };
      targetsTrained = Object.keys(rawMetrics.targets ?? {});
    } catch {
      // Keep fallback targets from trainingMetrics
    }
  }
  const targetsMissing = targetsExpected.filter((t) => !targetsTrained.includes(t));

  let modelFiles: string[] = [];
  if (modelsDir && existsSync(modelsDir)) {
    try {
      modelFiles = await fs.readdir(modelsDir);
    } catch {
      modelFiles = [];
    }
  }

  const missingModels = targetsExpected.map((target) => {
    const missing: string[] = [];
    const gbm = modelFiles.some((name) => name.startsWith('gbm_') && name.includes(`_${target}`) && name.endsWith('.pkl'));
    const lstm = modelFiles.some((name) => name.startsWith('lstm_') && name.includes(`_${target}`) && name.endsWith('.pt'));
    const tcn = modelFiles.some((name) => name.startsWith('tcn_') && name.includes(`_${target}`) && name.endsWith('.pt'));
    if (!gbm) missing.push('gbm');
    if (!lstm) missing.push('lstm');
    if (!tcn) missing.push('tcn');
    return { target, missing };
  }).filter((entry) => entry.missing.length > 0);

  return {
    features_path: featuresPathRaw ?? null,
    features_exists: featuresPath ? existsSync(featuresPath) : false,
    targets_expected: targetsExpected,
    targets_trained: targetsTrained,
    targets_missing: targetsMissing,
    models_dir: modelsDirRaw ?? null,
    missing_models: missingModels,
  };
}

async function loadMetricsBundle(reportsDir: string): Promise<{
  metrics: ForecastMetrics[];
  metricsBacktest: ForecastMetrics[];
  metricsSource: MetricsSource;
}> {
  const metricsBacktest = await loadForecastMetrics(reportsDir);
  const training = await loadTrainingMetrics(reportsDir);
  if (training.length) {
    return { metrics: training, metricsBacktest, metricsSource: 'week2_metrics' };
  }
  if (metricsBacktest.length) {
    return { metrics: metricsBacktest, metricsBacktest, metricsSource: 'forecast_point_metrics' };
  }
  return { metrics: [], metricsBacktest, metricsSource: 'missing' };
}

async function loadRegionBundle(params: {
  id: string;
  label: string;
  reportsDir: string;
  rootDir: string;
  configPath: string;
  ignoreDirs?: string[];
}): Promise<RegionReports> {
  const warnings: string[] = [];
  let reports: ReportFile[] = [];
  let metrics: ForecastMetrics[] = [];
  let metricsBacktest: ForecastMetrics[] = [];
  let metricsSource: MetricsSource = 'missing';
  let impact: ImpactSummary | null = null;
  let robustness: RobustnessSummary | null = null;
  let trainingStatus: TrainingStatus | null = null;
  let trainingMetrics: ForecastMetrics[] = [];

  try {
    reports = await loadReportList(params.reportsDir, { rootDir: params.rootDir, ignoreDirs: params.ignoreDirs });
  } catch (err) {
    warnings.push(`Failed to read report list: ${String(err)}`);
  }

  try {
    const bundle = await loadMetricsBundle(params.reportsDir);
    metrics = bundle.metrics;
    trainingMetrics = bundle.metrics;
    metricsBacktest = bundle.metricsBacktest;
    metricsSource = bundle.metricsSource;
  } catch (err) {
    warnings.push(`Failed to read metrics: ${String(err)}`);
  }

  try {
    impact = await loadImpactSummary(params.reportsDir);
  } catch (err) {
    warnings.push(`Failed to read impact summary: ${String(err)}`);
  }

  try {
    robustness = await loadRobustnessSummary(params.reportsDir);
  } catch (err) {
    warnings.push(`Failed to read robustness summary: ${String(err)}`);
  }

  try {
    trainingStatus = await loadTrainingStatus(params.configPath, params.reportsDir, trainingMetrics);
  } catch (err) {
    warnings.push(`Failed to read training status: ${String(err)}`);
  }

  const lastUpdated = reports.length ? reports[0].date : undefined;
  const source = reports.length ? 'reports' : 'missing';

  return {
    id: params.id,
    label: params.label,
    reports,
    metrics,
    metrics_backtest: metricsBacktest.length ? metricsBacktest : undefined,
    impact,
    robustness,
    training_status: trainingStatus,
    meta: {
      source,
      last_updated: lastUpdated,
      metrics_source: metricsSource,
      warnings: warnings.length ? warnings : undefined,
    },
  };
}

export async function loadImpactSummary(reportsDir = resolveReportsDir()): Promise<ImpactSummary | null> {
  const summaryPath = path.join(reportsDir, 'impact_summary.csv');
  if (!(await fileExists(summaryPath))) {
    return null;
  }

  const raw = await fs.readFile(summaryPath, 'utf-8');
  const rows = parseCsv(raw);
  if (!rows.length) return null;
  const row = rows[0];

  const baselineCost = toNumber(row.baseline_cost_usd);
  const gridCost = toNumber(row.gridpulse_cost_usd);
  const baselineCarbon = toNumber(row.baseline_carbon_kg);
  const gridCarbon = toNumber(row.gridpulse_carbon_kg);
  const baselinePeak = toNumber(row.baseline_peak_mw);
  const gridPeak = toNumber(row.gridpulse_peak_mw);

  return {
    cost_savings_pct: toNumber(row.cost_savings_pct),
    carbon_reduction_pct: toNumber(row.carbon_reduction_pct),
    peak_shaving_pct: toNumber(row.peak_shaving_pct),
    cost_savings_usd:
      baselineCost !== null && gridCost !== null ? Math.max(0, baselineCost - gridCost) : null,
    carbon_reduction_kg:
      baselineCarbon !== null && gridCarbon !== null ? Math.max(0, baselineCarbon - gridCarbon) : null,
    peak_shaving_mw:
      baselinePeak !== null && gridPeak !== null ? Math.max(0, baselinePeak - gridPeak) : null,
  };
}

export async function loadRobustnessSummary(reportsDir = resolveReportsDir()): Promise<RobustnessSummary | null> {
  const summaryPath = path.join(reportsDir, 'metrics', 'robustness_summary.csv');
  if (!(await fileExists(summaryPath))) {
    return null;
  }

  const raw = await fs.readFile(summaryPath, 'utf-8');
  const rows = parseCsv(raw);
  if (!rows.length) return null;

  const best = rows.reduce((acc, row) => {
    const accPct = toNumber(acc.perturbation_pct) ?? 0;
    const rowPct = toNumber(row.perturbation_pct) ?? 0;
    return rowPct >= accPct ? row : acc;
  }, rows[0]);

  return {
    perturbation_pct: toNumber(best.perturbation_pct) ?? 0,
    infeasible_rate: toNumber(best.infeasible_rate),
    mean_regret: toNumber(best.mean_regret),
    p95_regret: toNumber(best.p95_regret),
  };
}

export async function loadReportsOverview(): Promise<ReportsApiResponse> {
  const reportsDir = resolveReportsDir();
  const warnings: string[] = [];

  try {
    const stat = await fs.stat(reportsDir);
    if (!stat.isDirectory()) {
      return {
        reports: [],
        metrics: [],
        metrics_backtest: [],
        impact: null,
        robustness: null,
        regions: {},
        meta: {
          source: 'missing',
          metrics_source: 'missing',
          warnings: [`Reports path is not a directory: ${reportsDir}`],
        },
      };
    }
  } catch {
    return {
      reports: [],
      metrics: [],
      metrics_backtest: [],
      impact: null,
      robustness: null,
      regions: {},
      meta: {
        source: 'missing',
        metrics_source: 'missing',
        warnings: [`Reports directory not found: ${reportsDir}`],
      },
    };
  }

  let reports: ReportFile[] = [];

  try {
    reports = await loadReportList(reportsDir, { rootDir: reportsDir });
  } catch (err) {
    warnings.push(`Failed to read report list: ${String(err)}`);
  }

  const deBundle = await loadRegionBundle({
    id: 'DE',
    label: 'Germany (OPSD)',
    reportsDir,
    rootDir: reportsDir,
    configPath: path.join(REPO_ROOT, 'configs', 'train_forecast.yaml'),
    ignoreDirs: ['eia930'],
  });

  const usReportsDir = path.join(reportsDir, 'eia930');
  const usBundle = await loadRegionBundle({
    id: 'US',
    label: 'USA (EIA-930)',
    reportsDir: usReportsDir,
    rootDir: reportsDir,
    configPath: path.join(REPO_ROOT, 'configs', 'train_forecast_eia930.yaml'),
  });

  const publicationTablesDir = path.join(reportsDir, 'publication', 'tables');
  const publicationReports = await loadReportList(publicationTablesDir, { rootDir: reportsDir });
  const usPublicationReports = publicationReports.filter((report) =>
    report.path.toLowerCase().includes('_us')
  );
  if (usPublicationReports.length) {
    usBundle.reports = mergeReportLists(usBundle.reports, usPublicationReports);
    usBundle.meta.last_updated = latestDate([
      usBundle.meta.last_updated,
      ...usPublicationReports.map((report) => report.date),
    ]) ?? usBundle.meta.last_updated;
  }

  const usPublicationMetricsPath = path.join(publicationTablesDir, 'table3_forecast_metrics_us.csv');
  const usPublicationMetrics = await loadPublicationMetrics(usPublicationMetricsPath);
  if (usPublicationMetrics.length) {
    usBundle.metrics = usPublicationMetrics;
    usBundle.meta.metrics_source = 'publication_table';
  }

  const regions: Record<string, RegionReports> = {
    DE: deBundle,
    US: usBundle,
  };

  const primary = deBundle.metrics.length ? deBundle : usBundle.metrics.length ? usBundle : deBundle;
  const lastUpdated = reports.length ? reports[0].date : primary.meta.last_updated;

  return {
    reports,
    metrics: primary.metrics,
    metrics_backtest: primary.metrics_backtest,
    impact: primary.impact,
    robustness: primary.robustness,
    regions,
    meta: {
      source: 'reports',
      last_updated: lastUpdated,
      metrics_source: primary.meta.metrics_source,
      warnings: warnings.length ? warnings : undefined,
    },
  };
}
