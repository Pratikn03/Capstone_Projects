import 'server-only';

import { existsSync, type Dirent } from 'fs';
import fs from 'fs/promises';
import path from 'path';

import type { ForecastMetrics } from '@/lib/api/schema';
import type { ImpactSummary, ReportFile, ReportsApiResponse } from '@/lib/api/report-types';

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

export async function loadReportList(reportsDir = resolveReportsDir()): Promise<ReportFile[]> {
  let entries: Dirent[] = [];
  try {
    entries = await fs.readdir(reportsDir, { withFileTypes: true });
  } catch {
    return [];
  }

  const files = entries.filter((entry) => entry.isFile() && !entry.name.startsWith('.'));
  const results = await Promise.all(
    files.map(async (entry) => {
      const filePath = path.join(reportsDir, entry.name);
      const stat = await fs.stat(filePath);
      const ext = path.extname(entry.name).toLowerCase();
      const base = path.basename(entry.name, ext);
      const label = REPORT_LABELS[entry.name];
      return {
        name: entry.name,
        title: label?.title ?? toTitle(base),
        description: label?.description ?? 'Generated report artifact from pipeline runs.',
        type: TYPE_MAP[ext] ?? ext.replace('.', '').toUpperCase(),
        date: new Date(stat.mtime).toISOString().slice(0, 10),
        path: entry.name,
        size_bytes: stat.size,
        _mtimeMs: stat.mtimeMs,
      };
    })
  );

  return results
    .sort((a, b) => b._mtimeMs - a._mtimeMs)
    .map(({ _mtimeMs, ...rest }) => rest);
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

export async function loadReportsOverview(): Promise<ReportsApiResponse> {
  const reportsDir = resolveReportsDir();
  const warnings: string[] = [];

  try {
    const stat = await fs.stat(reportsDir);
    if (!stat.isDirectory()) {
      return {
        reports: [],
        metrics: [],
        impact: null,
        meta: { source: 'missing', warnings: [`Reports path is not a directory: ${reportsDir}`] },
      };
    }
  } catch {
    return {
      reports: [],
      metrics: [],
      impact: null,
      meta: { source: 'missing', warnings: [`Reports directory not found: ${reportsDir}`] },
    };
  }

  let reports: ReportFile[] = [];
  let metrics: ForecastMetrics[] = [];
  let impact: ImpactSummary | null = null;

  try {
    reports = await loadReportList(reportsDir);
  } catch (err) {
    warnings.push(`Failed to read report list: ${String(err)}`);
  }

  try {
    metrics = await loadForecastMetrics(reportsDir);
  } catch (err) {
    warnings.push(`Failed to read forecast metrics: ${String(err)}`);
  }

  try {
    impact = await loadImpactSummary(reportsDir);
  } catch (err) {
    warnings.push(`Failed to read impact summary: ${String(err)}`);
  }

  const lastUpdated = reports.length ? reports[0].date : undefined;

  return {
    reports,
    metrics,
    impact,
    meta: {
      source: 'reports',
      last_updated: lastUpdated,
      warnings: warnings.length ? warnings : undefined,
    },
  };
}
