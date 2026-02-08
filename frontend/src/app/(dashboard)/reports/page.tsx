'use client';

import { useEffect, useState } from 'react';
import { Panel } from '@/components/ui/Panel';
import { KPICard } from '@/components/ui/KPICard';
import { FileText, Download, TrendingUp, BarChart3, Leaf, Zap } from 'lucide-react';
import { useReportsData } from '@/lib/api/reports-client';
import { useRegion } from '@/components/ui/RegionContext';

const fallbackReportsList = [
  {
    title: 'Formal Evaluation Report',
    description: 'Comprehensive ML model evaluation with walk-forward validation and conformal PI coverage.',
    date: '2026-02-07',
    type: 'PDF',
    path: '',
  },
  {
    title: 'Walk-Forward Backtest',
    description: 'Rolling 7-day backtest across all targets (load, wind, solar) with 50-epoch models.',
    date: '2026-02-07',
    type: 'JSON',
    path: '',
  },
  {
    title: 'Dispatch Validation',
    description: 'Cost-carbon trade-off analysis with Pareto frontier and battery optimization results.',
    date: '2026-02-06',
    type: 'Markdown',
    path: '',
  },
  {
    title: 'Data Quality Report',
    description: 'Missing data analysis, outlier detection, and feature completeness for OPSD + EIA-930.',
    date: '2026-02-05',
    type: 'Markdown',
    path: '',
  },
  {
    title: 'Model Cards',
    description: 'Per-target model cards with architecture, hyperparameters, and fairness/bias analysis.',
    date: '2026-02-07',
    type: 'Markdown',
    path: '',
  },
  {
    title: 'Monitoring & Drift Report',
    description: 'KS-statistic drift monitoring, rolling RMSE degradation, and retraining triggers.',
    date: '2026-02-07',
    type: 'PDF',
    path: '',
  },
];

export default function ReportsPage() {
  const { metrics, metricsBacktest, metricsSource, impact, reports, regions, meta } = useReportsData();
  const { region } = useRegion();
  const [dataset, setDataset] = useState<'ALL' | 'DE' | 'US'>(region);

  useEffect(() => {
    setDataset(region);
  }, [region]);
  const regionData = dataset === 'ALL' ? null : regions[dataset];
  const list =
    regionData?.reports?.length
      ? regionData.reports
      : dataset === 'ALL'
      ? reports.length
        ? reports
        : fallbackReportsList
      : [];
  const displayPath = (path: string | undefined) => {
    if (!path) return '';
    if (dataset === 'US' && path.startsWith('eia930/')) {
      return path.slice('eia930/'.length);
    }
    return path;
  };
  const grouped = list.reduce<Record<string, typeof list>>((acc, report) => {
    const path = displayPath(report.path);
    const parts = path.split('/').filter(Boolean);
    const group = parts.length > 1 ? parts[0] : 'root';
    if (!acc[group]) acc[group] = [];
    acc[group].push(report);
    return acc;
  }, {});
  const groupOrder = Object.keys(grouped).sort((a, b) => {
    if (a === 'root') return -1;
    if (b === 'root') return 1;
    return a.localeCompare(b);
  });
  const metricsActive = regionData?.metrics?.length ? regionData.metrics : metrics;
  const backtestActive = regionData?.metrics_backtest ?? metricsBacktest;
  const metricsSourceActive = regionData?.meta?.metrics_source ?? metricsSource;
  const impactActive = regionData?.impact ?? impact;
  const loadMetrics = metricsActive.filter((m) => m.target === 'load_mw');
  const bestLoad = loadMetrics.length
    ? loadMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b))
    : null;
  const bestR2 = loadMetrics.map((m) => m.r2).filter((v): v is number => v !== undefined);
  const coverage90 = loadMetrics.map((m) => m.coverage_90).find((v) => v !== undefined);
  const carbonReductionPct = impactActive?.carbon_reduction_pct ?? 32.6;
  const carbonTons =
    impactActive?.carbon_reduction_kg !== null && impactActive?.carbon_reduction_kg !== undefined
      ? impactActive.carbon_reduction_kg / 1000
      : 47.8;
  const activeMeta = regionData?.meta ?? meta;
  const isDemo = activeMeta.source !== 'reports' || !list.length;
  const sourceLabel = isDemo ? 'demo' : dataset === 'US' ? 'reports/eia930' : 'reports/';
  const lastUpdated = !isDemo && activeMeta.last_updated ? ` • Updated ${activeMeta.last_updated}` : '';
  const metricsLabel =
    metricsSourceActive === 'week2_metrics'
      ? 'Training metrics (week2_metrics.json)'
      : metricsSourceActive === 'forecast_point_metrics'
      ? 'Backtest metrics (forecast_point_metrics.csv)'
      : 'Demo metrics';
  const formatMaybe = (value: number | undefined, digits: number) => (value === undefined ? 'N/A' : value.toFixed(digits));
  const trainingRows = dataset === 'ALL'
    ? Object.values(regions)
    : regionData
    ? [regionData]
    : [];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Reports & Documentation</h1>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/5 border border-white/10 text-xs">
            {[
              { id: 'ALL', label: 'All' },
              { id: 'DE', label: 'Germany' },
              { id: 'US', label: 'USA' },
            ].map((option) => (
              <button
                key={option.id}
                onClick={() => setDataset(option.id as 'ALL' | 'DE' | 'US')}
                className={`px-2.5 py-1 rounded-md transition-all ${
                  dataset === option.id
                    ? 'bg-energy-info/20 text-energy-info'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <FileText className="w-4 h-4" />
            <span>
              {list.length} reports • Source: {sourceLabel}
              {lastUpdated}
            </span>
          </div>
        </div>
      </div>

      {/* Summary KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard
          label="Best RMSE"
          value={bestLoad ? bestLoad.rmse.toFixed(0) : 'N/A'}
          unit={bestLoad ? 'MW' : undefined}
          icon={<BarChart3 className="w-4 h-4 text-energy-info" />}
          color="info"
        />
        <KPICard
          label="Best R²"
          value={bestR2.length ? Math.max(...bestR2).toFixed(3) : 'N/A'}
          icon={<TrendingUp className="w-4 h-4 text-energy-primary" />}
          color="primary"
        />
        <KPICard
          label="90% PI Coverage"
          value={coverage90 !== undefined ? coverage90.toFixed(1) : 'N/A'}
          unit={coverage90 !== undefined ? '%' : undefined}
          icon={<Zap className="w-4 h-4 text-energy-warn" />}
          color="warn"
        />
        <KPICard
          label="Carbon Saved"
          value={carbonTons.toFixed(0)}
          unit="tCO₂"
          change={carbonReductionPct}
          icon={<Leaf className="w-4 h-4 text-energy-primary" />}
          color="primary"
        />
      </div>

      {/* Reports List */}
      <Panel
        title="Generated Reports"
        subtitle="Auto-generated from pipeline runs"
        badge={`${list.length} reports`}
        badgeColor="info"
      >
        <div className="space-y-4">
          {groupOrder.map((group) => (
            <div key={group} className="space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 px-2">
                {group === 'root' ? 'root' : group}
              </div>
              {grouped[group].map((report) => {
                const href = report.path ? `/api/reports/file?path=${encodeURIComponent(report.path)}` : undefined;
                const shownPath = displayPath(report.path);
                const row = (
                  <>
                    <div className="w-10 h-10 rounded-lg bg-energy-info/10 border border-energy-info/20 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-5 h-5 text-energy-info" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white group-hover:text-energy-primary transition-colors">
                        {report.title}
                      </div>
                      <div className="text-xs text-slate-500 truncate">{report.description}</div>
                      {shownPath && shownPath.includes('/') && (
                        <div className="text-[10px] text-slate-600 font-mono">{shownPath}</div>
                      )}
                    </div>

                    <div className="flex items-center gap-3 flex-shrink-0">
                      <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-white/5 text-slate-400 border border-white/10">
                        {report.type}
                      </span>
                      <span className="text-xs text-slate-500 font-mono">{report.date}</span>
                      <Download className="w-4 h-4 text-slate-600 group-hover:text-energy-primary transition-colors" />
                    </div>
                  </>
                );

                return href ? (
                  <a
                    key={report.path || report.title}
                    href={href}
                    className="flex items-center gap-4 px-4 py-3 rounded-lg bg-white/3 hover:bg-white/5 transition-colors cursor-pointer group"
                  >
                    {row}
                  </a>
                ) : (
                  <div
                    key={report.path || report.title}
                    className="flex items-center gap-4 px-4 py-3 rounded-lg bg-white/3 hover:bg-white/5 transition-colors group"
                  >
                    {row}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </Panel>

      {/* Model Performance Summary */}
      <Panel title="Model Performance Summary" subtitle={metricsLabel}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Target</th>
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Model</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">RMSE</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">MAE</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">R²</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">90% PICP</th>
              </tr>
            </thead>
            <tbody>
              {metricsActive.map((m, i) => (
                <tr key={i} className="border-b border-white/5 hover:bg-white/3 transition-colors">
                  <td className="py-2.5 px-3 text-white">
                    {m.target === 'load_mw' ? '⚡ Load' : m.target === 'wind_mw' ? '💨 Wind' : '☀️ Solar'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">{m.model}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-white">{m.rmse.toFixed(1)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-300">{m.mae.toFixed(1)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-energy-primary">{formatMaybe(m.r2, 3)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-energy-info">
                    {m.coverage_90 === undefined ? 'N/A' : `${m.coverage_90.toFixed(1)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {backtestActive.length > 0 && (
        <Panel title="Backtest Metrics" subtitle="forecast_point_metrics.csv">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 px-3 text-slate-500 font-medium">Target</th>
                  <th className="text-left py-2 px-3 text-slate-500 font-medium">Model</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">RMSE</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">MAE</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">MAPE</th>
                  <th className="text-right py-2 px-3 text-slate-500 font-medium">90% PICP</th>
                </tr>
              </thead>
              <tbody>
                {backtestActive.map((m) => (
                  <tr key={`${m.target}-${m.model}`} className="border-b border-white/5 hover:bg-white/3 transition-colors">
                    <td className="py-2.5 px-3 text-white">
                      {m.target === 'load_mw' ? '⚡ Load' : m.target === 'wind_mw' ? '💨 Wind' : '☀️ Solar'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">{m.model}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-white">{m.rmse.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{m.mae.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{formatMaybe(m.mape, 1)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-energy-info">
                      {m.coverage_90 === undefined ? 'N/A' : `${m.coverage_90.toFixed(1)}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {trainingRows.length > 0 && (
        <Panel title="Training Coverage" subtitle="Expected targets vs available artifacts">
          <div className="space-y-3">
            {trainingRows.map((region) => {
              const status = region.training_status;
              const missingTargets = status?.targets_missing ?? [];
              const missingModels = status?.missing_models ?? [];
              return (
                <div key={region.id} className="rounded-lg bg-white/3 p-3">
                  <div className="flex items-center justify-between text-xs mb-2">
                    <span className="text-slate-300 font-medium">{region.label}</span>
                    <span className="text-slate-500">
                      {status?.features_exists ? 'Features found' : 'Features missing'}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Targets expected: <span className="text-slate-300">{status?.targets_expected?.join(', ') || 'N/A'}</span>
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Targets trained: <span className="text-slate-300">{status?.targets_trained?.join(', ') || 'N/A'}</span>
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Missing targets: <span className="text-slate-300">{missingTargets.length ? missingTargets.join(', ') : 'None'}</span>
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Missing model artifacts: <span className="text-slate-300">{missingModels.length ? missingModels.map((m) => `${m.target} (${m.missing.join(', ')})`).join('; ') : 'None'}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </Panel>
      )}

      {/* Footer */}
      <div className="text-center text-[11px] text-slate-600 py-4">
        Reports auto-generated by GridPulse ML Pipeline • Datasets: OPSD Germany, EIA-930 USA • 50 epochs, CosineAnnealingLR
      </div>
    </div>
  );
}
