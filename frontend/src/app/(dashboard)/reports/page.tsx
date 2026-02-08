'use client';

import { Panel } from '@/components/ui/Panel';
import { KPICard } from '@/components/ui/KPICard';
import { FileText, Download, TrendingUp, BarChart3, Leaf, Zap } from 'lucide-react';
import { useReportsData } from '@/lib/api/reports-client';

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
  const { metrics, impact, reports, meta } = useReportsData();
  const list = reports.length ? reports : fallbackReportsList;
  const loadMetrics = metrics.filter((m) => m.target === 'load_mw');
  const bestLoad = loadMetrics.length
    ? loadMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b))
    : null;
  const bestR2 = loadMetrics.map((m) => m.r2).filter((v): v is number => v !== undefined);
  const coverage90 = loadMetrics.map((m) => m.coverage_90).find((v) => v !== undefined);
  const carbonReductionPct = impact?.carbon_reduction_pct ?? 32.6;
  const carbonTons =
    impact?.carbon_reduction_kg !== null && impact?.carbon_reduction_kg !== undefined
      ? impact.carbon_reduction_kg / 1000
      : 47.8;
  const isDemo = meta.source !== 'reports' || !reports.length;
  const sourceLabel = isDemo ? 'demo' : 'reports/';
  const lastUpdated = !isDemo && meta.last_updated ? ` • Updated ${meta.last_updated}` : '';
  const formatMaybe = (value: number | undefined, digits: number) => (value === undefined ? 'N/A' : value.toFixed(digits));

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Reports & Documentation</h1>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <FileText className="w-4 h-4" />
          <span>
            {list.length} reports generated • Source: {sourceLabel}
            {lastUpdated}
          </span>
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
        <div className="space-y-2">
          {list.map((report, i) => {
            const href = report.path ? `/api/reports/file?path=${encodeURIComponent(report.path)}` : undefined;
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
                key={i}
                href={href}
                className="flex items-center gap-4 px-4 py-3 rounded-lg bg-white/3 hover:bg-white/5 transition-colors cursor-pointer group"
              >
                {row}
              </a>
            ) : (
              <div
                key={i}
                className="flex items-center gap-4 px-4 py-3 rounded-lg bg-white/3 hover:bg-white/5 transition-colors group"
              >
                {row}
              </div>
            );
          })}
        </div>
      </Panel>

      {/* Model Performance Summary */}
      <Panel title="Model Performance Summary" subtitle="All targets × all models">
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
              {metrics.map((m, i) => (
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

      {/* Footer */}
      <div className="text-center text-[11px] text-slate-600 py-4">
        Reports auto-generated by GridPulse ML Pipeline • Datasets: OPSD Germany, EIA-930 USA • 50 epochs, CosineAnnealingLR
      </div>
    </div>
  );
}
