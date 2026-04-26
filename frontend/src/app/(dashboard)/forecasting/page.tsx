'use client';

import { useEffect, useState } from 'react';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { Panel } from '@/components/ui/Panel';
import { StatusBanner } from '@/components/ui/StatusBanner';
import { useRegion } from '@/components/ui/RegionContext';
import { useReportsData } from '@/lib/api/reports-client';
import { useDatasetData } from '@/lib/api/dataset-client';
import Link from 'next/link';
import { ShieldCheck, ChevronRight } from 'lucide-react';
import { DOMAIN_OPTIONS, getDomainOption, isBatteryDomain, targetOptionsForDomain } from '@/lib/domain-options';

export default function ForecastingPage() {
  const [selectedTarget, setSelectedTarget] = useState<string>('load_mw');
  const { region, setRegion } = useRegion();
  const currentDomain = getDomainOption(region);
  const targetOptions = targetOptionsForDomain(region);
  const batteryDomain = isBatteryDomain(region);
  const { metrics, regions } = useReportsData();
  const dataset = useDatasetData(region);

  useEffect(() => {
    if (!targetOptions.some((target) => target.id === selectedTarget)) {
      setSelectedTarget(targetOptions[0]?.id ?? currentDomain.primaryTarget);
    }
  }, [currentDomain.primaryTarget, selectedTarget, targetOptions]);

  const metricsActive = batteryDomain && (region === 'DE' || region === 'US') && regions[region]?.metrics?.length ? regions[region].metrics : metrics;
  const realMetrics = dataset.metrics;

  // Real forecast data from extracted parquet
  const forecastData = dataset.forecast?.[selectedTarget] ?? [];

  const formatMaybe = (value: number | undefined | null, digits: number) =>
    value === undefined || value === null ? 'N/A' : value.toFixed(digits);

  const selectedMetrics = metricsActive.filter((m) => m.target === selectedTarget);
  const bestMetric = selectedMetrics.length
    ? selectedMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b))
    : null;

  // All models for the selected target (prefer real metrics)
  const realTargetMetrics = realMetrics.filter((m) => m.target === selectedTarget);
  const displayMetrics: Array<{ target: string; model: string; rmse: number; mae: number; mape?: number | null; r2?: number; coverage_90?: number }> =
    realTargetMetrics.length
      ? realTargetMetrics
      : metricsActive.filter((m) => m.target === selectedTarget);
  const statusMessages = [
    dataset.error ? `Dataset view error: ${dataset.error}` : null,
    !forecastData.length ? 'No extracted forecast trace is available for the selected region/target.' : null,
    !realTargetMetrics.length && displayMetrics.length ? 'Model table is using report-level metrics rather than dataset-level extracted metrics.' : null,
    !displayMetrics.length ? 'No model metrics are available for the selected target.' : null,
  ].filter((message): message is string => Boolean(message));

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Probabilistic Forecasting</h1>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/5 border border-white/10">
            {targetOptions.map((t) => (
              <button
                key={t.id}
                onClick={() => setSelectedTarget(t.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  selectedTarget === t.id
                    ? 'bg-energy-primary/20 text-energy-primary'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {t.icon} {t.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/5 border border-white/10">
            {DOMAIN_OPTIONS.map((r) => (
              <button
                key={r.id}
                onClick={() => setRegion(r.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  region === r.id
                    ? 'bg-energy-info/20 text-energy-info'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {r.flag} {r.shortLabel}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ORIUS Domain Context */}
      <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-gradient-to-r from-sky-500/5 via-transparent to-transparent border border-sky-500/10">
        <ShieldCheck className="w-4 h-4 text-sky-400 shrink-0" />
        <span className="text-[11px] text-slate-400">
          <span className="text-sky-400 font-semibold">{currentDomain.label}</span> · {batteryDomain ? 'Reference Witness Row · Conformal prediction intervals backed by Prop 1 · DC3S Stage 2 calibration active' : 'Tracked runtime artifact trace · no generated demo values · domain denominator stays in repo artifacts'}
        </span>
        <Link href="/safety" className="ml-auto flex items-center gap-1 text-[10px] text-sky-400/70 hover:text-sky-400 transition-colors shrink-0">
          Safety pipeline <ChevronRight className="w-3 h-3" />
        </Link>
      </div>

      <StatusBanner title="Forecasting Status" messages={statusMessages} />

      {/* Dataset info badge */}
      {dataset.stats && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="text-slate-300 font-medium">{dataset.stats.label}</span>
          <span>•</span>
          <span>{dataset.stats.rows.toLocaleString()} records</span>
          <span>•</span>
          <span>{dataset.stats.total_features} features</span>
          <span>•</span>
          <span>{forecastData.length ? `Showing ${forecastData.length} real artifact rows` : 'No trace extracted'}</span>
        </div>
      )}

      <ForecastChart
        data={forecastData.length ? forecastData : undefined}
        target={selectedTarget}
        zoneId={region}
        unit={targetOptions.find((target) => target.id === selectedTarget)?.unit}
        metrics={
          bestMetric
            ? { rmse: bestMetric.rmse, coverage_90: bestMetric.coverage_90, model: bestMetric.model }
            : undefined
        }
      />

      {/* Metrics table */}
      <Panel title="Model Comparison" subtitle={realTargetMetrics.length ? 'Real training metrics' : 'All models for selected target'} badge="Conformal PI" badgeColor="info">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-slate-500 font-medium">Model</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">RMSE (MW)</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">MAE (MW)</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">MAPE (%)</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">R²</th>
                <th className="text-right py-2 px-3 text-slate-500 font-medium">90% PICP</th>
              </tr>
            </thead>
            <tbody>
              {displayMetrics.map((m, i) => {
                const isBest = m.rmse === Math.min(...displayMetrics.map((x) => x.rmse));
                return (
                  <tr key={i} className={`border-b border-white/5 ${isBest ? 'bg-energy-primary/5' : ''}`}>
                    <td className="py-2.5 px-3 text-white font-medium">
                      {isBest && <span className="text-energy-primary mr-1">★</span>}
                      {m.model}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-white">{m.rmse.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{m.mae.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{formatMaybe(m.mape, 1)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-energy-primary">{formatMaybe(m.r2, 3)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-energy-info">
                      {m.coverage_90 === undefined ? 'N/A' : `${m.coverage_90.toFixed(1)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Hourly profiles if available */}
      {dataset.profiles?.[selectedTarget]?.length > 0 && (
        <Panel title="Average Hourly Profile" subtitle={`${selectedTarget} — mean ± std by hour of day`}>
          <div className="grid grid-cols-12 gap-1">
            {dataset.profiles[selectedTarget].map((p) => {
              const maxMean = Math.max(...dataset.profiles[selectedTarget].map((x) => x.mean));
              const pct = maxMean > 0 ? (p.mean / maxMean) * 100 : 0;
              return (
                <div key={p.hour} className="flex flex-col items-center gap-1">
                  <div className="w-full h-16 bg-white/3 rounded-md relative overflow-hidden">
                    <div
                      className="absolute bottom-0 w-full rounded-md bg-energy-primary/30"
                      style={{ height: `${pct}%` }}
                    />
                  </div>
                  <span className="text-[9px] text-slate-500">{p.hour}</span>
                  <span className="text-[8px] text-slate-600 font-mono">{p.mean.toFixed(0)}</span>
                </div>
              );
            })}
          </div>
          <div className="mt-2 text-[10px] text-slate-600 text-center">Hour of day (UTC)</div>
        </Panel>
      )}
    </div>
  );
}
