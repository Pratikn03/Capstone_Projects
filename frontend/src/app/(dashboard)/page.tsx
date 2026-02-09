'use client';

import { BarChart3, Zap, Leaf, TrendingDown, Activity, Database } from 'lucide-react';
import { KPICard } from '@/components/ui/KPICard';
import { Panel } from '@/components/ui/Panel';
import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { AnomalyTimeline } from '@/components/charts/AnomalyTimeline';
import { AnomalyList } from '@/components/charts/AnomalyList';
import { MLOpsMonitor } from '@/components/charts/MLOpsMonitor';
import {
  mockBatterySchedule,
  mockAnomalies,
  mockAnomalyZScores,
  mockDriftData,
  mockParetoFrontier,
} from '@/lib/api/mock-data';
import { useDispatchCompare } from '@/lib/api/dispatch-client';
import { useReportsData } from '@/lib/api/reports-client';
import { useDatasetData } from '@/lib/api/dataset-client';
import { formatCurrency, formatMW, formatPercent } from '@/lib/utils';
import { useRegion } from '@/components/ui/RegionContext';

export default function DashboardPage() {
  const { region } = useRegion();
  const dispatch = useDispatchCompare(region, 24);
  const battery = mockBatterySchedule(region);
  const anomalies = mockAnomalies();
  const anomalyZScores = mockAnomalyZScores(72);
  const driftData = mockDriftData(30);
  const pareto = mockParetoFrontier();
  const { metrics, impact, robustness, regions } = useReportsData();
  const dataset = useDatasetData(region as 'DE' | 'US');

  const regionBundle = regions[region];
  const metricsActive = regionBundle?.metrics?.length ? regionBundle.metrics : metrics;
  const impactActive = regionBundle?.impact ?? impact;
  const robustnessActive = regionBundle?.robustness ?? robustness;

  // Use real dataset metrics when available
  const realMetrics = dataset.metrics;
  const realImpact = dataset.impact;
  const realStats = dataset.stats;

  // Use real forecast data from extracted dataset
  const forecastData = dataset.forecast?.['load_mw'] ?? [];

  const loadMetrics = metricsActive.filter((m) => m.target === 'load_mw');
  const bestLoadMetric = loadMetrics.length
    ? loadMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b))
    : null;
  const bestRMSE = realMetrics.find((m) => m.target === 'load_mw')?.rmse ?? bestLoadMetric?.rmse ?? null;

  const costSavingsPct = realImpact?.cost_savings_pct ?? impactActive?.cost_savings_pct ?? null;
  const costSavingsRaw =
    realImpact
      ? (realImpact.baseline_cost_usd ?? 0) - (realImpact.gridpulse_cost_usd ?? 0)
      : impactActive?.cost_savings_usd ?? null;
  const carbonReductionPct = realImpact?.carbon_reduction_pct ?? impactActive?.carbon_reduction_pct ?? null;
  const carbonKg = realImpact?.baseline_carbon_kg != null && realImpact?.gridpulse_carbon_kg != null
    ? realImpact.baseline_carbon_kg - realImpact.gridpulse_carbon_kg
    : (impactActive as Record<string, unknown>)?.carbon_reduction_kg as number | null ?? null;
  const carbonTons = carbonKg !== null && carbonKg !== undefined ? carbonKg / 1000 : null;
  const peakShavingPct = realImpact?.peak_shaving_pct ?? impactActive?.peak_shaving_pct ?? null;
  const peakShavingMw = realImpact?.baseline_peak_mw != null && realImpact?.gridpulse_peak_mw != null
    ? realImpact.baseline_peak_mw - realImpact.gridpulse_peak_mw
    : (impactActive as Record<string, unknown>)?.peak_shaving_mw as number | null ?? null;
  const regionLabel = region === 'US' ? 'USA (EIA-930)' : 'Germany (OPSD)';
  const p95Regret = robustnessActive?.p95_regret ?? null;
  const infeasibleRate = robustnessActive?.infeasible_rate ?? null;
  const robustnessPct = robustnessActive?.perturbation_pct ?? null;

  const formatMaybePercent = (value: number | null | undefined) =>
    value === null || value === undefined ? 'N/A' : formatPercent(value);
  const formatMaybeCurrency = (value: number | null | undefined) =>
    value === null || value === undefined ? 'N/A' : formatCurrency(value, 'USD');

  return (
    <div className="p-6 space-y-6">
      {/* ─── Dataset Badge ─── */}
      {realStats && (
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <Database className="w-3.5 h-3.5" />
          <span className="font-medium text-slate-300">{regionLabel}</span>
          <span>•</span>
          <span>{realStats.rows.toLocaleString()} rows × {realStats.columns} features</span>
          <span>•</span>
          <span>
            {realStats.date_range.start?.slice(0, 10)} → {realStats.date_range.end?.slice(0, 10)}
          </span>
          <span>•</span>
          <span>{realMetrics.length} models trained</span>
        </div>
      )}

      {/* ─── KPI Row ─── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard
          label="Forecast RMSE"
          value={bestRMSE !== null ? bestRMSE.toFixed(0) : 'N/A'}
          unit={bestRMSE !== null ? 'MW' : undefined}
          icon={<BarChart3 className="w-4 h-4 text-energy-info" />}
          color="info"
          delay={0}
        />
        <KPICard
          label="Cost Savings"
          value={costSavingsPct !== null ? formatPercent(costSavingsPct) : 'N/A'}
          unit={costSavingsRaw !== null ? formatCurrency(costSavingsRaw, 'USD') : undefined}
          change={costSavingsPct ?? undefined}
          icon={<TrendingDown className="w-4 h-4 text-energy-primary" />}
          color="primary"
          delay={0.05}
        />
        <KPICard
          label="Carbon Reduction"
          value={carbonReductionPct !== null ? formatPercent(carbonReductionPct) : 'N/A'}
          unit={carbonTons !== null ? `${carbonTons.toFixed(0)} tCO₂` : undefined}
          change={carbonReductionPct ?? undefined}
          icon={<Leaf className="w-4 h-4 text-energy-primary" />}
          color="primary"
          delay={0.1}
        />
        <KPICard
          label="Peak Shaving"
          value={peakShavingPct !== null ? formatPercent(peakShavingPct) : 'N/A'}
          unit={peakShavingMw !== null ? formatMW(peakShavingMw) : undefined}
          change={peakShavingPct ?? undefined}
          changeLabel="peak reduced"
          icon={<Zap className="w-4 h-4 text-energy-warn" />}
          color="warn"
          delay={0.15}
        />
      </div>

      {/* ─── Row 1: Forecast + Dispatch ─── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ForecastChart
          data={forecastData.length ? forecastData : undefined}
          target="load_mw"
          zoneId={region}
          metrics={bestLoadMetric ? { rmse: bestLoadMetric.rmse, coverage_90: bestLoadMetric.coverage_90 } : undefined}
        />
        <DispatchChart
          optimized={
            dataset.dispatch.length
              ? dataset.dispatch.map((d) => ({
                  ...d,
                  battery_dispatch: 0,
                  price_eur_mwh: d.price_eur_mwh ?? undefined,
                }))
              : dispatch.optimized
          }
          baseline={dispatch.baseline}
          title={`24h Dispatch — ${regionLabel}`}
          showBaseline={!!dispatch.baseline}
        />
      </div>

      {/* ─── Row 2: Battery SOC + Cost-Carbon Tradeoff ─── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <BatterySOCChart schedule={battery.schedule} metrics={battery.metrics} />
        <CarbonCostPanel data={pareto} zoneId={region} summary={impactActive ?? undefined} />
      </div>

      {/* ─── Row 3: Impact & Robustness ─── */}
      <Panel
        title="Impact & Robustness"
        subtitle={realImpact ? 'Real pipeline results' : 'Stress-tested dispatch performance'}
        badge={robustnessPct !== null ? `±${robustnessPct}%` : 'Robustness'}
        badgeColor="info"
      >
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-3 rounded-lg bg-white/3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Cost Savings</div>
            <div className="text-lg font-semibold text-energy-primary">{formatMaybePercent(costSavingsPct)}</div>
            <div className="text-xs text-slate-500 font-mono">{costSavingsRaw !== null ? formatCurrency(costSavingsRaw, 'USD') : 'N/A'}</div>
          </div>
          <div className="p-3 rounded-lg bg-white/3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Carbon Reduction</div>
            <div className="text-lg font-semibold text-energy-primary">{formatMaybePercent(carbonReductionPct)}</div>
            <div className="text-xs text-slate-500 font-mono">
              {carbonTons !== null ? `${carbonTons.toFixed(1)} tCO₂` : 'N/A'}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-white/3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">P95 Regret</div>
            <div className="text-lg font-semibold text-white">{formatMaybeCurrency(p95Regret)}</div>
            <div className="text-xs text-slate-500">Worst-case cost gap</div>
          </div>
          <div className="p-3 rounded-lg bg-white/3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Infeasible Rate</div>
            <div className="text-lg font-semibold text-white">
              {infeasibleRate === null ? 'N/A' : `${(infeasibleRate * 100).toFixed(1)}%`}
            </div>
            <div className="text-xs text-slate-500">Across stress trials</div>
          </div>
        </div>
      </Panel>

      {/* ─── Row 4: Real Model Registry ─── */}
      <Panel title="Model Registry" subtitle="Production Models" badge={`${realMetrics.length || metricsActive.length} models`} delay={0.35}>
        <div className="space-y-2">
          {(['load_mw', 'wind_mw', 'solar_mw'] as const).map((target) => {
            const targetRealMetrics = realMetrics.filter((m) => m.target === target);
            const targetReportsMetrics = metricsActive.filter((m) => m.target === target);
            const targetMetrics = targetRealMetrics.length ? targetRealMetrics : targetReportsMetrics;
            if (!targetMetrics.length) return null;
            const best = targetMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b));
            const registryEntry = dataset.registry.find(
              (r) => r.target === target && r.model === best.model
            );
            return (
              <div key={target} className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-white/3">
                <div>
                  <span className="text-xs font-medium text-white">
                    {target === 'load_mw' ? '⚡ Load' : target === 'wind_mw' ? '💨 Wind' : '☀️ Solar'}
                  </span>
                  <span className="text-[10px] text-slate-500 ml-2">Best: {best.model}</span>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <div>
                    <span className="text-slate-500">RMSE</span>
                    <span className="ml-1 text-white font-mono">{best.rmse.toFixed(1)}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">MAE</span>
                    <span className="ml-1 text-energy-primary font-mono">{best.mae.toFixed(1)}</span>
                  </div>
                  {registryEntry && (
                    <div>
                      <span className="text-slate-500">Size</span>
                      <span className="ml-1 text-slate-300 font-mono">{registryEntry.size_mb} MB</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <div className="mt-3 flex items-center gap-3 text-[10px] text-slate-600">
          <span>
            {dataset.registry.length > 0
              ? `Last trained: ${dataset.registry[0]?.modified?.slice(0, 10) ?? 'N/A'}`
              : 'Last trained: 2026-02-07'}
          </span>
          <span>•</span>
          <span>50 epochs, CosineAnnealingLR</span>
          <span>•</span>
          <span>Datasets: OPSD • EIA-930</span>
        </div>
      </Panel>

      {/* ─── Row 5: Anomalies + Dataset Stats ─── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="space-y-4">
          <AnomalyTimeline data={anomalyZScores} />
          <Panel
            title="Anomaly Status"
            badge={`${anomalies.filter((a) => a.status === 'active').length} Active`}
            badgeColor="alert"
            delay={0.3}
          >
            <AnomalyList anomalies={anomalies} />
          </Panel>
        </div>
        <div className="space-y-4">
          <MLOpsMonitor data={driftData} />
          {realStats && (
            <Panel title="Dataset Overview" subtitle={regionLabel} delay={0.4}>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="px-3 py-2.5 rounded-lg bg-white/3">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">Records</div>
                  <div className="text-sm font-bold text-white font-mono">{realStats.rows.toLocaleString()}</div>
                </div>
                <div className="px-3 py-2.5 rounded-lg bg-white/3">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">Features</div>
                  <div className="text-sm font-bold text-white font-mono">{realStats.total_features}</div>
                </div>
                <div className="px-3 py-2.5 rounded-lg bg-white/3">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">Weather</div>
                  <div className="text-sm font-bold text-white font-mono">{realStats.weather_features}</div>
                </div>
                <div className="px-3 py-2.5 rounded-lg bg-white/3">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">Lag/Rolling</div>
                  <div className="text-sm font-bold text-white font-mono">{realStats.lag_features}</div>
                </div>
              </div>
              <div className="mt-3 space-y-1">
                {Object.entries(realStats.targets_summary).map(([target, s]) => (
                  <div key={target} className="flex items-center justify-between text-xs px-2 py-1.5 rounded bg-white/2">
                    <span className="text-slate-400">
                      {target === 'load_mw' ? '⚡' : target === 'wind_mw' ? '💨' : '☀️'} {target}
                    </span>
                    <div className="flex gap-3 font-mono text-slate-300">
                      <span>μ={s.mean.toFixed(0)}</span>
                      <span>σ={s.std.toFixed(0)}</span>
                      <span>max={s.max.toFixed(0)}</span>
                      <span className="text-energy-primary">{s.non_zero_pct}% non-zero</span>
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>
      </div>

      {/* ─── Footer ─── */}
      <div className="flex items-center justify-center gap-3 py-4 text-[11px] text-slate-600">
        <span className="flex items-center gap-1.5">
          <Activity className="w-3 h-3" />
          Forecast
        </span>
        <span>→</span>
        <span>Optimize</span>
        <span>→</span>
        <span>Measure Impact</span>
        <span className="ml-2">🔋</span>
      </div>
    </div>
  );
}
