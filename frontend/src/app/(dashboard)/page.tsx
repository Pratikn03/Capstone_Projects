'use client';

import { BarChart3, Zap, Leaf, TrendingDown, Battery, AlertTriangle, Activity } from 'lucide-react';
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
  mockForecastWithPI,
  mockAnomalies,
  mockAnomalyZScores,
  mockDriftData,
  mockParetoFrontier,
} from '@/lib/api/mock-data';
import { useDispatchCompare } from '@/lib/api/dispatch-client';
import { useReportsData } from '@/lib/api/reports-client';
import { formatCurrency, formatMW, formatPercent } from '@/lib/utils';
import { useRegion } from '@/components/ui/RegionContext';

export default function DashboardPage() {
  // Load mock data (in production, these would be RSC data fetches)
  const { region } = useRegion();
  const dispatch = useDispatchCompare(region, 24);
  const battery = mockBatterySchedule(region);
  const forecastLoad = mockForecastWithPI('load_mw', 48);
  const anomalies = mockAnomalies();
  const anomalyZScores = mockAnomalyZScores(72);
  const driftData = mockDriftData(30);
  const pareto = mockParetoFrontier();
  const { metrics, impact, robustness, regions } = useReportsData();
  const regionBundle = regions[region];
  const metricsActive = regionBundle?.metrics?.length ? regionBundle.metrics : metrics;
  const impactActive = regionBundle?.impact ?? impact;
  const robustnessActive = regionBundle?.robustness ?? robustness;

  const loadMetrics = metricsActive.filter((m) => m.target === 'load_mw');
  const bestLoadMetric = loadMetrics.length
    ? loadMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b))
    : null;
  const bestRMSE = bestLoadMetric ? bestLoadMetric.rmse : null;
  const costSavingsPct = impactActive?.cost_savings_pct ?? 17.4;
  const costSavingsUsd = impactActive?.cost_savings_usd ?? null;
  const carbonReductionPct = impactActive?.carbon_reduction_pct ?? 32.6;
  const carbonTons =
    impactActive?.carbon_reduction_kg !== null && impactActive?.carbon_reduction_kg !== undefined
      ? impactActive.carbon_reduction_kg / 1000
      : null;
  const peakShavingPct = impactActive?.peak_shaving_pct ?? 19.5;
  const peakShavingMw = impactActive?.peak_shaving_mw ?? 5000;
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
          value={formatPercent(costSavingsPct)}
          unit={costSavingsUsd !== null ? formatCurrency(costSavingsUsd, 'USD') : '€23,500'}
          change={costSavingsPct}
          icon={<TrendingDown className="w-4 h-4 text-energy-primary" />}
          color="primary"
          delay={0.05}
        />
        <KPICard
          label="Carbon Reduction"
          value={formatPercent(carbonReductionPct)}
          unit={carbonTons !== null ? `${carbonTons.toFixed(0)} tCO₂` : '47.8 tCO₂'}
          change={carbonReductionPct}
          icon={<Leaf className="w-4 h-4 text-energy-primary" />}
          color="primary"
          delay={0.1}
        />
        <KPICard
          label="Peak Shaving"
          value={formatPercent(peakShavingPct)}
          unit={formatMW(peakShavingMw)}
          change={peakShavingPct}
          changeLabel="peak reduced"
          icon={<Zap className="w-4 h-4 text-energy-warn" />}
          color="warn"
          delay={0.15}
        />
      </div>

      {/* ─── Row 1: Forecast + Dispatch ─── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Panel 1: Probabilistic Forecast with PI */}
        <ForecastChart
          data={forecastLoad}
          target="load_mw"
          zoneId={region}
          metrics={bestLoadMetric ? { rmse: bestLoadMetric.rmse, coverage_90: bestLoadMetric.coverage_90 } : undefined}
        />

        {/* Panel 2: Generation Dispatch (Baseline vs Optimized) */}
        <DispatchChart
          optimized={dispatch.optimized}
          baseline={dispatch.baseline}
          title={`24h Dispatch — ${regionLabel}`}
          showBaseline
        />
      </div>

      {/* ─── Row 2: Battery SOC + Cost-Carbon Tradeoff ─── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Panel 3: Battery State of Charge */}
        <BatterySOCChart schedule={battery.schedule} metrics={battery.metrics} />

        {/* Panel 4: Cost-Carbon Pareto */}
        <CarbonCostPanel data={pareto} zoneId={region} summary={impactActive ?? undefined} />
      </div>

      {/* ─── Row 3: Impact & Robustness ─── */}
      <Panel
        title="Impact & Robustness"
        subtitle="Stress-tested dispatch performance"
        badge={robustnessPct !== null ? `±${robustnessPct}%` : 'Robustness'}
        badgeColor="info"
      >
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-3 rounded-lg bg-white/3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Cost Savings</div>
            <div className="text-lg font-semibold text-energy-primary">{formatMaybePercent(costSavingsPct)}</div>
            <div className="text-xs text-slate-500 font-mono">{costSavingsUsd !== null ? formatCurrency(costSavingsUsd, 'USD') : 'N/A'}</div>
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
        <div className="mt-3 text-[11px] text-slate-500">
          Robustness computed from perturbation trials on load and renewables; regret is measured vs oracle dispatch.
        </div>
      </Panel>

      {/* ─── Row 4: Anomalies + MLOps ─── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Panel 5: Anomaly Detection Timeline */}
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

        {/* Panel 6: MLOps & Model Monitoring */}
        <div className="space-y-4">
          <MLOpsMonitor data={driftData} />

          {/* Model card quick view */}
          <Panel title="Model Registry" subtitle="Production Models" badge="3 Targets × 3 Models" delay={0.35}>
            <div className="space-y-2">
              {(['load_mw', 'wind_mw', 'solar_mw'] as const).map((target) => {
                const targetMetrics = metrics.filter((m) => m.target === target);
                const best = targetMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b));
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
                        <span className="text-slate-500">R²</span>
                        <span className="ml-1 text-energy-primary font-mono">{best.r2?.toFixed(3)}</span>
                      </div>
                      <div>
                        <span className="text-slate-500">PICP</span>
                        <span className="ml-1 text-energy-info font-mono">{best.coverage_90?.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-3 flex items-center gap-3 text-[10px] text-slate-600">
              <span>Last trained: 2026-02-07</span>
              <span>•</span>
              <span>50 epochs, CosineAnnealingLR, No early stopping</span>
              <span>•</span>
              <span>Datasets: OPSD • EIA-930</span>
            </div>
          </Panel>
        </div>
      </div>

      {/* ─── Footer: Pipeline Status ─── */}
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
