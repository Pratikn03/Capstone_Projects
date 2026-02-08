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
  mockDispatchForecast,
  mockBatterySchedule,
  mockForecastWithPI,
  mockAnomalies,
  mockAnomalyZScores,
  mockDriftData,
  mockParetoFrontier,
} from '@/lib/api/mock-data';
import { useReportsData } from '@/lib/api/reports-client';
import { formatCurrency, formatMW, formatPercent } from '@/lib/utils';

export default function DashboardPage() {
  // Load mock data (in production, these would be RSC data fetches)
  const dispatch = mockDispatchForecast('DE', 24);
  const battery = mockBatterySchedule('DE');
  const forecastLoad = mockForecastWithPI('load_mw', 48);
  const anomalies = mockAnomalies();
  const anomalyZScores = mockAnomalyZScores(72);
  const driftData = mockDriftData(30);
  const pareto = mockParetoFrontier();
  const { metrics, impact } = useReportsData();

  const loadMetrics = metrics.filter((m) => m.target === 'load_mw');
  const bestRMSE = loadMetrics.length ? Math.min(...loadMetrics.map((m) => m.rmse)) : null;
  const costSavingsPct = impact?.cost_savings_pct ?? 17.4;
  const costSavingsUsd = impact?.cost_savings_usd ?? null;
  const carbonReductionPct = impact?.carbon_reduction_pct ?? 32.6;
  const carbonTons =
    impact?.carbon_reduction_kg !== null && impact?.carbon_reduction_kg !== undefined
      ? impact.carbon_reduction_kg / 1000
      : null;
  const peakShavingPct = impact?.peak_shaving_pct ?? 19.5;
  const peakShavingMw = impact?.peak_shaving_mw ?? 5000;

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
        <ForecastChart data={forecastLoad} target="load_mw" zoneId="DE" />

        {/* Panel 2: Generation Dispatch (Baseline vs Optimized) */}
        <DispatchChart data={dispatch.data} title="24h Dispatch — Germany (OPSD)" />
      </div>

      {/* ─── Row 2: Battery SOC + Cost-Carbon Tradeoff ─── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Panel 3: Battery State of Charge */}
        <BatterySOCChart schedule={battery.schedule} metrics={battery.metrics} />

        {/* Panel 4: Cost-Carbon Pareto */}
        <CarbonCostPanel data={pareto} zoneId="DE" />
      </div>

      {/* ─── Row 3: Anomalies + MLOps ─── */}
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
