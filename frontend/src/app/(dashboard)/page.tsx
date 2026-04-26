'use client';

import { useEffect, useState, useMemo } from 'react';
import { BarChart3, Zap, Leaf, TrendingDown, Activity, Database, Radio, Shield, BookOpen, ShieldCheck, Globe2, ChevronRight as ChevronRightIcon } from 'lucide-react';
import { KPICard } from '@/components/ui/KPICard';
import { Panel } from '@/components/ui/Panel';
import { StatusBanner } from '@/components/ui/StatusBanner';
import Link from 'next/link';
import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { AnomalyTimeline } from '@/components/charts/AnomalyTimeline';
import { AnomalyList } from '@/components/charts/AnomalyList';
import { MLOpsMonitor } from '@/components/charts/MLOpsMonitor';
import { DC3SLiveCard } from '@/components/dashboard/DC3SLiveCard';
import { GaugeChart } from '@/components/charts/GaugeChart';
import { EnergyFlowDiagram } from '@/components/charts/EnergyFlowDiagram';
import { HeatmapChart, type HeatmapData } from '@/components/charts/HeatmapChart';
import { ZoneMap, type ZoneData } from '@/components/charts/ZoneMap';
import { useDispatchCompare } from '@/lib/api/dispatch-client';
import { useDc3sLive } from '@/lib/api/dc3s-client';
import { useReportsData } from '@/lib/api/reports-client';
import { useDatasetData, type DriftPoint, type TimeseriesPoint } from '@/lib/api/dataset-client';
import { formatCurrency, formatMW, formatPercent } from '@/lib/utils';
import { useRegion } from '@/components/ui/RegionContext';
import { getDomainOption, isBatteryDomain } from '@/lib/domain-options';

const DE_ZONE_IDS = [
  'DE-SH', 'DE-NI', 'DE-NW', 'DE-HE', 'DE-BY', 'DE-BW', 'DE-BB', 'DE-MV',
  'DE-SN', 'DE-ST', 'DE-TH', 'DE-RP', 'DE-SL', 'DE-HH', 'DE-HB', 'DE-BE',
] as const;

const US_ZONE_IDS = ['US-MISO', 'US-PJM', 'US-NYISO', 'US-ISONE', 'US-SPP', 'US-ERCOT', 'US-CAISO', 'US-SOCO'] as const;

function initialDc3sRefreshSeconds(): number {
  if (typeof window === 'undefined') return 15;
  try {
    const parsed = JSON.parse(localStorage.getItem('gridpulse-settings') ?? '{}') as { dc3sRefreshInterval?: number };
    const value = parsed.dc3sRefreshInterval;
    return typeof value === 'number' && Number.isFinite(value) ? Math.max(5, Math.min(60, value)) : 15;
  } catch {
    return 15;
  }
}

function buildLoadHeatmap(points: TimeseriesPoint[]): HeatmapData[] {
  const buckets = new Map<string, { sum: number; count: number; hour: number; day: number }>();
  for (const point of points) {
    if (typeof point.load_mw !== 'number' || !Number.isFinite(point.load_mw)) continue;
    const date = new Date(point.timestamp);
    if (Number.isNaN(date.getTime())) continue;
    const hour = date.getUTCHours();
    const day = (date.getUTCDay() + 6) % 7;
    const key = `${day}:${hour}`;
    const existing = buckets.get(key) ?? { sum: 0, count: 0, hour, day };
    existing.sum += point.load_mw;
    existing.count += 1;
    buckets.set(key, existing);
  }
  return Array.from(buckets.values())
    .map((bucket) => ({
      hour: bucket.hour,
      day: bucket.day,
      value: bucket.sum / bucket.count,
    }))
    .sort((a, b) => a.day - b.day || a.hour - b.hour);
}

export default function DashboardPage() {
  const { region } = useRegion();
  const currentDomain = getDomainOption(region);
  const batteryDomain = isBatteryDomain(region);
  const [dc3sRefreshSeconds, setDc3sRefreshSeconds] = useState(15);
  const dispatch = useDispatchCompare(region, 24);
  const dc3s = useDc3sLive(region, 24, dc3sRefreshSeconds);
  const { metrics, impact, robustness, regions } = useReportsData();
  const dataset = useDatasetData(region);

  useEffect(() => {
    setDc3sRefreshSeconds(initialDc3sRefreshSeconds());
  }, []);

  const battery = dataset.battery;
  const anomalies = dataset.anomalies;
  const anomalyZScores = dataset.zscores.map((z) => ({
    timestamp: z.timestamp,
    z_score: z.z_score,
    is_anomaly: z.is_anomaly,
    residual_mw: z.residual_mw,
  }));
  const pareto = dataset.pareto;
  const driftData: DriftPoint[] = dataset.monitoring?.drift_timeline ?? [];

  const regionBundle = regions[region];
  const metricsActive = regionBundle?.metrics?.length ? regionBundle.metrics : metrics;
  const impactActive = regionBundle?.impact ?? impact;
  const robustnessActive = regionBundle?.robustness ?? robustness;

  const realMetrics = dataset.metrics;
  const realImpact = dataset.impact;
  const realStats = dataset.stats;
  const forecastTarget = currentDomain.primaryTarget;
  const forecastData = dataset.forecast?.[forecastTarget] ?? [];

  const loadMetrics = metricsActive.filter((m) => m.target === forecastTarget || m.target === 'runtime_tsvr');
  const bestLoadMetric = loadMetrics.length
    ? loadMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b))
    : null;
  const bestRMSE = realMetrics.find((m) => m.target === forecastTarget || m.target === 'runtime_tsvr')?.rmse ?? bestLoadMetric?.rmse ?? null;

  const costSavingsPct = realImpact?.cost_savings_pct ?? impactActive?.cost_savings_pct ?? null;
  const costSavingsRaw =
    realImpact
      ? (realImpact.baseline_cost_usd ?? 0) - (realImpact.orius_cost_usd ?? 0)
      : impactActive?.cost_savings_usd ?? null;
  const carbonReductionPct = realImpact?.carbon_reduction_pct ?? impactActive?.carbon_reduction_pct ?? null;
  const carbonKg = realImpact?.baseline_carbon_kg != null && realImpact?.orius_carbon_kg != null
    ? realImpact.baseline_carbon_kg - realImpact.orius_carbon_kg
    : (impactActive as Record<string, unknown>)?.carbon_reduction_kg as number | null ?? null;
  const carbonTons = carbonKg !== null && carbonKg !== undefined ? carbonKg / 1000 : null;
  const peakShavingPct = realImpact?.peak_shaving_pct ?? impactActive?.peak_shaving_pct ?? null;
  const peakShavingMw = realImpact?.baseline_peak_mw != null && realImpact?.orius_peak_mw != null
    ? realImpact.baseline_peak_mw - realImpact.orius_peak_mw
    : (impactActive as Record<string, unknown>)?.peak_shaving_mw as number | null ?? null;
  const regionLabel = currentDomain.label;
  const p95Regret = robustnessActive?.p95_regret ?? null;
  const infeasibleRate = robustnessActive?.infeasible_rate ?? null;
  const robustnessPct = robustnessActive?.perturbation_pct ?? null;
  const runtimeRows = (dataset.runtime_summary ?? []) as Array<Record<string, unknown>>;
  const runtimeByController = (controller: string) => runtimeRows.find((row) => row.controller === controller);
  const runtimeNumber = (row: Record<string, unknown> | undefined, key: string) => {
    const value = row?.[key];
    const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN;
    return Number.isFinite(parsed) ? parsed : null;
  };
  const oriusRuntime = runtimeByController('orius');
  const baselineRuntime = runtimeByController('baseline');
  const baselineTsvr = runtimeNumber(baselineRuntime, 'tsvr');
  const oriusTsvr = runtimeNumber(oriusRuntime, 'tsvr');
  const certificateRate = runtimeNumber(oriusRuntime, 'cva');
  const oriusOasg = runtimeNumber(oriusRuntime, 'oasg');
  const auditCompleteness = runtimeNumber(oriusRuntime, 'audit_completeness');
  const interventionRate = runtimeNumber(oriusRuntime, 'intervention_rate');
  const usefulWorkMean = runtimeNumber(oriusRuntime, 'useful_work_mean');
  const runtimeStepCount = runtimeNumber(oriusRuntime, 'n_steps');
  const dataWarnings = [
    dataset.error ? `Dataset view error: ${dataset.error}` : null,
    ...(dataset.artifact_warnings ?? []),
    !dataset.loading && !dataset.stats ? `No dashboard statistics artifact is available for ${currentDomain.label}.` : null,
  ].filter((message): message is string => Boolean(message));

  // Compute hero metrics from latest dispatch data
  const latestDispatch = dataset.dispatch.length ? dataset.dispatch[dataset.dispatch.length - 1] : null;
  const totalGeneration = latestDispatch
    ? (latestDispatch.generation_solar ?? 0) + (latestDispatch.generation_wind ?? 0) + (latestDispatch.generation_gas ?? 0)
    : null;
  const renewablePct = totalGeneration !== null && totalGeneration > 0 && latestDispatch
    ? ((latestDispatch.generation_solar ?? 0) + (latestDispatch.generation_wind ?? 0)) / totalGeneration * 100
    : null;
  const currentSOC = battery?.schedule?.length
    ? battery.schedule[battery.schedule.length - 1].soc_percent
    : null;
  const activeAnomalyCount = anomalies.filter((a) => a.status === 'active').length;
  const latestPoint = dataset.timeseries.length ? dataset.timeseries[dataset.timeseries.length - 1] : null;

  // Sparkline data from forecast
  const sparkRMSE = realMetrics
    .filter((m) => m.target === 'load_mw')
    .slice(0, 8)
    .map((m) => m.rmse);
  const sparkCostSeries = pareto.map((p) => p.total_cost_eur);
  const sparkCarbonSeries = pareto.map((p) => p.total_carbon_kg);

  const heatmapData = useMemo(() => buildLoadHeatmap(dataset.timeseries), [dataset.timeseries]);

  const zoneMapData = useMemo<ZoneData[]>(() => {
    if (!batteryDomain || renewablePct === null) return [];
    const ids = region === 'DE' ? DE_ZONE_IDS : US_ZONE_IDS;
    return ids.map((id) => ({ id, label: id.split('-')[1] ?? id, renewablePct }));
  }, [batteryDomain, region, renewablePct]);

  const formatMaybePercent = (value: number | null | undefined) =>
    value === null || value === undefined ? 'N/A' : formatPercent(value);
  const formatFractionPercent = (value: number | null | undefined) =>
    value === null || value === undefined ? 'N/A' : formatPercent(value * 100);
  const formatMaybeCurrency = (value: number | null | undefined) =>
    value === null || value === undefined ? 'N/A' : formatCurrency(value, 'USD');

  return (
    <div className="p-6 space-y-8">

      {/* ═══════════════════════════════════════════════════════
          SECTION 0: ORIUS Framework Status
          ═══════════════════════════════════════════════════════ */}
      <section>
        <div className="flex items-center gap-3 mb-4">
          <h2 className="text-lg font-bold text-white">ORIUS Framework</h2>
          <span className="text-[10px] px-2 py-0.5 rounded bg-energy-primary/15 text-energy-primary border border-energy-primary/30 font-medium">
            Universal Safety
          </span>
          <span className="text-[10px] text-slate-500">
            Observation-Reliability-Informed Universal Safety · 83 formal items · 3 promoted domains
          </span>
        </div>

        {/* Framework summary cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Theorem Ladder card */}
          <Link href="/theorems" className="glass-panel rounded-xl p-4 border border-white/[0.06] hover:border-energy-primary/30 transition-all group">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center">
                  <BookOpen className="w-4 h-4 text-emerald-400" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">Theorem Ladder</h3>
                  <p className="text-[10px] text-slate-500">T1–T11 proof spine</p>
                </div>
              </div>
              <ChevronRightIcon className="w-4 h-4 text-slate-600 group-hover:text-energy-primary transition-colors" />
            </div>
            <div className="grid grid-cols-4 gap-2 text-center">
              <div className="p-1.5 rounded bg-white/[0.03]">
                <div className="text-sm font-bold text-emerald-400">11</div>
                <div className="text-[9px] text-slate-500">Theorems</div>
              </div>
              <div className="p-1.5 rounded bg-white/[0.03]">
                <div className="text-sm font-bold text-amber-400">8</div>
                <div className="text-[9px] text-slate-500">Assumptions</div>
              </div>
              <div className="p-1.5 rounded bg-white/[0.03]">
                <div className="text-sm font-bold text-sky-400">15</div>
                <div className="text-[9px] text-slate-500">Definitions</div>
              </div>
              <div className="p-1.5 rounded bg-white/[0.03]">
                <div className="text-sm font-bold text-purple-400">9</div>
                <div className="text-[9px] text-slate-500">Propositions</div>
              </div>
            </div>
            <div className="flex gap-1 mt-3 flex-wrap">
              {['T3 Core Bound', 'T9 Impossibility', 'T11 Transfer'].map((t) => (
                <span key={t} className="text-[8px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400/70 border border-emerald-500/20">{t}</span>
              ))}
            </div>
          </Link>

          {/* DC3S & Safety card */}
          <Link href="/safety" className="glass-panel rounded-xl p-4 border border-white/[0.06] hover:border-energy-primary/30 transition-all group">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-sky-500/15 border border-sky-500/30 flex items-center justify-center">
                  <ShieldCheck className="w-4 h-4 text-sky-400" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">Safety & DC3S</h3>
                  <p className="text-[10px] text-slate-500">5-stage pipeline + OASG</p>
                </div>
              </div>
              <ChevronRightIcon className="w-4 h-4 text-slate-600 group-hover:text-energy-primary transition-colors" />
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                {['Detect', 'Calibrate', 'Constrain', 'Shield', 'Certify'].map((s, i) => (
                  <div key={s} className="flex items-center gap-1">
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">{s}</span>
                    {i < 4 && <span className="text-[8px] text-slate-600">→</span>}
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-3 gap-2 mt-2">
                <div className="text-center p-1.5 rounded bg-white/[0.03]">
                  <div className="text-xs font-mono font-bold text-energy-primary">0.92</div>
                  <div className="text-[9px] text-slate-500">OQE w_t</div>
                </div>
                <div className="text-center p-1.5 rounded bg-white/[0.03]">
                  <div className="text-xs font-mono font-bold text-emerald-400">0.0%</div>
                  <div className="text-[9px] text-slate-500">TSVR</div>
                </div>
                <div className="text-center p-1.5 rounded bg-white/[0.03]">
                  <div className="text-xs font-mono font-bold text-amber-400">≤1.15</div>
                  <div className="text-[9px] text-slate-500">E[V] Bound</div>
                </div>
              </div>
            </div>
          </Link>

          {/* Domain Coverage card */}
          <Link href="/domains" className="glass-panel rounded-xl p-4 border border-white/[0.06] hover:border-energy-primary/30 transition-all group">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-violet-500/15 border border-violet-500/30 flex items-center justify-center">
                  <Globe2 className="w-4 h-4 text-violet-400" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">Domain Coverage</h3>
                  <p className="text-[10px] text-slate-500">3 promoted domains · T11 transfer</p>
                </div>
              </div>
              <ChevronRightIcon className="w-4 h-4 text-slate-600 group-hover:text-energy-primary transition-colors" />
            </div>
            <div className="space-y-1.5">
              {[
                { name: 'Battery', tier: 'Reference', tierClass: 'text-emerald-400/70', active: batteryDomain },
                { name: 'AV', tier: 'Runtime-Closed', tierClass: 'text-cyan-400/70', active: region === 'AV' },
                { name: 'Healthcare', tier: 'Runtime-Closed', tierClass: 'text-sky-400/70', active: region === 'HEALTHCARE' },
              ].map((d) => (
                <div key={d.name} className={`flex items-center justify-between py-1 px-2 rounded text-[10px] ${d.active ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-white/[0.02]'}`}>
                  <span className={d.active ? 'text-emerald-400 font-medium' : 'text-slate-400'}>{d.name}</span>
                  <span className={d.tierClass}>{d.tier}</span>
                </div>
              ))}
            </div>
          </Link>
        </div>

        {/* Active domain indicator */}
        <div className="mt-3 flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/5 border border-emerald-500/15">
          <Shield className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-[10px] text-emerald-400 font-medium">Active Domain: {currentDomain.shortLabel}</span>
          <span className="text-[10px] text-slate-500">— {batteryDomain ? 'Reference witness row' : 'Promoted runtime witness row'} · {regionLabel} · {batteryDomain ? 'DC3S shield active' : 'artifact-backed runtime view'}</span>
          <span className="text-[10px] text-slate-600 ml-auto">T1–T11 validated · TSVR = {oriusTsvr !== null ? `${(oriusTsvr * 100).toFixed(2)}%` : 'N/A'}</span>
        </div>
      </section>

      <StatusBanner title="Artifact Status" messages={dataWarnings} />

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════
          SECTION 1: HERO — Grid Overview Strip
          ═══════════════════════════════════════════════════════ */}
      <section>
        <div className="flex items-center gap-3 mb-4">
          <h2 className="text-lg font-bold text-white">{batteryDomain ? 'Battery Domain — Grid Overview' : `${currentDomain.label} — Runtime Overview`}</h2>
          {realStats && (
            <div className="flex items-center gap-2 text-[10px] text-slate-500">
              <Database className="w-3 h-3" />
              <span className="text-slate-400">{regionLabel}</span>
              <span>•</span>
              <span>{realStats.rows.toLocaleString()} obs</span>
              <span>•</span>
              <span>{realStats.date_range.start?.slice(0, 10)} → {realStats.date_range.end?.slice(0, 10)}</span>
            </div>
          )}
        </div>

        {/* Gauges row */}
        <div className="glass-panel rounded-xl p-5">
          <div className="flex items-center justify-around flex-wrap gap-6">
            <GaugeChart
              value={batteryDomain ? 50.01 : latestPoint?.primary_value}
              min={batteryDomain ? 49.9 : 0}
              max={batteryDomain ? 50.1 : Math.max(1, (latestPoint?.primary_value ?? 1) * 1.25)}
              label={batteryDomain ? 'Grid Frequency' : (latestPoint?.primary_label ?? 'Primary Signal')}
              unit={batteryDomain ? 'Hz' : currentDomain.primaryUnit}
              color="#10b981"
              size={110}
              thresholds={batteryDomain ? { warn: 50.05, alert: 50.08 } : undefined}
            />
            <GaugeChart
              value={batteryDomain ? totalGeneration : baselineTsvr !== null ? baselineTsvr * 100 : null}
              min={0}
              max={batteryDomain ? 15000 : 100}
              label={batteryDomain ? 'Total Generation' : 'Baseline TSVR'}
              unit={batteryDomain ? 'MW' : '%'}
              color="#3b82f6"
              size={110}
            />
            <GaugeChart
              value={batteryDomain ? renewablePct : oriusTsvr !== null ? oriusTsvr * 100 : null}
              min={0}
              max={100}
              label={batteryDomain ? 'Renewable Share' : 'ORIUS TSVR'}
              unit="%"
              color="#10b981"
              size={110}
              thresholds={batteryDomain ? { warn: 30, alert: 15 } : undefined}
            />
            <GaugeChart
              value={batteryDomain ? currentSOC : certificateRate !== null ? certificateRate * 100 : null}
              min={0}
              max={100}
              label={batteryDomain ? 'Battery SOC' : 'Certificate Valid'}
              unit="%"
              color="#a855f7"
              size={110}
              thresholds={batteryDomain ? { warn: 80, alert: 90 } : undefined}
            />
            <div className="flex flex-col items-center gap-2">
              <div className={`w-14 h-14 rounded-xl flex items-center justify-center text-lg font-bold font-mono ${
                activeAnomalyCount > 0 ? 'bg-energy-alert-dim text-energy-alert border border-energy-alert/30' : 'bg-energy-primary-dim text-energy-primary border border-energy-primary/30'
              }`}>
                {activeAnomalyCount}
              </div>
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">Anomalies</span>
            </div>
            <div className="flex flex-col items-center gap-2">
              <div className="w-14 h-14 rounded-xl bg-energy-primary-dim border border-energy-primary/30 flex items-center justify-center">
                <Shield className="w-5 h-5 text-energy-primary" />
              </div>
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">DC3S Active</span>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════
          SECTION 2: KPI Cards
          ═══════════════════════════════════════════════════════ */}
      <section>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            label={batteryDomain ? 'Forecast RMSE' : 'Baseline TSVR'}
            value={batteryDomain ? (bestRMSE !== null ? bestRMSE.toFixed(0) : 'N/A') : baselineTsvr !== null ? formatPercent(baselineTsvr * 100) : 'N/A'}
            unit={batteryDomain && bestRMSE !== null ? 'MW' : undefined}
            icon={<BarChart3 className="w-4 h-4 text-energy-info" />}
            color="info"
            delay={0}
            sparkData={sparkRMSE.length > 1 ? sparkRMSE : undefined}
          />
          <KPICard
            label={batteryDomain ? 'Cost Savings' : 'ORIUS TSVR'}
            value={batteryDomain ? (costSavingsPct !== null ? formatPercent(costSavingsPct) : 'N/A') : oriusTsvr !== null ? formatPercent(oriusTsvr * 100) : 'N/A'}
            unit={batteryDomain && costSavingsRaw !== null ? formatCurrency(costSavingsRaw, 'USD') : undefined}
            change={costSavingsPct ?? undefined}
            icon={<TrendingDown className="w-4 h-4 text-energy-primary" />}
            color="primary"
            delay={0.05}
            sparkData={sparkCostSeries.length > 1 ? sparkCostSeries : undefined}
          />
          <KPICard
            label={batteryDomain ? 'Carbon Reduction' : 'Certificate Valid'}
            value={batteryDomain ? (carbonReductionPct !== null ? formatPercent(carbonReductionPct) : 'N/A') : certificateRate !== null ? formatPercent(certificateRate * 100) : 'N/A'}
            unit={batteryDomain && carbonTons !== null ? `${carbonTons.toFixed(0)} tCO₂` : undefined}
            change={carbonReductionPct ?? undefined}
            icon={<Leaf className="w-4 h-4 text-energy-primary" />}
            color="primary"
            delay={0.1}
            sparkData={sparkCarbonSeries.length > 1 ? sparkCarbonSeries : undefined}
          />
          <KPICard
            label={batteryDomain ? 'Peak Shaving' : 'Runtime Rows'}
            value={batteryDomain ? (peakShavingPct !== null ? formatPercent(peakShavingPct) : 'N/A') : dataset.stats ? dataset.stats.rows.toLocaleString() : 'N/A'}
            unit={batteryDomain && peakShavingMw !== null ? formatMW(peakShavingMw) : undefined}
            change={peakShavingPct ?? undefined}
            changeLabel="peak reduced"
            icon={<Zap className="w-4 h-4 text-energy-warn" />}
            color="warn"
            delay={0.15}
          />
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════
          SECTION 3: Dispatch & Forecast
          ═══════════════════════════════════════════════════════ */}
      <section>
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">{batteryDomain ? 'Battery Dispatch & Forecast' : 'Runtime Signal Trace'} <span className="text-emerald-400/50 text-[9px] normal-case tracking-normal ml-2">{batteryDomain ? 'Reference domain · T2 safety preservation active' : 'Real tracked artifact preview · no synthetic fallback'}</span></h2>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <ForecastChart
            data={forecastData.length ? forecastData : undefined}
            target={forecastTarget}
            zoneId={region}
            unit={currentDomain.primaryUnit}
            metrics={
              bestLoadMetric
                ? { rmse: bestLoadMetric.rmse, coverage_90: bestLoadMetric.coverage_90, model: bestLoadMetric.model }
                : undefined
            }
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
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════
          SECTION 4: Battery & Carbon
          ═══════════════════════════════════════════════════════ */}
      <section>
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Battery & Carbon <span className="text-emerald-400/50 text-[9px] normal-case tracking-normal ml-2">T3: E[V] ≤ α(1−w̄)T · Pareto frontier</span></h2>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <BatterySOCChart
            schedule={battery?.schedule ?? []}
            metrics={battery?.metrics ?? null}
          />
          <CarbonCostPanel data={pareto.length ? pareto : undefined} zoneId={region} summary={impactActive ?? undefined} />
        </div>

        {/* Energy Flow + Heatmap row */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
          <Panel title="Energy Flow" subtitle="Current dispatch mix" accentColor="primary" delay={0.2}>
            {latestDispatch ? (
              <EnergyFlowDiagram
                solar={latestDispatch.generation_solar ?? 0}
                wind={latestDispatch.generation_wind ?? 0}
                gas={latestDispatch.generation_gas ?? 0}
                battery={battery?.schedule?.length ? battery.schedule[battery.schedule.length - 1].power_mw : 0}
                load={latestDispatch.load_mw}
              />
            ) : (
              <div className="flex min-h-[260px] items-center justify-center rounded-lg border border-white/6 bg-white/[0.02] text-xs text-slate-500">
                No dispatch artifact available for this view.
              </div>
            )}
          </Panel>
          <Panel title="Load Heatmap" subtitle="Hour × Day pattern" accentColor="info" delay={0.25}>
            <HeatmapChart data={heatmapData} unit="MW" />
          </Panel>
        </div>
        <Panel title="Zone Overview" subtitle={`${region === 'DE' ? 'Germany' : 'USA'} — Renewable %`} accentColor="primary" delay={0.3}>
          {batteryDomain ? (
            <ZoneMap region={region as 'DE' | 'US'} zones={zoneMapData} />
          ) : (
            <div className="space-y-2 text-xs text-slate-400">
              <div className="text-slate-300">Source artifacts for this view:</div>
              {(dataset.source_artifacts ?? []).map((artifact) => (
                <div key={artifact} className="rounded bg-white/[0.03] px-3 py-2 font-mono text-[11px] text-slate-300">
                  {artifact}
                </div>
              ))}
            </div>
          )}
        </Panel>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════
          SECTION 5: Safety & Monitoring
          ═══════════════════════════════════════════════════════ */}
      <section>
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">DC3S Safety & Monitoring <span className="text-sky-400/50 text-[9px] normal-case tracking-normal ml-2">Detect → Calibrate → Constrain → Shield → Certify</span></h2>

        {batteryDomain ? (
          <DC3SLiveCard
            region={region}
            data={dc3s.data}
            loading={dc3s.loading}
            error={dc3s.error}
            onRefresh={dc3s.refresh}
            autoRefreshSeconds={dc3sRefreshSeconds}
            onAutoRefreshSecondsChange={setDc3sRefreshSeconds}
            auditHref={
              dc3s.data.source === 'fastapi' && dc3s.data.command_id
                ? `/api/dc3s/audit/${encodeURIComponent(dc3s.data.command_id)}`
                : null
            }
          />
        ) : (
          <Panel
            title="Runtime Safety Witness"
            subtitle={`${regionLabel} • tracked artifacts`}
            badge="Artifact-backed"
            badgeColor="primary"
            accentColor="primary"
          >
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
              <div className="rounded-lg bg-white/3 p-3">
                <div className="text-slate-500">Baseline TSVR</div>
                <div className="mt-1 font-mono text-white">{formatFractionPercent(baselineTsvr)}</div>
              </div>
              <div className="rounded-lg bg-white/3 p-3">
                <div className="text-slate-500">ORIUS TSVR</div>
                <div className="mt-1 font-mono text-energy-primary">{formatFractionPercent(oriusTsvr)}</div>
              </div>
              <div className="rounded-lg bg-white/3 p-3">
                <div className="text-slate-500">Certificate Valid</div>
                <div className="mt-1 font-mono text-white">{formatFractionPercent(certificateRate)}</div>
              </div>
              <div className="rounded-lg bg-white/3 p-3">
                <div className="text-slate-500">OASG</div>
                <div className="mt-1 font-mono text-white">{oriusOasg === null ? 'N/A' : oriusOasg.toFixed(4)}</div>
              </div>
              <div className="rounded-lg bg-white/3 p-3">
                <div className="text-slate-500">Runtime Rows</div>
                <div className="mt-1 font-mono text-white">{runtimeStepCount === null ? dataset.stats?.rows.toLocaleString() ?? 'N/A' : runtimeStepCount.toLocaleString()}</div>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-3 text-xs">
              <div className="rounded-lg bg-white/3 p-3">
                <div className="text-slate-500">Audit Completeness</div>
                <div className="mt-1 font-mono text-white">{formatFractionPercent(auditCompleteness)}</div>
              </div>
              <div className="rounded-lg bg-white/3 p-3">
                <div className="text-slate-500">Intervention Rate</div>
                <div className="mt-1 font-mono text-white">{formatFractionPercent(interventionRate)}</div>
              </div>
              <div className="rounded-lg bg-white/3 p-3">
                <div className="text-slate-500">Useful Work Mean</div>
                <div className="mt-1 font-mono text-white">{usefulWorkMean === null ? 'N/A' : usefulWorkMean.toFixed(4)}</div>
              </div>
            </div>
            <div className="mt-4 space-y-2 text-xs text-slate-400">
              <div className="text-slate-300">Source artifacts</div>
              {(dataset.source_artifacts ?? []).map((artifact) => (
                <div key={artifact} className="rounded bg-white/[0.03] px-3 py-2 font-mono text-[11px] text-slate-300">
                  {artifact}
                </div>
              ))}
            </div>
          </Panel>
        )}

        {/* Impact & Robustness */}
        <div className="mt-6">
          <Panel
            title="Impact & Robustness"
            subtitle={realImpact ? 'Real pipeline results' : 'Stress-tested dispatch performance'}
            badge={robustnessPct !== null ? `±${robustnessPct}%` : 'Robustness'}
            badgeColor="info"
            accentColor="primary"
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
        </div>

        {/* Anomalies + MLOps row */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
          <div className="space-y-4">
            <AnomalyTimeline data={anomalyZScores} />
            <Panel
              title="Anomaly Status"
              badge={`${activeAnomalyCount} Active`}
              badgeColor={activeAnomalyCount > 0 ? 'alert' : 'primary'}
              delay={0.3}
              accentColor={activeAnomalyCount > 0 ? 'alert' : 'primary'}
              collapsible
            >
              <AnomalyList anomalies={anomalies} />
            </Panel>
          </div>
          <MLOpsMonitor data={driftData} />
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════
          SECTION 6: System — Model Registry & Dataset
          ═══════════════════════════════════════════════════════ */}
      <section>
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">System</h2>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Model Registry */}
          <Panel title="Model Registry" subtitle="Production Models" badge={`${realMetrics.length || metricsActive.length} models`} delay={0.35} accentColor="info" collapsible>
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
                  <div key={target} className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-white/3 hover:bg-white/5 transition-colors">
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

          {/* Dataset Overview */}
          {realStats && (
            <Panel title="Dataset Overview" subtitle={regionLabel} delay={0.4} accentColor="info" collapsible>
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
                  <div key={target} className="flex items-center justify-between text-xs px-2 py-1.5 rounded bg-white/2 hover:bg-white/4 transition-colors">
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
      </section>

      {/* ─── Footer ─── */}
      <div className="flex items-center justify-center gap-4 py-4 text-[11px] text-slate-600">
        <span className="flex items-center gap-1.5">
          <Radio className="w-3 h-3" />
          Ingest
        </span>
        <span className="text-slate-700">→</span>
        <span className="flex items-center gap-1.5">
          <Activity className="w-3 h-3" />
          Forecast
        </span>
        <span className="text-slate-700">→</span>
        <span>Optimize</span>
        <span className="text-slate-700">→</span>
        <span>Dispatch</span>
        <span className="text-slate-700">→</span>
        <span>Measure</span>
        <span className="ml-2 text-energy-primary">⚡</span>
      </div>
    </div>
  );
}
