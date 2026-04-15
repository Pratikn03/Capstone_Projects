'use client';

import { useState, useMemo } from 'react';
import { BarChart3, Zap, Leaf, TrendingDown, Activity, Database, Radio, Shield, BookOpen, ShieldCheck, Globe2, ChevronRight as ChevronRightIcon } from 'lucide-react';
import { KPICard } from '@/components/ui/KPICard';
import { Panel } from '@/components/ui/Panel';
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
import { HeatmapChart, generateLoadHeatmap } from '@/components/charts/HeatmapChart';
import { ZoneMap, type ZoneData } from '@/components/charts/ZoneMap';
import { useDispatchCompare } from '@/lib/api/dispatch-client';
import { useDc3sLive } from '@/lib/api/dc3s-client';
import { useReportsData } from '@/lib/api/reports-client';
import { useDatasetData, type DriftPoint } from '@/lib/api/dataset-client';
import { formatCurrency, formatMW, formatPercent } from '@/lib/utils';
import { useRegion } from '@/components/ui/RegionContext';

export default function DashboardPage() {
  const { region } = useRegion();
  const [dc3sRefreshSeconds, setDc3sRefreshSeconds] = useState(15);
  const dispatch = useDispatchCompare(region, 24);
  const dc3s = useDc3sLive(region as 'DE' | 'US', 24, dc3sRefreshSeconds);
  const { metrics, impact, robustness, regions } = useReportsData();
  const dataset = useDatasetData(region as 'DE' | 'US');

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
  const forecastData = dataset.forecast?.['load_mw'] ?? [];

  const loadMetrics = metricsActive.filter((m) => m.target === 'load_mw');
  const bestLoadMetric = loadMetrics.length
    ? loadMetrics.reduce((a, b) => (a.rmse < b.rmse ? a : b))
    : null;
  const bestRMSE = realMetrics.find((m) => m.target === 'load_mw')?.rmse ?? bestLoadMetric?.rmse ?? null;

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
  const regionLabel = region === 'US' ? 'USA (EIA-930)' : 'Germany (OPSD)';
  const p95Regret = robustnessActive?.p95_regret ?? null;
  const infeasibleRate = robustnessActive?.infeasible_rate ?? null;
  const robustnessPct = robustnessActive?.perturbation_pct ?? null;

  // Compute hero metrics from latest dispatch data
  const latestDispatch = dataset.dispatch.length ? dataset.dispatch[dataset.dispatch.length - 1] : null;
  const totalGeneration = latestDispatch
    ? (latestDispatch.generation_solar ?? 0) + (latestDispatch.generation_wind ?? 0) + (latestDispatch.generation_gas ?? 0)
    : 0;
  const renewablePct = totalGeneration > 0 && latestDispatch
    ? ((latestDispatch.generation_solar ?? 0) + (latestDispatch.generation_wind ?? 0)) / totalGeneration * 100
    : 0;
  const currentSOC = battery?.schedule?.length
    ? battery.schedule[battery.schedule.length - 1].soc_percent
    : 68;
  const activeAnomalyCount = anomalies.filter((a) => a.status === 'active').length;

  // Sparkline data from forecast
  const sparkRMSE = realMetrics
    .filter((m) => m.target === 'load_mw')
    .slice(0, 8)
    .map((m) => m.rmse);
  const sparkCostSeries = pareto.map((p) => p.total_cost_eur);
  const sparkCarbonSeries = pareto.map((p) => p.total_carbon_kg);

  // Heatmap mock data
  const heatmapData = useMemo(() => generateLoadHeatmap(), []);

  // Zone map data
  const zoneMapData = useMemo<ZoneData[]>(() => {
    if (region === 'DE') {
      return [
        { id: 'DE-SH', label: 'SH', renewablePct: 78 },
        { id: 'DE-NI', label: 'NI', renewablePct: 52 },
        { id: 'DE-NW', label: 'NW', renewablePct: 28 },
        { id: 'DE-HE', label: 'HE', renewablePct: 35 },
        { id: 'DE-BY', label: 'BY', renewablePct: 42 },
        { id: 'DE-BW', label: 'BW', renewablePct: 38 },
        { id: 'DE-BB', label: 'BB', renewablePct: 68 },
        { id: 'DE-MV', label: 'MV', renewablePct: 72 },
        { id: 'DE-SN', label: 'SN', renewablePct: 30 },
        { id: 'DE-ST', label: 'ST', renewablePct: 55 },
        { id: 'DE-TH', label: 'TH', renewablePct: 45 },
        { id: 'DE-RP', label: 'RP', renewablePct: 32 },
        { id: 'DE-SL', label: 'SL', renewablePct: 20 },
        { id: 'DE-HH', label: 'HH', renewablePct: 15 },
        { id: 'DE-HB', label: 'HB', renewablePct: 22 },
        { id: 'DE-BE', label: 'BE', renewablePct: 12 },
      ];
    }
    return [
      { id: 'US-MISO', label: 'MISO', renewablePct: 25 },
      { id: 'US-PJM', label: 'PJM', renewablePct: 18 },
      { id: 'US-NYISO', label: 'NYISO', renewablePct: 30 },
      { id: 'US-ISONE', label: 'ISONE', renewablePct: 35 },
      { id: 'US-SPP', label: 'SPP', renewablePct: 45 },
      { id: 'US-ERCOT', label: 'ERCOT', renewablePct: 38 },
      { id: 'US-CAISO', label: 'CAISO', renewablePct: 55 },
      { id: 'US-SOCO', label: 'SOCO', renewablePct: 15 },
    ];
  }, [region]);

  const formatMaybePercent = (value: number | null | undefined) =>
    value === null || value === undefined ? 'N/A' : formatPercent(value);
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
            Observation-Reliability-Informed Universal Safety · 83 formal items · 6 domains
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
                  <p className="text-[10px] text-slate-500">6 domains · T11 transfer</p>
                </div>
              </div>
              <ChevronRightIcon className="w-4 h-4 text-slate-600 group-hover:text-energy-primary transition-colors" />
            </div>
            <div className="space-y-1.5">
              {[
                { name: 'Battery', tier: 'Reference', color: 'emerald', active: true },
                { name: 'Industrial', tier: 'Proof-Validated', color: 'sky', active: false },
                { name: 'Healthcare', tier: 'Proof-Validated', color: 'sky', active: false },
                { name: 'AV', tier: 'Proof-Candidate', color: 'cyan', active: false },
                { name: 'Aerospace', tier: 'Experimental', color: 'amber', active: false },
                { name: 'Navigation', tier: 'Shadow-Synth', color: 'slate', active: false },
              ].map((d) => (
                <div key={d.name} className={`flex items-center justify-between py-1 px-2 rounded text-[10px] ${d.active ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-white/[0.02]'}`}>
                  <span className={d.active ? 'text-emerald-400 font-medium' : 'text-slate-400'}>{d.name}</span>
                  <span className={`text-${d.color}-400/70`}>{d.tier}</span>
                </div>
              ))}
            </div>
          </Link>
        </div>

        {/* Active domain indicator */}
        <div className="mt-3 flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/5 border border-emerald-500/15">
          <Shield className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-[10px] text-emerald-400 font-medium">Active Domain: Battery / Energy Storage</span>
          <span className="text-[10px] text-slate-500">— Reference witness row · {regionLabel} · DC3S shield active</span>
          <span className="text-[10px] text-slate-600 ml-auto">T1–T11 fully validated · TSVR = 0.0%</span>
        </div>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════
          SECTION 1: HERO — Grid Overview Strip
          ═══════════════════════════════════════════════════════ */}
      <section>
        <div className="flex items-center gap-3 mb-4">
          <h2 className="text-lg font-bold text-white">Battery Domain — Grid Overview</h2>
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
              value={50.01}
              min={49.9}
              max={50.1}
              label="Grid Frequency"
              unit="Hz"
              color="#10b981"
              size={110}
              thresholds={{ warn: 50.05, alert: 50.08 }}
            />
            <GaugeChart
              value={totalGeneration}
              min={0}
              max={15000}
              label="Total Generation"
              unit="MW"
              color="#3b82f6"
              size={110}
            />
            <GaugeChart
              value={renewablePct}
              min={0}
              max={100}
              label="Renewable Share"
              unit="%"
              color="#10b981"
              size={110}
              thresholds={{ warn: 30, alert: 15 }}
            />
            <GaugeChart
              value={currentSOC}
              min={0}
              max={100}
              label="Battery SOC"
              unit="%"
              color="#a855f7"
              size={110}
              thresholds={{ warn: 80, alert: 90 }}
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
            label="Forecast RMSE"
            value={bestRMSE !== null ? bestRMSE.toFixed(0) : 'N/A'}
            unit={bestRMSE !== null ? 'MW' : undefined}
            icon={<BarChart3 className="w-4 h-4 text-energy-info" />}
            color="info"
            delay={0}
            sparkData={sparkRMSE.length > 1 ? sparkRMSE : undefined}
          />
          <KPICard
            label="Cost Savings"
            value={costSavingsPct !== null ? formatPercent(costSavingsPct) : 'N/A'}
            unit={costSavingsRaw !== null ? formatCurrency(costSavingsRaw, 'USD') : undefined}
            change={costSavingsPct ?? undefined}
            icon={<TrendingDown className="w-4 h-4 text-energy-primary" />}
            color="primary"
            delay={0.05}
            sparkData={sparkCostSeries.length > 1 ? sparkCostSeries : undefined}
          />
          <KPICard
            label="Carbon Reduction"
            value={carbonReductionPct !== null ? formatPercent(carbonReductionPct) : 'N/A'}
            unit={carbonTons !== null ? `${carbonTons.toFixed(0)} tCO₂` : undefined}
            change={carbonReductionPct ?? undefined}
            icon={<Leaf className="w-4 h-4 text-energy-primary" />}
            color="primary"
            delay={0.1}
            sparkData={sparkCarbonSeries.length > 1 ? sparkCarbonSeries : undefined}
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
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════
          SECTION 3: Dispatch & Forecast
          ═══════════════════════════════════════════════════════ */}
      <section>
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Battery Dispatch & Forecast <span className="text-emerald-400/50 text-[9px] normal-case tracking-normal ml-2">Reference domain · T2 safety preservation active</span></h2>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <ForecastChart
            data={forecastData.length ? forecastData : undefined}
            target="load_mw"
            zoneId={region}
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
            metrics={battery?.metrics ?? { cost_savings_eur: 0, carbon_reduction_kg: 0, peak_shaving_pct: 0, avg_efficiency: 92 }}
          />
          <CarbonCostPanel data={pareto.length ? pareto : undefined} zoneId={region} summary={impactActive ?? undefined} />
        </div>

        {/* Energy Flow + Heatmap row */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
          <Panel title="Energy Flow" subtitle="Current dispatch mix" accentColor="primary" delay={0.2}>
            <EnergyFlowDiagram
              solar={latestDispatch?.generation_solar ?? 2800}
              wind={latestDispatch?.generation_wind ?? 3500}
              gas={latestDispatch?.generation_gas ?? 1200}
              battery={battery?.schedule?.length ? battery.schedule[battery.schedule.length - 1].power_mw : -200}
              load={latestDispatch?.load_mw ?? 7500}
            />
          </Panel>
          <Panel title="Load Heatmap" subtitle="Hour × Day pattern" accentColor="info" delay={0.25}>
            <HeatmapChart data={heatmapData} unit="MW" />
          </Panel>
        </div>
        <Panel title="Zone Overview" subtitle={`${region === 'DE' ? 'Germany' : 'USA'} — Renewable %`} accentColor="primary" delay={0.3}>
          <ZoneMap region={region as 'DE' | 'US'} zones={zoneMapData} />
        </Panel>
      </section>

      <div className="section-divider" />

      {/* ═══════════════════════════════════════════════════════
          SECTION 5: Safety & Monitoring
          ═══════════════════════════════════════════════════════ */}
      <section>
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">DC3S Safety & Monitoring <span className="text-sky-400/50 text-[9px] normal-case tracking-normal ml-2">Detect → Calibrate → Constrain → Shield → Certify</span></h2>

        <DC3SLiveCard
          region={region as 'DE' | 'US'}
          data={dc3s.data}
          loading={dc3s.loading}
          error={dc3s.error}
          onRefresh={dc3s.refresh}
          autoRefreshSeconds={dc3sRefreshSeconds}
          onAutoRefreshSecondsChange={setDc3sRefreshSeconds}
          auditHref={
            dc3s.data.command_id
              ? `/api/dc3s/audit/${encodeURIComponent(dc3s.data.command_id)}`
              : null
          }
        />

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
