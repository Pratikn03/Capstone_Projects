'use client';

import { useState } from 'react';
import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { Panel } from '@/components/ui/Panel';
import { StatusBanner } from '@/components/ui/StatusBanner';
import { useRegion } from '@/components/ui/RegionContext';
import { useDatasetData } from '@/lib/api/dataset-client';
import { useDispatchCompare } from '@/lib/api/dispatch-client';
import { useReportsData } from '@/lib/api/reports-client';
import { formatCurrency, formatPercent } from '@/lib/utils';
import { ArrowLeftRight, ShieldCheck, ChevronRight, Scale, Activity, Database } from 'lucide-react';
import Link from 'next/link';
import { getDomainOption, isBatteryDomain } from '@/lib/domain-options';

function finiteNumber(value: unknown): number | null {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN;
  return Number.isFinite(parsed) ? parsed : null;
}

function formatMaybeCurrency(value: number | null | undefined): string {
  return value === null || value === undefined ? 'N/A' : formatCurrency(value, 'USD');
}

function formatMaybePercent(value: number | null | undefined): string {
  return value === null || value === undefined ? 'N/A' : formatPercent(value);
}

function formatFractionPercent(value: number | null | undefined): string {
  return value === null || value === undefined ? 'N/A' : formatPercent(value * 100);
}

export default function OptimizationPage() {
  const { region } = useRegion();
  const dataset = useDatasetData(region);
  const { impact: reportsImpact, regions } = useReportsData();
  const { baseline: baselineDispatch } = useDispatchCompare(region);
  const [showBaseline, setShowBaseline] = useState(true);

  // Use real extracted data
  const battery = dataset.battery;
  const pareto = dataset.pareto;
  const currentDomain = getDomainOption(region);
  const regionLabel = currentDomain.label;
  const batteryDomain = isBatteryDomain(region);

  const realImpact = dataset.impact;
  const regionImpact = region === 'DE' || region === 'US' ? regions[region]?.impact : undefined;

  // Use real dispatch data from dataset
  const dispatchData = dataset.dispatch.length
    ? dataset.dispatch.map((d) => ({
        timestamp: d.timestamp,
        load_mw: d.load_mw,
        generation_solar: d.generation_solar,
        generation_wind: d.generation_wind,
        generation_gas: d.generation_gas,
        battery_dispatch: 0,
        price_eur_mwh: d.price_eur_mwh ?? undefined,
      }))
    : [];

  const baselineCost = finiteNumber(realImpact?.baseline_cost_usd);
  const oriusCost = finiteNumber(realImpact?.orius_cost_usd);
  const costSaved =
    baselineCost !== null && oriusCost !== null
      ? baselineCost - oriusCost
      : finiteNumber(regionImpact?.cost_savings_usd ?? reportsImpact?.cost_savings_usd);
  const costSavingsPct = finiteNumber(realImpact?.cost_savings_pct ?? regionImpact?.cost_savings_pct ?? reportsImpact?.cost_savings_pct);
  const carbonReductionPct = finiteNumber(realImpact?.carbon_reduction_pct ?? regionImpact?.carbon_reduction_pct ?? reportsImpact?.carbon_reduction_pct);
  const carbonReductionKg =
    finiteNumber(realImpact?.baseline_carbon_kg) !== null && finiteNumber(realImpact?.orius_carbon_kg) !== null
      ? finiteNumber(realImpact?.baseline_carbon_kg)! - finiteNumber(realImpact?.orius_carbon_kg)!
      : finiteNumber(regionImpact?.carbon_reduction_kg ?? reportsImpact?.carbon_reduction_kg);
  const peakShavingPct = finiteNumber(realImpact?.peak_shaving_pct ?? regionImpact?.peak_shaving_pct ?? reportsImpact?.peak_shaving_pct);
  const peakShavingMw =
    finiteNumber(realImpact?.baseline_peak_mw) !== null && finiteNumber(realImpact?.orius_peak_mw) !== null
      ? finiteNumber(realImpact?.baseline_peak_mw)! - finiteNumber(realImpact?.orius_peak_mw)!
      : finiteNumber(regionImpact?.peak_shaving_mw ?? reportsImpact?.peak_shaving_mw);

  const carbonCostSummary = {
    cost_savings_pct: costSavingsPct,
    cost_savings_usd: costSaved,
    carbon_reduction_pct: carbonReductionPct,
    carbon_reduction_kg: carbonReductionKg,
    peak_shaving_pct: peakShavingPct,
    peak_shaving_mw: peakShavingMw,
  };

  const runtimeRows = (dataset.runtime_summary ?? []) as Array<Record<string, unknown>>;
  const runtimeFor = (controller: string) => runtimeRows.find((row) => row.controller === controller);
  const baselineRuntime = runtimeFor('baseline');
  const oriusRuntime = runtimeFor('orius');
  const runtimeEvidence = [
    { label: 'Baseline TSVR', value: formatFractionPercent(finiteNumber(baselineRuntime?.tsvr)), tone: 'text-sky-300' },
    { label: 'ORIUS TSVR', value: formatFractionPercent(finiteNumber(oriusRuntime?.tsvr)), tone: 'text-emerald-300' },
    { label: 'Certificate Valid', value: formatFractionPercent(finiteNumber(oriusRuntime?.cva)), tone: 'text-emerald-300' },
    { label: 'Audit Complete', value: formatFractionPercent(finiteNumber(oriusRuntime?.audit_completeness)), tone: 'text-emerald-300' },
    { label: 'OASG', value: finiteNumber(oriusRuntime?.oasg) === null ? 'N/A' : String(finiteNumber(oriusRuntime?.oasg)), tone: 'text-amber-300' },
    { label: 'Intervention Rate', value: formatFractionPercent(finiteNumber(oriusRuntime?.intervention_rate)), tone: 'text-violet-300' },
  ];

  const statusMessages = [
    dataset.error ? `Dataset view error: ${dataset.error}` : null,
    ...(dataset.artifact_warnings ?? []),
    batteryDomain && !dataset.loading && !dispatchData.length ? 'No extracted dispatch trace is available; optimization charts are waiting for real artifacts.' : null,
    batteryDomain && !dataset.loading && !pareto.length ? 'No Pareto frontier artifact is available for this battery domain.' : null,
    !batteryDomain ? `${regionLabel} does not use battery dispatch charts; optimization evidence is shown through runtime cost-safety tradeoff metrics.` : null,
    batteryDomain && !dataset.loading && !realImpact ? 'Impact cards are using report-level fallbacks when available.' : null,
  ].filter((message): message is string => Boolean(message));

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Optimization / Cost-Safety Tradeoff</h1>
        <div className="flex items-center gap-3">
          {baselineDispatch && baselineDispatch.length > 0 && (
            <button
              onClick={() => setShowBaseline((v) => !v)}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                showBaseline
                  ? 'border-energy-primary/40 bg-energy-primary/10 text-energy-primary'
                  : 'border-white/10 bg-white/5 text-slate-400 hover:text-white'
              }`}
            >
              <ArrowLeftRight className="w-3.5 h-3.5" />
              {showBaseline ? 'Baseline On' : 'Baseline Off'}
            </button>
          )}
          <div className="flex items-center gap-2 text-xs text-slate-500">
          {batteryDomain ? (
            <>
              <span>Solver: Pyomo + GLPK</span>
              <span>•</span>
              <span>Battery: 20 GWh / 5 GW</span>
              {(realImpact || regionImpact) && (
                <>
                  <span>•</span>
                  <span className="text-energy-primary">
                    Savings: {formatMaybePercent(costSavingsPct)}
                    {costSaved !== null ? ` (${formatCurrency(costSaved, 'USD')})` : ''}
                  </span>
                </>
              )}
            </>
          ) : (
            <>
              <span>Runtime contract</span>
              <span>•</span>
              <span>{currentDomain.shortLabel} cost-safety evidence</span>
              <span>•</span>
              <span className="text-energy-primary">TSVR / OASG / certificate tradeoff</span>
            </>
          )}
        </div>
        </div>
      </div>

      {/* ORIUS Safety Guarantee Context */}
      <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500/5 via-transparent to-transparent border border-emerald-500/10">
        <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
        <span className="text-[11px] text-slate-400">
          <span className="text-emerald-400 font-semibold">T3 Core Bound</span> · E[V] ≤ α(1−w̄)T guarantees bounded safety violations · <span className="text-white/70">T4 No-Free-Safety</span> tradeoff on Pareto frontier · DC3S Stage 3 constraint enforcement
        </span>
        <Link href="/theorems" className="ml-auto flex items-center gap-1 text-[10px] text-emerald-400/70 hover:text-emerald-400 transition-colors shrink-0">
          Theorem ladder <ChevronRight className="w-3 h-3" />
        </Link>
      </div>

      <StatusBanner title="Optimization Status" messages={statusMessages} tone={batteryDomain ? 'warn' : 'info'} />

      {batteryDomain ? (
        <>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <DispatchChart
              optimized={dispatchData}
              baseline={baselineDispatch?.map((d) => ({
                timestamp: d.timestamp,
                load_mw: d.load_mw,
                generation_solar: d.generation_solar,
                generation_wind: d.generation_wind,
                generation_gas: d.generation_gas,
                battery_dispatch: d.battery_dispatch ?? 0,
                price_eur_mwh: d.price_eur_mwh ?? undefined,
              }))}
              showBaseline={showBaseline}
              title={`Optimized Dispatch — ${regionLabel}`}
            />
            <CarbonCostPanel data={pareto.length ? pareto : undefined} zoneId={region} summary={carbonCostSummary} />
          </div>

          <BatterySOCChart
            schedule={battery?.schedule ?? []}
            metrics={battery?.metrics ?? null}
          />
        </>
      ) : (
        <Panel
          title="Runtime Cost-Safety Tradeoff"
          subtitle={`${regionLabel} - domain runtime evidence`}
          badge="No battery dispatch"
          badgeColor="info"
          accentColor="info"
        >
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[0.85fr_1.15fr]">
            <div className="rounded-xl border border-white/[0.06] bg-white/[0.03] p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Scale className="h-4 w-4 text-energy-info" />
                Evidence Scope
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-400">
                {currentDomain.shortLabel} optimization is not a battery dispatch schedule. The relevant cost-safety tradeoff is whether ORIUS reduces true-state violations, preserves certificate validity, and maintains audit completeness under runtime constraints.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Link href="/safety" className="rounded-lg border border-energy-primary/20 bg-energy-primary/10 px-3 py-1.5 text-xs text-energy-primary hover:bg-energy-primary/15">
                  Safety evidence
                </Link>
                <Link href="/domains" className="rounded-lg border border-energy-info/20 bg-energy-info/10 px-3 py-1.5 text-xs text-energy-info hover:bg-energy-info/15">
                  Domain evidence
                </Link>
                <Link href="/reports" className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10">
                  Reports
                </Link>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
              {runtimeEvidence.map((metric) => (
                <div key={metric.label} className="rounded-lg border border-white/[0.06] bg-white/[0.03] p-3">
                  <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-slate-500">
                    <Activity className="h-3 w-3" />
                    {metric.label}
                  </div>
                  <div className={`mt-2 font-mono text-sm font-semibold ${metric.tone}`}>{metric.value}</div>
                </div>
              ))}
            </div>
          </div>
          {dataset.source_artifacts && dataset.source_artifacts.length > 0 && (
            <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-2">
              {dataset.source_artifacts.slice(0, 4).map((artifact) => (
                <div key={artifact} className="flex items-center gap-2 truncate rounded-lg bg-black/15 px-3 py-2 font-mono text-[11px] text-slate-300" title={artifact}>
                  <Database className="h-3.5 w-3.5 shrink-0 text-energy-info" />
                  <span className="truncate">{artifact}</span>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}

      <Panel title="Optimization Parameters" subtitle="Current Configuration">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          {[
            { label: 'Capacity', value: '20,000 MWh' },
            { label: 'Max Power', value: '5,000 MW' },
            { label: 'Efficiency', value: '92%' },
            { label: 'Carbon Weight', value: '20' },
            { label: 'SOC Min', value: '10%' },
            { label: 'SOC Max', value: '90%' },
            { label: 'Ramp Limit', value: '2,500 MW/h' },
            { label: 'Objective', value: 'Balanced' },
          ].map((p) => (
            <div key={p.label} className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">{p.label}</div>
              <div className="text-white font-mono font-medium">{p.value}</div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Cost Impact Summary */}
      {batteryDomain && (realImpact || regionImpact) && (
        <Panel title="Cost Impact Summary" subtitle={`${regionLabel} — real pipeline results`}>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-xs">
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Baseline Cost</div>
              <div className="text-white font-mono font-medium">
                {formatMaybeCurrency(baselineCost)}
              </div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">ORIUS Cost</div>
              <div className="text-energy-primary font-mono font-medium">
                {formatMaybeCurrency(oriusCost)}
              </div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Net Savings</div>
              <div className="text-energy-primary font-mono font-medium">
                {formatMaybeCurrency(costSaved)}
                {costSavingsPct !== null ? ` (${costSavingsPct.toFixed(2)}%)` : ''}
              </div>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}
