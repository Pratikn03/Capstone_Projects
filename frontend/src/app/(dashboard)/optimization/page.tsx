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
import { formatCurrency } from '@/lib/utils';
import { ArrowLeftRight, ShieldCheck, ChevronRight } from 'lucide-react';
import Link from 'next/link';
import { getDomainOption } from '@/lib/domain-options';

export default function OptimizationPage() {
  const { region } = useRegion();
  const dataset = useDatasetData(region);
  const { regions } = useReportsData();
  const { baseline: baselineDispatch } = useDispatchCompare(region);
  const [showBaseline, setShowBaseline] = useState(true);

  // Use real extracted data
  const battery = dataset.battery;
  const pareto = dataset.pareto;
  const regionLabel = getDomainOption(region).label;

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

  const costSavingsPct = realImpact?.cost_savings_pct ?? regionImpact?.cost_savings_pct ?? null;
  const baselineCost = realImpact?.baseline_cost_usd ?? null;
  const oriusCost = realImpact?.orius_cost_usd ?? null;
  const costSaved = baselineCost !== null && oriusCost !== null ? baselineCost - oriusCost : null;
  const statusMessages = [
    dataset.error ? `Dataset view error: ${dataset.error}` : null,
    ...(dataset.artifact_warnings ?? []),
    !dataset.loading && !dispatchData.length ? 'No extracted dispatch trace is available; optimization charts are waiting for real artifacts.' : null,
    !dataset.loading && !realImpact ? 'Impact cards are using report-level fallbacks when available.' : null,
  ].filter((message): message is string => Boolean(message));

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Dispatch Optimization</h1>
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
          <span>Solver: Pyomo + GLPK</span>
          <span>•</span>
          <span>Battery: 20 GWh / 5 GW</span>
          {realImpact && (
            <>
              <span>•</span>
              <span className="text-energy-primary">
                Savings: {costSavingsPct !== null ? `${costSavingsPct.toFixed(2)}%` : 'N/A'}
                {costSaved !== null ? ` (${formatCurrency(costSaved, 'USD')})` : ''}
              </span>
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

      <StatusBanner title="Optimization Status" messages={statusMessages} />

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
        <CarbonCostPanel data={pareto.length ? pareto : undefined} zoneId={region} />
      </div>

      <BatterySOCChart 
        schedule={battery?.schedule ?? []} 
        metrics={battery?.metrics ?? null}
      />

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
      {realImpact && (
        <Panel title="Cost Impact Summary" subtitle={`${regionLabel} — real pipeline results`}>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-xs">
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Baseline Cost</div>
              <div className="text-white font-mono font-medium">
                {baselineCost !== null ? formatCurrency(baselineCost, 'USD') : 'N/A'}
              </div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">ORIUS Cost</div>
              <div className="text-energy-primary font-mono font-medium">
                {oriusCost !== null ? formatCurrency(oriusCost, 'USD') : 'N/A'}
              </div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Net Savings</div>
              <div className="text-energy-primary font-mono font-medium">
                {costSaved !== null ? formatCurrency(costSaved, 'USD') : 'N/A'}
                {costSavingsPct !== null ? ` (${costSavingsPct.toFixed(2)}%)` : ''}
              </div>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}
