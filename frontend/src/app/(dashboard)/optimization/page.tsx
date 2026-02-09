'use client';

import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { Panel } from '@/components/ui/Panel';
import { useRegion } from '@/components/ui/RegionContext';
import { useDatasetData } from '@/lib/api/dataset-client';
import { useReportsData } from '@/lib/api/reports-client';
import { formatCurrency } from '@/lib/utils';

export default function OptimizationPage() {
  const { region } = useRegion();
  const dataset = useDatasetData(region as 'DE' | 'US');
  const { impact: reportsImpact, regions } = useReportsData();
  
  // Use real extracted data
  const battery = dataset.battery;
  const pareto = dataset.pareto;
  const regionLabel = region === 'US' ? 'USA' : 'Germany';

  const realImpact = dataset.impact;
  const regionImpact = regions[region]?.impact;

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
  const gridpulseCost = realImpact?.gridpulse_cost_usd ?? null;
  const costSaved = baselineCost !== null && gridpulseCost !== null ? baselineCost - gridpulseCost : null;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Dispatch Optimization</h1>
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

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <DispatchChart
          optimized={dispatchData}
          title={`Optimized Dispatch — ${regionLabel}`}
        />
        <CarbonCostPanel data={pareto.length ? pareto : undefined} zoneId={region} />
      </div>

      <BatterySOCChart 
        schedule={battery?.schedule ?? []} 
        metrics={battery?.metrics ?? { cost_savings_eur: 0, carbon_reduction_kg: 0, peak_shaving_pct: 0, avg_efficiency: 92 }} 
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
              <div className="text-slate-500 mb-1">GridPulse Cost</div>
              <div className="text-energy-primary font-mono font-medium">
                {gridpulseCost !== null ? formatCurrency(gridpulseCost, 'USD') : 'N/A'}
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
