'use client';

import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { KPICard } from '@/components/ui/KPICard';
import { Leaf, TrendingDown, Factory } from 'lucide-react';
import { mockParetoFrontier } from '@/lib/api/mock-data';
import { useReportsData } from '@/lib/api/reports-client';
import { useDatasetData } from '@/lib/api/dataset-client';
import { useRegion } from '@/components/ui/RegionContext';
import { Panel } from '@/components/ui/Panel';
import { formatCurrency } from '@/lib/utils';

export default function CarbonPage() {
  const pareto = mockParetoFrontier();
  const { region, setRegion } = useRegion();
  const { impact: reportsImpact, regions } = useReportsData();
  const dataset = useDatasetData(region as 'DE' | 'US');
  const realImpact = dataset.impact;
  const regionImpact = regions[region]?.impact;

  const carbonReductionPct = realImpact?.carbon_reduction_pct ?? regionImpact?.carbon_reduction_pct ?? reportsImpact?.carbon_reduction_pct ?? null;
  const baselineCarbon = realImpact?.baseline_carbon_kg ?? null;
  const gridpulseCarbon = realImpact?.gridpulse_carbon_kg ?? null;
  const carbonReductionKg = baselineCarbon !== null && gridpulseCarbon !== null ? baselineCarbon - gridpulseCarbon : null;
  const carbonTons = carbonReductionKg !== null ? carbonReductionKg / 1000 : null;
  const carbonIntensity = baselineCarbon !== null && dataset.stats?.rows
    ? baselineCarbon / (dataset.stats.rows || 1) / 1000  // rough estimate kgCO2/MWh
    : null;
  const regionLabel = region === 'US' ? 'USA (EIA-930)' : 'Germany (OPSD)';

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Carbon Impact Analysis</h1>
        <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/5 border border-white/10">
          {[
            { id: 'DE', label: '🇩🇪 Germany' },
            { id: 'US', label: '🇺🇸 USA' },
          ].map((r) => (
            <button
              key={r.id}
              onClick={() => setRegion(r.id as 'DE' | 'US')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                region === r.id
                  ? 'bg-energy-info/20 text-energy-info'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPICard
          label="Total Carbon Reduction"
          value={carbonTons !== null ? carbonTons.toFixed(0) : 'N/A'}
          unit="tCO₂"
          change={carbonReductionPct ?? undefined}
          icon={<Leaf className="w-4 h-4 text-energy-primary" />}
          color="primary"
        />
        <KPICard
          label="Baseline Carbon"
          value={baselineCarbon !== null ? (baselineCarbon / 1e6).toFixed(1) : 'N/A'}
          unit="ktCO₂"
          icon={<Factory className="w-4 h-4 text-energy-warn" />}
          color="warn"
        />
        <KPICard
          label="Optimised Carbon"
          value={gridpulseCarbon !== null ? (gridpulseCarbon / 1e6).toFixed(1) : 'N/A'}
          unit="ktCO₂"
          change={carbonReductionPct ? -carbonReductionPct : undefined}
          changeLabel="reduction"
          icon={<TrendingDown className="w-4 h-4 text-energy-info" />}
          color="info"
        />
      </div>

      <CarbonCostPanel data={pareto} zoneId={region} />

      {/* Carbon breakdown */}
      {realImpact && (
        <Panel title="Carbon Impact Breakdown" subtitle={regionLabel}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Baseline Emissions</div>
              <div className="text-white font-mono font-medium">
                {baselineCarbon !== null ? `${(baselineCarbon / 1e6).toFixed(2)} Mt CO₂` : 'N/A'}
              </div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">GridPulse Emissions</div>
              <div className="text-energy-primary font-mono font-medium">
                {gridpulseCarbon !== null ? `${(gridpulseCarbon / 1e6).toFixed(2)} Mt CO₂` : 'N/A'}
              </div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Reduction</div>
              <div className="text-energy-primary font-mono font-medium">
                {carbonReductionPct !== null ? `${carbonReductionPct.toFixed(2)}%` : 'N/A'}
              </div>
            </div>
            <div className="px-3 py-2.5 rounded-lg bg-white/3">
              <div className="text-slate-500 mb-1">Avoided (tCO₂)</div>
              <div className="text-energy-primary font-mono font-medium">
                {carbonTons !== null ? carbonTons.toLocaleString(undefined, { maximumFractionDigits: 0 }) : 'N/A'}
              </div>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}
