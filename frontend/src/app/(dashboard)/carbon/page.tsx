'use client';

import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { KPICard } from '@/components/ui/KPICard';
import { Leaf, TrendingDown, Factory } from 'lucide-react';
import { useReportsData } from '@/lib/api/reports-client';
import { useDatasetData } from '@/lib/api/dataset-client';
import { useRegion } from '@/components/ui/RegionContext';
import { Panel } from '@/components/ui/Panel';
import { StatusBanner } from '@/components/ui/StatusBanner';
import Link from 'next/link';
import { ShieldCheck, ChevronRight } from 'lucide-react';
import { getDomainOption } from '@/lib/domain-options';

export default function CarbonPage() {
  const { region, setRegion } = useRegion();
  const { impact: reportsImpact, regions } = useReportsData();
  const dataset = useDatasetData(region);
  
  // Use real extracted data
  const pareto = dataset.pareto;
  const realImpact = dataset.impact;
  const regionImpact = region === 'DE' || region === 'US' ? regions[region]?.impact : undefined;

  const carbonReductionPct = realImpact?.carbon_reduction_pct ?? regionImpact?.carbon_reduction_pct ?? reportsImpact?.carbon_reduction_pct ?? null;
  const baselineCarbon = realImpact?.baseline_carbon_kg ?? null;
  const oriusCarbon = realImpact?.orius_carbon_kg ?? null;
  const carbonReductionKg = baselineCarbon !== null && oriusCarbon !== null ? baselineCarbon - oriusCarbon : null;
  const carbonTons = carbonReductionKg !== null ? carbonReductionKg / 1000 : null;
  const regionLabel = getDomainOption(region).label;
  const statusMessages = [
    dataset.error ? `Dataset view error: ${dataset.error}` : null,
    ...(dataset.artifact_warnings ?? []),
    !dataset.loading && !realImpact ? 'Carbon impact metrics are falling back to report-level summaries where available.' : null,
    !dataset.loading && !pareto.length ? 'No Pareto frontier artifact is available for this region.' : null,
  ].filter((message): message is string => Boolean(message));

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

      <StatusBanner title="Carbon View Status" messages={statusMessages} />

      {/* ORIUS Tradeoff Context */}
      <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-gradient-to-r from-green-500/5 via-transparent to-transparent border border-green-500/10">
        <ShieldCheck className="w-4 h-4 text-green-400 shrink-0" />
        <span className="text-[11px] text-slate-400">
          <span className="text-green-400 font-semibold">T4 No-Free-Safety</span> · Carbon–cost Pareto frontier quantifies the irreducible safety–performance tradeoff · <span className="text-white/70">T5 monotonic tightening</span> ensures bounds improve with data · Battery domain reference witness
        </span>
        <Link href="/theorems" className="ml-auto flex items-center gap-1 text-[10px] text-green-400/70 hover:text-green-400 transition-colors shrink-0">
          Theorem ladder <ChevronRight className="w-3 h-3" />
        </Link>
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
          value={oriusCarbon !== null ? (oriusCarbon / 1e6).toFixed(1) : 'N/A'}
          unit="ktCO₂"
          change={carbonReductionPct ? -carbonReductionPct : undefined}
          changeLabel="reduction"
          icon={<TrendingDown className="w-4 h-4 text-energy-info" />}
          color="info"
        />
      </div>

      <CarbonCostPanel data={pareto.length ? pareto : undefined} zoneId={region} />

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
              <div className="text-slate-500 mb-1">ORIUS Emissions</div>
              <div className="text-energy-primary font-mono font-medium">
                {oriusCarbon !== null ? `${(oriusCarbon / 1e6).toFixed(2)} Mt CO₂` : 'N/A'}
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
