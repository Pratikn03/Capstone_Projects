'use client';

import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { KPICard } from '@/components/ui/KPICard';
import { Leaf, TrendingDown, Factory } from 'lucide-react';
import { mockParetoFrontier } from '@/lib/api/mock-data';

export default function CarbonPage() {
  const pareto = mockParetoFrontier();

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-bold text-white">Carbon Impact Analysis</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPICard
          label="Total Carbon Reduction"
          value="47.8"
          unit="tCO₂"
          change={32.6}
          icon={<Leaf className="w-4 h-4 text-energy-primary" />}
          color="primary"
        />
        <KPICard
          label="Carbon Intensity"
          value="185"
          unit="kgCO₂/MWh"
          change={-18.2}
          changeLabel="vs grid avg"
          icon={<Factory className="w-4 h-4 text-energy-warn" />}
          color="warn"
        />
        <KPICard
          label="Offsetting Cost"
          value="€4,780"
          change={-32.6}
          changeLabel="avoided"
          icon={<TrendingDown className="w-4 h-4 text-energy-info" />}
          color="info"
        />
      </div>

      <CarbonCostPanel data={pareto} zoneId="DE" />
    </div>
  );
}
