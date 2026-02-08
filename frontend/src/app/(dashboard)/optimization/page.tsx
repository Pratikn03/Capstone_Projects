'use client';

import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { Panel } from '@/components/ui/Panel';
import { useRegion } from '@/components/ui/RegionContext';
import {
  mockDispatchForecast,
  mockBatterySchedule,
  mockParetoFrontier,
} from '@/lib/api/mock-data';

export default function OptimizationPage() {
  const { region } = useRegion();
  const dispatch = mockDispatchForecast(region, 24);
  const battery = mockBatterySchedule(region);
  const pareto = mockParetoFrontier();
  const regionLabel = region === 'US' ? 'USA' : 'Germany';

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Dispatch Optimization</h1>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>Solver: Pyomo + GLPK</span>
          <span>•</span>
          <span>Battery: 20 GWh / 5 GW</span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <DispatchChart optimized={dispatch.data} title={`Optimized Dispatch — ${regionLabel}`} />
        <CarbonCostPanel data={pareto} zoneId={region} />
      </div>

      <BatterySOCChart schedule={battery.schedule} metrics={battery.metrics} />

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
    </div>
  );
}
