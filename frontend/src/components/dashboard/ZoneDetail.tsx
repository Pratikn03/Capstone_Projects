'use client';

import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { GridStatusCard } from '@/components/ai/tools/GridStatus';
import {
  mockDispatchForecast,
  mockForecastWithPI,
  mockZoneSummaries,
} from '@/lib/api/mock-data';
import { useDatasetData } from '@/lib/api/dataset-client';

interface ZoneDetailProps {
  zoneId: string;
}

export function ZoneDetail({ zoneId }: ZoneDetailProps) {
  const dataset = useDatasetData(zoneId.toUpperCase() as 'DE' | 'US');
  const zone = mockZoneSummaries().find((z) => z.zone_id === zoneId.toUpperCase())
    || mockZoneSummaries()[0];
  const dispatch = mockDispatchForecast(zoneId, 24);
  
  // Use real extracted data
  const battery = dataset.battery;
  const forecast = mockForecastWithPI('load_mw', 48);

  return (
    <div className="space-y-6">
      <GridStatusCard zone={zone} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ForecastChart data={forecast} target="load_mw" zoneId={zoneId.toUpperCase()} />
        <DispatchChart optimized={dispatch.data} title={`Dispatch — ${zone.name}`} />
      </div>

      <BatterySOCChart 
        schedule={battery?.schedule ?? []} 
        metrics={battery?.metrics ?? { cost_savings_eur: 0, carbon_reduction_kg: 0, peak_shaving_pct: 0, avg_efficiency: 92 }} 
      />
    </div>
  );
}
