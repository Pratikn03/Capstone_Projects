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
import { useDispatchCompare } from '@/lib/api/dispatch-client';

interface ZoneDetailProps {
  zoneId: string;
}

export function ZoneDetail({ zoneId }: ZoneDetailProps) {
  const dataset = useDatasetData(zoneId.toUpperCase() as 'DE' | 'US');
  const liveDispatch = useDispatchCompare(zoneId.toUpperCase(), 24);
  const zone = mockZoneSummaries().find((z) => z.zone_id === zoneId.toUpperCase())
    || mockZoneSummaries()[0];
  const dispatch = mockDispatchForecast(zoneId, 24);
  
  // Prefer real extracted data over static mock payloads.
  const battery = dataset.battery;
  const forecast = dataset.forecast?.['load_mw']?.length
    ? dataset.forecast['load_mw']
    : mockForecastWithPI('load_mw', 48);
  const optimizedDispatch = dataset.dispatch.length
    ? dataset.dispatch.map((d) => ({
        ...d,
        battery_dispatch: 0,
        price_eur_mwh: d.price_eur_mwh ?? undefined,
      }))
    : (liveDispatch.optimized.length ? liveDispatch.optimized : dispatch.data);

  return (
    <div className="space-y-6">
      <GridStatusCard zone={zone} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ForecastChart data={forecast} target="load_mw" zoneId={zoneId.toUpperCase()} />
        <DispatchChart optimized={optimizedDispatch} title={`Dispatch — ${zone.name}`} />
      </div>

      <BatterySOCChart 
        schedule={battery?.schedule ?? []} 
        metrics={battery?.metrics ?? { cost_savings_eur: 0, carbon_reduction_kg: 0, peak_shaving_pct: 0, avg_efficiency: 92 }} 
      />
    </div>
  );
}
