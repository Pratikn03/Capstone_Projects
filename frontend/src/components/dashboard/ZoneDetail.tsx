'use client';

import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { GridStatusCard } from '@/components/ai/tools/GridStatus';
import {
  mockDispatchForecast,
  mockBatterySchedule,
  mockForecastWithPI,
  mockZoneSummaries,
} from '@/lib/api/mock-data';

interface ZoneDetailProps {
  zoneId: string;
}

export function ZoneDetail({ zoneId }: ZoneDetailProps) {
  const zone = mockZoneSummaries().find((z) => z.zone_id === zoneId.toUpperCase())
    || mockZoneSummaries()[0];
  const dispatch = mockDispatchForecast(zoneId, 24);
  const battery = mockBatterySchedule(zoneId);
  const forecast = mockForecastWithPI('load_mw', 48);

  return (
    <div className="space-y-6">
      <GridStatusCard zone={zone} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ForecastChart data={forecast} target="load_mw" zoneId={zoneId.toUpperCase()} />
        <DispatchChart data={dispatch.data} title={`Dispatch — ${zone.name}`} />
      </div>

      <BatterySOCChart schedule={battery.schedule} metrics={battery.metrics} />
    </div>
  );
}
