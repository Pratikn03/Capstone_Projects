'use client';

import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { GridStatusCard } from '@/components/ai/tools/GridStatus';
import { StatusBanner } from '@/components/ui/StatusBanner';
import { useDatasetData } from '@/lib/api/dataset-client';
import { useDispatchCompare } from '@/lib/api/dispatch-client';

interface ZoneDetailProps {
  zoneId: string;
}

export function ZoneDetail({ zoneId }: ZoneDetailProps) {
  const normalizedZone = zoneId.toUpperCase() as 'DE' | 'US';
  const dataset = useDatasetData(normalizedZone);
  const liveDispatch = useDispatchCompare(normalizedZone, 24);
  const latestPoint = dataset.timeseries.at(-1);
  const latestLoad = latestPoint?.load_mw ?? 0;
  const latestRenewables = (latestPoint?.wind_mw ?? 0) + (latestPoint?.solar_mw ?? 0);
  const zone = {
    zone_id: normalizedZone,
    name: dataset.stats?.label ?? normalizedZone,
    status: dataset.error ? 'warning' as const : 'normal' as const,
    load_mw: latestLoad,
    generation_mw: latestRenewables,
    renewable_pct: latestLoad > 0 ? (latestRenewables / latestLoad) * 100 : 0,
    frequency_hz: normalizedZone === 'US' ? 60.0 : 50.0,
    anomaly_count: dataset.anomalies.length,
  };
  
  const battery = dataset.battery;
  const forecast = dataset.forecast?.['load_mw']?.length ? dataset.forecast['load_mw'] : undefined;
  const optimizedDispatch = dataset.dispatch.length
    ? dataset.dispatch.map((d) => ({
        ...d,
        battery_dispatch: 0,
        price_eur_mwh: d.price_eur_mwh ?? undefined,
      }))
    : liveDispatch.optimized;
  const statusMessages = [
    dataset.error ? `Dataset view error: ${dataset.error}` : null,
    !forecast?.length ? 'No forecast artifact is available for this zone.' : null,
    !optimizedDispatch.length ? 'No dispatch artifact or live optimization response is available for this zone.' : null,
  ].filter((message): message is string => Boolean(message));

  return (
    <div className="space-y-6">
      <StatusBanner title="Zone Status" messages={statusMessages} />
      <GridStatusCard zone={zone} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ForecastChart data={forecast} target="load_mw" zoneId={normalizedZone} />
        <DispatchChart optimized={optimizedDispatch} title={`Dispatch — ${zone.name}`} />
      </div>

      <BatterySOCChart 
        schedule={battery?.schedule ?? []} 
        metrics={battery?.metrics ?? { cost_savings_eur: 0, carbon_reduction_kg: 0, peak_shaving_pct: 0, avg_efficiency: 92 }} 
      />
    </div>
  );
}
