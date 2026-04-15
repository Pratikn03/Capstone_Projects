'use client';

import { DispatchChart } from '@/components/ai/tools/DispatchChart';
import { BatterySOCChart } from '@/components/ai/tools/BatterySOCChart';
import { ForecastChart } from '@/components/ai/tools/ForecastChart';
import { GridStatusCard } from '@/components/ai/tools/GridStatus';
import { CarbonCostPanel } from '@/components/ai/tools/CarbonCostPanel';
import { StatusBanner } from '@/components/ui/StatusBanner';
import { KPICard } from '@/components/ui/KPICard';
import { Panel } from '@/components/ui/Panel';
import { useDatasetData } from '@/lib/api/dataset-client';
import { useDispatchCompare } from '@/lib/api/dispatch-client';
import { AlertTriangle, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

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
      <div className="flex items-center gap-3">
        <Link href="/" className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
          <ArrowLeft className="w-4 h-4 text-slate-400" />
        </Link>
        <StatusBanner title="Zone Status" messages={statusMessages} />
      </div>

      <GridStatusCard zone={zone} />

      {/* Zone KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard label="Load" value={`${(zone.load_mw / 1000).toFixed(1)} GW`} color="info" />
        <KPICard label="Renewable %" value={`${zone.renewable_pct.toFixed(1)}%`} color="primary" />
        <KPICard label="Frequency" value={`${zone.frequency_hz.toFixed(2)} Hz`} color="warn" />
        <KPICard label="Anomalies" value={`${zone.anomaly_count}`} color={zone.anomaly_count > 0 ? 'alert' : 'primary'} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ForecastChart data={forecast} target="load_mw" zoneId={normalizedZone} />
        <DispatchChart
          optimized={optimizedDispatch}
          baseline={liveDispatch.baseline?.map((d) => ({
            timestamp: d.timestamp,
            load_mw: d.load_mw,
            generation_solar: d.generation_solar,
            generation_wind: d.generation_wind,
            generation_gas: d.generation_gas,
            battery_dispatch: d.battery_dispatch ?? 0,
            price_eur_mwh: d.price_eur_mwh ?? undefined,
          }))}
          showBaseline={true}
          title={`Dispatch — ${zone.name}`}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <BatterySOCChart 
          schedule={battery?.schedule ?? []} 
          metrics={battery?.metrics ?? { cost_savings_eur: 0, carbon_reduction_kg: 0, peak_shaving_pct: 0, avg_efficiency: 92 }} 
        />
        <CarbonCostPanel data={dataset.pareto.length ? dataset.pareto : undefined} zoneId={normalizedZone} />
      </div>

      {/* Anomaly list */}
      {dataset.anomalies.length > 0 && (
        <Panel title="Zone Anomalies" subtitle={`${dataset.anomalies.length} detected`} accentColor="alert">
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {dataset.anomalies.slice(0, 20).map((a, i) => (
              <div key={i} className="flex items-start gap-3 px-3 py-2 rounded-lg bg-white/3 hover:bg-white/5 transition-colors">
                <AlertTriangle className="w-4 h-4 text-energy-alert mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-white">
                    {'feature' in a ? (a as any).feature : 'timestamp' in a ? (a as any).timestamp : `Anomaly #${i + 1}`}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    {'z_score' in a ? `Z-score: ${((a as any).z_score as number).toFixed(2)}` : ''}
                    {'severity' in a ? ` • Severity: ${(a as any).severity}` : ''}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}
