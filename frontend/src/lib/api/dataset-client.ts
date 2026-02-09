import { useEffect, useState } from 'react';

import type {
  RegionDashboardData,
  DatasetStats,
  TimeseriesPoint,
  ForecastPoint,
  DispatchPoint,
  HourlyProfile,
  ModelMetric,
  ImpactData,
  ModelRegistryEntry,
} from '@/lib/server/dataset';

export type { DatasetStats, TimeseriesPoint, ForecastPoint, DispatchPoint, HourlyProfile, ModelMetric, ImpactData, ModelRegistryEntry };

const EMPTY: RegionDashboardData = {
  stats: null,
  timeseries: [],
  forecast: {},
  dispatch: [],
  profiles: {},
  metrics: [],
  impact: null,
  registry: [],
};

export function useDatasetData(region: 'DE' | 'US') {
  const [data, setData] = useState<RegionDashboardData>(EMPTY);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      try {
        const res = await fetch(`/api/data?region=${region}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`Data API error: ${res.status}`);
        const payload = (await res.json()) as RegionDashboardData;
        if (active) setData(payload);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (active) setLoading(false);
      }
    }

    load();
    return () => { active = false; };
  }, [region]);

  return { ...data, loading, error };
}
