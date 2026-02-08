import { useEffect, useState } from 'react';

import { mockDispatchForecast } from './mock-data';
import type { DispatchCompareResponse, DispatchSeriesPoint } from './dispatch-types';

export function useDispatchCompare(region = 'DE', horizon = 24) {
  const fallback = mockDispatchForecast(region, horizon).data as DispatchSeriesPoint[];
  const [data, setData] = useState<DispatchCompareResponse>({
    optimized: fallback,
    baseline: undefined,
    meta: { source: 'missing', horizon_hours: horizon },
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const res = await fetch(`/api/dispatch/compare?region=${region}&horizon=${horizon}`, { cache: 'no-store' });
        if (!res.ok) {
          throw new Error(`Dispatch API error: ${res.status}`);
        }
        const payload = (await res.json()) as DispatchCompareResponse;
        if (active && payload.optimized.length) {
          setData(payload);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    load();

    return () => {
      active = false;
    };
  }, [region, horizon]);

  return {
    optimized: data.optimized,
    baseline: data.baseline,
    meta: data.meta,
    loading,
    error,
  };
}
