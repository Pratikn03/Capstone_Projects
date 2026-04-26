import { useEffect, useState } from 'react';

import type { DispatchCompareResponse } from './dispatch-types';
import { isBatteryDomain, type DomainId } from '@/lib/domain-options';

export function useDispatchCompare(region: DomainId = 'DE', horizon = 24) {
  const [data, setData] = useState<DispatchCompareResponse>({
    optimized: [],
    baseline: undefined,
    meta: { source: 'missing', horizon_hours: horizon },
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function load() {
      if (!isBatteryDomain(region)) {
        if (active) {
          setData({
            optimized: [],
            baseline: undefined,
            meta: {
              source: 'missing',
              horizon_hours: horizon,
              warnings: ['Dispatch comparison is only available for Battery DE/US datasets.'],
            },
          });
          setLoading(false);
          setError(null);
        }
        return;
      }
      try {
        const res = await fetch(`/api/dispatch/compare?region=${region}&horizon=${horizon}`, { cache: 'no-store' });
        if (!res.ok) {
          throw new Error(`Dispatch API error: ${res.status}`);
        }
        const payload = (await res.json()) as DispatchCompareResponse;
        if (active) {
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
