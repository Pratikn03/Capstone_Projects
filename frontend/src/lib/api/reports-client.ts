import { useEffect, useState } from 'react';

import { mockForecastMetrics } from './mock-data';
import type { ReportsApiResponse } from './report-types';

export function useReportsData() {
  const [data, setData] = useState<ReportsApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const res = await fetch('/api/reports', { cache: 'no-store' });
        if (!res.ok) {
          throw new Error(`Reports API error: ${res.status}`);
        }
        const payload = (await res.json()) as ReportsApiResponse;
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
  }, []);

  return {
    reports: data?.reports ?? [],
    metrics: data?.metrics?.length ? data.metrics : mockForecastMetrics(),
    metricsBacktest: data?.metrics_backtest ?? [],
    impact: data?.impact ?? null,
    robustness: data?.robustness ?? null,
    regions: data?.regions ?? {},
    meta: data?.meta ?? { source: 'missing' as const },
    metricsSource: data?.meta?.metrics_source ?? 'missing',
    loading,
    error,
  };
}
