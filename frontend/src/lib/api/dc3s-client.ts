import { useCallback, useEffect, useState } from 'react';
import { isBatteryDomain, type DomainId } from '@/lib/domain-options';

export type Dc3sPreviewRow = {
  h: number;
  lower: number;
  upper: number;
  width: number;
};

export type Dc3sLivePayload = {
  ok: boolean;
  degraded?: boolean;
  evidence_status?: 'certificate_backed' | 'shadow_only_not_certificate_backed';
  generated_at?: string;
  region?: DomainId;
  source?: 'fastapi' | 'local_artifact_shadow';
  backend_error?: string;
  command_id?: string;
  certificate_id?: string | null;
  certificate_hash?: string | null;
  controller?: string;
  proposed_action?: { charge_mw?: number; discharge_mw?: number };
  safe_action?: { charge_mw?: number; discharge_mw?: number };
  repaired?: boolean;
  reliability_w_t?: number | null;
  drift_flag?: boolean | null;
  inflation?: number | null;
  mean_interval_width_mw?: number | null;
  uncertainty_preview?: Dc3sPreviewRow[];
  error?: string;
};

const EMPTY: Dc3sLivePayload = { ok: false };

export function useDc3sLive(region: DomainId, horizon = 24, autoRefreshSeconds = 0) {
  const [data, setData] = useState<Dc3sLivePayload>(EMPTY);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshIndex, setRefreshIndex] = useState(0);

  const refresh = useCallback(() => {
    setRefreshIndex((x) => x + 1);
  }, []);

  useEffect(() => {
    let active = true;
    async function load() {
      if (!isBatteryDomain(region)) {
        if (active) {
          setData(EMPTY);
          setError('Live DC3S endpoint is battery-grid only; this view is showing tracked runtime artifacts.');
          setLoading(false);
        }
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`/api/dc3s/live?region=${region}&horizon=${horizon}`, {
          cache: 'no-store',
        });
        const payload = (await res.json()) as Dc3sLivePayload;
        if (payload.source === 'local_artifact_shadow' && payload.degraded) {
          if (active) {
            setData(payload);
            setError(payload.error || 'Local artifact shadow only; not certificate-backed live evidence.');
          }
          return;
        }
        if (!res.ok || !payload.ok) {
          throw new Error(payload.error || `DC3S API error: ${res.status}`);
        }
        if (active) {
          setData(payload);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : String(err));
          setData(EMPTY);
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
  }, [region, horizon, refreshIndex]);

  useEffect(() => {
    if (!Number.isFinite(autoRefreshSeconds) || autoRefreshSeconds <= 0) {
      return;
    }
    const timer = window.setInterval(() => {
      refresh();
    }, Math.trunc(autoRefreshSeconds) * 1000);
    return () => {
      window.clearInterval(timer);
    };
  }, [autoRefreshSeconds, refresh]);

  return { data, loading, error, refresh };
}
