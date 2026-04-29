import { NextResponse } from 'next/server';

import { fetchFastApi } from '@/lib/server/config';
import { loadRegionData, type ForecastPoint, type RegionDashboardData } from '@/lib/server/dataset';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type Dc3sStepResponse = {
  command_id?: string;
  certificate_id?: string;
  proposed_action?: { charge_mw?: number; discharge_mw?: number };
  safe_action?: { charge_mw?: number; discharge_mw?: number };
  uncertainty?: {
    lower?: number[];
    upper?: number[];
    meta?: {
      inflation?: number;
      drift_flag?: boolean;
      w_t_used?: number;
    };
  };
  certificate?: {
    certificate_hash?: string;
    uncertainty?: {
      shield_repair?: {
        repaired?: boolean;
      };
    };
  };
};

type Dc3sAuditResponse = {
  certificate_hash?: string;
  reliability?: { w_t?: number };
  uncertainty?: {
    meta?: { inflation?: number; drift_flag?: boolean; w_t_used?: number };
    shield_repair?: { repaired?: boolean };
  };
};

type BatteryRegion = 'DE' | 'US';

function mean(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function toNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function forecastCenter(point: ForecastPoint | undefined, fallback: number): number {
  return toNumber(point?.forecast, toNumber(point?.predicted, toNumber(point?.actual, fallback)));
}

function buildShadowUncertainty(
  dataset: RegionDashboardData,
  horizon: number,
  fallbackLoad: number
): Array<{ h: number; lower: number; upper: number; width: number }> {
  const forecastRows = (dataset.forecast?.load_mw ?? []).slice(0, horizon);
  const latestForecast = dataset.forecast?.load_mw?.at(-1);
  const rows = forecastRows.length ? forecastRows : Array.from({ length: Math.min(6, horizon) }, () => latestForecast);

  return rows.slice(0, 6).map((point, index) => {
    const center = forecastCenter(point, fallbackLoad);
    const lower = toNumber(point?.lower_90, center - Math.max(25, Math.abs(center) * 0.04));
    const upper = toNumber(point?.upper_90, center + Math.max(25, Math.abs(center) * 0.04));
    return {
      h: index + 1,
      lower,
      upper,
      width: upper - lower,
    };
  });
}

async function localArtifactShadowResponse(
  region: BatteryRegion,
  controller: string,
  horizon: number,
  backendError: string
) {
  const dataset = await loadRegionData(region);
  const latestTs = dataset.timeseries.at(-1);
  const latestForecast = dataset.forecast?.load_mw?.at(-1);
  const latestBattery = dataset.battery?.schedule?.at(-1);
  const latestLoad = toNumber(latestTs?.load_mw, forecastCenter(latestForecast, 45_000));
  const forecastLoad = forecastCenter(latestForecast, latestLoad);
  const relativeError = Math.abs(latestLoad - forecastLoad) / Math.max(Math.abs(latestLoad), 1);
  const reliabilityWT = Math.max(0, Math.min(1, relativeError));
  const proposedPower = toNumber(latestBattery?.power_mw, 0);
  const proposedAction = {
    charge_mw: Math.max(0, -proposedPower),
    discharge_mw: Math.max(0, proposedPower),
  };
  const soc = toNumber(latestBattery?.soc_percent, 50);
  const safeAction = {
    charge_mw: soc >= 95 ? 0 : proposedAction.charge_mw,
    discharge_mw: soc <= 5 ? 0 : proposedAction.discharge_mw,
  };
  const repaired =
    safeAction.charge_mw !== proposedAction.charge_mw ||
    safeAction.discharge_mw !== proposedAction.discharge_mw;
  const uncertaintyPreview = buildShadowUncertainty(dataset, horizon, latestLoad);
  const widths = uncertaintyPreview.map((row) => row.width).filter((value) => Number.isFinite(value));

  return NextResponse.json({
    ok: false,
    degraded: true,
    evidence_status: 'shadow_only_not_certificate_backed',
    error: 'Live DC3S backend unavailable; serving local artifact shadow only.',
    generated_at: new Date().toISOString(),
    region,
    source: 'local_artifact_shadow',
    backend_error: backendError.slice(0, 300),
    command_id: null,
    certificate_id: null,
    certificate_hash: null,
    controller,
    proposed_action: proposedAction,
    safe_action: safeAction,
    repaired,
    reliability_w_t: reliabilityWT,
    drift_flag: dataset.monitoring?.summary.data_drift_detected ?? null,
    inflation: 1 + reliabilityWT,
    mean_interval_width_mw: mean(widths),
    uncertainty_preview: uncertaintyPreview,
  });
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const region = (searchParams.get('region') || 'DE').toUpperCase() as BatteryRegion;
  const horizon = Math.max(1, Math.min(168, Number(searchParams.get('horizon') || 24)));
  const deviceId = searchParams.get('device_id') || `dashboard-${region.toLowerCase()}-001`;
  const controller = searchParams.get('controller') || 'deterministic';

  if (region !== 'DE' && region !== 'US') {
    return NextResponse.json({ ok: false, error: 'Invalid region. Use DE or US.' }, { status: 400 });
  }

  try {
    const dataset = await loadRegionData(region);
    const latestTs = dataset.timeseries.at(-1);
    const latestForecast = dataset.forecast?.load_mw?.at(-1);
    const latestBattery = dataset.battery?.schedule?.at(-1);
    const capacityMwh = toNumber(latestBattery?.capacity_mwh, 10);
    const currentSocMwh = latestBattery
      ? Math.max(0, (toNumber(latestBattery.soc_percent, 50) / 100.0) * capacityMwh)
      : 5.0;

    const payload = {
      device_id: deviceId,
      zone_id: region,
      current_soc_mwh: currentSocMwh,
      telemetry_event: {
        ts_utc: latestTs?.timestamp || new Date().toISOString(),
        load_mw: toNumber(latestTs?.load_mw, 45000),
        wind_mw: toNumber(latestTs?.wind_mw, 3000),
        solar_mw: toNumber(latestTs?.solar_mw, 1200),
      },
      last_actual_load_mw: toNumber(latestTs?.load_mw, 45000),
      last_pred_load_mw: toNumber(latestForecast?.forecast, toNumber(latestTs?.load_mw, 45000)),
      horizon,
      controller,
      include_certificate: true,
    };

    const stepRes = await fetchFastApi('/dc3s/step', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    if (!stepRes.ok) {
      const detail = await stepRes.text();
      return localArtifactShadowResponse(
        region,
        controller,
        horizon,
        `DC3S step failed (${stepRes.status}): ${detail}`
      );
    }

    const step = (await stepRes.json()) as Dc3sStepResponse;
    const commandId = step.command_id || '';
    let audit: Dc3sAuditResponse | null = null;
    if (commandId) {
      const auditRes = await fetchFastApi(`/dc3s/audit/${encodeURIComponent(commandId)}`);
      if (auditRes.ok) {
        audit = (await auditRes.json()) as Dc3sAuditResponse;
      }
    }

    const lower = step.uncertainty?.lower ?? [];
    const upper = step.uncertainty?.upper ?? [];
    const widths = lower.map((l, i) => (upper[i] ?? l) - l).filter((w) => Number.isFinite(w));
    const meanWidth = mean(widths as number[]);
    const uncertaintyPreview = lower.slice(0, 6).map((l, i) => ({
      h: i + 1,
      lower: l,
      upper: upper[i] ?? l,
      width: (upper[i] ?? l) - l,
    }));

    const meta = step.uncertainty?.meta || audit?.uncertainty?.meta || {};
    const repaired =
      step.certificate?.uncertainty?.shield_repair?.repaired ??
      audit?.uncertainty?.shield_repair?.repaired ??
      false;
    const certificateHash = step.certificate?.certificate_hash || audit?.certificate_hash || null;

    return NextResponse.json({
      ok: true,
      generated_at: new Date().toISOString(),
      region,
      source: 'fastapi',
      command_id: commandId,
      certificate_id: step.certificate_id || null,
      certificate_hash: certificateHash,
      controller,
      proposed_action: step.proposed_action || { charge_mw: 0, discharge_mw: 0 },
      safe_action: step.safe_action || { charge_mw: 0, discharge_mw: 0 },
      repaired: Boolean(repaired),
      reliability_w_t:
        audit?.reliability?.w_t ??
        (typeof meta.w_t_used === 'number' ? meta.w_t_used : null),
      drift_flag: typeof meta.drift_flag === 'boolean' ? meta.drift_flag : null,
      inflation: typeof meta.inflation === 'number' ? meta.inflation : null,
      mean_interval_width_mw: meanWidth,
      uncertainty_preview: uncertaintyPreview,
    });
  } catch (error) {
    return localArtifactShadowResponse(
      region,
      controller,
      horizon,
      error instanceof Error ? error.message : String(error)
    );
  }
}
