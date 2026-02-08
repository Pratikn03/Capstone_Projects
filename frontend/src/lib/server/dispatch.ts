import 'server-only';

import type { DispatchCompareResponse, DispatchSeriesPoint } from '@/lib/api/dispatch-types';

type ForecastPayload = {
  generated_at?: string;
  horizon_hours?: number;
  forecasts?: Record<string, { timestamp?: string[]; forecast?: number[] }>;
};

type OptimizePayload = {
  dispatch_plan?: {
    grid_mw?: number[];
    battery_charge_mw?: number[];
    battery_discharge_mw?: number[];
    renewables_used_mw?: number[];
  };
};

const DEFAULT_BASE = 'http://localhost:8000';

function resolveApiBase(): string {
  return process.env.FASTAPI_URL || DEFAULT_BASE;
}

function minLength(...arrays: Array<ReadonlyArray<unknown> | undefined>): number {
  const sizes = arrays.filter(Boolean).map((arr) => arr!.length);
  return sizes.length ? Math.min(...sizes) : 0;
}

function buildDispatchSeries(
  timestamps: string[],
  load: number[],
  wind: number[],
  solar: number[],
  plan: OptimizePayload['dispatch_plan']
): DispatchSeriesPoint[] {
  if (!plan) return [];
  const grid = plan.grid_mw ?? [];
  const charge = plan.battery_charge_mw ?? [];
  const discharge = plan.battery_discharge_mw ?? [];
  const renewUsed = plan.renewables_used_mw ?? [];
  const length = minLength(load, wind, solar, grid, charge, discharge, renewUsed, timestamps);

  const data: DispatchSeriesPoint[] = [];
  for (let i = 0; i < length; i += 1) {
    const totalRenew = wind[i] + solar[i];
    const used = totalRenew > 0 ? renewUsed[i] / totalRenew : 0;
    const solarUsed = totalRenew > 0 ? solar[i] * used : 0;
    const windUsed = totalRenew > 0 ? wind[i] * used : 0;
    data.push({
      timestamp: timestamps[i],
      load_mw: load[i],
      generation_solar: solarUsed,
      generation_wind: windUsed,
      generation_gas: grid[i],
      battery_dispatch: (discharge[i] ?? 0) - (charge[i] ?? 0),
    });
  }
  return data;
}

function buildFallbackTimestamps(horizon: number): string[] {
  const base = new Date();
  return Array.from({ length: horizon }, (_, i) => {
    const t = new Date(base.getTime() + (i + 1) * 3600000);
    return t.toISOString();
  });
}

export async function loadDispatchCompare(
  region = 'DE',
  horizon = 24
): Promise<DispatchCompareResponse> {
  const apiBase = resolveApiBase();
  const warnings: string[] = [];

  try {
    const forecastRes = await fetch(
      `${apiBase}/forecast?targets=load_mw,wind_mw,solar_mw&horizon=${horizon}`,
      { cache: 'no-store' }
    );
    if (!forecastRes.ok) {
      throw new Error(`Forecast API error: ${forecastRes.status}`);
    }
    const forecastPayload = (await forecastRes.json()) as ForecastPayload;
    const forecasts = forecastPayload.forecasts ?? {};
    const loadForecast = forecasts.load_mw?.forecast ?? [];
    const windForecast = forecasts.wind_mw?.forecast ?? [];
    const solarForecast = forecasts.solar_mw?.forecast ?? [];
    const timestamps =
      forecasts.load_mw?.timestamp ?? buildFallbackTimestamps(loadForecast.length || horizon);

    const length = minLength(loadForecast, windForecast, solarForecast, timestamps);
    if (!length) {
      throw new Error('Missing forecast series for optimization.');
    }

    const load = loadForecast.slice(0, length);
    const wind = windForecast.slice(0, length);
    const solar = solarForecast.slice(0, length);
    const renewables = load.map((_, idx) => wind[idx] + solar[idx]);

    const optimizeBody = JSON.stringify({
      forecast_load_mw: load,
      forecast_renewables_mw: renewables,
    });

    const [optimizedRes, baselineRes] = await Promise.all([
      fetch(`${apiBase}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: optimizeBody,
      }),
      fetch(`${apiBase}/optimize/baseline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: optimizeBody,
      }),
    ]);

    if (!optimizedRes.ok) {
      throw new Error(`Optimize API error: ${optimizedRes.status}`);
    }
    if (!baselineRes.ok) {
      warnings.push(`Baseline API error: ${baselineRes.status}`);
    }

    const optimizedPayload = (await optimizedRes.json()) as OptimizePayload;
    const baselinePayload = baselineRes.ok ? ((await baselineRes.json()) as OptimizePayload) : null;

    return {
      optimized: buildDispatchSeries(timestamps, load, wind, solar, optimizedPayload.dispatch_plan),
      baseline: baselinePayload
        ? buildDispatchSeries(timestamps, load, wind, solar, baselinePayload.dispatch_plan)
        : undefined,
      meta: {
        source: 'fastapi',
        horizon_hours: length,
        generated_at: forecastPayload.generated_at,
        warnings: warnings.length ? warnings : undefined,
      },
    };
  } catch (err) {
    return {
      optimized: [],
      baseline: undefined,
      meta: {
        source: 'missing',
        horizon_hours: horizon,
        warnings: [String(err)],
      },
    };
  }
}
