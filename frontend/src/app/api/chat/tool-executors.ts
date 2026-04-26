import { fetchFastApi } from '@/lib/server/config';

const CARBON_WEIGHTS = [0, 5, 10, 20, 40] as const;

type ForecastSeries = {
  timestamp?: string[];
  forecast?: number[];
  quantiles?: Record<string, number[]>;
};

type ForecastResponse = {
  generated_at?: string;
  horizon_hours?: number;
  forecasts?: Record<string, ForecastSeries>;
  meta?: Record<string, unknown>;
};

type ForecastIntervalsResponse = {
  yhat?: number[];
  pi90_lower?: number[] | null;
  pi90_upper?: number[] | null;
};

type OptimizeResponse = {
  dispatch_plan?: {
    grid_mw?: number[];
    battery_charge_mw?: number[];
    battery_discharge_mw?: number[];
    renewables_used_mw?: number[];
    curtailment_mw?: number[];
    unmet_load_mw?: number[];
    soc_mwh?: number[];
    peak_mw?: number;
  };
  expected_cost_usd?: number | null;
  carbon_kg?: number | null;
  carbon_cost_usd?: number | null;
};

type MonitorResponse = Record<string, unknown>;
type AnomalyResponse = {
  combined?: boolean[];
  note?: string;
};

async function fetchBackendJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = init.method ?? 'GET';

  try {
    const response = await fetchFastApi(path, init);

    if (!response.ok) {
      const detail = (await response.text().catch(() => '')).slice(0, 300);
      throw new Error(
        `Backend ${method} ${path} failed (${response.status})${detail ? `: ${detail}` : ''}`
      );
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof Error && error.message.startsWith('Backend ')) {
      throw error;
    }
    const msg = error instanceof Error ? error.message : String(error);
    throw new Error(`Backend ${method} ${path} request failed: ${msg}`);
  }
}

function toFixedArray(values: number[] | undefined, size: number): number[] {
  return (values ?? []).slice(0, size).map((v) => Number(v ?? 0));
}

function sum(values: number[]): number {
  return values.reduce((acc, value) => acc + value, 0);
}

function mean(values: number[]): number | null {
  return values.length ? sum(values) / values.length : null;
}

function buildFallbackTimestamps(length: number): string[] {
  const now = Date.now();
  return Array.from({ length }, (_, i) => new Date(now + (i + 1) * 3600000).toISOString());
}

async function loadForecastTriplet(horizonHours: number): Promise<{
  generatedAt: string | null;
  timestamps: string[];
  load: number[];
  wind: number[];
  solar: number[];
}> {
  const forecast = await fetchBackendJson<ForecastResponse>(
    `/forecast?targets=load_mw,wind_mw,solar_mw&horizon=${horizonHours}`
  );

  const loadForecast = forecast.forecasts?.load_mw?.forecast ?? [];
  const windForecast = forecast.forecasts?.wind_mw?.forecast ?? [];
  const solarForecast = forecast.forecasts?.solar_mw?.forecast ?? [];
  const tsFromBackend = forecast.forecasts?.load_mw?.timestamp ?? [];

  if (!loadForecast.length || !windForecast.length || !solarForecast.length) {
    throw new Error('Backend forecast payload missing load/wind/solar forecast arrays.');
  }

  const timeRef = tsFromBackend.length ? tsFromBackend : buildFallbackTimestamps(loadForecast.length);
  const length = Math.min(loadForecast.length, windForecast.length, solarForecast.length, timeRef.length);
  if (!length) {
    throw new Error('Backend forecast arrays are empty after alignment.');
  }

  return {
    generatedAt: forecast.generated_at ?? null,
    timestamps: timeRef.slice(0, length),
    load: toFixedArray(loadForecast, length),
    wind: toFixedArray(windForecast, length),
    solar: toFixedArray(solarForecast, length),
  };
}

function clampHorizon(value: number | undefined): number {
  if (!value || Number.isNaN(value)) return 24;
  return Math.min(168, Math.max(1, Math.trunc(value)));
}

export const chatToolExecutors = {
  get_dispatch_forecast: async ({
    zoneId,
    horizonHours,
  }: {
    zoneId: 'DE' | 'US';
    horizonHours: number;
  }) => {
    const horizon = clampHorizon(horizonHours);
    const { generatedAt, timestamps, load, wind, solar } = await loadForecastTriplet(horizon);
    const data = timestamps.map((timestamp, idx) => {
      const gas = Math.max(load[idx] - wind[idx] - solar[idx], 0);
      return {
        timestamp,
        load_mw: load[idx],
        generation_solar: solar[idx],
        generation_wind: wind[idx],
        generation_gas: gas,
        battery_dispatch: 0,
      };
    });

    const totalLoad = sum(load);
    const totalRenewables = sum(wind) + sum(solar);

    return {
      type: 'dispatch_chart',
      source: 'backend',
      zoneId,
      horizonHours: data.length,
      generated_at: generatedAt,
      data,
      summary: {
        total_load_mwh: totalLoad,
        renewable_fraction: totalLoad > 0 ? totalRenewables / totalLoad : null,
        peak_load_mw: load.length ? Math.max(...load) : null,
        avg_price: null,
        total_carbon_kg: null,
      },
    };
  },

  get_load_forecast: async ({
    target,
    zoneId,
    horizonHours,
  }: {
    target: 'load_mw' | 'wind_mw' | 'solar_mw';
    zoneId: 'DE' | 'US';
    horizonHours?: number;
  }) => {
    const horizon = clampHorizon(horizonHours);
    const payload = await fetchBackendJson<ForecastIntervalsResponse>(
      `/forecast/with-intervals?target=${encodeURIComponent(target)}&horizon=${horizon}`
    );

    const yhat = payload.yhat ?? [];
    const lower = payload.pi90_lower ?? [];
    const upper = payload.pi90_upper ?? [];
    if (!yhat.length || !lower.length || !upper.length) {
      throw new Error('Backend interval payload missing yhat/pi90_lower/pi90_upper arrays.');
    }

    const length = Math.min(yhat.length, lower.length, upper.length);
    if (!length) {
      throw new Error('Backend interval arrays are empty after alignment.');
    }

    const timestamps = buildFallbackTimestamps(length);
    const widths: number[] = [];
    const data = timestamps.map((timestamp, idx) => {
      const width = Math.max(upper[idx] - lower[idx], 0);
      widths.push(width);
      return {
        timestamp,
        forecast: Number(yhat[idx]),
        lower_90: Number(lower[idx]),
        upper_90: Number(upper[idx]),
      };
    });

    return {
      type: 'forecast_chart',
      source: 'backend',
      target,
      zoneId,
      horizonHours: length,
      data,
      metrics: {
        rmse: null,
        coverage_90: null,
        mean_interval_width_mw: mean(widths),
        sample_count: length,
      },
    };
  },

  get_battery_schedule: async ({
    zoneId,
    horizonHours,
  }: {
    zoneId: 'DE' | 'US';
    horizonHours?: number;
  }) => {
    const horizon = clampHorizon(horizonHours);
    const { generatedAt, timestamps, load, wind, solar } = await loadForecastTriplet(horizon);
    const renewables = load.map((_, idx) => wind[idx] + solar[idx]);
    const optimizeInput = {
      forecast_load_mw: load,
      forecast_renewables_mw: renewables,
      optimization_mode: 'deterministic' as const,
    };

    const [optimized, baseline] = await Promise.all([
      fetchBackendJson<OptimizeResponse>('/optimize', {
        method: 'POST',
        body: JSON.stringify(optimizeInput),
      }),
      fetchBackendJson<OptimizeResponse>('/optimize/baseline', {
        method: 'POST',
        body: JSON.stringify(optimizeInput),
      }),
    ]);

    const charge = optimized.dispatch_plan?.battery_charge_mw ?? [];
    const discharge = optimized.dispatch_plan?.battery_discharge_mw ?? [];
    const soc = optimized.dispatch_plan?.soc_mwh ?? [];
    const alignedLength = Math.min(timestamps.length, charge.length, discharge.length, soc.length);
    if (!alignedLength) {
      throw new Error('Backend optimization payload missing battery schedule arrays.');
    }

    const capacityMwh = Math.max(...soc.slice(0, alignedLength), 1);
    let cumulativeThroughputMwh = 0;
    const schedule = timestamps.slice(0, alignedLength).map((timestamp, idx) => {
      const powerMw = Number(discharge[idx] ?? 0) - Number(charge[idx] ?? 0);
      cumulativeThroughputMwh += Math.abs(powerMw);
      const cycles = cumulativeThroughputMwh / (2 * capacityMwh);
      return {
        timestamp,
        soc_percent: Math.max(0, Math.min(100, (Number(soc[idx] ?? 0) / capacityMwh) * 100)),
        power_mw: powerMw,
        capacity_mwh: capacityMwh,
        cycles_today: cycles,
      };
    });

    const optimizedCost = optimized.expected_cost_usd;
    const baselineCost = baseline.expected_cost_usd;
    const optimizedCarbon = optimized.carbon_kg;
    const baselineCarbon = baseline.carbon_kg;
    const optimizedGrid = optimized.dispatch_plan?.grid_mw ?? [];
    const baselineGrid = baseline.dispatch_plan?.grid_mw ?? [];

    const peakOptimized = optimizedGrid.length ? Math.max(...optimizedGrid) : null;
    const peakBaseline = baselineGrid.length ? Math.max(...baselineGrid) : null;
    const totalCharge = sum(charge.slice(0, alignedLength));
    const totalDischarge = sum(discharge.slice(0, alignedLength));
    const avgEfficiency = totalCharge > 0 ? Math.max(0, Math.min(100, (totalDischarge / totalCharge) * 100)) : null;

    return {
      type: 'battery_chart',
      source: 'backend',
      zoneId,
      generated_at: generatedAt,
      schedule,
      metrics: {
        cost_savings_usd:
          typeof baselineCost === 'number' && typeof optimizedCost === 'number'
            ? baselineCost - optimizedCost
            : null,
        carbon_reduction_kg:
          typeof baselineCarbon === 'number' && typeof optimizedCarbon === 'number'
            ? baselineCarbon - optimizedCarbon
            : null,
        peak_shaving_pct:
          typeof peakBaseline === 'number' && peakBaseline > 0 && typeof peakOptimized === 'number'
            ? ((peakBaseline - peakOptimized) / peakBaseline) * 100
            : null,
        avg_efficiency: avgEfficiency,
      },
      raw: {
        optimized,
        baseline,
      },
    };
  },

  get_grid_status: async () => {
    const [health, ready, monitor, anomaly] = await Promise.all([
      fetchBackendJson<Record<string, unknown>>('/health'),
      fetchBackendJson<Record<string, unknown>>('/ready'),
      fetchBackendJson<MonitorResponse>('/monitor'),
      fetchBackendJson<AnomalyResponse>('/anomaly'),
    ]);

    const anomalySeries = anomaly.combined ?? [];
    const anomalyCount = anomalySeries.filter(Boolean).length;

    return {
      type: 'grid_status',
      source: 'backend',
      health,
      ready,
      monitor,
      anomaly_summary: {
        total_points: anomalySeries.length,
        anomaly_count: anomalyCount,
        note: anomaly.note ?? null,
      },
      zones: [],
    };
  },

  get_cost_carbon_tradeoff: async ({
    zoneId,
    horizonHours,
  }: {
    zoneId: 'DE' | 'US';
    horizonHours?: number;
  }) => {
    const horizon = clampHorizon(horizonHours);
    const { generatedAt, load, wind, solar } = await loadForecastTriplet(horizon);
    const renewables = load.map((_, idx) => wind[idx] + solar[idx]);

    const results = await Promise.all(
      CARBON_WEIGHTS.map(async (carbonWeight) => {
        const payload = {
          forecast_load_mw: load,
          forecast_renewables_mw: renewables,
          optimization_mode: 'deterministic' as const,
          config: {
            objective: {
              cost_weight: 1.0,
              carbon_weight: carbonWeight,
            },
          },
        };
        const response = await fetchBackendJson<OptimizeResponse>('/optimize', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
        return {
          carbonWeight,
          expectedCostUsd: response.expected_cost_usd ?? null,
          carbonKg: response.carbon_kg ?? null,
        };
      })
    );

    const reference = results.find((item) => item.carbonWeight === 0) ?? null;
    const referenceCost = reference?.expectedCostUsd ?? null;
    const referenceCarbon = reference?.carbonKg ?? null;

    const data = results.map((item) => {
      const costSavingsPct =
        typeof referenceCost === 'number' &&
        referenceCost > 0 &&
        typeof item.expectedCostUsd === 'number'
          ? ((referenceCost - item.expectedCostUsd) / referenceCost) * 100
          : null;
      const carbonReductionPct =
        typeof referenceCarbon === 'number' &&
        referenceCarbon > 0 &&
        typeof item.carbonKg === 'number'
          ? ((referenceCarbon - item.carbonKg) / referenceCarbon) * 100
          : null;
      return {
        carbon_weight: item.carbonWeight,
        total_cost_eur: item.expectedCostUsd,
        total_carbon_kg: item.carbonKg,
        cost_savings_pct: costSavingsPct,
        carbon_reduction_pct: carbonReductionPct,
      };
    });

    const validPoints = data.filter(
      (point) =>
        typeof point.cost_savings_pct === 'number' && typeof point.carbon_reduction_pct === 'number'
    );
    const optimalPoint =
      validPoints.length > 0
        ? validPoints.reduce((best, current) =>
            (current.cost_savings_pct ?? 0) + (current.carbon_reduction_pct ?? 0) >
            (best.cost_savings_pct ?? 0) + (best.carbon_reduction_pct ?? 0)
              ? current
              : best
          )
        : null;

    return {
      type: 'pareto_chart',
      source: 'backend',
      zoneId,
      horizonHours: horizon,
      generated_at: generatedAt,
      data,
      optimal_point: optimalPoint,
    };
  },

  get_model_info: async () => {
    const payload = await fetchBackendJson<Record<string, unknown>>('/monitor/model-info');
    return {
      type: 'model_info',
      source: 'backend',
      ...payload,
    };
  },
};
