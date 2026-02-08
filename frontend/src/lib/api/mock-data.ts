/**
 * Realistic mock data generators for GridPulse dashboard.
 * Used in demo mode when FastAPI backend is unavailable.
 * Data patterns mirror actual OPSD Germany + EIA-930 USA datasets.
 */

import type {
  DispatchForecast,
  BatterySchedule,
  Anomaly,
  ForecastMetrics,
  ZoneSummary,
  OptimizationResult,
} from './schema';

// ─── Helpers ───
function generateTimestamps(hours: number, startHour: number = 6): string[] {
  const base = new Date('2026-02-07T00:00:00Z');
  base.setHours(startHour);
  return Array.from({ length: hours }, (_, i) => {
    const t = new Date(base.getTime() + i * 3600000);
    return t.toISOString();
  });
}

function solarProfile(hour: number): number {
  // Bell curve centered at 12:00, zero at night
  if (hour < 6 || hour > 20) return 0;
  const peak = 4200;
  return peak * Math.exp(-0.5 * ((hour - 13) / 3) ** 2) * (0.85 + 0.15 * Math.random());
}

function windProfile(hour: number): number {
  // Higher at night, moderate during day
  const base = 3800 + 1200 * Math.sin((hour / 24) * Math.PI * 2 + 1.5);
  return Math.max(400, base * (0.8 + 0.4 * Math.random()));
}

function loadProfile(hour: number): number {
  // Morning ramp, midday plateau, evening peak, night trough
  const pattern = [
    0.55, 0.52, 0.50, 0.48, 0.50, 0.55, // 0-5
    0.65, 0.78, 0.88, 0.92, 0.94, 0.95, // 6-11
    0.93, 0.90, 0.88, 0.87, 0.89, 0.95, // 12-17
    1.00, 0.97, 0.90, 0.82, 0.72, 0.62, // 18-23
  ];
  const base = 8500;
  return base * (pattern[hour % 24] ?? 0.7) * (0.95 + 0.1 * Math.random());
}

// ─── Forecast Dispatch ───
export function mockDispatchForecast(zoneId: string = 'DE', horizonHours: number = 24): DispatchForecast {
  const timestamps = generateTimestamps(horizonHours);
  const data = timestamps.map((ts) => {
    const h = new Date(ts).getUTCHours();
    const load = loadProfile(h);
    const solar = solarProfile(h);
    const wind = windProfile(h);
    const renewables = solar + wind;
    const gas = Math.max(0, load - renewables) * (0.9 + 0.2 * Math.random());
    const battery = load > renewables ? Math.min(500, load - renewables) * 0.3 : -Math.min(400, renewables - load) * 0.4;

    return {
      timestamp: ts,
      load_mw: Math.round(load),
      generation_solar: Math.round(solar),
      generation_wind: Math.round(wind),
      generation_gas: Math.round(gas),
      battery_dispatch: Math.round(battery),
      price_eur_mwh: 35 + 25 * (load / 8500) + 10 * Math.random(),
      carbon_kg_mwh: gas > 0 ? 180 + 120 * (gas / load) + 20 * Math.random() : 15 + 10 * Math.random(),
    };
  });

  const totalLoad = data.reduce((s, d) => s + d.load_mw, 0);
  const totalRenewable = data.reduce((s, d) => s + d.generation_solar + d.generation_wind, 0);

  return {
    zone_id: zoneId,
    horizon_hours: horizonHours,
    generated_at: new Date().toISOString(),
    data,
    summary: {
      total_load_mwh: totalLoad,
      renewable_fraction: totalRenewable / (totalLoad || 1),
      peak_load_mw: Math.max(...data.map((d) => d.load_mw)),
      avg_price: data.reduce((s, d) => s + (d.price_eur_mwh ?? 0), 0) / data.length,
      total_cost: totalLoad * 42.5,
      total_carbon_kg: data.reduce((s, d) => s + (d.carbon_kg_mwh ?? 0) * d.load_mw / 1000, 0),
    },
  };
}

// ─── Battery SOC Schedule ───
export function mockBatterySchedule(zoneId: string = 'DE'): BatterySchedule {
  const timestamps = generateTimestamps(24);
  let soc = 65;
  const schedule = timestamps.map((ts) => {
    const h = new Date(ts).getUTCHours();
    // Charge during solar peak (10-15), discharge during evening peak (17-21)
    let power = 0;
    if (h >= 10 && h <= 15) {
      power = -(2000 + 1500 * Math.random()); // charging
      soc = Math.min(95, soc + 4 + 2 * Math.random());
    } else if (h >= 17 && h <= 21) {
      power = 2500 + 1500 * Math.random(); // discharging
      soc = Math.max(15, soc - 5 - 3 * Math.random());
    } else if (h >= 1 && h <= 5) {
      power = -(800 + 400 * Math.random()); // off-peak charging
      soc = Math.min(95, soc + 2 + Math.random());
    } else {
      power = 200 * (Math.random() - 0.5);
      soc += 0.5 * (Math.random() - 0.5);
    }
    soc = Math.max(10, Math.min(95, soc));

    return {
      timestamp: ts,
      soc_percent: Math.round(soc * 10) / 10,
      power_mw: Math.round(power),
      capacity_mwh: 20000,
      cycles_today: h >= 20 ? 2 : h >= 15 ? 1 : 0,
    };
  });

  return {
    zone_id: zoneId,
    schedule,
    metrics: {
      cost_savings_eur: 23500,
      carbon_reduction_kg: 47800,
      peak_shaving_pct: 19.5,
      avg_efficiency: 92.1,
    },
  };
}

// ─── Anomalies ───
export function mockAnomalies(): Anomaly[] {
  return [
    {
      id: 'anom-001',
      timestamp: '2026-02-07T10:15:00Z',
      type: 'load_spike',
      severity: 'high',
      status: 'investigating',
      zone_id: 'DE',
      description: 'Load spike of +1,200 MW detected — 3.2σ above forecast. Possible industrial demand surge.',
      value: 9800,
      threshold: 8600,
    },
    {
      id: 'anom-002',
      timestamp: '2026-02-07T08:30:00Z',
      type: 'solar_drop',
      severity: 'medium',
      status: 'resolved',
      zone_id: 'DE',
      description: 'Solar generation dropped 40% below forecast due to unexpected cloud cover.',
      value: 1200,
      threshold: 2000,
    },
    {
      id: 'anom-003',
      timestamp: '2026-02-07T06:00:00Z',
      type: 'battery_fault',
      severity: 'critical',
      status: 'active',
      zone_id: 'DE',
      description: 'Battery inverter #3 communication timeout. SOC reading unreliable.',
    },
    {
      id: 'anom-004',
      timestamp: '2026-02-07T04:40:00Z',
      type: 'sensor_fault',
      severity: 'low',
      status: 'resolved',
      zone_id: 'DE',
      description: 'Temperature sensor fault on transformer T4-B. Replaced with redundant reading.',
    },
    {
      id: 'anom-005',
      timestamp: '2026-02-07T14:15:00Z',
      type: 'wind_ramp',
      severity: 'medium',
      status: 'resolved',
      zone_id: 'US',
      description: 'Wind ramp of +800 MW in 15 minutes. Generation forecast recalibrated.',
      value: 4200,
      threshold: 3400,
    },
  ];
}

// ─── Forecast Metrics ───
export function mockForecastMetrics(): ForecastMetrics[] {
  return [
    { target: 'load_mw', model: 'GBM (LightGBM)', rmse: 348.45, mae: 241.2, mape: 4.1, r2: 0.967, coverage_90: 93.2 },
    { target: 'load_mw', model: 'LSTM', rmse: 412.8, mae: 289.5, mape: 4.8, r2: 0.954, coverage_90: 91.8 },
    { target: 'load_mw', model: 'TCN', rmse: 385.1, mae: 265.3, mape: 4.4, r2: 0.960, coverage_90: 92.5 },
    { target: 'wind_mw', model: 'GBM (LightGBM)', rmse: 184.18, mae: 132.4, mape: 8.2, r2: 0.912, coverage_90: 91.4 },
    { target: 'wind_mw', model: 'LSTM', rmse: 225.6, mae: 161.8, mape: 10.1, r2: 0.883, coverage_90: 89.7 },
    { target: 'wind_mw', model: 'TCN', rmse: 198.3, mae: 142.1, mape: 8.9, r2: 0.901, coverage_90: 90.8 },
    { target: 'solar_mw', model: 'GBM (LightGBM)', rmse: 5.20, mae: 3.8, mape: 6.3, r2: 0.945, coverage_90: 94.1 },
    { target: 'solar_mw', model: 'LSTM', rmse: 6.85, mae: 4.9, mape: 8.1, r2: 0.928, coverage_90: 92.3 },
    { target: 'solar_mw', model: 'TCN', rmse: 5.90, mae: 4.2, mape: 7.0, r2: 0.938, coverage_90: 93.0 },
  ];
}

// ─── Zone Summaries ───
export function mockZoneSummaries(): ZoneSummary[] {
  return [
    {
      zone_id: 'DE',
      name: 'Germany (OPSD)',
      status: 'normal',
      load_mw: 8420,
      generation_mw: 8650,
      renewable_pct: 62.4,
      frequency_hz: 50.01,
      anomaly_count: 2,
    },
    {
      zone_id: 'US',
      name: 'USA (EIA-930)',
      status: 'warning',
      load_mw: 142500,
      generation_mw: 145200,
      renewable_pct: 38.7,
      frequency_hz: 60.00,
      anomaly_count: 1,
    },
  ];
}

// ─── Forecast with Prediction Intervals ───
export interface ForecastWithPI {
  timestamp: string;
  actual: number;
  forecast: number;
  lower_90: number;
  upper_90: number;
  lower_50: number;
  upper_50: number;
}

export function mockForecastWithPI(target: string = 'load_mw', hours: number = 48): ForecastWithPI[] {
  const timestamps = generateTimestamps(hours);
  return timestamps.map((ts) => {
    const h = new Date(ts).getUTCHours();
    let actual: number;
    let width: number;
    
    if (target === 'load_mw') {
      actual = loadProfile(h);
      width = 600;
    } else if (target === 'wind_mw') {
      actual = windProfile(h);
      width = 800;
    } else {
      actual = solarProfile(h);
      width = Math.max(50, solarProfile(h) * 0.3);
    }

    const forecast = actual * (0.97 + 0.06 * Math.random());
    const noise90 = width * (0.8 + 0.4 * Math.random());
    const noise50 = noise90 * 0.5;

    return {
      timestamp: ts,
      actual: Math.round(actual),
      forecast: Math.round(forecast),
      lower_90: Math.round(Math.max(0, forecast - noise90)),
      upper_90: Math.round(forecast + noise90),
      lower_50: Math.round(Math.max(0, forecast - noise50)),
      upper_50: Math.round(forecast + noise50),
    };
  });
}

// ─── Anomaly Z-Scores Timeline ───
export interface AnomalyZScore {
  timestamp: string;
  z_score: number;
  is_anomaly: boolean;
  residual_mw: number;
}

export function mockAnomalyZScores(hours: number = 72): AnomalyZScore[] {
  const timestamps = generateTimestamps(hours, 0);
  return timestamps.map((ts) => {
    const isAnomaly = Math.random() < 0.06;
    const z = isAnomaly
      ? (Math.random() > 0.5 ? 1 : -1) * (2.5 + 1.5 * Math.random())
      : (Math.random() - 0.5) * 3;

    return {
      timestamp: ts,
      z_score: Math.round(z * 100) / 100,
      is_anomaly: Math.abs(z) > 2.0,
      residual_mw: Math.round(z * 280),
    };
  });
}

// ─── MLOps Drift Monitoring ───
export interface DriftPoint {
  date: string;
  ks_statistic: number;
  rolling_rmse: number;
  threshold: number;
  is_drift: boolean;
}

export function mockDriftData(days: number = 30): DriftPoint[] {
  return Array.from({ length: days }, (_, i) => {
    const date = new Date('2026-01-08');
    date.setDate(date.getDate() + i);
    const ks = 0.03 + 0.02 * Math.random() + (i > 22 ? 0.04 * Math.random() : 0);
    const rmse = 320 + 30 * Math.random() + (i > 25 ? 80 * Math.random() : 0);
    return {
      date: date.toISOString().slice(0, 10),
      ks_statistic: Math.round(ks * 1000) / 1000,
      rolling_rmse: Math.round(rmse),
      threshold: 0.08,
      is_drift: ks > 0.08,
    };
  });
}

// ─── Pareto Frontier (Cost vs Carbon) ───
export interface ParetoPoint {
  carbon_weight: number;
  total_cost_eur: number;
  total_carbon_kg: number;
  cost_savings_pct: number;
  carbon_reduction_pct: number;
}

export function mockParetoFrontier(): ParetoPoint[] {
  return [
    { carbon_weight: 0, total_cost_eur: 285000, total_carbon_kg: 52000, cost_savings_pct: 22.1, carbon_reduction_pct: 12.3 },
    { carbon_weight: 5, total_cost_eur: 292000, total_carbon_kg: 46500, cost_savings_pct: 20.2, carbon_reduction_pct: 21.5 },
    { carbon_weight: 10, total_cost_eur: 301000, total_carbon_kg: 41200, cost_savings_pct: 17.7, carbon_reduction_pct: 30.5 },
    { carbon_weight: 15, total_cost_eur: 312000, total_carbon_kg: 37800, cost_savings_pct: 14.7, carbon_reduction_pct: 36.2 },
    { carbon_weight: 20, total_cost_eur: 323500, total_carbon_kg: 34100, cost_savings_pct: 11.6, carbon_reduction_pct: 42.5 },
    { carbon_weight: 30, total_cost_eur: 345000, total_carbon_kg: 30800, cost_savings_pct: 5.7, carbon_reduction_pct: 48.1 },
    { carbon_weight: 50, total_cost_eur: 378000, total_carbon_kg: 28200, cost_savings_pct: -3.3, carbon_reduction_pct: 52.4 },
  ];
}
