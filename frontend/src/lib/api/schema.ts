import { z } from 'zod';

// ─── Grid State (Real-time SCADA) ───
export const GridStateSchema = z.object({
  timestamp: z.string().datetime(),
  frequency_hz: z.number().min(59).max(61),
  active_power_mw: z.number(),
  reactive_power_mvar: z.number(),
  voltage_pu: z.number().optional(),
  zone_id: z.string().optional(),
});
export type GridState = z.infer<typeof GridStateSchema>;

// ─── Dispatch Forecast ───
export const DispatchPointSchema = z.object({
  timestamp: z.string(),
  load_mw: z.number(),
  generation_solar: z.number(),
  generation_wind: z.number(),
  generation_gas: z.number(),
  battery_dispatch: z.number(), // +Discharge, -Charge
  price_eur_mwh: z.number().optional(),
  carbon_kg_mwh: z.number().optional(),
});
export type DispatchPoint = z.infer<typeof DispatchPointSchema>;

export const DispatchForecastSchema = z.object({
  zone_id: z.string(),
  horizon_hours: z.number(),
  generated_at: z.string(),
  data: z.array(DispatchPointSchema),
  summary: z.object({
    total_load_mwh: z.number(),
    renewable_fraction: z.number(),
    peak_load_mw: z.number(),
    avg_price: z.number().optional(),
    total_cost: z.number().optional(),
    total_carbon_kg: z.number().optional(),
  }),
});
export type DispatchForecast = z.infer<typeof DispatchForecastSchema>;

// ─── Battery / SOC ───
export const BatteryStateSchema = z.object({
  timestamp: z.string(),
  soc_percent: z.number().min(0).max(100),
  power_mw: z.number(), // +Discharge, -Charge
  capacity_mwh: z.number(),
  cycles_today: z.number(),
});
export type BatteryState = z.infer<typeof BatteryStateSchema>;

export const BatteryScheduleSchema = z.object({
  zone_id: z.string(),
  schedule: z.array(BatteryStateSchema),
  metrics: z.object({
    cost_savings_eur: z.number(),
    carbon_reduction_kg: z.number(),
    peak_shaving_pct: z.number(),
    avg_efficiency: z.number(),
  }),
});
export type BatterySchedule = z.infer<typeof BatteryScheduleSchema>;

// ─── Anomaly Detection ───
export const AnomalySchema = z.object({
  id: z.string(),
  timestamp: z.string(),
  type: z.enum(['load_spike', 'load_drop', 'solar_drop', 'solar_surge', 'wind_ramp', 'wind_drop', 'frequency_deviation', 'battery_fault', 'sensor_fault']),
  severity: z.enum(['low', 'medium', 'high', 'critical']),
  status: z.enum(['active', 'investigating', 'resolved']),
  zone_id: z.string(),
  description: z.string(),
  value: z.number().optional(),
  threshold: z.number().optional(),
});
export type Anomaly = z.infer<typeof AnomalySchema>;

// ─── Forecast Metrics ───
export const ForecastMetricsSchema = z.object({
  target: z.string(),
  model: z.string(),
  rmse: z.number(),
  mae: z.number(),
  mape: z.number().optional(),
  r2: z.number().optional(),
  coverage_90: z.number().optional(),
});
export type ForecastMetrics = z.infer<typeof ForecastMetricsSchema>;

// ─── Zone Summary ───
export const ZoneSummarySchema = z.object({
  zone_id: z.string(),
  name: z.string(),
  status: z.enum(['normal', 'warning', 'critical']),
  load_mw: z.number(),
  generation_mw: z.number(),
  renewable_pct: z.number(),
  frequency_hz: z.number(),
  anomaly_count: z.number(),
});
export type ZoneSummary = z.infer<typeof ZoneSummarySchema>;

// ─── Optimization Result ───
export const OptimizationResultSchema = z.object({
  zone_id: z.string(),
  objective: z.enum(['cost', 'carbon', 'balanced']),
  status: z.enum(['optimal', 'feasible', 'infeasible']),
  total_cost_eur: z.number(),
  total_carbon_kg: z.number(),
  cost_savings_pct: z.number(),
  carbon_reduction_pct: z.number(),
  dispatch_schedule: z.array(DispatchPointSchema),
  battery_schedule: z.array(BatteryStateSchema),
});
export type OptimizationResult = z.infer<typeof OptimizationResultSchema>;
