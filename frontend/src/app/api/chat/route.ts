import { openai } from '@ai-sdk/openai';
import { streamText, tool } from 'ai';
import { z } from 'zod';

export const maxDuration = 60;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai('gpt-4o'),
    system: `You are GridPulse AI, an expert grid operator assistant for an energy management system.
You manage forecasting (load, wind, solar), battery dispatch optimization, carbon tracking, and anomaly detection.
Available regions: Germany (DE) using OPSD data, USA (US) using EIA-930 data.
Models: GBM (LightGBM), LSTM, TCN — all trained for 50 epochs with CosineAnnealingLR scheduler.

When the user asks about forecasts, grid status, optimization, or battery schedules, call the appropriate tool.
Keep responses concise and operator-focused. Use MW, MWh, €, tCO₂ units.`,
    messages,
    tools: {
      get_dispatch_forecast: tool({
        description: 'Get economic dispatch forecast showing generation mix vs load for a zone. Returns data for rendering a stacked area chart.',
        parameters: z.object({
          zoneId: z.enum(['DE', 'US']).describe('Grid zone'),
          horizonHours: z.number().default(24).describe('Forecast horizon in hours'),
        }),
        execute: async ({ zoneId, horizonHours }) => ({
          type: 'dispatch_chart',
          zoneId,
          horizonHours,
          summary: {
            peak_load_mw: zoneId === 'DE' ? 8950 : 145200,
            renewable_fraction: zoneId === 'DE' ? 0.624 : 0.387,
            avg_price: zoneId === 'DE' ? 42.5 : undefined,
            total_carbon_kg: zoneId === 'DE' ? 41200 : undefined,
          },
        }),
      }),

      get_load_forecast: tool({
        description: 'Get load/wind/solar forecast with 90% prediction intervals from conformal prediction.',
        parameters: z.object({
          target: z.enum(['load_mw', 'wind_mw', 'solar_mw']).describe('Forecast target'),
          zoneId: z.enum(['DE', 'US']).default('DE'),
        }),
        execute: async ({ target, zoneId }) => ({
          type: 'forecast_chart',
          target,
          zoneId,
          metrics: {
            rmse: target === 'load_mw' ? 348.45 : target === 'wind_mw' ? 184.18 : 5.20,
            coverage_90: target === 'load_mw' ? 93.2 : target === 'wind_mw' ? 91.4 : 94.1,
            model: 'GBM (LightGBM)',
          },
        }),
      }),

      get_battery_schedule: tool({
        description: 'Show battery state-of-charge and charge/discharge schedule for optimized dispatch.',
        parameters: z.object({
          zoneId: z.enum(['DE', 'US']).default('DE'),
        }),
        execute: async ({ zoneId }) => ({
          type: 'battery_chart',
          zoneId,
          metrics: {
            cost_savings_eur: 23500,
            carbon_reduction_kg: 47800,
            peak_shaving_pct: 19.5,
            capacity_mwh: 20000,
          },
        }),
      }),

      get_grid_status: tool({
        description: 'Get current grid status overview for all zones including load, renewable fraction, and anomalies.',
        parameters: z.object({}),
        execute: async () => ({
          type: 'grid_status',
          zones: [
            { zone_id: 'DE', name: 'Germany (OPSD)', status: 'normal', load_mw: 8420, renewable_pct: 62.4, frequency_hz: 50.01, anomaly_count: 2 },
            { zone_id: 'US', name: 'USA (EIA-930)', status: 'warning', load_mw: 142500, renewable_pct: 38.7, frequency_hz: 60.00, anomaly_count: 1 },
          ],
        }),
      }),

      get_cost_carbon_tradeoff: tool({
        description: 'Show cost vs carbon Pareto frontier from multi-objective dispatch optimization.',
        parameters: z.object({
          zoneId: z.enum(['DE', 'US']).default('DE'),
        }),
        execute: async ({ zoneId }) => ({
          type: 'pareto_chart',
          zoneId,
          optimal_point: { carbon_weight: 20, cost_savings_pct: 17.4, carbon_reduction_pct: 32.6 },
        }),
      }),

      get_model_info: tool({
        description: 'Get model registry information including training details and metrics.',
        parameters: z.object({}),
        execute: async () => ({
          type: 'model_info',
          models: [
            { target: 'load_mw', best_model: 'GBM', rmse: 348.45, r2: 0.967, coverage: 93.2 },
            { target: 'wind_mw', best_model: 'GBM', rmse: 184.18, r2: 0.912, coverage: 91.4 },
            { target: 'solar_mw', best_model: 'GBM', rmse: 5.20, r2: 0.945, coverage: 94.1 },
          ],
          training: { epochs: 50, scheduler: 'CosineAnnealingLR', early_stopping: false },
          datasets: ['OPSD Germany', 'EIA-930 USA'],
          last_trained: '2026-02-07',
        }),
      }),
    },
  });

  return result.toDataStreamResponse();
}
