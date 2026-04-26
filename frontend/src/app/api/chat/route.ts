import { openai } from '@ai-sdk/openai';
import { streamText, tool } from 'ai';
import { z } from 'zod';

import { chatModel, llmReady } from '@/lib/server/config';

import { chatToolExecutors } from './tool-executors';

export const maxDuration = 60;

const ChatRequestSchema = z.object({
  messages: z.array(z.unknown()).min(1).max(40),
});

export async function POST(req: Request) {
  if (!llmReady()) {
    return Response.json(
      { error: 'Chat assistant is not configured. Set OPENAI_API_KEY on the dashboard server.' },
      { status: 503 }
    );
  }

  const parsed = ChatRequestSchema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) {
    return Response.json({ error: 'Invalid chat payload.' }, { status: 400 });
  }

  const result = streamText({
    model: openai(chatModel()),
    system: `You are ORIUS AI, an expert grid operator assistant for an energy management system.
You manage forecasting (load, wind, solar), battery dispatch optimization, carbon tracking, and anomaly detection.
Available regions: Germany (DE) using OPSD data, USA (US) using EIA-930 data.
Models: GBM (LightGBM), LSTM, TCN — all trained for 50 epochs with CosineAnnealingLR scheduler.

When the user asks about forecasts, grid status, optimization, or battery schedules, call the appropriate tool.
Keep responses concise and operator-focused. Use MW, MWh, €, tCO₂ units.`,
    messages: parsed.data.messages as any,
    tools: {
      get_dispatch_forecast: tool({
        description:
          'Get economic dispatch forecast showing generation mix vs load for a zone. Returns data for rendering a stacked area chart.',
        parameters: z.object({
          zoneId: z.enum(['DE', 'US']).describe('Grid zone'),
          horizonHours: z.number().default(24).describe('Forecast horizon in hours'),
        }),
        execute: chatToolExecutors.get_dispatch_forecast,
      }),

      get_load_forecast: tool({
        description:
          'Get load/wind/solar forecast with 90% prediction intervals from conformal prediction.',
        parameters: z.object({
          target: z.enum(['load_mw', 'wind_mw', 'solar_mw']).describe('Forecast target'),
          zoneId: z.enum(['DE', 'US']).default('DE'),
          horizonHours: z.number().default(24),
        }),
        execute: chatToolExecutors.get_load_forecast,
      }),

      get_battery_schedule: tool({
        description: 'Show battery state-of-charge and charge/discharge schedule for optimized dispatch.',
        parameters: z.object({
          zoneId: z.enum(['DE', 'US']).default('DE'),
          horizonHours: z.number().default(24),
        }),
        execute: chatToolExecutors.get_battery_schedule,
      }),

      get_grid_status: tool({
        description:
          'Get current grid status overview for all zones including load, renewable fraction, and anomalies.',
        parameters: z.object({}),
        execute: chatToolExecutors.get_grid_status,
      }),

      get_cost_carbon_tradeoff: tool({
        description: 'Show cost vs carbon Pareto frontier from multi-objective dispatch optimization.',
        parameters: z.object({
          zoneId: z.enum(['DE', 'US']).default('DE'),
          horizonHours: z.number().default(24),
        }),
        execute: chatToolExecutors.get_cost_carbon_tradeoff,
      }),

      get_model_info: tool({
        description: 'Get model registry information including training details and metrics.',
        parameters: z.object({}),
        execute: chatToolExecutors.get_model_info,
      }),
    },
  });

  return result.toDataStreamResponse();
}
