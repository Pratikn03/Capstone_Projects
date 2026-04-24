import { NextResponse } from 'next/server';

import { backendTimeoutMs, fastApiBaseUrl, fetchFastApi, llmReady } from '@/lib/server/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const started = Date.now();
  let backend: { ok: boolean; status?: number; latency_ms?: number; error?: string } = { ok: false };
  let baseUrl = 'invalid';

  try {
    baseUrl = fastApiBaseUrl();
    const response = await fetchFastApi('/health', {}, { timeoutMs: Math.min(backendTimeoutMs(), 5_000) });
    backend = {
      ok: response.ok,
      status: response.status,
      latency_ms: Date.now() - started,
    };
  } catch (error) {
    backend = {
      ok: false,
      latency_ms: Date.now() - started,
      error: error instanceof Error ? error.message : String(error),
    };
  }

  return NextResponse.json({
    ok: true,
    service: 'orius-dashboard',
    generated_at: new Date().toISOString(),
    backend: {
      base_url: baseUrl,
      ...backend,
    },
    integrations: {
      openai_configured: llmReady(),
      dashboard_auth_enabled: Boolean(process.env.DASHBOARD_BASIC_AUTH),
    },
  });
}
