import { NextResponse } from 'next/server';

import { fetchFastApiJson } from '@/lib/server/config';
import { loadReportsOverview } from '@/lib/server/reports';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const data = await fetchFastApiJson<Record<string, unknown>>('/research/reports');
    return NextResponse.json(data);
  } catch (error) {
    const fallback = await loadReportsOverview();
    return NextResponse.json({
      ...fallback,
      meta: {
        ...fallback.meta,
        source: fallback.meta.source,
        warnings: [
          ...(fallback.meta.warnings ?? []),
          `FastAPI reports route unavailable; serving local artifacts: ${
            error instanceof Error ? error.message : String(error)
          }`,
        ],
      },
    });
  }
}
