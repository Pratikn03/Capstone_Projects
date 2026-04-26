import { NextResponse } from 'next/server';

import { loadDispatchCompare } from '@/lib/server/dispatch';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const region = (searchParams.get('region') || 'DE').toUpperCase();
  const horizonRaw = searchParams.get('horizon');
  const horizon = Math.max(1, Math.min(168, horizonRaw ? Number(horizonRaw) : 24));
  if (region !== 'DE' && region !== 'US') {
    return NextResponse.json({ error: 'Invalid region. Use DE or US.' }, { status: 400 });
  }
  const data = await loadDispatchCompare(region, Number.isFinite(horizon) ? horizon : 24);
  return NextResponse.json(data);
}
