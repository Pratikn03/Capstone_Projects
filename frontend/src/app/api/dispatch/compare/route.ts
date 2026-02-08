import { NextResponse } from 'next/server';

import { loadDispatchCompare } from '@/lib/server/dispatch';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const region = searchParams.get('region') || 'DE';
  const horizonRaw = searchParams.get('horizon');
  const horizon = horizonRaw ? Number(horizonRaw) : 24;
  const data = await loadDispatchCompare(region, Number.isFinite(horizon) ? horizon : 24);
  return NextResponse.json(data);
}
