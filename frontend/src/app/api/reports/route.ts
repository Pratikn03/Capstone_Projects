import { NextResponse } from 'next/server';

import { loadReportsOverview } from '@/lib/server/reports';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const data = await loadReportsOverview();
  return NextResponse.json(data);
}
