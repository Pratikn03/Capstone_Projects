import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function getFastApiBase(): string {
  return process.env.FASTAPI_URL || 'http://localhost:8000';
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const region = (searchParams.get('region') || 'DE').toUpperCase() as 'DE' | 'US';

  if (region !== 'DE' && region !== 'US') {
    return NextResponse.json({ error: 'Invalid region. Use DE or US.' }, { status: 400 });
  }

  const base = getFastApiBase();
  const [regionRes, manifestRes] = await Promise.all([
    fetch(`${base}/research/region/${region}`, { cache: 'no-store' }),
    fetch(`${base}/research/manifest`, { cache: 'no-store' }),
  ]);
  if (!regionRes.ok || !manifestRes.ok) {
    return NextResponse.json({ error: 'Research backend unavailable.' }, { status: 502 });
  }
  const [data, manifest] = await Promise.all([regionRes.json(), manifestRes.json()]);

  return NextResponse.json({
    ...data,
    manifest: manifest ? { generated_at: manifest.frozen_at_utc, regions: Object.keys(manifest.dataset_profiles ?? {}) } : null,
  });
}
