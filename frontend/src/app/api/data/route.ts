import { NextResponse } from 'next/server';

import { fetchFastApiJson } from '@/lib/server/config';
import { loadManifest, loadRegionData } from '@/lib/server/dataset';
import { isDomainId, type DomainId } from '@/lib/domain-options';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const region = (searchParams.get('region') || 'DE').toUpperCase() as DomainId;

  if (!isDomainId(region)) {
    return NextResponse.json({ error: 'Invalid region/domain. Use DE, US, AV, or HEALTHCARE.' }, { status: 400 });
  }

  try {
    const [data, manifest] = await Promise.all([
      fetchFastApiJson<Record<string, unknown>>(`/research/region/${region}`),
      fetchFastApiJson<Record<string, any>>('/research/manifest'),
    ]);

    return NextResponse.json({
      ...data,
      manifest: manifest
        ? { generated_at: manifest.frozen_at_utc, regions: Object.keys(manifest.dataset_profiles ?? {}) }
        : null,
      meta: { source: 'fastapi' },
    });
  } catch (error) {
    const [data, manifest] = await Promise.all([loadRegionData(region), loadManifest()]);
    return NextResponse.json({
      ...data,
      manifest: manifest
        ? { generated_at: manifest.generated_at, regions: Object.keys(manifest.regions ?? {}) }
        : null,
      meta: {
        source: 'local_artifacts',
        warning: error instanceof Error ? error.message : String(error),
      },
    });
  }
}
