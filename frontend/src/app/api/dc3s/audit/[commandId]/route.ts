import { NextResponse } from 'next/server';

import { fetchFastApi } from '@/lib/server/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ commandId: string }> }
) {
  const { commandId } = await params;
  if (!commandId || !/^[A-Za-z0-9._:-]{1,160}$/.test(commandId)) {
    return NextResponse.json({ ok: false, error: 'Missing commandId.' }, { status: 400 });
  }

  try {
    const upstream = await fetchFastApi(`/dc3s/audit/${encodeURIComponent(commandId)}`);

    const text = await upstream.text();
    let payload: unknown = null;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      payload = { raw: text };
    }

    if (!upstream.ok) {
      return NextResponse.json(
        {
          ok: false,
          error: `Audit fetch failed (${upstream.status})`,
          detail: payload,
        },
        { status: upstream.status }
      );
    }

    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
