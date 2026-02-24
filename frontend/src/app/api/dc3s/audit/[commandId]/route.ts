import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function resolveApiBase() {
  return process.env.FASTAPI_URL || 'http://localhost:8000';
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ commandId: string }> }
) {
  const { commandId } = await params;
  if (!commandId) {
    return NextResponse.json({ ok: false, error: 'Missing commandId.' }, { status: 400 });
  }

  try {
    const apiBase = resolveApiBase();
    const upstream = await fetch(`${apiBase}/dc3s/audit/${encodeURIComponent(commandId)}`, {
      cache: 'no-store',
    });

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
