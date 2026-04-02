import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function getFastApiBase(): string {
  return process.env.FASTAPI_URL || 'http://localhost:8000';
}

export async function GET() {
  const res = await fetch(`${getFastApiBase()}/research/reports`, { cache: 'no-store' });
  if (!res.ok) {
    return NextResponse.json({ error: 'Research backend unavailable.' }, { status: 502 });
  }
  const data = await res.json();
  return NextResponse.json(data);
}
