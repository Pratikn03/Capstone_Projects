export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function getFastApiBase(): string {
  return process.env.FASTAPI_URL || 'http://localhost:8000';
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const requested = searchParams.get('path');

  if (!requested) {
    return new Response('Missing path parameter.', { status: 400 });
  }

  const res = await fetch(
    `${getFastApiBase()}/research/reports/file?path=${encodeURIComponent(requested)}`,
    { cache: 'no-store' }
  );
  if (!res.ok) {
    return new Response('Not found.', { status: 404 });
  }

  const data = await res.arrayBuffer();
  return new Response(data, {
    headers: {
      'Content-Type': res.headers.get('content-type') ?? 'application/octet-stream',
      'Content-Disposition': res.headers.get('content-disposition') ?? 'attachment',
    },
  });
}
