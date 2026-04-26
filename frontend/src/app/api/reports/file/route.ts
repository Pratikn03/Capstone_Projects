import fs from 'fs/promises';
import path from 'path';

import { fetchFastApi } from '@/lib/server/config';
import { resolveReportsDir } from '@/lib/server/reports';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MIME_TYPES: Record<string, string> = {
  '.csv': 'text/csv; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml; charset=utf-8',
  '.tex': 'text/plain; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
};

function resolveSafeReportPath(requested: string): string | null {
  const reportsDir = path.resolve(resolveReportsDir());
  const normalized = requested.replace(/\\/g, '/').replace(/^\/+/, '');
  const candidate = path.resolve(reportsDir, normalized);
  if (candidate !== reportsDir && !candidate.startsWith(`${reportsDir}${path.sep}`)) {
    return null;
  }
  return candidate;
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const requested = searchParams.get('path');

  if (!requested) {
    return new Response('Missing path parameter.', { status: 400 });
  }

  try {
    const res = await fetchFastApi(`/research/reports/file?path=${encodeURIComponent(requested)}`);
    if (res.ok) {
      const data = await res.arrayBuffer();
      return new Response(data, {
        headers: {
          'Content-Type': res.headers.get('content-type') ?? 'application/octet-stream',
          'Content-Disposition': res.headers.get('content-disposition') ?? 'attachment',
        },
      });
    }
  } catch {
    // Fall through to local artifact serving. The dashboard must remain usable
    // from a frozen artifact bundle when the live research backend is offline.
  }

  const safePath = resolveSafeReportPath(requested);
  if (!safePath) {
    return new Response('Invalid report path.', { status: 400 });
  }

  try {
    const data = await fs.readFile(safePath);
    const fileName = path.basename(safePath).replace(/["\r\n]/g, '');
    return new Response(data, {
      headers: {
        'Content-Type': MIME_TYPES[path.extname(safePath).toLowerCase()] ?? 'application/octet-stream',
        'Content-Disposition': `attachment; filename="${fileName}"`,
      },
    });
  } catch {
    return new Response('Not found.', { status: 404 });
  }
}
