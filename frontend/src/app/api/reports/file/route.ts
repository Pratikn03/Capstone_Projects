import fs from 'fs/promises';
import path from 'path';

import { resolveReportsDir } from '@/lib/server/reports';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const CONTENT_TYPES: Record<string, string> = {
  '.md': 'text/markdown; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
};

function resolveSafePath(reportsDir: string, requestPath: string): string | null {
  const normalized = path.normalize(requestPath).replace(/^(\.\.(\/|\\|$))+/, '');
  const resolved = path.resolve(reportsDir, normalized);
  if (!resolved.startsWith(reportsDir + path.sep)) {
    return null;
  }
  return resolved;
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const requested = searchParams.get('path');

  if (!requested) {
    return new Response('Missing path parameter.', { status: 400 });
  }

  const reportsDir = resolveReportsDir();
  const safePath = resolveSafePath(reportsDir, requested);
  if (!safePath) {
    return new Response('Invalid path.', { status: 400 });
  }

  try {
    const stat = await fs.stat(safePath);
    if (!stat.isFile()) {
      return new Response('Not found.', { status: 404 });
    }
    const data = await fs.readFile(safePath);
    const ext = path.extname(safePath).toLowerCase();
    const contentType = CONTENT_TYPES[ext] ?? 'application/octet-stream';
    return new Response(data, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${path.basename(safePath)}"`,
      },
    });
  } catch {
    return new Response('Not found.', { status: 404 });
  }
}
