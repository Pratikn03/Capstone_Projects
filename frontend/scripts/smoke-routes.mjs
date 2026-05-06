#!/usr/bin/env node

const OK_ROUTES = [
  '/',
  '/theorems',
  '/safety',
  '/domains',
  '/forecasting',
  '/optimization',
  '/reports',
  '/data',
];

const REDIRECT_ROUTES = [
  { from: '/carbon', to: '/optimization' },
  { from: '/anomalies', to: '/safety' },
  { from: '/monitoring', to: '/safety' },
];

function printHelp() {
  console.log(`Usage: node scripts/smoke-routes.mjs [--base-url URL] [--timeout-ms N] [--wait]

Checks the ORIUS dashboard route contract:
  - research console pages return 2xx
  - legacy routes redirect to their ORIUS evidence surfaces
  - server HTML does not contain literal NaN values

Environment:
  SMOKE_BASE_URL or NEXT_PUBLIC_SITE_URL can provide the base URL.
`);
}

function parseArgs(argv) {
  const args = {
    baseUrl: process.env.SMOKE_BASE_URL || process.env.NEXT_PUBLIC_SITE_URL || 'http://127.0.0.1:3000',
    timeoutMs: 10_000,
    wait: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
    if (arg === '--wait') {
      args.wait = true;
      continue;
    }
    if (arg === '--base-url') {
      args.baseUrl = argv[index + 1] ?? args.baseUrl;
      index += 1;
      continue;
    }
    if (arg.startsWith('--base-url=')) {
      args.baseUrl = arg.slice('--base-url='.length);
      continue;
    }
    if (arg === '--timeout-ms') {
      args.timeoutMs = Number(argv[index + 1] ?? args.timeoutMs);
      index += 1;
      continue;
    }
    if (arg.startsWith('--timeout-ms=')) {
      args.timeoutMs = Number(arg.slice('--timeout-ms='.length));
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) {
    throw new Error('--timeout-ms must be a positive number');
  }

  return {
    ...args,
    baseUrl: args.baseUrl.replace(/\/+$/u, ''),
  };
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function waitForServer(baseUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;

  while (Date.now() < deadline) {
    try {
      const res = await fetchWithTimeout(new URL('/api/health', baseUrl), { redirect: 'manual' }, 2_000);
      if (res.ok) return;
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  throw new Error(`Server did not become healthy at ${baseUrl}: ${lastError?.message ?? 'health check failed'}`);
}

function redirectMatches(location, expectedPath, baseUrl) {
  if (!location) return false;
  try {
    return new URL(location, baseUrl).pathname === expectedPath;
  } catch {
    return location === expectedPath;
  }
}

async function checkOkRoute(baseUrl, route, timeoutMs) {
  const url = new URL(route, baseUrl);
  const res = await fetchWithTimeout(url, { redirect: 'manual' }, timeoutMs);
  const html = await res.text();
  if (res.status < 200 || res.status >= 300) {
    throw new Error(`${route} returned ${res.status}`);
  }
  if (/\bNaN\b/u.test(html)) {
    throw new Error(`${route} server HTML contains literal NaN`);
  }
  return `${route} ${res.status}`;
}

async function checkRedirectRoute(baseUrl, route, timeoutMs) {
  const url = new URL(route.from, baseUrl);
  const res = await fetchWithTimeout(url, { redirect: 'manual' }, timeoutMs);
  const location = res.headers.get('location');
  if (![301, 302, 303, 307, 308].includes(res.status) || !redirectMatches(location, route.to, baseUrl)) {
    throw new Error(`${route.from} expected redirect to ${route.to}, got ${res.status} ${location ?? ''}`.trim());
  }
  return `${route.from} ${res.status} ${location}`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.wait) {
    await waitForServer(args.baseUrl, Math.max(args.timeoutMs, 30_000));
  }

  const results = [];
  for (const route of OK_ROUTES) {
    results.push(await checkOkRoute(args.baseUrl, route, args.timeoutMs));
  }
  for (const route of REDIRECT_ROUTES) {
    results.push(await checkRedirectRoute(args.baseUrl, route, args.timeoutMs));
  }

  console.log(`ORIUS route smoke passed for ${args.baseUrl}`);
  for (const result of results) console.log(`  ${result}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
