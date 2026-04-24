import 'server-only';

export const DEFAULT_FASTAPI_URL = 'http://localhost:8000';
const DEFAULT_TIMEOUT_MS = 12_000;

export class UpstreamError extends Error {
  status: number;

  constructor(message: string, status = 502) {
    super(message);
    this.name = 'UpstreamError';
    this.status = status;
  }
}

function cleanBaseUrl(raw: string): string {
  const parsed = new URL(raw);
  return parsed.toString().replace(/\/$/, '');
}

export function fastApiBaseUrl(): string {
  const raw = process.env.FASTAPI_URL || DEFAULT_FASTAPI_URL;
  try {
    return cleanBaseUrl(raw);
  } catch {
    throw new Error(`FASTAPI_URL is not a valid URL: ${raw}`);
  }
}

export function backendTimeoutMs(): number {
  const raw = process.env.FASTAPI_TIMEOUT_MS;
  if (!raw) return DEFAULT_TIMEOUT_MS;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 1_000 || parsed > 120_000) {
    return DEFAULT_TIMEOUT_MS;
  }
  return Math.trunc(parsed);
}

export function apiSecretHeader(): Record<string, string> {
  const value = process.env.API_SECRET_KEY?.trim();
  return value ? { 'X-ORIUS-Key': value } : {};
}

export function llmReady(): boolean {
  return Boolean(process.env.OPENAI_API_KEY?.trim());
}

export function chatModel(): string {
  return process.env.OPENAI_MODEL?.trim() || 'gpt-4o';
}

export async function fetchFastApi(
  routePath: string,
  init: RequestInit = {},
  options: { timeoutMs?: number } = {}
): Promise<Response> {
  const controller = new AbortController();
  const timeout = windowlessSetTimeout(() => controller.abort(), options.timeoutMs ?? backendTimeoutMs());
  const headers = new Headers(init.headers);
  for (const [key, value] of Object.entries(apiSecretHeader())) {
    headers.set(key, value);
  }
  if (init.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  try {
    return await fetch(`${fastApiBaseUrl()}${routePath}`, {
      cache: 'no-store',
      ...init,
      headers,
      signal: controller.signal,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new UpstreamError(`FastAPI request failed for ${routePath}: ${message}`, 502);
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchFastApiJson<T>(
  routePath: string,
  init: RequestInit = {},
  options: { timeoutMs?: number } = {}
): Promise<T> {
  const response = await fetchFastApi(routePath, init, options);
  if (!response.ok) {
    const detail = (await response.text().catch(() => '')).slice(0, 400);
    throw new UpstreamError(
      `FastAPI ${routePath} failed (${response.status})${detail ? `: ${detail}` : ''}`,
      response.status === 404 ? 404 : 502
    );
  }
  return (await response.json()) as T;
}

function windowlessSetTimeout(callback: () => void, ms: number): ReturnType<typeof setTimeout> {
  return setTimeout(callback, ms);
}
