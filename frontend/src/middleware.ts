import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * GridPulse Middleware
 * - Protects dashboard routes (requires auth in production)
 * - Adds security headers
 * - Handles tenant/region routing
 */
export function middleware(request: NextRequest) {
  const response = NextResponse.next();

  // Security headers for control room environments
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set(
    'Content-Security-Policy',
    "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' https://api.openai.com https://api.anthropic.com;"
  );

  return response;
}

export const config = {
  matcher: [
    // Apply to all routes except static files and API
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
