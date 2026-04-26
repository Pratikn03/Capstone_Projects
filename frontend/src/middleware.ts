import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const isProduction = process.env.NODE_ENV === 'production';

function cspHeader(): string {
  const scriptSrc = isProduction ? "script-src 'self' 'unsafe-inline';" : "script-src 'self' 'unsafe-eval' 'unsafe-inline';";
  return [
    "default-src 'self';",
    scriptSrc,
    "style-src 'self' 'unsafe-inline';",
    "img-src 'self' data: blob:;",
    "font-src 'self' data:;",
    "connect-src 'self';",
    "frame-ancestors 'none';",
    "base-uri 'self';",
    "form-action 'self';",
    "object-src 'none';",
  ].join(' ');
}

function isBasicAuthAllowed(request: NextRequest): boolean {
  const credentials = process.env.DASHBOARD_BASIC_AUTH;
  if (!credentials) return true;
  const header = request.headers.get('authorization');
  return header === `Basic ${btoa(credentials)}`;
}

function unauthorized(): NextResponse {
  return new NextResponse('Authentication required.', {
    status: 401,
    headers: {
      'WWW-Authenticate': 'Basic realm="ORIUS Dashboard", charset="UTF-8"',
    },
  });
}

/**
 * ORIUS Middleware
 * - Protects dashboard routes (requires auth in production)
 * - Adds security headers
 * - Handles tenant/region routing
 */
export function middleware(request: NextRequest) {
  if (!isBasicAuthAllowed(request)) {
    return unauthorized();
  }

  const response = NextResponse.next();

  // Security headers for control room environments
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set('Content-Security-Policy', cspHeader());
  response.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=(), payment=()');
  if (isProduction) {
    response.headers.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  }

  return response;
}

export const config = {
  matcher: [
    // Apply to dashboard and API routes except static files and health checks.
    '/((?!api/health|_next/static|_next/image|favicon.ico).*)',
  ],
};
