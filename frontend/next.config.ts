/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow long-running server actions for AI + optimization roundtrips
  serverExternalPackages: [],
  experimental: {
    serverActions: {
      bodySizeLimit: '4mb',
    },
  },
  // Proxy API calls to FastAPI backend
  // Frontend calls /api/grid/* → FastAPI at http://localhost:8000/*
  async rewrites() {
    const backendUrl = process.env.FASTAPI_URL || 'http://localhost:8000';
    return [
      // Proxy all /api/grid/ calls to the FastAPI backend
      {
        source: '/api/grid/:path*',
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
