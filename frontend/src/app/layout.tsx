import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ORIUS Research Console',
  description:
    'Research evidence console for ORIUS theorem status, reproducibility, domain gates, runtime safety, and artifact lineage.',
  icons: {
    icon: '/icon.svg',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
