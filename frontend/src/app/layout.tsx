import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'GridPulse — Smart Energy Management',
  description:
    'Next-generation energy management system with ML forecasting, battery dispatch optimization, and natural language query interface.',
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
