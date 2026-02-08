'use client';

import { useState } from 'react';
import { Sidebar } from '@/components/ui/Sidebar';
import { TopBar } from '@/components/ui/TopBar';
import { Copilot } from '@/components/ai/Copilot';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [region, setRegion] = useState('DE');

  return (
    <div className="flex min-h-screen bg-grid-deeper">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <TopBar region={region} onRegionChange={setRegion} />
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
      <Copilot />
    </div>
  );
}
