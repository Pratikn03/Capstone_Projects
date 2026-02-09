'use client';

import { Sidebar } from '@/components/ui/Sidebar';
import { TopBar } from '@/components/ui/TopBar';
import { ChatAssistant } from '@/components/ai/ChatAssistant';
import { RegionProvider } from '@/components/ui/RegionContext';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <RegionProvider>
      <div className="flex min-h-screen bg-grid-deeper">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <TopBar />
          <main className="flex-1 overflow-auto">
            {children}
          </main>
        </div>
        <ChatAssistant />
      </div>
    </RegionProvider>
  );
}
