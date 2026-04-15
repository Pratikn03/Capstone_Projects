'use client';

import { Sidebar } from '@/components/ui/Sidebar';
import { TopBar } from '@/components/ui/TopBar';
import { ChatAssistant } from '@/components/ai/ChatAssistant';
import { RegionProvider } from '@/components/ui/RegionContext';
import { NotificationProvider } from '@/lib/notifications';
import { motion } from 'framer-motion';
import { usePathname } from 'next/navigation';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <NotificationProvider>
    <RegionProvider>
      <div className="flex min-h-screen bg-grid-deeper">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <TopBar />
          <main className="flex-1 overflow-auto">
            <motion.div
              key={pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
            >
              {children}
            </motion.div>
          </main>
        </div>
        <ChatAssistant />
      </div>
    </RegionProvider>
    </NotificationProvider>
  );
}
