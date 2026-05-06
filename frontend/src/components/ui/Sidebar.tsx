'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  BarChart3,
  Zap,
  FileText,
  Database,
  Settings,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  ShieldCheck,
  Globe2,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { dashboardConfig, operatorInitial } from '@/lib/dashboard-config';

const navItems = [
  { href: '/', label: 'Research Hub', icon: LayoutDashboard },
  { href: '/theorems', label: 'Theorem Ledger', icon: BookOpen },
  { href: '/safety', label: 'Safety / OASG / DC3S', icon: ShieldCheck },
  { href: '/domains', label: 'Domain Evidence', icon: Globe2 },
  { href: '/forecasting', label: 'Forecasting Evidence', icon: BarChart3 },
  { href: '/optimization', label: 'Optimization / Cost-Safety Tradeoff', icon: Zap },
  { href: '/reports', label: 'Reports', icon: FileText },
  { href: '/data', label: 'Data Explorer', icon: Database },
];

const bottomItems = [
  { href: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [isNarrow, setIsNarrow] = useState(false);
  const effectiveCollapsed = collapsed || isNarrow;

  useEffect(() => {
    const media = window.matchMedia('(max-width: 768px)');
    const update = () => setIsNarrow(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  return (
    <motion.aside
      animate={{ width: effectiveCollapsed ? 68 : 292 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="glass-sidebar flex flex-col h-screen sticky top-0 z-30 select-none"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-white/6">
        <div className="w-8 h-8 rounded-lg bg-energy-primary flex items-center justify-center flex-shrink-0">
          <ShieldCheck className="w-4 h-4 text-grid-dark" />
        </div>
        <AnimatePresence>
          {!effectiveCollapsed && (
            <motion.span
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              className="text-lg font-bold text-white tracking-tight overflow-hidden whitespace-nowrap"
            >
              {dashboardConfig.appLabel}
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-2 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all duration-200
                ${isActive
                  ? 'bg-energy-primary/15 text-energy-primary border border-energy-primary/20'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                }
              `}
            >
              <item.icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-energy-primary' : ''}`} />
              <AnimatePresence>
                {!effectiveCollapsed && (
                  <motion.span
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    className="overflow-hidden whitespace-nowrap leading-tight"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </Link>
          );
        })}
      </nav>

      {/* Bottom Items */}
      <div className="py-4 px-2 space-y-1 border-t border-white/6">
        {/* User Profile */}
        <div className="flex items-center gap-3 px-3 py-2.5 mb-2">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-energy-primary to-energy-info flex items-center justify-center text-xs font-bold text-white flex-shrink-0">
            {operatorInitial()}
          </div>
          <AnimatePresence>
            {!effectiveCollapsed && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                className="overflow-hidden whitespace-nowrap"
              >
                <div className="text-sm font-medium text-white">{dashboardConfig.operatorName}</div>
                <div className="text-[10px] text-slate-500">{dashboardConfig.operatorRole}</div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {bottomItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-500 hover:text-slate-300 hover:bg-white/5 transition-all"
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            <AnimatePresence>
              {!effectiveCollapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="overflow-hidden whitespace-nowrap"
                >
                  {item.label}
                </motion.span>
              )}
            </AnimatePresence>
          </Link>
        ))}

        {/* Collapse Toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-500 hover:text-slate-300 hover:bg-white/5 transition-all w-full"
        >
          {effectiveCollapsed ? (
            <ChevronRight className="w-5 h-5 flex-shrink-0" />
          ) : (
            <>
              <ChevronLeft className="w-5 h-5 flex-shrink-0" />
              <span className="overflow-hidden whitespace-nowrap">Collapse</span>
            </>
          )}
        </button>
      </div>
    </motion.aside>
  );
}
