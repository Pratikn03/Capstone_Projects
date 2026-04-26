'use client';

import { Globe, Clock, Bell, ChevronDown } from 'lucide-react';
import { useEffect, useState, useRef } from 'react';
import { useRegion } from './RegionContext';
import { NotificationPanel, type Notification as PanelNotification } from './NotificationPanel';
import { useNotifications } from '@/lib/notifications';
import { dashboardConfig, operatorInitial } from '@/lib/dashboard-config';
import { DOMAIN_OPTIONS } from '@/lib/domain-options';

export function TopBar() {
  const { region, setRegion } = useRegion();
  const { notifications: ctxNotifs, dismiss: ctxDismiss } = useNotifications();
  const [showRegions, setShowRegions] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [nowUtc, setNowUtc] = useState(() => new Date().toISOString().replace('T', ' ').slice(0, 16) + ' UTC');
  const currentRegion = DOMAIN_OPTIONS.find((r) => r.id === region) || DOMAIN_OPTIONS[0];
  const notifRef = useRef<HTMLDivElement>(null);

  // Build panel-compatible notifications from context
  const panelNotifications: PanelNotification[] = ctxNotifs
    .filter((n) => !n.dismissed)
    .map((n) => ({
      id: n.id,
      type: n.type as PanelNotification['type'],
      severity: n.severity as PanelNotification['severity'],
      title: n.title,
      message: n.message,
      timestamp: new Date(n.timestamp).toISOString(),
    }));

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNowUtc(new Date().toISOString().replace('T', ' ').slice(0, 16) + ' UTC');
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  // Close notification panel on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setShowNotifications(false);
      }
    }
    if (showNotifications) document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showNotifications]);

  const dismissNotification = (id: string) => {
    ctxDismiss(id);
  };

  const criticalCount = panelNotifications.filter((n) => n.severity === 'critical' || n.severity === 'warn').length;

  return (
    <header className="h-14 flex items-center justify-between px-6 border-b border-white/6 bg-grid-dark/80 backdrop-blur-sm sticky top-0 z-20">
      {/* Left: Status + Region */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs text-slate-300">
          <span className="w-2 h-2 rounded-full bg-energy-primary live-dot" />
          {dashboardConfig.appLabel}
        </div>

        {/* Region Selector */}
        <div className="relative">
          <button
            onClick={() => setShowRegions(!showRegions)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-sm text-slate-200 hover:bg-white/10 transition-all"
          >
            <Globe className="w-4 h-4 text-energy-info" />
            <span>{currentRegion.flag} {currentRegion.label}</span>
            <ChevronDown className={`w-3 h-3 text-slate-500 transition-transform ${showRegions ? 'rotate-180' : ''}`} />
          </button>
          {showRegions && (
            <div className="absolute top-full mt-1 left-0 glass-panel-elevated rounded-lg py-1 min-w-[200px] z-50">
              {DOMAIN_OPTIONS.map((r) => (
                <button
                  key={r.id}
                  onClick={() => {
                    setRegion(r.id);
                    try {
                      const current = JSON.parse(localStorage.getItem('gridpulse-settings') ?? '{}') as Record<string, unknown>;
                      localStorage.setItem('gridpulse-settings', JSON.stringify({ ...current, defaultRegion: r.id }));
                    } catch {
                      localStorage.setItem('gridpulse-settings', JSON.stringify({ defaultRegion: r.id }));
                    }
                    setShowRegions(false);
                  }}
                  className={`w-full text-left px-4 py-2 text-sm hover:bg-white/10 transition-colors ${
                    r.id === region ? 'text-energy-primary' : 'text-slate-300'
                  }`}
                >
                  {r.flag} {r.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Model Badge */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-energy-primary-dim border border-energy-primary/20 text-xs text-energy-primary">
          Shield: DC3S
        </div>
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/5 border border-white/10 text-xs text-slate-300">
          Data: Artifact + Live
        </div>
      </div>

      {/* Right: Time + Notifications + Profile */}
      <div className="flex items-center gap-4">
        <div className="hidden md:flex items-center gap-1.5 text-xs text-slate-400">
          <Clock className="w-3.5 h-3.5" />
          <span className="font-mono">{nowUtc}</span>
        </div>

        {/* Notifications */}
        <div className="relative" ref={notifRef}>
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            className={`relative p-2 rounded-lg transition-colors ${
              showNotifications ? 'bg-white/10' : 'hover:bg-white/5'
            }`}
          >
            <Bell className="w-4 h-4 text-slate-400" />
            {panelNotifications.length > 0 && (
              <span className={`absolute -top-0.5 -right-0.5 min-w-[16px] h-4 flex items-center justify-center rounded-full text-[9px] font-bold text-white px-1 ${
                criticalCount > 0 ? 'bg-energy-alert' : 'bg-energy-info'
              }`}>
                {panelNotifications.length}
              </span>
            )}
          </button>
          <NotificationPanel
            notifications={panelNotifications}
            open={showNotifications}
            onClose={() => setShowNotifications(false)}
            onDismiss={dismissNotification}
          />
        </div>

        <div className="flex items-center gap-2 pl-3 border-l border-white/10">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-energy-primary to-energy-info flex items-center justify-center text-xs font-bold text-white">
            {operatorInitial()}
          </div>
          <span className="hidden sm:inline text-sm text-slate-300">{dashboardConfig.operatorName}</span>
        </div>
      </div>
    </header>
  );
}
