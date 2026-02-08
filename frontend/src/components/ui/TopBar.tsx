'use client';

import { Globe, Clock, Bell, User, ChevronDown } from 'lucide-react';
import { useState } from 'react';

interface TopBarProps {
  region: string;
  onRegionChange?: (region: string) => void;
}

const regions = [
  { id: 'DE', label: 'Germany (OPSD)', flag: '🇩🇪' },
  { id: 'US', label: 'USA (EIA-930)', flag: '🇺🇸' },
];

export function TopBar({ region, onRegionChange }: TopBarProps) {
  const [showRegions, setShowRegions] = useState(false);
  const currentRegion = regions.find((r) => r.id === region) || regions[0];

  return (
    <header className="h-14 flex items-center justify-between px-6 border-b border-white/6 bg-grid-dark/80 backdrop-blur-sm sticky top-0 z-20">
      {/* Left: Status + Region */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs text-slate-300">
          <span className="w-2 h-2 rounded-full bg-energy-primary animate-pulse" />
          LIVE
        </div>

        {/* Region Selector */}
        <div className="relative">
          <button
            onClick={() => setShowRegions(!showRegions)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-sm text-slate-200 hover:bg-white/10 transition-all"
          >
            <Globe className="w-4 h-4 text-energy-info" />
            <span>{currentRegion.flag} {currentRegion.label}</span>
            <ChevronDown className="w-3 h-3 text-slate-500" />
          </button>
          {showRegions && (
            <div className="absolute top-full mt-1 left-0 glass-panel rounded-lg py-1 min-w-[200px]">
              {regions.map((r) => (
                <button
                  key={r.id}
                  onClick={() => {
                    onRegionChange?.(r.id);
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
          Model: GBM
        </div>
      </div>

      {/* Right: Time + Notifications + Profile */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Clock className="w-3.5 h-3.5" />
          <span className="font-mono">2026-02-07 05:46 UTC</span>
        </div>

        <button className="relative p-2 rounded-lg hover:bg-white/5 transition-colors">
          <Bell className="w-4 h-4 text-slate-400" />
          <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-energy-alert" />
        </button>

        <div className="flex items-center gap-2 pl-3 border-l border-white/10">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-energy-primary to-energy-info flex items-center justify-center text-xs font-bold text-white">
            P
          </div>
          <span className="text-sm text-slate-300">Pratik N</span>
        </div>
      </div>
    </header>
  );
}
