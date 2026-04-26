'use client';

import { useState, useEffect } from 'react';
import { Panel } from '@/components/ui/Panel';
import { useRegion } from '@/components/ui/RegionContext';
import { RefreshCw, Globe, Bell, Layout, BarChart3, Shield, Save } from 'lucide-react';
import { DOMAIN_OPTIONS, type DomainId } from '@/lib/domain-options';

interface Settings {
  dc3sRefreshInterval: number;
  defaultRegion: DomainId;
  notificationsEnabled: boolean;
  notifAnomalies: boolean;
  notifDrift: boolean;
  notifSafety: boolean;
  density: 'comfortable' | 'compact';
  animationsEnabled: boolean;
}

const defaults: Settings = {
  dc3sRefreshInterval: 15,
  defaultRegion: 'DE',
  notificationsEnabled: true,
  notifAnomalies: true,
  notifDrift: true,
  notifSafety: true,
  density: 'comfortable',
  animationsEnabled: true,
};

function loadSettings(): Settings {
  if (typeof window === 'undefined') return defaults;
  try {
    const raw = localStorage.getItem('gridpulse-settings');
    return raw ? { ...defaults, ...JSON.parse(raw) } : defaults;
  } catch {
    return defaults;
  }
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center justify-between py-2 cursor-pointer group">
      <span className="text-sm text-slate-300 group-hover:text-white transition-colors">{label}</span>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative w-10 h-5 rounded-full transition-colors ${
          checked ? 'bg-energy-primary' : 'bg-white/10'
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
            checked ? 'translate-x-5' : ''
          }`}
        />
      </button>
    </label>
  );
}

export default function SettingsPage() {
  const { setRegion } = useRegion();
  const [settings, setSettings] = useState<Settings>(defaults);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setSettings(loadSettings());
  }, []);

  const update = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const save = () => {
    localStorage.setItem('gridpulse-settings', JSON.stringify(settings));
    setRegion(settings.defaultRegion);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="p-6 space-y-6 max-w-3xl">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-xl font-bold text-white">Settings</h1>
          <p className="text-sm text-slate-400">Dashboard configuration and preferences</p>
        </div>
        <button
          onClick={save}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            saved
              ? 'bg-energy-primary/20 text-energy-primary border border-energy-primary/30'
              : 'bg-energy-primary text-grid-dark hover:bg-energy-primary/90'
          }`}
        >
          <Save className="w-4 h-4" />
          {saved ? 'Saved' : 'Save Settings'}
        </button>
      </div>

      {/* DC3S / Real-time */}
      <Panel title="Real-time Settings" accentColor="primary">
        <div className="space-y-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <RefreshCw className="w-4 h-4 text-energy-primary" />
              <span className="text-sm font-medium text-white">DC3S Refresh Interval</span>
              <span className="text-xs text-slate-500 ml-auto font-mono">{settings.dc3sRefreshInterval}s</span>
            </div>
            <input
              type="range"
              min={5}
              max={60}
              step={5}
              value={settings.dc3sRefreshInterval}
              onChange={(e) => update('dc3sRefreshInterval', Number(e.target.value))}
              className="w-full h-1.5 rounded-full appearance-none bg-white/10 accent-energy-primary cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-600 mt-1">
              <span>5s</span>
              <span>30s</span>
              <span>60s</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-energy-info" />
            <span className="text-sm text-slate-300">DC3S safety shield is always active in production.</span>
          </div>
        </div>
      </Panel>

      {/* Region */}
      <Panel title="Default Region" accentColor="info">
        <div className="flex items-center gap-2 mb-3">
          <Globe className="w-4 h-4 text-energy-info" />
          <span className="text-sm font-medium text-white">Primary Region</span>
        </div>
        <div className="grid grid-cols-2 gap-3">
          {DOMAIN_OPTIONS.map((r) => (
            <button
              key={r.id}
              onClick={() => update('defaultRegion', r.id)}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg border text-sm transition-all ${
                settings.defaultRegion === r.id
                  ? 'bg-energy-info/10 border-energy-info/30 text-white'
                  : 'bg-white/3 border-white/6 text-slate-400 hover:bg-white/5'
              }`}
            >
              <span className="text-lg">{r.flag}</span>
              <span>{r.label}</span>
            </button>
          ))}
        </div>
      </Panel>

      {/* Notifications */}
      <Panel title="Notifications" accentColor="warn">
        <div className="flex items-center gap-2 mb-3">
          <Bell className="w-4 h-4 text-energy-warn" />
          <span className="text-sm font-medium text-white">Alert Preferences</span>
        </div>
        <div className="space-y-1 divide-y divide-white/4">
          <Toggle
            checked={settings.notificationsEnabled}
            onChange={(v) => update('notificationsEnabled', v)}
            label="Enable notifications"
          />
          <Toggle
            checked={settings.notifAnomalies}
            onChange={(v) => update('notifAnomalies', v)}
            label="Anomaly alerts"
          />
          <Toggle
            checked={settings.notifDrift}
            onChange={(v) => update('notifDrift', v)}
            label="Model drift warnings"
          />
          <Toggle
            checked={settings.notifSafety}
            onChange={(v) => update('notifSafety', v)}
            label="Safety shield events"
          />
        </div>
      </Panel>

      {/* Display */}
      <Panel title="Display" accentColor="info">
        <div className="space-y-4">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Layout className="w-4 h-4 text-energy-info" />
              <span className="text-sm font-medium text-white">Density</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {(['comfortable', 'compact'] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => update('density', d)}
                  className={`px-4 py-2.5 rounded-lg border text-sm capitalize transition-all ${
                    settings.density === d
                      ? 'bg-energy-info/10 border-energy-info/30 text-white'
                      : 'bg-white/3 border-white/6 text-slate-400 hover:bg-white/5'
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-energy-info" />
            <Toggle
              checked={settings.animationsEnabled}
              onChange={(v) => update('animationsEnabled', v)}
              label="Chart animations"
            />
          </div>
        </div>
      </Panel>

      {/* Runtime notes - collapsed by default */}
      <Panel title="Runtime Surface" subtitle="Technical details" badge="Info" badgeColor="info" collapsible defaultCollapsed>
        <div className="space-y-3 text-sm text-slate-300">
          {[
            'The dashboard is backed by extracted dashboard JSON, report artifacts, and the live DC3S runtime route.',
            'The DC3S live shield card calls the FastAPI backend through /api/dc3s/live and falls back to an explicit artifact shadow when the live step route is unavailable.',
            'Other panels prefer repo artifacts and report exports over live streaming telemetry.',
            'Hardware validation and field commissioning remain future work.',
          ].map((note) => (
            <div key={note} className="rounded-lg bg-white/3 px-3 py-2.5">
              {note}
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
