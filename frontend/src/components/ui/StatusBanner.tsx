'use client';

import { AlertTriangle, Info } from 'lucide-react';

interface StatusBannerProps {
  title: string;
  messages: string[];
  tone?: 'warn' | 'info';
}

const toneClasses = {
  warn: 'border-energy-warn/20 bg-energy-warn/10 text-energy-warn',
  info: 'border-energy-info/20 bg-energy-info/10 text-energy-info',
};

export function StatusBanner({ title, messages, tone = 'warn' }: StatusBannerProps) {
  if (!messages.length) return null;

  const Icon = tone === 'warn' ? AlertTriangle : Info;

  return (
    <div className={`rounded-xl border px-4 py-3 ${toneClasses[tone]}`}>
      <div className="flex items-center gap-2 text-sm font-medium">
        <Icon className="h-4 w-4" />
        <span>{title}</span>
      </div>
      <div className="mt-2 space-y-1 text-xs text-slate-200">
        {messages.map((message) => (
          <div key={message}>{message}</div>
        ))}
      </div>
    </div>
  );
}
