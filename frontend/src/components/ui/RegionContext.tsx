'use client';

import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { isDomainId, type DomainId } from '@/lib/domain-options';

export type RegionId = DomainId;

interface RegionContextValue {
  region: RegionId;
  setRegion: (region: RegionId) => void;
}

const RegionContext = createContext<RegionContextValue | undefined>(undefined);

function initialSettingsRegion(fallback: RegionId): RegionId {
  if (typeof window === 'undefined') return fallback;
  try {
    const parsed = JSON.parse(localStorage.getItem('gridpulse-settings') ?? '{}') as { defaultRegion?: RegionId };
    return parsed.defaultRegion && isDomainId(parsed.defaultRegion) ? parsed.defaultRegion : fallback;
  } catch {
    return fallback;
  }
}

export function RegionProvider({ children, initialRegion = 'DE' }: { children: ReactNode; initialRegion?: RegionId }) {
  const [region, setRegion] = useState<RegionId>(initialRegion);

  useEffect(() => {
    setRegion(initialSettingsRegion(initialRegion));
  }, [initialRegion]);

  const value = useMemo(() => ({ region, setRegion }), [region]);

  return <RegionContext.Provider value={value}>{children}</RegionContext.Provider>;
}

export function useRegion() {
  const ctx = useContext(RegionContext);
  if (!ctx) {
    throw new Error('useRegion must be used within RegionProvider');
  }
  return ctx;
}
