'use client';

import { createContext, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

export type RegionId = 'DE' | 'US';

interface RegionContextValue {
  region: RegionId;
  setRegion: (region: RegionId) => void;
}

const RegionContext = createContext<RegionContextValue | undefined>(undefined);

export function RegionProvider({ children, initialRegion = 'DE' }: { children: ReactNode; initialRegion?: RegionId }) {
  const [region, setRegion] = useState<RegionId>(initialRegion);
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
