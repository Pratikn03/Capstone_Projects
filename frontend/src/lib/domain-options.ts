export type DomainId = 'DE' | 'US' | 'AV' | 'HEALTHCARE';

export type DomainOption = {
  id: DomainId;
  label: string;
  shortLabel: string;
  flag: string;
  family: 'battery' | 'av' | 'healthcare';
  primaryTarget: string;
  primaryUnit: string;
};

export const DOMAIN_OPTIONS: DomainOption[] = [
  {
    id: 'DE',
    label: 'Germany (OPSD)',
    shortLabel: 'Germany',
    flag: '🇩🇪',
    family: 'battery',
    primaryTarget: 'load_mw',
    primaryUnit: 'MW',
  },
  {
    id: 'US',
    label: 'USA (EIA-930)',
    shortLabel: 'USA',
    flag: '🇺🇸',
    family: 'battery',
    primaryTarget: 'load_mw',
    primaryUnit: 'MW',
  },
  {
    id: 'AV',
    label: 'Autonomous Vehicles',
    shortLabel: 'AV',
    flag: '🚗',
    family: 'av',
    primaryTarget: 'true_margin',
    primaryUnit: 'm',
  },
  {
    id: 'HEALTHCARE',
    label: 'Healthcare Monitoring',
    shortLabel: 'Healthcare',
    flag: '🏥',
    family: 'healthcare',
    primaryTarget: 'spo2_proxy',
    primaryUnit: 'index',
  },
];

export function getDomainOption(id: string): DomainOption {
  return DOMAIN_OPTIONS.find((option) => option.id === id) ?? DOMAIN_OPTIONS[0];
}

export function isDomainId(value: string): value is DomainId {
  return DOMAIN_OPTIONS.some((option) => option.id === value);
}

export function isBatteryDomain(id: DomainId): boolean {
  return getDomainOption(id).family === 'battery';
}

export function targetOptionsForDomain(id: DomainId): Array<{ id: string; label: string; icon: string; unit: string }> {
  if (id === 'AV') {
    return [
      { id: 'true_margin', label: 'True Margin', icon: '🛡️', unit: 'm' },
      { id: 'safe_acceleration_mps2', label: 'Safe Accel', icon: '🚗', unit: 'm/s²' },
      { id: 'reliability_w', label: 'Reliability', icon: '📡', unit: 'w_t' },
    ];
  }
  if (id === 'HEALTHCARE') {
    return [
      { id: 'spo2_proxy', label: 'SpO₂ Proxy', icon: '🫁', unit: 'index' },
      { id: 'forecast', label: 'Prediction', icon: '📈', unit: 'index' },
      { id: 'reliability', label: 'Reliability', icon: '📡', unit: 'w_t' },
    ];
  }
  return [
    { id: 'load_mw', label: 'Load', icon: '⚡', unit: 'MW' },
    { id: 'wind_mw', label: 'Wind', icon: '💨', unit: 'MW' },
    { id: 'solar_mw', label: 'Solar', icon: '☀️', unit: 'MW' },
  ];
}
