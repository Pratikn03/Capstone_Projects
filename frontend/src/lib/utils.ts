import { clsx, type ClassValue } from 'clsx';

/**
 * Merge class names with conditional logic.
 * Lightweight alternative to clsx+twMerge for our controlled design system.
 */
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

/**
 * Format MW values with appropriate precision.
 */
export function formatMW(value: number): string {
  if (Math.abs(value) >= 1000) {
    return `${(value / 1000).toFixed(1)} GW`;
  }
  return `${value.toFixed(0)} MW`;
}

/**
 * Format currency values.
 */
export function formatCurrency(value: number, currency = 'EUR'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

/**
 * Format percentage values.
 */
export function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

/**
 * Format timestamp for chart axes.
 */
export function formatChartTime(isoString: string): string {
  const d = new Date(isoString);
  return `${d.getHours().toString().padStart(2, '0')}:00`;
}

/**
 * Generate a unique ID for messages.
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Delay utility for loading states.
 */
export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
