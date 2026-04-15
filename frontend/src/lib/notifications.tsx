'use client';

import { createContext, useCallback, useContext, useReducer, type ReactNode } from 'react';

export type NotifSeverity = 'info' | 'warn' | 'critical';
export type NotifType = 'anomaly' | 'drift' | 'safety' | 'system';

export interface Notification {
  id: string;
  type: NotifType;
  severity: NotifSeverity;
  title: string;
  message: string;
  timestamp: number;
  dismissed: boolean;
}

interface NotificationState {
  items: Notification[];
}

type Action =
  | { type: 'ADD'; payload: Omit<Notification, 'id' | 'timestamp' | 'dismissed'> }
  | { type: 'DISMISS'; id: string }
  | { type: 'DISMISS_ALL' }
  | { type: 'CLEAR' };

let counter = 0;

function reducer(state: NotificationState, action: Action): NotificationState {
  switch (action.type) {
    case 'ADD':
      return {
        items: [
          {
            ...action.payload,
            id: `notif-${++counter}-${Date.now()}`,
            timestamp: Date.now(),
            dismissed: false,
          },
          ...state.items,
        ].slice(0, 50), // cap at 50
      };
    case 'DISMISS':
      return {
        items: state.items.map((n) => (n.id === action.id ? { ...n, dismissed: true } : n)),
      };
    case 'DISMISS_ALL':
      return { items: state.items.map((n) => ({ ...n, dismissed: true })) };
    case 'CLEAR':
      return { items: [] };
    default:
      return state;
  }
}

interface NotifContextValue {
  notifications: Notification[];
  activeCount: number;
  add: (n: Omit<Notification, 'id' | 'timestamp' | 'dismissed'>) => void;
  dismiss: (id: string) => void;
  dismissAll: () => void;
  clear: () => void;
}

const NotifContext = createContext<NotifContextValue | null>(null);

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { items: [] });

  const add = useCallback(
    (n: Omit<Notification, 'id' | 'timestamp' | 'dismissed'>) => dispatch({ type: 'ADD', payload: n }),
    [],
  );
  const dismiss = useCallback((id: string) => dispatch({ type: 'DISMISS', id }), []);
  const dismissAll = useCallback(() => dispatch({ type: 'DISMISS_ALL' }), []);
  const clear = useCallback(() => dispatch({ type: 'CLEAR' }), []);

  const activeCount = state.items.filter((n) => !n.dismissed).length;

  return (
    <NotifContext.Provider value={{ notifications: state.items, activeCount, add, dismiss, dismissAll, clear }}>
      {children}
    </NotifContext.Provider>
  );
}

export function useNotifications() {
  const ctx = useContext(NotifContext);
  if (!ctx) throw new Error('useNotifications must be used within NotificationProvider');
  return ctx;
}
