/**
 * Next.js Instrumentation
 * 
 * Patches Node.js v22+ built-in localStorage which doesn't work
 * without a valid --localstorage-file path. Some libraries check
 * for localStorage existence during SSR and crash when its methods
 * throw instead of being absent.
 */
export async function register() {
  if (typeof globalThis.localStorage !== 'undefined') {
    const storage = new Map<string, string>();
    
    // Replace the broken Node.js built-in localStorage with a simple Map-backed shim
    Object.defineProperty(globalThis, 'localStorage', {
      value: {
        getItem(key: string) { return storage.get(key) ?? null; },
        setItem(key: string, value: string) { storage.set(key, String(value)); },
        removeItem(key: string) { storage.delete(key); },
        clear() { storage.clear(); },
        key(index: number) { return [...storage.keys()][index] ?? null; },
        get length() { return storage.size; },
      },
      writable: true,
      configurable: true,
    });
  }
}
