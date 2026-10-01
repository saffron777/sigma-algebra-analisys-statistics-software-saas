// Live synchronisation between the windows of one account: polls /api/sync and tells the pages what changed.
// A scope is "datasets", "results", "drafts", "templates", "notebooks" or "workspace:<module>".

import { api } from "./api.js";

const POLL_MS = 3000;
const SLOW_POLL_MS = 15000;

/** @param {Record<string, any>} snapshot @param {string} scope */
const valueOf = (snapshot, scope) =>
  scope.startsWith("workspace:") ? (snapshot.workspace?.[scope.slice(10)] ?? 0) : snapshot[scope];

export function defineSyncStore() {
  // not kept on the reactive store: Alpine would wrap the Map and every callback in proxies
  const listeners = new Map(); // scope -> Set<(value, previous) => void>
  let timer = null;
  let failures = 0;
  let polling = false;

  Alpine.store("sync", {
    snapshot: null,

    /** Subscribe to a scope; returns the function that unsubscribes. */
    on(scope, callback) {
      if (!listeners.has(scope)) listeners.set(scope, new Set());
      listeners.get(scope).add(callback);
      return () => listeners.get(scope)?.delete(callback);
    },

    start() {
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") this.pollSoon(0);
      });
      this.pollSoon(0); // take the baseline right away: a change made while the page is still loading must be seen
    },

    pollSoon(milliseconds) {
      clearTimeout(timer);
      timer = setTimeout(() => this.poll(), milliseconds);
    },

    async poll() {
      if (polling) return;
      if (document.visibilityState !== "visible") { this.pollSoon(POLL_MS); return; } // hidden tabs wait for the event
      polling = true;
      try {
        const next = await api("/api/sync");
        failures = 0;
        const previous = this.snapshot;
        this.snapshot = next;
        if (previous) {
          for (const [scope, callbacks] of listeners) {
            const before = valueOf(previous, scope);
            const now = valueOf(next, scope);
            if (before !== now) for (const callback of [...callbacks]) callback(now, before);
          }
        }
      } catch {
        failures += 1; // server restarting or offline: try again, less often
      } finally {
        polling = false;
        this.pollSoon(failures > 3 ? SLOW_POLL_MS : POLL_MS);
      }
    },
  });
}
