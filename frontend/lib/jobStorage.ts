/**
 * jobStorage.ts — Per-client, per-session active job persistence.
 *
 * Each browser tab / window stores its own active job under a unique
 * tab-scoped key so multiple concurrent users never clobber each other's state.
 *
 * Storage strategy:
 *  - sessionStorage  → cleared automatically when the tab/window is closed.
 *  - localStorage    → persists across tab restores (e.g. session-restore on browser restart).
 *
 * We write to BOTH storages so that:
 *  1. A refresh within the same tab recovers the job from sessionStorage.
 *  2. A browser-session-restore still finds the job via localStorage (best-effort).
 *
 * Each tab generates a unique CLIENT_TAB_ID on first load so the storage keys
 * never collide between different browser tabs opened by different users.
 */

import type { ParamValues } from "./types";

/** Shape of the persisted active job record */
export interface ActiveJobRecord {
  jobId: string;
  worker: string | null;
  mode: "standard" | "production";
  startedAt: number; // unix ms
  scenario?: ParamValues;
  trackerIds?: string[];
}

const TAB_ID_KEY = "tracker_tab_id";
const JOB_KEY_PREFIX = "tracker_active_job_";

/** Get or create a stable tab-unique identifier. */
function getTabId(): string {
  // sessionStorage is tab-scoped; perfect for a unique tab ID
  try {
    let id = sessionStorage.getItem(TAB_ID_KEY);
    if (!id) {
      id = `tab_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      sessionStorage.setItem(TAB_ID_KEY, id);
    }
    return id;
  } catch {
    // If storage is blocked (e.g. private browsing restrictions), fall back to
    // a module-level variable that survives for the lifetime of the page.
    return _fallbackTabId;
  }
}

// Module-level fallback (reset on every page load, so acts like sessionStorage)
const _fallbackTabId = `tab_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

function storageKey(): string {
  return `${JOB_KEY_PREFIX}${getTabId()}`;
}

/** Persist the active job record for this browser tab. */
export function saveActiveJob(record: ActiveJobRecord): void {
  const key = storageKey();
  const serialised = JSON.stringify(record);
  try { sessionStorage.setItem(key, serialised); } catch { /* ignore */ }
  try { localStorage.setItem(key, serialised); } catch { /* ignore */ }
}

/**
 * Read back the active job for this tab.
 * Falls through sessionStorage → localStorage so a page refresh still recovers it.
 * Returns null if no active job is stored or the record has expired (> 2 hours old).
 */
export function getActiveJob(): ActiveJobRecord | null {
  const key = storageKey();
  let raw: string | null = null;
  try { raw = sessionStorage.getItem(key); } catch { /* ignore */ }
  if (!raw) {
    try { raw = localStorage.getItem(key); } catch { /* ignore */ }
  }
  if (!raw) return null;
  try {
    const record = JSON.parse(raw) as ActiveJobRecord;
    // Discard stale records older than 2 hours to avoid zombie state
    const TWO_HOURS_MS = 2 * 60 * 60 * 1000;
    if (Date.now() - record.startedAt > TWO_HOURS_MS) {
      clearActiveJob();
      return null;
    }
    return record;
  } catch {
    clearActiveJob();
    return null;
  }
}

/**
 * Remove the active job record for this tab.
 * Call this when the job is finished, cancelled, or on explicit user clear.
 */
export function clearActiveJob(): void {
  const key = storageKey();
  try { sessionStorage.removeItem(key); } catch { /* ignore */ }
  try { localStorage.removeItem(key); } catch { /* ignore */ }
}
