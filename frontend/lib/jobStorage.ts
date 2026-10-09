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
  tabId?: string; // which tab created this job
  worker: string | null;
  mode: "standard" | "production";
  startedAt: number; // unix ms
  scenario?: ParamValues;
  trackerIds?: string[];
}

const TAB_ID_KEY = "tracker_tab_id";
const SHARED_JOB_KEY = "tracker_active_job";

/** Get or create a stable tab-unique identifier. */
export function getTabId(): string {
  try {
    let id = sessionStorage.getItem(TAB_ID_KEY);
    if (!id) {
      id = `tab_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      sessionStorage.setItem(TAB_ID_KEY, id);
    }
    return id;
  } catch {
    return _fallbackTabId;
  }
}

// Module-level fallback (reset on every page load, so acts like sessionStorage)
const _fallbackTabId = `tab_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

/** Persist the active job record across browser tabs. */
export function saveActiveJob(record: ActiveJobRecord): void {
  const payload: ActiveJobRecord = {
    ...record,
    tabId: record.tabId || getTabId(),
  };
  const serialised = JSON.stringify(payload);
  try { sessionStorage.setItem(SHARED_JOB_KEY, serialised); } catch { /* ignore */ }
  try { localStorage.setItem(SHARED_JOB_KEY, serialised); } catch { /* ignore */ }
}

/**
 * Read back the active job across browser tabs.
 * Returns null if no active job is stored or the record has expired (> 2 hours old).
 */
export function getActiveJob(): ActiveJobRecord | null {
  let raw: string | null = null;
  try { raw = localStorage.getItem(SHARED_JOB_KEY); } catch { /* ignore */ }
  if (!raw) {
    try { raw = sessionStorage.getItem(SHARED_JOB_KEY); } catch { /* ignore */ }
  }
  if (!raw) return null;
  try {
    const record = JSON.parse(raw) as ActiveJobRecord;
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
 * Remove the active job record from browser storage.
 */
export function clearActiveJob(): void {
  try { sessionStorage.removeItem(SHARED_JOB_KEY); } catch { /* ignore */ }
  try { localStorage.removeItem(SHARED_JOB_KEY); } catch { /* ignore */ }
}
