import type { Catalog, JobStatus, ParamValues, ScenarioMeta, SimulationResult } from "./types";

/**
 * Resolves the API base URL dynamically at runtime:
 * 1. In Vercel server functions/SSR: reads internal service binding `BACKEND_URL`.
 * 2. In browser client: uses relative URL "" (routed to backend via Vercel rewrites),
 *    or NEXT_PUBLIC_API_URL if explicitly specified.
 * 3. Local fallback: http://localhost:8000.
 */
export function getApiBase(): string {
  if (typeof window === "undefined" && process.env.BACKEND_URL) {
    return process.env.BACKEND_URL.replace(/\/$/, "");
  }
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    return "";
  }
  return "http://localhost:8000";
}

export const API = process.env.NEXT_PUBLIC_API_URL || "";

export function mediaUrl(p: string | null | undefined): string {
  if (!p) return "";
  if (p.startsWith("http://") || p.startsWith("https://") || p.startsWith("data:") || p.startsWith("blob:")) return p;
  const base = getApiBase();
  const normalizedPath = p.startsWith("/") ? p : `/${p}`;
  return base ? `${base}${normalizedPath}` : normalizedPath;
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const base = getApiBase();
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = base ? `${base}${normalizedPath}` : normalizedPath;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export function getCatalog(): Promise<Catalog> {
  return http<Catalog>("/api/trackers");
}

export function scenarioPreview(params: ParamValues): Promise<{ scenario_id: string; meta: ScenarioMeta }> {
  return http("/api/scenarios/preview", { method: "POST", body: JSON.stringify(params) });
}

export function startSimulation(payload: {
  scenario: ParamValues;
  detection: ParamValues;
  trackers: { tracker_id: string; params: ParamValues }[];
}): Promise<{ job_id: string; worker?: string }> {
  return http("/api/simulations", { method: "POST", body: JSON.stringify(payload) });
}

export function getJob(jobId: string): Promise<JobStatus> {
  return http(`/api/jobs/${jobId}`);
}

export async function uploadVideo(file: File): Promise<{ upload_id: string; size: number }> {
  const base = getApiBase();
  const url = base ? `${base}/api/real/upload` : `/api/real/upload`;
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "upload failed");
  return res.json();
}

export function startRealJob(payload: {
  upload_id: string;
  tracker_id: string;
  params: ParamValues;
  det_params: ParamValues;
}): Promise<{ job_id: string }> {
  return http("/api/real/jobs", { method: "POST", body: JSON.stringify(payload) });
}

export async function pollUntilDone(jobId: string, onProgress: (j: JobStatus) => void) {
  let j: JobStatus;
  let notFoundRetries = 0;
  for (;;) {
    try {
      j = await getJob(jobId);
      notFoundRetries = 0;
      onProgress(j);
      if (j.status === "done" || j.status === "error") return j;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // If serverless container just spun up or hasn't synced the job file yet, retry up to 6 times (6 * 800ms ~ 5s)
      if (msg.includes("404") && notFoundRetries < 6) {
        notFoundRetries++;
        await new Promise((r) => setTimeout(r, 800));
        continue;
      }
      throw err;
    }
    await new Promise((r) => setTimeout(r, 700));
  }
}

export function getLogFiles(worker?: string): Promise<{ files: import("./types").LogFileInfo[] }> {
  const q = worker ? `?worker=${encodeURIComponent(worker)}` : "";
  return http(`/api/logs/files${q}`);
}

export function getLogLines(params: {
  file?: string;
  cursor?: number | null;
  limit?: number;
  worker?: string;
}): Promise<import("./types").LogPage> {
  const q = new URLSearchParams();
  if (params.file) q.set("file", params.file);
  if (params.cursor !== undefined && params.cursor !== null) q.set("cursor", String(params.cursor));
  if (params.limit) q.set("limit", String(params.limit));
  if (params.worker) q.set("worker", params.worker);
  const query = q.toString();
  return http(`/api/logs${query ? `?${query}` : ""}`);
}

export function cleanupGeneratedFiles(params?: {
  include_uploads?: boolean;
  include_jobs?: boolean;
  include_scenarios?: boolean;
  worker?: string;
}): Promise<{
  ok: boolean;
  deleted_count: number;
  freed_bytes: number;
  freed_mb: number;
  details: Record<string, number>;
  worker?: string;
}> {
  const q = new URLSearchParams();
  if (params?.include_uploads) q.set("include_uploads", "true");
  if (params?.include_jobs !== undefined) q.set("include_jobs", String(params.include_jobs));
  if (params?.include_scenarios !== undefined) q.set("include_scenarios", String(params.include_scenarios));
  if (params?.worker) q.set("worker", params.worker);
  const qs = q.toString();
  return http(`/api/cleanup${qs ? `?${qs}` : ""}`, { method: "POST" });
}