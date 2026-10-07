import type { Catalog, JobStatus, ParamValues, ScenarioMeta, SimulationResult } from "./types";

export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function mediaUrl(p: string | null | undefined): string {
  if (!p) return "";
  return p.startsWith("http://") || p.startsWith("https://") ? p : `${API}${p}`;
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
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
}): Promise<{ job_id: string }> {
  return http("/api/simulations", { method: "POST", body: JSON.stringify(payload) });
}

export function getJob(jobId: string): Promise<JobStatus> {
  return http(`/api/jobs/${jobId}`);
}

export async function uploadVideo(file: File): Promise<{ upload_id: string; size: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/api/real/upload`, { method: "POST", body: form });
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
  for (;;) {
    j = await getJob(jobId);
    onProgress(j);
    if (j.status === "done" || j.status === "error") return j;
    await new Promise((r) => setTimeout(r, 700));
  }
}

export function getLogFiles(): Promise<{ files: import("./types").LogFileInfo[] }> {
  return http("/api/logs/files");
}

export function getLogLines(params: {
  file?: string;
  cursor?: number | null;
  limit?: number;
}): Promise<import("./types").LogPage> {
  const q = new URLSearchParams();
  if (params.file) q.set("file", params.file);
  if (params.cursor !== undefined && params.cursor !== null) q.set("cursor", String(params.cursor));
  if (params.limit) q.set("limit", String(params.limit));
  const query = q.toString();
  return http(`/api/logs${query ? `?${query}` : ""}`);
}