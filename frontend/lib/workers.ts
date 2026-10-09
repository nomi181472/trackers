// Dedicated single-worker proxy configuration (worker-n only).

export type WorkerId = "worker-n";

export interface WorkerConfig {
  id: WorkerId;
  name: string;
  baseUrl: string;
  timeoutMs: number;
}

// Configured to worker-n (from env or local backend fallback)
const LOCAL_URL = "http://127.0.0.1:8000";
const rawWorkerUrl = (
  process.env.WORKER_N_URL?.trim() ||
  process.env.BACKEND_URL?.trim() ||
  LOCAL_URL
).replace(/\/$/, "");

const NORTHFLANK_URL =
  rawWorkerUrl.startsWith("http://") || rawWorkerUrl.startsWith("https://")
    ? rawWorkerUrl
    : `https://${rawWorkerUrl}`;

export const WORKER_N: WorkerConfig = {
  id: "worker-n",
  name: "worker-n",
  baseUrl: NORTHFLANK_URL,
  timeoutMs: 45000,
};

export const WORKERS: Record<WorkerId, WorkerConfig> = {
  "worker-n": WORKER_N,
};

export const WORKER_LIST: WorkerId[] = ["worker-n"];

// Sliding window rate limiter state: max 3 requests per second
interface RateLimiterState {
  timestamps: number[];
}

const rateLimiter: RateLimiterState = {
  timestamps: [],
};

const MAX_REQ_PER_SEC = 3;
const WINDOW_MS = 1000;

export function checkRateLimit(): { allowed: boolean; message?: string } {
  // Allow disabling rate limiting via environment variable (e.g. in local dev or Docker)
  if (process.env.DISABLE_RATE_LIMIT === "true" || process.env.DISABLE_RATE_LIMIT === "1") {
    return { allowed: true };
  }

  const now = Date.now();
  // Filter out timestamps older than 1 second
  rateLimiter.timestamps = rateLimiter.timestamps.filter((ts) => now - ts < WINDOW_MS);

  if (rateLimiter.timestamps.length >= MAX_REQ_PER_SEC) {
    return {
      allowed: false,
      message:
        "Too many candidates, please wait. It is running on free version.\n\n" +
        "curl -fsSL https://raw.githubusercontent.com/nomi181472/trackers/main/docker-compose.yml -o docker-compose.yml && docker compose pull && docker compose up -d\n\n" +
        "you can also run in your local",
    };
  }

  rateLimiter.timestamps.push(now);
  return { allowed: true };
}

/**
 * Returns worker-n directly.
 */
export async function getNextAvailableWorker(): Promise<WorkerConfig> {
  return WORKER_N;
}

/**
 * Resolves worker-n.
 */
export function resolveWorkerFromTarget(_target: string | null): WorkerConfig {
  return WORKER_N;
}

/**
 * No prefix encoding needed — passes clean original jobId to Northflank.
 */
export function encodeJobId(_workerId: WorkerId, originalJobId: string): string {
  // If it already had wn_, strip it cleanly
  return originalJobId.replace(/^wn_/, "");
}

/**
 * Decodes jobId by stripping any legacy prefix.
 */
export function decodeJobId(jobId: string): { worker: WorkerConfig; rawJobId: string } {
  const cleanId = jobId.replace(/^wn_/, "").replace(/^wr_/, "").replace(/^wv_/, "");
  return {
    worker: WORKER_N,
    rawJobId: cleanId,
  };
}
