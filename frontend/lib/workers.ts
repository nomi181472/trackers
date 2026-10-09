// Multi-worker configuration supporting worker-n and worker-v.

export type WorkerId = "worker-n" | "worker-v";

export interface WorkerConfig {
  id: WorkerId;
  name: string;
  baseUrl: string;
  prefix: string;
  timeoutMs: number;
}

const LOCAL_URL = "http://127.0.0.1:8000";

// WORKER_N: Northflank worker or fallback
const rawWorkerN = (
  process.env.WORKER_N_URL?.trim() ||
  process.env.BACKEND_URL?.trim() ||
  LOCAL_URL
).replace(/\/$/, "");

const NORTHFLANK_URL =
  rawWorkerN.startsWith("http://") || rawWorkerN.startsWith("https://")
    ? rawWorkerN
    : `https://${rawWorkerN}`;

export const WORKER_N: WorkerConfig = {
  id: "worker-n",
  name: "worker-n",
  baseUrl: NORTHFLANK_URL,
  prefix: "wn_",
  timeoutMs: 45000,
};

// WORKER_V: Vercel / serverless worker or fallback
const rawWorkerV = (
  process.env.WORKER_V_URL?.trim() ||
  process.env.BACKEND_URL?.trim() ||
  LOCAL_URL
).replace(/\/$/, "");

const VERCEL_WORKER_URL =
  rawWorkerV.startsWith("http://") || rawWorkerV.startsWith("https://")
    ? rawWorkerV
    : `https://${rawWorkerV}`;

export const WORKER_V: WorkerConfig = {
  id: "worker-v",
  name: "worker-v",
  baseUrl: VERCEL_WORKER_URL,
  prefix: "wv_",
  timeoutMs: 30000,
};

export const WORKERS: Record<WorkerId, WorkerConfig> = {
  "worker-n": WORKER_N,
  "worker-v": WORKER_V,
};

// If WORKER_V_URL is specifically configured, include both; default active is selectable
export const WORKER_LIST: WorkerId[] = process.env.WORKER_V_URL?.trim()
  ? ["worker-n", "worker-v"]
  : ["worker-n"];

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
 * Returns default or specified worker.
 */
export async function getNextAvailableWorker(preferredId?: WorkerId | null): Promise<WorkerConfig> {
  if (preferredId && WORKERS[preferredId]) {
    return WORKERS[preferredId];
  }
  // Default to worker-v if explicitly configured and requested, otherwise worker-n
  if (process.env.DEFAULT_WORKER === "worker-v") {
    return WORKER_V;
  }
  return WORKER_N;
}

/**
 * Resolves worker from target string or prefix (e.g., 'worker-v', 'wv_', 'worker-n', 'wn_').
 */
export function resolveWorkerFromTarget(target: string | null): WorkerConfig {
  if (!target) return WORKER_N;
  const lower = target.toLowerCase();
  if (lower === "worker-v" || lower.startsWith("wv_")) {
    return WORKER_V;
  }
  return WORKER_N;
}

/**
 * Encodes original backend jobId with worker prefix if multiple workers in use.
 */
export function encodeJobId(workerId: WorkerId, originalJobId: string): string {
  const clean = originalJobId.replace(/^(wn_|wr_|wv_)/, "");
  return `${WORKERS[workerId].prefix}${clean}`;
}

/**
 * Decodes jobId by detecting worker prefix.
 */
export function decodeJobId(jobId: string): { worker: WorkerConfig; rawJobId: string } {
  if (jobId.startsWith("wv_")) {
    return {
      worker: WORKER_V,
      rawJobId: jobId.replace(/^wv_/, ""),
    };
  }
  const cleanId = jobId.replace(/^wn_/, "").replace(/^wr_/, "");
  return {
    worker: WORKER_N,
    rawJobId: cleanId,
  };
}
