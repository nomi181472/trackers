// Server-only worker router, rate limiter, and job affinity coordinator.

export type WorkerId = "worker-n" | "worker-r" | "worker-v";

export interface WorkerConfig {
  id: WorkerId;
  name: string;
  baseUrl: string;
  prefix: string;
  timeoutMs: number;
}

// Environment variables
const rawEnvN = process.env.WORKER_N_URL?.trim();
const rawEnvR = process.env.WORKER_R_URL?.trim();
const rawEnvV = (process.env.WORKER_V_URL || process.env.BACKEND_URL)?.trim();

const provided = [
  rawEnvN ? "WORKER_N_URL" : null,
  rawEnvR ? "WORKER_R_URL" : null,
  rawEnvV ? "WORKER_V_URL (or BACKEND_URL)" : null,
].filter(Boolean) as string[];

const missing = [
  !rawEnvN ? "WORKER_N_URL" : null,
  !rawEnvR ? "WORKER_R_URL" : null,
  !rawEnvV ? "WORKER_V_URL (or BACKEND_URL)" : null,
].filter(Boolean) as string[];

// Case 1: All are null/empty => Local development / debug mode (e.g. ./run)
export const IS_LOCAL_DEV = provided.length === 0;

// Case 2: Partial configuration error (1 or 2 are null)
export const ENV_CONFIG_ERROR =
  provided.length > 0 && missing.length > 0
    ? `Worker configuration error: Incomplete environment variables detected. Configured: [${provided.join(", ")}], Missing: [${missing.join(", ")}]. For local development, leave all empty. For cluster mode, define all three.`
    : null;

const LOCAL_BACKEND = rawEnvV || "http://127.0.0.1:8000";

export const WORKERS: Record<WorkerId, WorkerConfig> = {
  "worker-n": {
    id: "worker-n",
    name: "worker-n",
    baseUrl: rawEnvN || LOCAL_BACKEND,
    prefix: "wn_",
    timeoutMs: 30000,
  },
  "worker-r": {
    id: "worker-r",
    name: "worker-r",
    baseUrl: rawEnvR || LOCAL_BACKEND,
    prefix: "wr_",
    timeoutMs: 65000,
  },
  "worker-v": {
    id: "worker-v",
    name: "worker-v",
    baseUrl: rawEnvV || LOCAL_BACKEND,
    prefix: "wv_",
    timeoutMs: 25000,
  },
};

export const WORKER_LIST: WorkerId[] = IS_LOCAL_DEV ? ["worker-v"] : ["worker-n", "worker-r", "worker-v"];

// Sliding window rate limiter state: max 3 requests per second
interface RateLimiterState {
  timestamps: number[];
  activeExecutions: number;
}

const rateLimiter: RateLimiterState = {
  timestamps: [],
  activeExecutions: 0,
};

const MAX_REQ_PER_SEC = 3;
const WINDOW_MS = 1000;

export function checkRateLimit(): { allowed: boolean; message?: string } {
  const now = Date.now();
  // Filter out timestamps older than 1 second
  rateLimiter.timestamps = rateLimiter.timestamps.filter((ts) => now - ts < WINDOW_MS);

  if (rateLimiter.timestamps.length >= MAX_REQ_PER_SEC) {
    return {
      allowed: false,
      message: "Too many candidates, please wait. It is running on free version.",
    };
  }

  rateLimiter.timestamps.push(now);
  return { allowed: true };
}

// Track health status cache for fast non-blocking selection
interface HealthCache {
  healthy: boolean;
  checkedAt: number;
}

const healthCache: Record<WorkerId, HealthCache> = {
  "worker-n": { healthy: true, checkedAt: 0 },
  "worker-r": { healthy: true, checkedAt: 0 },
  "worker-v": { healthy: true, checkedAt: 0 },
};

const HEALTH_TTL_MS = 25000;

export async function isWorkerHealthy(workerId: WorkerId): Promise<boolean> {
  const now = Date.now();
  const cached = healthCache[workerId];
  if (cached && now - cached.checkedAt < HEALTH_TTL_MS) {
    return cached.healthy;
  }

  const worker = WORKERS[workerId];
  if (!worker) return false;

  try {
    const controller = new AbortController();
    // Short timeout for health checks so user requests are not held up
    const timeout = setTimeout(() => controller.abort(), 4000);
    const res = await fetch(`${worker.baseUrl}/api/health`, {
      headers: { Accept: "*/*" },
      signal: controller.signal,
    });
    clearTimeout(timeout);
    const ok = res.ok;
    healthCache[workerId] = { healthy: ok, checkedAt: now };
    return ok;
  } catch {
    healthCache[workerId] = { healthy: false, checkedAt: now };
    return false;
  }
}

// Round-robin / priority pointer
let workerTurn = 0;

/**
 * Selects an available healthy worker using priority fallback.
 * Checks worker-n -> worker-r -> worker-v (or cycles through them).
 */
export async function getNextAvailableWorker(): Promise<WorkerConfig | null> {
  if (IS_LOCAL_DEV) {
    return WORKERS["worker-v"];
  }

  const order: WorkerId[] = ["worker-n", "worker-r", "worker-v"];

  // Reorder slightly starting from current turn to distribute load fairly
  const startIndex = workerTurn % order.length;
  workerTurn = (workerTurn + 1) % order.length;

  const candidateOrder = [
    order[startIndex],
    ...order.filter((_, idx) => idx !== startIndex),
  ];

  for (const wid of candidateOrder) {
    const healthy = await isWorkerHealthy(wid);
    if (healthy) {
      return WORKERS[wid];
    }
  }

  // Fallback to worker-n if all health checks timed out
  return WORKERS["worker-n"];
}

/**
 * Extracts worker affinity from a jobId or path prefix.
 * e.g., "wn_abc123" -> worker-n, "wr_abc123" -> worker-r, "wv_abc123" -> worker-v
 */
export function resolveWorkerFromTarget(target: string | null): WorkerConfig | null {
  if (!target) return null;
  const lower = target.toLowerCase();
  if (lower === "worker-n" || lower.startsWith("wn_")) return WORKERS["worker-n"];
  if (lower === "worker-r" || lower.startsWith("wr_")) return WORKERS["worker-r"];
  if (lower === "worker-v" || lower.startsWith("wv_")) return WORKERS["worker-v"];
  return null;
}

/**
 * Encodes original backend jobId with worker prefix so polling sticks to the same worker.
 */
export function encodeJobId(workerId: WorkerId, originalJobId: string): string {
  const prefix = WORKERS[workerId].prefix;
  if (originalJobId.startsWith(prefix)) return originalJobId;
  return `${prefix}${originalJobId}`;
}

/**
 * Decodes the prefixed jobId back to the worker's internal jobId.
 */
export function decodeJobId(prefixedJobId: string): { worker: WorkerConfig; rawJobId: string } {
  for (const wid of WORKER_LIST) {
    const config = WORKERS[wid];
    if (prefixedJobId.startsWith(config.prefix)) {
      return {
        worker: config,
        rawJobId: prefixedJobId.slice(config.prefix.length),
      };
    }
  }
  // Default to worker-n
  return { worker: WORKERS["worker-n"], rawJobId: prefixedJobId };
}
