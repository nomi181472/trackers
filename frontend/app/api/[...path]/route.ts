import { NextRequest, NextResponse } from "next/server";
import {
  WORKERS,
  WORKER_LIST,
  WorkerConfig,
  ENV_CONFIG_ERROR,
  checkRateLimit,
  getNextAvailableWorker,
  resolveWorkerFromTarget,
  encodeJobId,
  decodeJobId,
} from "@/lib/workers";

export const dynamic = "force-dynamic";

/**
 * Strips headers that reveal infrastructure or cloud providers.
 */
function sanitizeHeaders(headers: Headers): Headers {
  const clean = new Headers(headers);
  clean.delete("server");
  clean.delete("x-render-origin-server");
  clean.delete("x-powered-by");
  clean.delete("x-matched-path");
  clean.delete("x-vercel-id");
  clean.delete("x-vercel-cache");
  clean.delete("x-nf-request-id");
  return clean;
}

async function handleProxy(req: NextRequest, params: { path: string[] }) {
  // 0. Validate Environment Variable Setup
  if (ENV_CONFIG_ERROR) {
    return NextResponse.json(
      { detail: ENV_CONFIG_ERROR },
      { status: 500 }
    );
  }

  const pathSegments = params.path || [];
  const rawPath = "/" + pathSegments.join("/");
  const urlObj = new URL(req.url);
  const searchParams = urlObj.searchParams;

  // 1. Check Rate Limit (3 req / sec)
  const rateCheck = checkRateLimit();
  if (!rateCheck.allowed) {
    return NextResponse.json(
      { detail: rateCheck.message || "Too many candidates, please wait. It is running on free version." },
      { status: 429, headers: { "Retry-After": "1" } }
    );
  }

  // 2. Determine target worker
  let targetWorker: WorkerConfig | null = null;
  let rewrittenPath = rawPath;

  // A. Check explicit target from query or header (used by Logs & Clear Records)
  const explicitTarget = searchParams.get("worker") || req.headers.get("x-target-worker");
  if (explicitTarget) {
    // If clearing across all workers
    if (explicitTarget === "all" && rawPath === "/cleanup") {
      let totalDeleted = 0;
      let totalFreedMb = 0;
      const details: Record<string, number> = {};

      for (const wid of WORKER_LIST) {
        try {
          const w = WORKERS[wid];
          const query = searchParams.toString() ? `?${searchParams.toString()}` : "";
          const targetUrl = `${w.baseUrl}/api/cleanup${query}`;
          const res = await fetch(targetUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          });
          if (res.ok) {
            const data = await res.json();
            totalDeleted += data.deleted_count || 0;
            totalFreedMb += data.freed_mb || 0;
            details[wid] = data.deleted_count || 0;
          }
        } catch {
          // ignore offline worker during cleanup
        }
      }

      return NextResponse.json({
        ok: true,
        deleted_count: totalDeleted,
        freed_bytes: Math.round(totalFreedMb * 1024 * 1024),
        freed_mb: Number(totalFreedMb.toFixed(2)),
        details,
        worker: "all",
      });
    }

    targetWorker = resolveWorkerFromTarget(explicitTarget);
  }

  // B. Job Affinity: /jobs/{jobId}
  if (!targetWorker && pathSegments[0] === "jobs" && pathSegments[1]) {
    const prefixedJobId = pathSegments[1];
    const { worker, rawJobId } = decodeJobId(prefixedJobId);
    targetWorker = worker;
    rewrittenPath = `/jobs/${rawJobId}`;
  }

  // C. Media Affinity: /media/{filename}
  if (!targetWorker && pathSegments[0] === "media" && pathSegments[1]) {
    const filename = pathSegments[1];
    // Check if filename has prefix wn_, wr_, wv_
    targetWorker = resolveWorkerFromTarget(filename) || WORKERS["worker-n"];
  }

  // D. General dispatch (new simulation or preview) -> pick available worker
  if (!targetWorker) {
    targetWorker = await getNextAvailableWorker();
  }

  if (!targetWorker) {
    return NextResponse.json(
      { detail: "All workers are currently unavailable. Please try again shortly." },
      { status: 503 }
    );
  }

  // Construct target URL
  // Remove 'worker' param from upstream request so FastAPI doesn't complain about unexpected query params
  const forwardParams = new URLSearchParams(searchParams);
  forwardParams.delete("worker");
  const queryString = forwardParams.toString() ? `?${forwardParams.toString()}` : "";
  const targetUrl = `${targetWorker.baseUrl}/api${rewrittenPath}${queryString}`;

  // Forward request headers
  const forwardHeaders = new Headers(req.headers);
  forwardHeaders.delete("host");
  forwardHeaders.delete("x-target-worker");
  forwardHeaders.set("Accept", forwardHeaders.get("accept") || "*/*");

  const reqBody = ["GET", "HEAD"].includes(req.method) ? undefined : await req.arrayBuffer();

  try {
    const upstreamRes = await fetch(targetUrl, {
      method: req.method,
      headers: forwardHeaders,
      body: reqBody,
      redirect: "follow",
    });

    const contentType = upstreamRes.headers.get("content-type") || "";
    const responseHeaders = sanitizeHeaders(upstreamRes.headers);
    responseHeaders.set("X-Worker", targetWorker.name);

    // If response is JSON, check if we need to encode job_id with worker prefix
    if (contentType.includes("application/json")) {
      const data = await upstreamRes.json();

      // Encode job_id in /simulations response
      if (data && typeof data === "object") {
        if ("job_id" in data && typeof data.job_id === "string") {
          data.job_id = encodeJobId(targetWorker.id, data.job_id);
        }
        // Always return which worker executed the request
        data.worker = targetWorker.name;
      }

      return NextResponse.json(data, {
        status: upstreamRes.status,
        headers: responseHeaders,
      });
    }

    // For media, video streaming or text
    const responseBody = upstreamRes.body;
    return new NextResponse(responseBody, {
      status: upstreamRes.status,
      headers: responseHeaders,
    });
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      {
        detail: `Worker ${targetWorker.name} connection error: ${errMsg}`,
        worker: targetWorker.name,
      },
      { status: 502 }
    );
  }
}

export async function GET(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const resolvedParams = await params;
  return handleProxy(req, resolvedParams);
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const resolvedParams = await params;
  return handleProxy(req, resolvedParams);
}

export async function DELETE(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const resolvedParams = await params;
  return handleProxy(req, resolvedParams);
}

export async function PUT(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const resolvedParams = await params;
  return handleProxy(req, resolvedParams);
}
