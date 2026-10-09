import { NextRequest, NextResponse } from "next/server";
import {
  WORKER_N,
  checkRateLimit,
  resolveWorkerFromTarget,
  decodeJobId,
  encodeJobId,
  getNextAvailableWorker,
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
  clean.delete("content-encoding");
  clean.delete("content-length");
  clean.delete("transfer-encoding");
  return clean;
}

async function handleProxy(req: NextRequest, params: { path: string[] }) {
  const pathSegments = params.path || [];
  let rewrittenPath = "/" + pathSegments.join("/");
  const urlObj = new URL(req.url);
  const searchParams = urlObj.searchParams;

  // 1. Check Rate Limit (3 req / sec)
  const rateCheck = checkRateLimit();
  if (!rateCheck.allowed) {
    return NextResponse.json(
      {
        detail:
          rateCheck.message ||
          "Too many candidates, please wait. It is running on free version.\n\ncurl -fsSL https://raw.githubusercontent.com/nomi181472/trackers/main/docker-compose.yml -o docker-compose.yml && docker compose pull && docker compose up -d\n\nyou can also run in your local",
      },
      { status: 429, headers: { "Retry-After": "1" } }
    );
  }

  // 2. Resolve target worker (job affinity, explicit worker param, or default)
  let targetWorker = WORKER_N;
  const targetParam = searchParams.get("worker") || req.headers.get("x-target-worker");

  // A. Explicit worker request
  if (targetParam) {
    targetWorker = resolveWorkerFromTarget(targetParam);
  }

  // B. Job Affinity: /jobs/{jobId}
  if (pathSegments[0] === "jobs" && pathSegments[1]) {
    const { worker, rawJobId } = decodeJobId(pathSegments[1]);
    targetWorker = worker;
    const rest = pathSegments.slice(2).join("/");
    rewrittenPath = rest ? `/jobs/${rawJobId}/${rest}` : `/jobs/${rawJobId}`;
  }

  // C. Media Affinity: /media/{filename}
  if (pathSegments[0] === "media" && pathSegments[1]) {
    targetWorker = resolveWorkerFromTarget(pathSegments[1]);
  }

  // D. General dispatch (new simulation or preview)
  if (!targetParam && pathSegments[0] !== "jobs" && pathSegments[0] !== "media") {
    targetWorker = await getNextAvailableWorker();
  }

  // 3. Construct target URL to chosen worker
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

    // If response is JSON
    if (contentType.includes("application/json")) {
      const data = await upstreamRes.json();
      if (data && typeof data === "object") {
        if ("job_id" in data && typeof data.job_id === "string") {
          // If using worker-v, prefix with wv_ so job polling routes back to worker-v
          if (targetWorker.id === "worker-v") {
            data.job_id = encodeJobId("worker-v", data.job_id);
          } else {
            // Strip any prefix for default worker-n
            data.job_id = data.job_id.replace(/^(wn_|wr_|wv_)/, "");
          }
        }
        data.worker = targetWorker.name;
      }

      return NextResponse.json(data, {
        status: upstreamRes.status,
        headers: responseHeaders,
      });
    }

    // For media, video streaming or raw text
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
