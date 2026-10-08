import { NextRequest, NextResponse } from "next/server";
import {
  WORKER_N,
  checkRateLimit,
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
  const pathSegments = params.path || [];
  let rewrittenPath = "/" + pathSegments.join("/");
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

  // 2. Clean jobId if prefixed
  if (pathSegments[0] === "jobs" && pathSegments[1]) {
    const rawJobId = pathSegments[1].replace(/^(wn_|wr_|wv_)/, "");
    rewrittenPath = `/jobs/${rawJobId}`;
  }

  // 3. Construct target URL to worker-n directly
  const forwardParams = new URLSearchParams(searchParams);
  forwardParams.delete("worker");
  const queryString = forwardParams.toString() ? `?${forwardParams.toString()}` : "";
  const targetUrl = `${WORKER_N.baseUrl}/api${rewrittenPath}${queryString}`;

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
    responseHeaders.set("X-Worker", "worker-n");

    // If response is JSON
    if (contentType.includes("application/json")) {
      const data = await upstreamRes.json();
      if (data && typeof data === "object") {
        if ("job_id" in data && typeof data.job_id === "string") {
          // Strip any prefix, keep raw job_id
          data.job_id = data.job_id.replace(/^(wn_|wr_|wv_)/, "");
        }
        data.worker = "worker-n";
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
        detail: `Worker worker-n connection error: ${errMsg}`,
        worker: "worker-n",
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
