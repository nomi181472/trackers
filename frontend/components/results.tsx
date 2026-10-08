"use client";

import { useState, useRef } from "react";
import { mediaUrl } from "@/lib/api";
import type { SimulationResult, TrackerResult } from "@/lib/types";

const FMT = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });

export function ResultsView({ result }: { result: SimulationResult }) {
  return (
    <div>
      <div className="panel" style={{ marginBottom: 14 }}>
        <h3>Scenario</h3>
        <div className="legend">
          <span>seed {result.scenario.seed}</span>
          <span>{result.scenario.num_objects} objects</span>
          <span>{result.scenario.frames} frames</span>
          {result.scenario.crossing ? <span>✖️ crossing</span> : null}
          {result.scenario.occlusion ? <span>🧱 occlusion</span> : null}
          {result.scenario.camera_shake ? <span>📳 shake</span> : null}
          {result.scenario.blur ? <span>🌫️ blur</span> : null}
          {result.scenario.similar_colors ? <span>🧑‍🤝‍🧑 look-alikes</span> : null}
        </div>
        {result.scenario.preview_data_url || result.scenario.preview_url ? (
          <video src={mediaUrl(result.scenario.preview_data_url || result.scenario.preview_url)} controls loop muted style={{ width: "100%", borderRadius: 8, border: "1px solid var(--line)" }} />
        ) : (
          <div className="err">Scenario preview unavailable for this run.</div>
        )}
        <small>The scene both trackers watched — identical for everyone.</small>
      </div>

      <div className="result-grid">
        {result.results.map((r) => (
          <ResultCard key={r.tracker_id} r={r} fps={result.scenario.fps || 15} />
        ))}
      </div>

      {result.results.length > 0 ? (
        <MultiTrackerComparisonChart results={result.results} chartUrl={result.chart_url} />
      ) : null}
    </div>
  );
}

export function ResultCard({ r, fps = 15 }: { r: TrackerResult; fps?: number }) {
  const m = r.metrics;
  return (
    <div className="result-card">
      <div className="result-head">
        <h3>{r.name}</h3>
        <span className={`grade grade-${r.report?.grade || "F"}`}>{r.report?.grade || "✖"}</span>
        <span style={{ marginLeft: "auto" }} className="tag mono">{r.codec}</span>
      </div>
      {r.video_data_url || r.video_url ? (
        <video
          src={mediaUrl(r.video_data_url || r.video_url)}
          controls
          loop
          muted
          poster={r.thumbnails?.[0]?.data_url ? mediaUrl(r.thumbnails[0].data_url) : (r.thumbnails?.[0]?.url ? mediaUrl(r.thumbnails[0].url) : undefined)}
        />
      ) : null}

      <div className="metrics">
        {typeof m.mota === "number" && isFinite(m.mota) ? <MSum label="MOTA" v={m.mota} best="high" /> : null}
        {typeof m.motp === "number" && isFinite(m.motp) ? <MSum label="MOTP" v={m.motp} best="low" /> : null}
        {typeof m.idf1 === "number" && isFinite(m.idf1) ? <MSum label="IDF1" v={m.idf1} best="high" /> : null}
        {typeof m.idsw === "number" && isFinite(m.idsw) ? <MSum label="ID switches" v={m.idsw} best="low" /> : null}
        {typeof m.fp === "number" && isFinite(m.fp) ? <MSum label="FP (Ghosts)" v={m.fp} best="low" /> : null}
        {typeof m.fn === "number" && isFinite(m.fn) ? <MSum label="FN (Misses)" v={m.fn} best="low" /> : null}
        {typeof m.mt === "number" && isFinite(m.mt) ? <MSum label="MT" v={m.mt} best="high" /> : null}
        {typeof m.ml === "number" && isFinite(m.ml) ? <MSum label="ML" v={m.ml} best="low" /> : null}
        {typeof m.accuracy === "number" && isFinite(m.accuracy) ? <MSum label="Accuracy" v={m.accuracy} best="high" /> : null}
        {typeof m.total_visible === "number" ? <MSum label="Missed frames" v={m.lost_frames ?? 0} best="low" /> : null}
        {typeof m.avg_time_ms === "number" && isFinite(m.avg_time_ms) ? <MSum label="Avg latency" v={m.avg_time_ms} unit="ms" best="low" /> : null}
        {typeof m.fps === "number" && isFinite(m.fps) ? <MSum label="Tracker speed" v={m.fps} unit="FPS" best="high" /> : null}
      </div>

      {m.motmetrics ? (
        <div style={{ display: "flex", alignItems: "center", gap: 6, margin: "6px 14px 2px", padding: "4px 8px", background: "rgba(56, 189, 248, 0.08)", border: "1px solid rgba(56, 189, 248, 0.2)", borderRadius: 6, fontSize: 11 }}>
          <span style={{ fontWeight: 600, color: "#38bdf8" }}>✓ py-motmetrics verified:</span>
          <span className="mono" style={{ color: "var(--muted)" }}>
            MOTA {(m.motmetrics.mota * 100).toFixed(1)}% · IDF1 {(m.motmetrics.idf1 * 100).toFixed(1)}% · IDSW {m.motmetrics.idsw} · MT {m.motmetrics.mt} · ML {m.motmetrics.ml}
          </span>
        </div>
      ) : null}

      {(typeof m.fp === "number" || typeof m.fn === "number" || typeof m.idsw === "number") ? (
        <ErrorBreakdownBar fp={m.fp ?? 0} fn={m.fn ?? 0} idsw={m.idsw ?? 0} />
      ) : null}

      {m.latencies?.length ? <LatencyGraph latencies={m.latencies} avgMs={m.avg_time_ms} /> : null}

      {r.report ? (
        <div className="report">
          <div className="verdict">{r.report.verdict}</div>
          {r.report.sections.map((s) => (
            <div className="sec" key={s.name}>
              <h4>{s.name}</h4>
              {s.text ? <p>{s.text}</p> : null}
              {s.bullets?.length ? (
                <ul>
                  {s.bullets.map((b, idx) => (
                    <li key={`${b}-${idx}`}>{b}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}

      {r.thumbnails?.length ? (
        <div className="thumbs">
          {r.thumbnails.map((t, idx) => (
            <div key={`${r.tracker_id}-${t.frame}-${t.type}-${idx}`} className="thumb" title={t.type} onClick={() => seek(r.video_data_url || r.video_url, t.frame, fps)}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={mediaUrl(t.data_url || t.url)} alt={t.type} loading="lazy" />
              <div className="lbl">
                f{t.frame} {t.type}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {r.events?.length ? (
        <div className="events">
          {r.events.slice(0, 12).map((e, i) => (
            <div className="event" key={i}>
              <span className={`sev sev-${e.severity}`}>{e.severity}</span>
              <div>
                <span className="mono">f{e.frame}</span> {e.text}
                {e.blame ? <span className="blame mono">blame: {e.blame}</span> : null}
                <small>Fix: {e.fix || "—"}</small>
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {r.error ? <div className="err" style={{ margin: 12 }}>{r.error}</div> : null}
    </div>
  );
}

function LatencyGraph({ latencies, avgMs }: { latencies: number[]; avgMs?: number }) {
  const width = 300;
  const height = 64;
  const pad = 6;
  const maxVal = Math.max(...latencies, 1);
  const minVal = 0;

  const points = latencies.map((val, idx) => {
    const x = pad + (idx / Math.max(latencies.length - 1, 1)) * (width - 2 * pad);
    const y = height - pad - ((val - minVal) / (maxVal - minVal)) * (height - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  const avgY = avgMs != null
    ? height - pad - ((avgMs - minVal) / (maxVal - minVal)) * (height - 2 * pad)
    : null;

  return (
    <div style={{ padding: "4px 14px 10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4, fontSize: 11, color: "var(--muted)" }}>
        <span>Frame Latency Curve</span>
        <span>Peak: <b style={{ color: "var(--text)" }}>{maxVal.toFixed(1)} ms</b></span>
      </div>
      <div style={{ background: "var(--bg-3)", border: "1px solid var(--line)", borderRadius: 6, padding: "4px 6px" }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: 50, display: "block" }}>
          {/* Average reference line */}
          {avgY != null && (
            <line
              x1={pad}
              y1={avgY}
              x2={width - pad}
              y2={avgY}
              stroke="var(--line)"
              strokeDasharray="3 3"
              strokeWidth="1"
            />
          )}
          {/* Latency line */}
          <polyline
            fill="none"
            stroke="var(--accent)"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={points}
          />
        </svg>
      </div>
    </div>
  );
}

function ErrorBreakdownBar({ fp, fn, idsw }: { fp: number; fn: number; idsw: number }) {
  const total = fp + fn + idsw;
  if (total === 0) {
    return (
      <div style={{ padding: "0 14px 8px", fontSize: 11, color: "var(--accent)" }}>
        ✓ Zero errors (0 FP, 0 Misses, 0 ID switches)
      </div>
    );
  }

  const pFp = ((fp / total) * 100).toFixed(1);
  const pFn = ((fn / total) * 100).toFixed(1);
  const pIdsw = ((idsw / total) * 100).toFixed(1);

  return (
    <div style={{ padding: "0 14px 10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4, fontSize: 11, color: "var(--muted)" }}>
        <span>Error Breakdown ({total} total errors)</span>
        <span>
          <span style={{ color: "#ff7d8b" }}>● {fp} FP</span>{" "}
          <span style={{ color: "#ffd07a", marginLeft: 6 }}>● {fn} FN</span>{" "}
          <span style={{ color: "#8bd0ff", marginLeft: 6 }}>● {idsw} IDSW</span>
        </span>
      </div>
      <div style={{ height: 8, background: "var(--bg-3)", borderRadius: 4, overflow: "hidden", display: "flex" }}>
        {fp > 0 && <div style={{ width: `${pFp}%`, background: "#ff7d8b" }} title={`False Positives: ${fp} (${pFp}%)`} />}
        {fn > 0 && <div style={{ width: `${pFn}%`, background: "#ffd07a" }} title={`Misses (FN): ${fn} (${pFn}%)`} />}
        {idsw > 0 && <div style={{ width: `${pIdsw}%`, background: "#8bd0ff" }} title={`ID Switches: ${idsw} (${pIdsw}%)`} />}
      </div>
    </div>
  );
}

const TRACKER_COLORS = [
  "#38bdf8", // Sky blue
  "#34d399", // Emerald
  "#f472b6", // Pink
  "#fbbf24", // Amber
  "#a78bfa", // Purple
  "#fb923c", // Orange
  "#4ade80", // Green
  "#818cf8", // Indigo
];

function MultiTrackerComparisonChart({
  results,
  chartUrl,
}: {
  results: TrackerResult[];
  chartUrl?: string;
}) {
  const [metricTab, setMetricTab] = useState<"tradeoff" | "errors" | "rates" | "latency">("tradeoff");
  const [activeTracker, setActiveTracker] = useState<string | null>(null);
  const [useLogScale, setUseLogScale] = useState<boolean>(true);
  const [exportingPng, setExportingPng] = useState<boolean>(false);
  const svgRef = useRef<SVGSVGElement>(null);

  // Trackers with errors, rates, and latency to compare
  const items = results.map((r, i) => {
    const totalErrors = (r.metrics.fp ?? 0) + (r.metrics.fn ?? 0) + (r.metrics.idsw ?? 0);
    return {
      id: r.tracker_id,
      name: r.name,
      color: TRACKER_COLORS[i % TRACKER_COLORS.length],
      fp: r.metrics.fp ?? 0,
      fn: r.metrics.fn ?? 0,
      idsw: r.metrics.idsw ?? 0,
      totalErrors,
      mota: r.metrics.mota != null ? Math.max(0, r.metrics.mota * 100) : 0,
      idf1: r.metrics.idf1 != null ? Math.max(0, r.metrics.idf1 * 100) : 0,
      avg_time_ms: r.metrics.avg_time_ms ?? 0,
      fps: r.metrics.fps ?? 0,
    };
  });

  const maxError = Math.max(...items.flatMap((it) => [it.fp, it.fn, it.idsw]), 1);
  const maxTotalErrors = Math.max(...items.map((it) => it.totalErrors), 1);
  const maxLatency = Math.max(...items.map((it) => it.avg_time_ms), 1);
  const slowestLatency = Math.max(...items.map((it) => it.avg_time_ms));

  // Trade-off 2D plot bounds
  const plotWidth = 620;
  const plotHeight = 260;
  const padLeft = 60;
  const padRight = 40;
  const padTop = 30;
  const padBottom = 48;

  const downloadChartPng = () => {
    if (chartUrl) {
      // Directly download the server-rendered high-res PNG
      const a = document.createElement("a");
      a.href = mediaUrl(chartUrl);
      a.download = "simulation-chart.png";
      a.target = "_blank";
      a.click();
      return;
    }

    const svgEl = svgRef.current;
    if (!svgEl) return;
    setExportingPng(true);

    try {
      const serializer = new XMLSerializer();
      const svgStr = serializer.serializeToString(svgEl);
      const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const img = new Image();

      img.onload = () => {
        const canvas = document.createElement("canvas");
        const scale = 2; // high resolution retina
        canvas.width = plotWidth * scale;
        canvas.height = plotHeight * scale;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);

        const pngUrl = canvas.toDataURL("image/png");
        const a = document.createElement("a");
        a.download = `tracker-comparison-chart.png`;
        a.href = pngUrl;
        a.click();
        setExportingPng(false);
      };

      img.onerror = () => {
        setExportingPng(false);
      };

      img.src = url;
    } catch {
      setExportingPng(false);
    }
  };

  // Log vs linear coordinate mapping for latency (X-axis)
  const minValLog = 0.05;
  const getNormX = (val: number) => {
    if (!useLogScale) {
      return val / Math.max(maxLatency, 0.01);
    }
    const logMin = Math.log10(minValLog);
    const logMax = Math.log10(Math.max(maxLatency, 1));
    const logVal = Math.log10(Math.max(val, minValLog));
    return Math.max(0, Math.min(1, (logVal - logMin) / (logMax - logMin)));
  };

  const activeItem = items.find((it) => it.id === activeTracker);

  return (
    <div className="panel" style={{ marginTop: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <h3 style={{ margin: 0 }}>📊 Multi-Tracker Comparison Chart</h3>
          <button
            type="button"
            className="btn ghost"
            style={{
              padding: "4px 10px",
              fontSize: 11.5,
              fontWeight: 600,
              color: "#38bdf8",
              borderColor: "rgba(56, 189, 248, 0.3)",
              background: "rgba(56, 189, 248, 0.08)",
            }}
            onClick={downloadChartPng}
            disabled={exportingPng}
            title="Download high-resolution comparison chart as PNG"
          >
            {exportingPng ? "Exporting…" : "📸 Export Chart (PNG)"}
          </button>
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <button
            className={`btn ghost ${metricTab === "tradeoff" ? "active" : ""}`}
            style={{ padding: "4px 10px", fontSize: 11, background: metricTab === "tradeoff" ? "var(--bg-3)" : undefined }}
            onClick={() => setMetricTab("tradeoff")}
          >
            ⚖️ Trade-off (Latency vs Accuracy)
          </button>
          <button
            className={`btn ghost ${metricTab === "errors" ? "active" : ""}`}
            style={{ padding: "4px 10px", fontSize: 11, background: metricTab === "errors" ? "var(--bg-3)" : undefined }}
            onClick={() => setMetricTab("errors")}
          >
            Error Breakdown (FP / FN / IDSW)
          </button>
          <button
            className={`btn ghost ${metricTab === "rates" ? "active" : ""}`}
            style={{ padding: "4px 10px", fontSize: 11, background: metricTab === "rates" ? "var(--bg-3)" : undefined }}
            onClick={() => setMetricTab("rates")}
          >
            Accuracy (MOTA / IDF1 %)
          </button>
          <button
            className={`btn ghost ${metricTab === "latency" ? "active" : ""}`}
            style={{ padding: "4px 10px", fontSize: 11, background: metricTab === "latency" ? "var(--bg-3)" : undefined }}
            onClick={() => setMetricTab("latency")}
          >
            ⚡ Latency & Speed (ms / FPS)
          </button>
        </div>
      </div>

      {metricTab === "tradeoff" ? (
        <div>
          {/* Subheader controls for Trade-off */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, fontSize: 12, color: "var(--muted)" }}>
            <span>Hover or click any bubble to inspect tracker stats:</span>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span style={{ fontSize: 11 }}>Scale:</span>
              <button
                className={`btn ghost ${useLogScale ? "active" : ""}`}
                style={{ padding: "2px 8px", fontSize: 10.5, borderRadius: 4 }}
                onClick={() => setUseLogScale(true)}
              >
                Logarithmic (Separated)
              </button>
              <button
                className={`btn ghost ${!useLogScale ? "active" : ""}`}
                style={{ padding: "2px 8px", fontSize: 10.5, borderRadius: 4 }}
                onClick={() => setUseLogScale(false)}
              >
                Linear
              </button>
            </div>
          </div>

          <div style={{ background: "var(--bg-3)", border: "1px solid var(--line)", borderRadius: 8, padding: "14px", position: "relative" }}>
            <svg
              ref={svgRef}
              viewBox={`0 0 ${plotWidth} ${plotHeight}`}
              style={{ width: "100%", height: "auto", display: "block", overflow: "visible" }}
            >
              {/* Optimal quadrant (High Accuracy + Low Latency = Top Left) */}
              <rect
                x={padLeft}
                y={padTop}
                width={(plotWidth - padLeft - padRight) * 0.45}
                height={(plotHeight - padTop - padBottom) * 0.5}
                fill="rgba(97, 224, 169, 0.07)"
                rx={6}
              />
              <text x={padLeft + 8} y={padTop + 15} fill="#61e0a9" fontSize={10} fontWeight={600}>
                ★ SWEET SPOT (Fast & Accurate)
              </text>

              {/* Grid lines */}
              <line x1={padLeft} y1={plotHeight - padBottom} x2={plotWidth - padRight} y2={plotHeight - padBottom} stroke="var(--line)" strokeWidth={1.5} />
              <line x1={padLeft} y1={padTop} x2={padLeft} y2={plotHeight - padBottom} stroke="var(--line)" strokeWidth={1.5} />

              <line x1={padLeft} y1={(padTop + (plotHeight - padBottom)) / 2} x2={plotWidth - padRight} y2={(padTop + (plotHeight - padBottom)) / 2} stroke="var(--line)" strokeDasharray="3 3" strokeWidth={0.8} />

              {/* Axis Labels */}
              <text x={(padLeft + plotWidth - padRight) / 2} y={plotHeight - 12} fill="var(--muted)" fontSize={11} textAnchor="middle">
                Average Latency per frame (ms) {useLogScale ? "[Log Scale — separated view]" : "[Linear]"} → [Lower is Faster]
              </text>
              <text
                x={-(padTop + (plotHeight - padBottom) / 2)}
                y={18}
                fill="var(--muted)"
                fontSize={11}
                textAnchor="middle"
                transform="rotate(-90)"
              >
                MOTA Accuracy (%) → [Higher is Better]
              </text>

              {/* Y-axis Ticks */}
              <text x={padLeft - 6} y={padTop + 4} fill="var(--muted)" fontSize={9} textAnchor="end">100%</text>
              <text x={padLeft - 6} y={(padTop + (plotHeight - padBottom)) / 2 + 3} fill="var(--muted)" fontSize={9} textAnchor="end">50%</text>
              <text x={padLeft - 6} y={plotHeight - padBottom + 2} fill="var(--muted)" fontSize={9} textAnchor="end">0%</text>

              {/* X-axis Ticks */}
              {useLogScale ? (
                <>
                  <text x={padLeft} y={plotHeight - padBottom + 16} fill="var(--muted)" fontSize={9} textAnchor="middle">0.1ms</text>
                  <text x={padLeft + (plotWidth - padLeft - padRight) * 0.35} y={plotHeight - padBottom + 16} fill="var(--muted)" fontSize={9} textAnchor="middle">1ms</text>
                  <text x={padLeft + (plotWidth - padLeft - padRight) * 0.70} y={plotHeight - padBottom + 16} fill="var(--muted)" fontSize={9} textAnchor="middle">10ms</text>
                  <text x={plotWidth - padRight} y={plotHeight - padBottom + 16} fill="var(--muted)" fontSize={9} textAnchor="middle">
                    {maxLatency.toFixed(0)}ms
                  </text>
                </>
              ) : (
                <>
                  <text x={padLeft} y={plotHeight - padBottom + 16} fill="var(--muted)" fontSize={9} textAnchor="middle">0ms</text>
                  <text x={(padLeft + (plotWidth - padRight)) / 2} y={plotHeight - padBottom + 16} fill="var(--muted)" fontSize={9} textAnchor="middle">
                    {(maxLatency / 2).toFixed(1)}ms
                  </text>
                  <text x={plotWidth - padRight} y={plotHeight - padBottom + 16} fill="var(--muted)" fontSize={9} textAnchor="middle">
                    {maxLatency.toFixed(1)}ms
                  </text>
                </>
              )}

              {/* Tracker Bubbles */}
              {items.map((it) => {
                const normX = getNormX(it.avg_time_ms);
                const normY = it.mota / 100;
                const cx = padLeft + normX * (plotWidth - padLeft - padRight);
                const cy = (plotHeight - padBottom) - normY * (plotHeight - padTop - padBottom);
                // Bubble radius scales with total errors (FP + FN + IDSW)
                const baseR = 7 + (it.totalErrors / Math.max(maxTotalErrors, 1)) * 13;
                const isSelected = activeTracker === it.id;
                const isHovered = activeTracker === it.id;
                const isDimmed = activeTracker !== null && !isSelected;

                return (
                  <g
                    key={it.id}
                    style={{ cursor: "pointer", transition: "opacity 0.2s" }}
                    opacity={isDimmed ? 0.25 : 1}
                    onMouseEnter={() => setActiveTracker(it.id)}
                    onClick={() => setActiveTracker((prev) => (prev === it.id ? null : it.id))}
                  >
                    {/* Pulsing ring for selected bubble */}
                    {isSelected && (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={baseR + 7}
                        fill="none"
                        stroke={it.color}
                        strokeWidth={2}
                        strokeDasharray="4 2"
                      />
                    )}
                    {/* Error halo ring */}
                    <circle
                      cx={cx}
                      cy={cy}
                      r={baseR}
                      fill={it.color}
                      fillOpacity={isSelected ? 0.45 : 0.22}
                      stroke={it.color}
                      strokeWidth={isSelected ? 2.5 : 1.5}
                    />
                    {/* Core center dot */}
                    <circle cx={cx} cy={cy} r={isSelected ? 5 : 3.5} fill={it.color} />

                    {/* Clean compact label tag when hovered/selected */}
                    {isHovered ? (
                      <g>
                        <rect
                          x={cx - 50}
                          y={Math.max(10, cy - baseR - 22)}
                          width={100}
                          height={18}
                          rx={4}
                          fill="rgba(15, 23, 42, 0.92)"
                          stroke={it.color}
                          strokeWidth={1}
                        />
                        <text
                          x={cx}
                          y={Math.max(10, cy - baseR - 22) + 12}
                          fill="#ffffff"
                          fontSize={9.5}
                          fontWeight={700}
                          textAnchor="middle"
                        >
                          {it.name}
                        </text>
                      </g>
                    ) : null}
                  </g>
                );
              })}
            </svg>

            {/* Interactive Inspector Card */}
            {activeItem ? (
              <div
                style={{
                  marginTop: 10,
                  padding: "8px 12px",
                  background: "var(--bg-2)",
                  border: `1px solid ${activeItem.color}`,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexWrap: "wrap",
                  gap: 10,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: activeItem.color }} />
                  <span style={{ fontWeight: 700, fontSize: 13, color: "var(--text)" }}>{activeItem.name}</span>
                </div>
                <div style={{ display: "flex", gap: 14, fontSize: 12 }}>
                  <span>MOTA: <b style={{ color: "#61e0a9" }}>{activeItem.mota.toFixed(1)}%</b></span>
                  <span>Latency: <b style={{ color: "#fbbf24" }}>{activeItem.avg_time_ms.toFixed(2)} ms</b></span>
                  <span>Speed: <b style={{ color: "var(--text)" }}>{activeItem.fps ? `${activeItem.fps.toFixed(0)} FPS` : "—"}</b></span>
                  <span>Errors: <b style={{ color: "#ff7d8b" }}>{activeItem.totalErrors} total</b> ({activeItem.fp} FP, {activeItem.fn} FN, {activeItem.idsw} IDSW)</span>
                </div>
              </div>
            ) : (
              <div style={{ marginTop: 8, fontSize: 11, color: "var(--muted)", textAlign: "center" }}>
                💡 Click or hover any tracker badge below or bubble above to inspect details without clutter.
              </div>
            )}
          </div>

          {/* Explanatory Guide Box */}
          <div style={{ marginTop: 12, padding: "14px 16px", background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 8, fontSize: 12 }}>
            <div style={{ fontWeight: 700, color: "var(--text)", marginBottom: 10, fontSize: 13, display: "flex", alignItems: "center", gap: 8 }}>
              <span>💡 Comprehensive Guide: Understanding Tracker Trade-offs</span>
            </div>

            {/* Core Metrics Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 14, color: "var(--muted)" }}>
              <div style={{ background: "var(--bg-3)", padding: "10px 12px", borderRadius: 6, border: "1px solid var(--line)" }}>
                <div style={{ color: "var(--text)", fontWeight: 700, marginBottom: 4, display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#38bdf8", display: "inline-block" }} />
                  1. Bubble Radius (Error Burden)
                </div>
                <div style={{ lineHeight: 1.45 }}>
                  Measures total tracking breakdown:
                  <div className="mono" style={{ color: "#ffd07a", margin: "4px 0", background: "rgba(0,0,0,0.25)", padding: "2px 6px", borderRadius: 4 }}>
                    Radius ∝ FP + FN + ID Switches
                  </div>
                  • <b>Small Bubble:</b> Smooth, continuous tracking with minimal ID swaps or lost targets.
                  <br />
                  • <b>Large Bubble:</b> High churn rate (tracker constantly loses objects, swaps IDs, or creates phantom tracks).
                </div>
              </div>

              <div style={{ background: "var(--bg-3)", padding: "10px 12px", borderRadius: 6, border: "1px solid var(--line)" }}>
                <div style={{ color: "var(--text)", fontWeight: 700, marginBottom: 4, display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ color: "#61e0a9" }}>▲</span>
                  2. Vertical Axis (Y: Accuracy)
                </div>
                <div style={{ lineHeight: 1.45 }}>
                  Standard MOTA (Multiple Object Tracking Accuracy):
                  <div className="mono" style={{ color: "#61e0a9", margin: "4px 0", background: "rgba(0,0,0,0.25)", padding: "2px 6px", borderRadius: 4 }}>
                    MOTA = 1 - (FP + FN + IDSW) / Total_GT
                  </div>
                  • <b>Near 100%:</b> High spatial IoU overlap & consistent identity maintenance across occlusions.
                  <br />
                  • <b>Low / Near 0%:</b> Target trajectory was broken or lost for majority of frames.
                </div>
              </div>

              <div style={{ background: "var(--bg-3)", padding: "10px 12px", borderRadius: 6, border: "1px solid var(--line)" }}>
                <div style={{ color: "var(--text)", fontWeight: 700, marginBottom: 4, display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ color: "#fbbf24" }}>◀</span>
                  3. Horizontal Axis (X: Speed)
                </div>
                <div style={{ lineHeight: 1.45 }}>
                  Average per-frame computational overhead:
                  <div className="mono" style={{ color: "#fbbf24", margin: "4px 0", background: "rgba(0,0,0,0.25)", padding: "2px 6px", borderRadius: 4 }}>
                    Latency (ms/frame) = 1000 / FPS
                  </div>
                  • <b>Far Left (&lt;2ms, &gt;500 FPS):</b> Ideal for edge devices, embedded cameras, robotics.
                  <br />
                  • <b>Far Right (&gt;20ms, &lt;50 FPS):</b> Heavy appearance models (DeepSORT/ReID) requiring GPU acceleration.
                </div>
              </div>
            </div>

            {/* Quadrant Map & Selection Advice */}
            <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px dashed var(--line)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
              <div>
                <b style={{ color: "var(--text)" }}>🗺️ How to Choose Based on Quadrants:</b>
                <ul style={{ margin: "4px 0 0", paddingLeft: 18, color: "var(--muted)", lineHeight: 1.45 }}>
                  <li><b style={{ color: "#61e0a9" }}>Top-Left (Sweet Spot):</b> Fast and accurate. Best default choice for real-time production pipelines (e.g., ByteTrack, BoT-SORT).</li>
                  <li><b style={{ color: "#ffd07a" }}>Top-Right (High Accuracy, High Cost):</b> Heavy Deep Learning/ReID trackers. Recommended only if offline processing or GPU is available.</li>
                  <li><b style={{ color: "#ff9b9b" }}>Bottom-Left (Fast but Fragile):</b> Simple centroid or greedy matching. Only suitable for uncrowded scenes with zero occlusion.</li>
                </ul>
              </div>
              <div>
                <b style={{ color: "var(--text)" }}>⚙️ Interactive Tips:</b>
                <ul style={{ margin: "4px 0 0", paddingLeft: 18, color: "var(--muted)", lineHeight: 1.45 }}>
                  <li>Toggle between <b>Logarithmic</b> (spreads out fast trackers) and <b>Linear</b> scales using the buttons at the top right.</li>
                  <li>Hover or click any bubble to view exact numbers (FPS, total errors, latency).</li>
                  <li>Click any tracker name in the bottom legend to highlight its location on the chart.</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Bottom Trade-off Legends */}
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 20, marginTop: 12, paddingTop: 10, borderTop: "1px dashed var(--line)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, background: "#61e0a9", borderRadius: 2 }} />
              <span><b>Top-Left</b>: Sweet Spot (High Accuracy & Fast Speed)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 8, height: 8, border: "2px solid #38bdf8", borderRadius: "50%", background: "rgba(56,189,248,0.3)" }} />
              <span><b>Small Bubble</b>: Clean run (Low FP/FN/IDSW)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 16, height: 16, border: "2px solid #ff7d8b", borderRadius: "50%", background: "rgba(255,125,139,0.3)" }} />
              <span><b>Large Bubble</b>: Heavy errors (High FP/FN/IDSW)</span>
            </div>
          </div>
        </div>
      ) : metricTab === "errors" ? (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${items.length}, 1fr)`, gap: 16, alignItems: "flex-end", height: 160, padding: "10px 10px 0", background: "var(--bg-3)", borderRadius: 8, border: "1px solid var(--line)" }}>
            {items.map((it) => (
              <div key={it.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", height: "100%", justifyContent: "flex-end" }}>
                <div style={{ display: "flex", gap: 4, alignItems: "flex-end", width: "100%", justifyContent: "center", height: 120 }}>
                  {/* FP bar */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "26%" }} title={`${it.name} FP: ${it.fp}`}>
                    <span style={{ fontSize: 9.5, color: "#ff7d8b", marginBottom: 2 }}>{it.fp}</span>
                    <div style={{ width: "100%", height: `${Math.max(4, (it.fp / maxError) * 100)}px`, background: "#ff7d8b", borderRadius: "3px 3px 0 0" }} />
                    <span style={{ fontSize: 9, color: "var(--muted)", marginTop: 2 }}>FP</span>
                  </div>
                  {/* FN bar */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "26%" }} title={`${it.name} FN (Misses): ${it.fn}`}>
                    <span style={{ fontSize: 9.5, color: "#ffd07a", marginBottom: 2 }}>{it.fn}</span>
                    <div style={{ width: "100%", height: `${Math.max(4, (it.fn / maxError) * 100)}px`, background: "#ffd07a", borderRadius: "3px 3px 0 0" }} />
                    <span style={{ fontSize: 9, color: "var(--muted)", marginTop: 2 }}>FN</span>
                  </div>
                  {/* IDSW bar */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "26%" }} title={`${it.name} IDSW: ${it.idsw}`}>
                    <span style={{ fontSize: 9.5, color: "#8bd0ff", marginBottom: 2 }}>{it.idsw}</span>
                    <div style={{ width: "100%", height: `${Math.max(4, (it.idsw / maxError) * 100)}px`, background: "#8bd0ff", borderRadius: "3px 3px 0 0" }} />
                    <span style={{ fontSize: 9, color: "var(--muted)", marginTop: 2 }}>ID</span>
                  </div>
                </div>
                <div style={{ marginTop: 6, fontSize: 11, fontWeight: 600, color: it.color, textAlign: "center", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", width: "100%" }}>
                  {it.name}
                </div>
              </div>
            ))}
          </div>

          {/* Bottom Legends */}
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 18, marginTop: 14, paddingTop: 10, borderTop: "1px dashed var(--line)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, background: "#ff7d8b", borderRadius: 2 }} />
              <span><b>False Positives (FP)</b> — Hallucinated / ghost detections</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, background: "#ffd07a", borderRadius: 2 }} />
              <span><b>False Negatives (FN)</b> — Missed / lost target objects</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, background: "#8bd0ff", borderRadius: 2 }} />
              <span><b>ID Switches (IDSW)</b> — Identity swaps across targets</span>
            </div>
          </div>
        </div>
      ) : metricTab === "rates" ? (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${items.length}, 1fr)`, gap: 16, alignItems: "flex-end", height: 160, padding: "10px 10px 0", background: "var(--bg-3)", borderRadius: 8, border: "1px solid var(--line)" }}>
            {items.map((it) => (
              <div key={it.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", height: "100%", justifyContent: "flex-end" }}>
                <div style={{ display: "flex", gap: 6, alignItems: "flex-end", width: "100%", justifyContent: "center", height: 120 }}>
                  {/* MOTA bar */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "36%" }} title={`${it.name} MOTA: ${it.mota.toFixed(1)}%`}>
                    <span style={{ fontSize: 9.5, color: "#61e0a9", marginBottom: 2 }}>{it.mota.toFixed(0)}%</span>
                    <div style={{ width: "100%", height: `${Math.max(4, (it.mota / 100) * 100)}px`, background: "#61e0a9", borderRadius: "3px 3px 0 0" }} />
                    <span style={{ fontSize: 9, color: "var(--muted)", marginTop: 2 }}>MOTA</span>
                  </div>
                  {/* IDF1 bar */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "36%" }} title={`${it.name} IDF1: ${it.idf1.toFixed(1)}%`}>
                    <span style={{ fontSize: 9.5, color: "#38bdf8", marginBottom: 2 }}>{it.idf1.toFixed(0)}%</span>
                    <div style={{ width: "100%", height: `${Math.max(4, (it.idf1 / 100) * 100)}px`, background: "#38bdf8", borderRadius: "3px 3px 0 0" }} />
                    <span style={{ fontSize: 9, color: "var(--muted)", marginTop: 2 }}>IDF1</span>
                  </div>
                </div>
                <div style={{ marginTop: 6, fontSize: 11, fontWeight: 600, color: it.color, textAlign: "center", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", width: "100%" }}>
                  {it.name}
                </div>
              </div>
            ))}
          </div>

          {/* Bottom Legends */}
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 18, marginTop: 14, paddingTop: 10, borderTop: "1px dashed var(--line)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, background: "#61e0a9", borderRadius: 2 }} />
              <span><b>MOTA</b> — Overall Multi-Object Tracking Accuracy (higher is better)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, background: "#38bdf8", borderRadius: 2 }} />
              <span><b>IDF1</b> — Identity F1 consistency score (higher is better)</span>
            </div>
          </div>
        </div>
      ) : (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${items.length}, 1fr)`, gap: 16, alignItems: "flex-end", height: 160, padding: "10px 10px 0", background: "var(--bg-3)", borderRadius: 8, border: "1px solid var(--line)" }}>
            {items.map((it) => {
              const isSlowest = slowestLatency > 0 && it.avg_time_ms === slowestLatency && items.length > 1;
              return (
                <div key={it.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", height: "100%", justifyContent: "flex-end" }}>
                  <div style={{ display: "flex", gap: 6, alignItems: "flex-end", width: "100%", justifyContent: "center", height: 120 }}>
                    {/* Latency bar */}
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "42%" }} title={`${it.name} Latency: ${it.avg_time_ms} ms (${it.fps} FPS)`}>
                      <span style={{ fontSize: 9.5, color: isSlowest ? "#ff7d8b" : "#fbbf24", marginBottom: 2, fontWeight: isSlowest ? 700 : 500 }}>
                        {it.avg_time_ms.toFixed(1)} ms
                      </span>
                      <div
                        style={{
                          width: "100%",
                          height: `${Math.max(4, (it.avg_time_ms / maxLatency) * 95)}px`,
                          background: isSlowest ? "#ff7d8b" : "#fbbf24",
                          borderRadius: "3px 3px 0 0",
                          boxShadow: isSlowest ? "0 0 8px rgba(255, 125, 139, 0.4)" : undefined,
                        }}
                      />
                      <span style={{ fontSize: 9, color: "var(--muted)", marginTop: 2 }}>{it.fps ? `${it.fps.toFixed(0)} FPS` : "ms"}</span>
                    </div>
                  </div>
                  <div style={{ marginTop: 6, fontSize: 11, fontWeight: 600, color: it.color, textAlign: "center", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", width: "100%" }}>
                    {it.name}
                    {isSlowest && (
                      <span style={{ display: "block", fontSize: 9, color: "#ff7d8b", fontWeight: 700, letterSpacing: "0.03em" }}>
                        ⚠️ SLOWEST
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Bottom Legends */}
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 18, marginTop: 14, paddingTop: 10, borderTop: "1px dashed var(--line)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, background: "#fbbf24", borderRadius: 2 }} />
              <span><b>Avg Latency (ms)</b> — Average computation time per frame (lower is faster)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, background: "#ff7d8b", borderRadius: 2 }} />
              <span><b>⚠️ SLOWEST Highlight</b> — Tracker with the highest average latency overhead</span>
            </div>
          </div>
        </div>
      )}

      {/* Tracker identity color legends at the very bottom */}
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 10, marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--line)" }}>
        <span style={{ fontSize: 11.5, color: "var(--muted)", marginRight: 4, alignSelf: "center" }}>Highlight:</span>
        {items.map((it) => {
          const isSelected = activeTracker === it.id;
          return (
            <div
              key={it.id}
              onClick={() => setActiveTracker((prev) => (prev === it.id ? null : it.id))}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: 11.5,
                cursor: "pointer",
                padding: "3px 8px",
                borderRadius: 5,
                background: isSelected ? "var(--bg-2)" : "transparent",
                border: `1px solid ${isSelected ? it.color : "transparent"}`,
                transition: "all 0.15s ease",
              }}
            >
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: it.color }} />
              <span style={{ color: isSelected ? "var(--text)" : "var(--muted)", fontWeight: isSelected ? 700 : 400 }}>{it.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function seek(url: string, frame: number, fps: number) {
  const target = mediaUrl(url);
  const videos = document.querySelectorAll("video");
  for (const v of videos) {
    if (v.src === target || v.getAttribute("src") === target) {
      v.currentTime = frame / Math.max(fps, 1);
      break;
    }
  }
}

function MSum({ label, v, unit, best }: { label: string; v: number; unit?: string; best: "high" | "low" }) {
  return (
    <span className="metric">
      <b>{FMT.format(v)}{unit ? ` ${unit}` : ""}</b>
      {label}
    </span>
  );
}

export function CompareTable({ history }: { history: SimulationResult[] }) {
  type Row = { name: string; mota: number | null; idf1: number | null; idsw: number | null; acc: number | null; avg_time_ms: number | null; fps: number | null };
  const latest = history[0];
  const prev = history[1];
  const nameOf = (r: TrackerResult) => r.name;
  const cells: Row[] = latest.results.map((r) => {
    return {
      name: nameOf(r),
      mota: r.metrics.mota ?? null,
      idf1: r.metrics.idf1 ?? null,
      idsw: r.metrics.idsw ?? null,
      acc: r.mode === "single" ? r.metrics.accuracy ?? null : null,
      avg_time_ms: r.metrics.avg_time_ms ?? null,
      fps: r.metrics.fps ?? null,
    };
  });
  const withMota = cells.filter((c) => c.mota != null);
  const best = (key: keyof Row) =>
    key === "idsw" || key === "avg_time_ms"
      ? Math.min(...cells.map((c) => (c[key] as number) ?? Infinity))
      : Math.max(...cells.map((c) => (c[key] as number) ?? -Infinity));

  return (
    <div className="compare">
      <div className="panel">
        <h3>Compare runs (latest vs previous)</h3>
        <table className="tbl">
          <thead>
            <tr>
              <th>Tracker</th>
              <th>MOTA</th>
              <th>IDF1</th>
              <th>ID switches</th>
              <th>Single-obj accuracy</th>
              <th>Avg Latency</th>
              <th>Speed</th>
            </tr>
          </thead>
          <tbody>
            {cells.map((c) => (
              <tr key={c.name} className={c.mota != null && c.mota === best("mota") ? "best" : ""}>
                <td>{c.name}</td>
                <td>{c.mota != null ? FMT.format(c.mota) : "—"}</td>
                <td>{c.idf1 != null ? FMT.format(c.idf1) : "—"}</td>
                <td>{c.idsw != null ? FMT.format(c.idsw) : "—"}</td>
                <td>{c.acc != null ? FMT.format(c.acc) : "—"}</td>
                <td className="mono">{c.avg_time_ms != null ? `${FMT.format(c.avg_time_ms)} ms` : "—"}</td>
                <td className="mono">{c.fps != null ? `${FMT.format(c.fps)} FPS` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <small>▲ higher is better · ▼ lower is better · highlighted = best MOTA</small>
      </div>
    </div>
  );
}