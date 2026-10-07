"use client";

import { mediaUrl } from "@/lib/api";
import type { RealResult, SimulationResult, TrackerResult } from "@/lib/types";

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
        {result.scenario.preview_url ? (
          <video src={mediaUrl(result.scenario.preview_url)} controls loop muted style={{ width: "100%", borderRadius: 8, border: "1px solid var(--line)" }} />
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
      {r.video_url ? (
        <video src={mediaUrl(r.video_url)} controls loop muted poster={r.thumbnails?.[0]?.url ? mediaUrl(r.thumbnails[0].url) : undefined} />
      ) : null}

      <div className="metrics">
        {typeof m.mota === "number" && isFinite(m.mota) ? <MSum label="MOTA" v={m.mota} best="high" /> : null}
        {typeof m.motp === "number" && isFinite(m.motp) ? <MSum label="MOTP" v={m.motp} best="low" /> : null}
        {typeof m.idf1 === "number" && isFinite(m.idf1) ? <MSum label="IDF1" v={m.idf1} best="high" /> : null}
        {typeof m.idsw === "number" && isFinite(m.idsw) ? <MSum label="ID switches" v={m.idsw} best="low" /> : null}
        {typeof m.fp === "number" && isFinite(m.fp) ? <MSum label="FP" v={m.fp} best="low" /> : null}
        {typeof m.fn === "number" && isFinite(m.fn) ? <MSum label="FN" v={m.fn} best="low" /> : null}
        {typeof m.mt === "number" && isFinite(m.mt) ? <MSum label="MT" v={m.mt} best="high" /> : null}
        {typeof m.ml === "number" && isFinite(m.ml) ? <MSum label="ML" v={m.ml} best="low" /> : null}
        {typeof m.accuracy === "number" && isFinite(m.accuracy) ? <MSum label="Accuracy" v={m.accuracy} best="high" /> : null}
        {typeof m.total_visible === "number" ? <MSum label="Missed frames" v={m.lost_frames ?? 0} best="low" /> : null}
        {typeof m.avg_time_ms === "number" && isFinite(m.avg_time_ms) ? <MSum label="Avg latency" v={m.avg_time_ms} unit="ms" best="low" /> : null}
        {typeof m.fps === "number" && isFinite(m.fps) ? <MSum label="Tracker speed" v={m.fps} unit="FPS" best="high" /> : null}
      </div>

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
            <div key={`${r.tracker_id}-${t.frame}-${t.type}-${idx}`} className="thumb" title={t.type} onClick={() => seek(r.video_url, t.frame, fps)}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={mediaUrl(t.url)} alt={t.type} loading="lazy" />
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

function seek(url: string, frame: number, fps: number) {
  const v = document.querySelector(`video[src="${mediaUrl(url)}"]`) as HTMLVideoElement | null;
  if (v) v.currentTime = frame / Math.max(fps, 1);
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

export function RealResultsView({ result }: { result: RealResult }) {
  const r = result.results[0];
  return <ResultCard r={r} fps={15} />;
}