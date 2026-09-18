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
      </div>

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

function seek(url: string, frame: number, fps: number) {
  const v = document.querySelector(`video[src="${mediaUrl(url)}"]`) as HTMLVideoElement | null;
  if (v) v.currentTime = frame / Math.max(fps, 1);
}

function MSum({ label, v, best }: { label: string; v: number; best: "high" | "low" }) {
  return (
    <span className="metric">
      <b>{FMT.format(v)}</b>
      {label}
    </span>
  );
}

export function CompareTable({ history }: { history: SimulationResult[] }) {
  type Row = { name: string; mota: number | null; idf1: number | null; idsw: number | null; acc: number | null };
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
    };
  });
  const withMota = cells.filter((c) => c.mota != null);
  const best = (key: keyof Row) =>
    key === "idsw" ? Math.min(...withMota.map((c) => (c[key] as number) ?? Infinity)) : Math.max(...withMota.map((c) => (c[key] as number) ?? -Infinity));

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