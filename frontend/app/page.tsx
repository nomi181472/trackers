"use client";

import { useEffect, useState } from "react";
import type { Catalog, ParamValues, ScenarioMeta, SimulationResult, TrackerMeta } from "@/lib/types";
import { getCatalog, scenarioPreview, startSimulation, pollUntilDone, uploadVideo, startRealJob, mediaUrl } from "@/lib/api";
import { SliderRow, BoolRow, ParamControl, SelectRow } from "@/components/controls";
import { TrackerPicker } from "@/components/trackerPicker";
import { ResultsView, CompareTable, RealResultsView } from "@/components/results";
import type { JobStatus } from "@/lib/types";

/* ------------------------------------------------------------------ */
/* Presets & defaults                                                  */
/* ------------------------------------------------------------------ */

const PRESETS: Record<string, { label: string; desc: string; scenario: ParamValues }> = {
  clean: {
    label: "☀️ Clean scene",
    desc: "Nothing to hide. Everyone separated.",
    scenario: { seed: 2, num_objects: 3, crossing: false, occlusion: false, camera_shake: false, blur: false, similar_colors: false },
  },
  occlusion: {
    label: "🧱 Occlusion wall",
    desc: "Objects walk behind a wall.",
    scenario: { seed: 4, num_objects: 3, crossing: false, occlusion: true, occluder_width: 90, camera_shake: false, blur: false, similar_colors: false },
  },
  crossing: {
    label: "✖️ Crossing chaos",
    desc: "Two objects swap sides at the exact same moment.",
    scenario: { seed: 4, num_objects: 3, crossing: true, occlusion: false, camera_shake: false, blur: false, similar_colors: false },
  },
  lookalike: {
    label: "🧑‍🤝‍🧑 Look-alikes",
    desc: "Everyone identical — no appearance cues.",
    scenario: { seed: 5, num_objects: 4, crossing: true, occlusion: true, similar_colors: true, camera_shake: false, blur: false },
  },
  shake: {
    label: "📳 Camera shake",
    desc: "The whole frame jumps around.",
    scenario: { seed: 3, num_objects: 3, crossing: true, occlusion: false, camera_shake: true, shake_px: 12, blur: false, similar_colors: false },
  },
  blur: {
    label: "🌫️ Motion blur",
    desc: "Detector confidence drops to nothing.",
    scenario: { seed: 8, num_objects: 4, crossing: true, occlusion: true, blur: true, blur_sigma: 4, camera_shake: false, similar_colors: false },
  },
};

const SCENARIO_DEFAULTS: ParamValues = {
  seed: 4, fps: 15, duration_seconds: 6, num_objects: 3, object_type: "person", crossing: true,
  occlusion: true, occluder_width: 70, camera_shake: false, shake_px: 10,
  blur: false, blur_sigma: 3, similar_colors: false,
};

/* ------------------------------------------------------------------ */

import { LogsView } from "@/components/logsView";

export default function Home() {
  const [tab, setTab] = useState<"sim" | "real" | "logs">("sim");
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [catalogErr, setCatalogErr] = useState<string | null>(null);

  useEffect(() => {
    getCatalog()
      .then(setCatalog)
      .catch((e) => setCatalogErr(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="wrap">
      <div className="tabs">
        <div className={`tab ${tab === "sim" ? "active" : ""}`} onClick={() => setTab("sim")}>
          🧪 Simulator
        </div>
        <div className={`tab ${tab === "real" ? "active" : ""}`} onClick={() => setTab("real")}>
          🎥 Real video
        </div>
        <div className={`tab ${tab === "logs" ? "active" : ""}`} onClick={() => setTab("logs")}>
          📋 Server Logs
        </div>
      </div>
      {catalogErr ? <div className="err">Can&apos;t reach the backend: {catalogErr}. Start it with <span className="mono">uvicorn app.main:app --port 8000</span> in backend/.</div> : null}
      {tab === "logs" ? (
        <LogsView />
      ) : catalog ? (
        tab === "sim" ? (
          <SimulatorTab catalog={catalog} />
        ) : (
          <RealTab catalog={catalog} />
        )
      ) : !catalogErr ? (
        <div className="panel">Loading tracker catalog…</div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Simulator tab                                                       */
/* ------------------------------------------------------------------ */

function SimulatorTab({ catalog }: { catalog: Catalog }) {
  const [scenario, setScenario] = useState<ParamValues>({ ...SCENARIO_DEFAULTS });
  const [detection, setDetection] = useState<ParamValues>({});
  const [selected, setSelected] = useState<Record<string, ParamValues | null>>({});
  const [preview, setPreview] = useState<ScenarioMeta | null>(null);
  const [running, setRunning] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [history, setHistory] = useState<SimulationResult[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [preset, setPreset] = useState<string>("crossing");
  const [showDetNoise, setShowDetNoise] = useState(false);

  useEffect(() => {
    const det: ParamValues = {};
    [...catalog.detector_params, ...catalog.scenario_detection_params].forEach((p) => (det[p.key] = p.default));
    setDetection(det);
    const defaults: Record<string, ParamValues | null> = {};
    catalog.trackers.filter((t) => t.available).forEach((t) => (defaults[t.id] = { ...(catalog.defaults[t.id] || {}) }));
    setSelected(defaults);
  }, [catalog]);

  const applyPreset = (name: string) => {
    setPreset(name);
    setScenario({ ...SCENARIO_DEFAULTS, ...PRESETS[name].scenario });
  };

  const run = async () => {
    setErr(null);
    setResult(null);
    const chosen = catalog.trackers.filter((t) => selected[t.id]);
    if (!chosen.length) return setErr("Pick at least one tracker.");
    const unavailable = chosen.filter((t) => !t.available);
    if (unavailable.length)
      return setErr(
        `"${unavailable.map((t) => t.name).join(", ")}" not available in this OpenCV/Python build — enable a different tracker.`
      );
    try {
      const { job_id } = await startSimulation({
        scenario,
        detection,
        trackers: chosen.map((t) => ({ tracker_id: t.id, params: selected[t.id] || {} })),
      });
      const job = await pollUntilDone(job_id, setRunning);
      if (job.status === "error") throw new Error(job.error || "job failed");
      const res = job.result as SimulationResult;
      setResult(res);
      setHistory((h) => [res, ...h].slice(0, 3));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
    setRunning(null);
  };

  const previewIt = async () => {
    try {
      const { meta } = await scenarioPreview(scenario);
      setPreview(meta);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="grid">
      <div>
        <div className="panel">
          <h3>Scenario — pick what breaks trackers</h3>
          <div className="presets">
            {Object.entries(PRESETS).map(([k, v]) => (
              <div key={k} className={`preset ${preset === k ? "active" : ""}`} onClick={() => applyPreset(k)}>
                <b>{v.label}</b>
                {v.desc}
              </div>
            ))}
          </div>

          <SelectRow
            label="Object type"
            value={String(scenario.object_type || "person")}
            options={[
              { value: "person", label: "🏃 Person (Skeleton)" },
              { value: "car", label: "🚗 Car" },
              { value: "ball", label: "⚽ Ball" },
            ]}
            hint="Choose what to simulate: animated walking skeleton, car, or ball."
            onChange={(v) => setScenario((s) => ({ ...s, object_type: v }))}
          />

          <SliderRow label="Seed" min={0} max={99} step={1} value={Number(scenario.seed)} hint="Changes the random layout of the scene." onChange={(v) => setScenario((s) => ({ ...s, seed: v }))} />
          <SliderRow label="Objects" min={1} max={6} step={1} value={Number(scenario.num_objects)} hint="How many objects to track." onChange={(v) => setScenario((s) => ({ ...s, num_objects: v }))} />
          <SliderRow label="Duration" min={2} max={15} step={1} value={Number(scenario.duration_seconds)} unit="s" hint="Keep it short while experimenting." onChange={(v) => setScenario((s) => ({ ...s, duration_seconds: v }))} />
          <SliderRow label="Frame rate" min={5} max={30} step={1} value={Number(scenario.fps)} unit="fps" onChange={(v) => setScenario((s) => ({ ...s, fps: v }))} />

          <BoolRow label="Crossing objects" text="Two objects swap sides, overlapping at the midpoint." value={Boolean(scenario.crossing)} onChange={(v) => setScenario((s) => ({ ...s, crossing: v }))} />
          <BoolRow label="Occlusion wall" text="Objects disappear behind a wall." value={Boolean(scenario.occlusion)} onChange={(v) => setScenario((s) => ({ ...s, occlusion: v }))} />
          {scenario.occlusion ? (
            <SliderRow label="Wall thickness" min={30} max={140} step={2} value={Number(scenario.occluder_width)} unit="px" hint="Thicker = longer hidden = harder to re-find." onChange={(v) => setScenario((s) => ({ ...s, occluder_width: v }))} />
          ) : null}
          <BoolRow label="Camera shake" text="The whole frame translates randomly → tests motion-compensation." value={Boolean(scenario.camera_shake)} onChange={(v) => setScenario((s) => ({ ...s, camera_shake: v }))} />
          {scenario.camera_shake ? (
            <SliderRow label="Shake strength" min={2} max={30} step={1} value={Number(scenario.shake_px)} unit="px" onChange={(v) => setScenario((s) => ({ ...s, shake_px: v }))} />
          ) : null}
          <BoolRow label="Motion blur" text="Whole frame blurred → detector confidence melts." value={Boolean(scenario.blur)} onChange={(v) => setScenario((s) => ({ ...s, blur: v }))} />
          {scenario.blur ? (
            <SliderRow label="Blur strength" min={1} max={7} step={0.5} value={Number(scenario.blur_sigma)} onChange={(v) => setScenario((s) => ({ ...s, blur_sigma: v }))} />
          ) : null}
          <BoolRow label="Look-alikes" text="All objects the same colour → appearance gives no clues." value={Boolean(scenario.similar_colors)} onChange={(v) => setScenario((s) => ({ ...s, similar_colors: v }))} />

          <button className="btn ghost" style={{ marginTop: 4 }} onClick={previewIt} disabled={running !== null}>
            Preview scene
          </button>
          {preview ? (
            <div style={{ marginTop: 10 }}>
              <video src={mediaUrl(preview.preview_url)} controls muted loop style={{ width: "100%", borderRadius: 8, border: "1px solid var(--line)" }} />
              <small>
                {preview.frames} frames · {preview.width}×{preview.height} — boxes = ground truth that each tracker is judged against.
              </small>
            </div>
          ) : null}
        </div>

        <div className="panel" style={{ marginTop: 16 }}>
          <h3>Detector — the eyes your trackers see through</h3>
          {catalog.detector_params.map((p) => (
            <ParamControl key={p.key} p={p} value={detection[p.key] ?? p.default} onChange={(v) => setDetection((d) => ({ ...d, [p.key]: v }))} />
          ))}
          <div className="collapse-head" onClick={() => setShowDetNoise((s) => !s)}>
            <span style={{ transform: `rotate(${showDetNoise ? 90 : 0}deg)`, display: "inline-block" }}>▸</span>
            Simulated detection imperfections {showDetNoise ? "▲" : "▼"}
          </div>
          {showDetNoise &&
            catalog.scenario_detection_params.map((p) => (
              <ParamControl key={p.key} p={p} value={detection[p.key] ?? p.default} onChange={(v) => setDetection((d) => ({ ...d, [p.key]: v }))} />
            ))}
        </div>

        <div className="panel" style={{ marginTop: 16 }}>
          <h3>Trackers — pick who to torture</h3>
          <div className="legend">
            <span className="badge badge-ul">ultralytics</span>
            <span className="badge badge-cu">bespoke</span>
            <span className="badge badge-cv">opencv</span>
          </div>
          <TrackerPicker catalog={catalog} selected={selected} setSelected={setSelected} />
        </div>

        {err ? <div className="err" style={{ marginTop: 12 }}>{err}</div> : null}

        <button className="btn" style={{ width: "100%", marginTop: 14 }} onClick={run} disabled={running !== null}>
          {running ? "Running…" : "▶ Run simulation"}
        </button>
        {running ? (
          <div style={{ marginTop: 10 }}>
            <div className="progress"><div style={{ width: `${Math.round((running.progress || 0) * 100)}%` }} /></div>
            <small style={{ display: "block", marginTop: 6 }}>
              <span className="spin" /> <span className="mono">{running.message}</span> ({Math.round((running.progress || 0) * 100)}%)
            </small>
          </div>
        ) : null}
      </div>

      <div>
        {result ? <ResultsView result={result} /> : null}
        {history.length > 1 ? <CompareTable history={history} /> : null}
        {!result && !history.length ? (
          <div className="panel" style={{ textAlign: "center", padding: "60px 20px", color: "var(--muted)" }}>
            <div style={{ fontSize: 30 }}>🎬</div>
            Pick a scenario, choose trackers, hit <b>Run simulation</b>. We&apos;ll show exactly where each tracker loses its mind — and why.
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Real video tab                                                      */
/* ------------------------------------------------------------------ */

function RealTab({ catalog }: { catalog: Catalog }) {
  const [file, setFile] = useState<File | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [trackerId, setTrackerId] = useState<string>("bytetrack");
  const [params, setParams] = useState<ParamValues>({});
  const [detParams, setDetParams] = useState<ParamValues>({});
  const [running, setRunning] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const det: ParamValues = {};
    catalog.detector_params.forEach((p) => (det[p.key] = p.default));
    setDetParams(det);
  }, [catalog]);

  const onFile = async (f: File | null) => {
    setFile(f);
    setUploadId(null);
    if (f) {
      try {
        const { upload_id } = await uploadVideo(f);
        setUploadId(upload_id);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    }
  };

  const tracker = catalog.trackers.find((t) => t.id === trackerId);

  const run = async () => {
    setErr(null);
    setResult(null);
    if (!uploadId) return setErr("Upload a video first.");
    if (!tracker || !tracker.available) return setErr("That tracker is not available in this build.");
    try {
      const { job_id } = await startRealJob({ upload_id: uploadId, tracker_id: trackerId, params, det_params: detParams });
      const job = await pollUntilDone(job_id, setRunning);
      if (job.status === "error") throw new Error(job.error || "job failed");
      setResult(job.result as SimulationResult);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
    setRunning(null);
  };

  return (
    <div className="grid">
      <div>
        <div className="panel">
          <h3>Your video</h3>
          <input type="file" accept="video/*" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
          {file ? (
            <small style={{ display: "block", marginTop: 6 }}>
              {file.name} · {(file.size / 1e6).toFixed(1)} MB {uploadId ? "· uploaded ✓" : "· uploading…"}
            </small>
          ) : null}
          <div className="collapse-head" style={{ marginTop: 12 }}>Notes</div>
          <small>
            Supports the formats OpenCV can read (mp4/mov/avi…). YOLO detection runs frame-by-frame, so long clips are sampled to ≤600 frames. There is no ground truth here — the report card is heuristic, watching for dropped/swapped ids.
          </small>
        </div>

        <div className="panel" style={{ marginTop: 16 }}>
          <h3>Tracker</h3>
          <div className="field">
            <select value={trackerId} onChange={(e) => { setTrackerId(e.target.value); setParams({}); }}>
              {catalog.trackers.filter((t) => t.available && t.mode === "multi").map((t) => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>
          </div>
          {tracker ? (
            <div className="section-row" style={{ margin: 0 }}>
              <b>{tracker.name}</b> — {tracker.tagline}
              {tracker.params.map((p) => (
                <ParamControl key={p.key} p={p} value={params[p.key] ?? p.default} onChange={(v) => setParams((s) => ({ ...s, [p.key]: v }))} />
              ))}
            </div>
          ) : null}
        </div>

        <div className="panel" style={{ marginTop: 16 }}>
          <h3>Detector</h3>
          {catalog.detector_params.filter((p) => p.key !== "model").map((p) => (
            <ParamControl key={p.key} p={p} value={detParams[p.key] ?? p.default} onChange={(v) => setDetParams((d) => ({ ...d, [p.key]: v }))} />
          ))}
        </div>

        {err ? <div className="err" style={{ marginTop: 12 }}>{err}</div> : null}
        <button className="btn" style={{ width: "100%", marginTop: 14 }} onClick={run} disabled={running !== null}>
          {running ? "Running…" : "▶ Run tracker on video"}
        </button>
        {running ? (
          <div style={{ marginTop: 10 }}>
            <div className="progress"><div style={{ width: `${Math.round((running.progress || 0) * 100)}%` }} /></div>
            <small style={{ display: "block", marginTop: 6 }}>
              <span className="spin" /> <span className="mono">{running.message}</span>
            </small>
          </div>
        ) : null}
      </div>

      <div>{result ? <RealResultsView result={result} /> : <div className="panel" style={{ textAlign: "center", padding: 60, color: "var(--muted)" }}>Upload a clip to find real-world failures.</div>}</div>
    </div>
  );
}