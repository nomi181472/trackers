"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Catalog, ParamValues, ScenarioMeta, SimulationResult } from "@/lib/types";
import { getCatalog, scenarioPreview, startSimulation, pollUntilDone, mediaUrl, cleanupGeneratedFiles, cancelJob, sendBeaconCancel } from "@/lib/api";
import { SliderRow, BoolRow, ParamControl, SelectRow } from "@/components/controls";
import { TrackerPicker } from "@/components/trackerPicker";
import { ResultsView, CompareTable } from "@/components/results";
import { LogsView } from "@/components/logsView";
import type { JobStatus } from "@/lib/types";
import { saveActiveJob, getActiveJob, clearActiveJob } from "@/lib/jobStorage";

/* ------------------------------------------------------------------ */
/* Presets: Standard & Production                                     */
/* ------------------------------------------------------------------ */

interface PresetDef {
  label: string;
  badge?: string;
  desc: string;
  scenario: ParamValues;
  detection?: ParamValues;
}

const STANDARD_PRESETS: Record<string, PresetDef> = {
  clean: {
    label: "☀️ Clean Scene",
    badge: "Baseline",
    desc: "Zero occlusions, separated paths. Verify ideal baseline tracking.",
    scenario: { seed: 2, num_objects: 3, object_type: "person", crossing: false, occlusion: false, camera_shake: false, blur: false, similar_colors: false },
  },
  occlusion: {
    label: "🧱 Occlusion Wall",
    badge: "Stress",
    desc: "Objects walk behind a physical wall. Tests re-identification vs track loss.",
    scenario: { seed: 4, num_objects: 3, object_type: "person", crossing: false, occlusion: true, occluder_width: 85, camera_shake: false, blur: false, similar_colors: false },
  },
  crossing: {
    label: "✖️ Trajectory Crossing",
    badge: "Identity",
    desc: "Objects cross and swap sides at the exact same frame. Tests ID switch resilience.",
    scenario: { seed: 4, num_objects: 3, object_type: "person", crossing: true, occlusion: false, camera_shake: false, blur: false, similar_colors: false },
  },
  lookalike: {
    label: "🧑‍🤝‍🧑 Look-alikes (Disguise)",
    badge: "Appearance",
    desc: "All objects share identical colors. Forces reliance on motion dynamics.",
    scenario: { seed: 5, num_objects: 4, object_type: "person", crossing: true, occlusion: true, similar_colors: true, camera_shake: false, blur: false },
  },
  shake: {
    label: "📳 Severe Camera Shake",
    badge: "Camera Motion",
    desc: "Frame violently translates. Tests Global Motion Compensation (GMC).",
    scenario: { seed: 3, num_objects: 3, object_type: "person", crossing: true, occlusion: false, camera_shake: true, shake_px: 14, blur: false, similar_colors: false },
  },
  blur: {
    label: "🌫️ High Motion Blur",
    badge: "Detector Stress",
    desc: "Simulated rapid movement blur degrades bounding box detections.",
    scenario: { seed: 8, num_objects: 4, object_type: "person", crossing: true, occlusion: true, blur: true, blur_sigma: 4.5, camera_shake: false, similar_colors: false },
  },
};

const PRODUCTION_PRESETS: Record<string, PresetDef> = {
  cctv_surveillance: {
    label: "📹 Security CCTV Corridor",
    badge: "Surveillance",
    desc: "15 FPS surveillance feed with pedestrian crossings, moderate occlusions, and real-world detector confidence.",
    scenario: { seed: 12, fps: 15, duration_seconds: 6, num_objects: 4, object_type: "person", crossing: true, occlusion: true, occluder_width: 65, camera_shake: false, blur: false, similar_colors: false },
    detection: { conf: 0.35, iou: 0.65, det_dropout: 0.03, det_noise_pos: 2.0 },
  },
  traffic_intersection: {
    label: "🚦 Urban Vehicle Traffic",
    badge: "Smart City",
    desc: "Fast moving vehicles crossing intersections with occasional truck occlusions.",
    scenario: { seed: 21, fps: 24, duration_seconds: 7, num_objects: 5, object_type: "car", crossing: true, occlusion: true, occluder_width: 90, camera_shake: false, blur: false, similar_colors: false },
    detection: { conf: 0.45, iou: 0.7, det_dropout: 0.02, det_noise_pos: 1.5 },
  },
  dense_retail: {
    label: "🛍️ Dense Retail / Store Crowd",
    badge: "High Density",
    desc: "High density of shoppers (6 objects) in tight proximity with look-alike clothing and frequent overlaps.",
    scenario: { seed: 44, fps: 20, duration_seconds: 8, num_objects: 6, object_type: "person", crossing: true, occlusion: true, occluder_width: 75, camera_shake: false, blur: false, similar_colors: true },
    detection: { conf: 0.30, iou: 0.6, det_dropout: 0.05, det_noise_pos: 3.0 },
  },
  edge_robotics: {
    label: "🤖 Mobile Robot / Drone Feed",
    badge: "Edge AI",
    desc: "Active camera vibration + motion blur. Simulates real-time edge embedded processors with strict latency requirements.",
    scenario: { seed: 33, fps: 30, duration_seconds: 5, num_objects: 3, object_type: "ball", crossing: true, occlusion: false, camera_shake: true, shake_px: 12, blur: true, blur_sigma: 3.5, similar_colors: false },
    detection: { conf: 0.40, iou: 0.65, det_dropout: 0.04, det_noise_pos: 2.5 },
  },
};

const SCENARIO_DEFAULTS: ParamValues = {
  seed: 4,
  fps: 15,
  duration_seconds: 6,
  num_objects: 3,
  object_type: "person",
  crossing: true,
  occlusion: true,
  occluder_width: 70,
  camera_shake: false,
  shake_px: 10,
  blur: false,
  blur_sigma: 3,
  similar_colors: false,
};

/* ------------------------------------------------------------------ */
/* Main Application Component                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [mode, setMode] = useState<"standard" | "production" | "logs">("standard");
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [catalogErr, setCatalogErr] = useState<string | null>(null);
  const [theme, setTheme] = useState<"dark" | "light" | "midnight">("dark");
  const [showClearModal, setShowClearModal] = useState<boolean>(false);
  const [clearing, setClearing] = useState<boolean>(false);
  const [clearStatus, setClearStatus] = useState<string | null>(null);
  const [clearError, setClearError] = useState<string | null>(null);
  const [selectedClearWorker, setSelectedClearWorker] = useState<string>("worker-n");

  useEffect(() => {
    const savedTheme = (localStorage.getItem("tracker_theme") as "dark" | "light" | "midnight") || "dark";
    setTheme(savedTheme);
    document.documentElement.setAttribute("data-theme", savedTheme);
  }, []);

  const changeTheme = (newTheme: "dark" | "light" | "midnight") => {
    setTheme(newTheme);
    localStorage.setItem("tracker_theme", newTheme);
    document.documentElement.setAttribute("data-theme", newTheme);
  };

  useEffect(() => {
    getCatalog()
      .then(setCatalog)
      .catch((e) => setCatalogErr(e instanceof Error ? e.message : String(e)));
  }, []);

  const handleClearAllRecords = async () => {
    setClearing(true);
    setClearError(null);
    try {
      const res = await cleanupGeneratedFiles({
        include_jobs: true,
        include_scenarios: true,
        include_uploads: true,
        worker: selectedClearWorker,
      });
      setShowClearModal(false);
      const workerLabel = selectedClearWorker === "all" ? "all workers" : selectedClearWorker;
      setClearStatus(
        `Successfully cleared ${res.deleted_count} files (${res.freed_mb} MB freed) on ${workerLabel}.`
      );
      setTimeout(() => setClearStatus(null), 5000);
    } catch (e) {
      setClearError(e instanceof Error ? e.message : String(e));
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="wrap">
      {/* Top Header Bar */}
      <div className="topbar" style={{ borderRadius: "12px", marginBottom: "16px", flexWrap: "wrap", gap: "12px" }}>
        <div className="topbar-left">
          <div className="dot" />
          <div>
            <h1>
              <span>🎯 Tracker Lab</span>
              <span style={{ fontSize: "12px", fontWeight: 400, color: "var(--muted)", borderLeft: "1px solid var(--line)", paddingLeft: "8px" }}>
                Multi-Object Tracking Evaluation Platform
              </span>
            </h1>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
          {/* Mode Switcher */}
          <div className="mode-switcher" style={{ margin: 0 }}>
            <button
              type="button"
              className={`mode-btn ${mode === "standard" ? "active" : ""}`}
              onClick={() => setMode("standard")}
            >
              <span>🧪 Standard</span>
              <span className="mode-pill mode-pill-std">Diagnostics</span>
            </button>
            <button
              type="button"
              className={`mode-btn prod ${mode === "production" ? "active" : ""}`}
              onClick={() => setMode("production")}
            >
              <span>🚀 Production</span>
              <span className="mode-pill mode-pill-prod">SLAs</span>
            </button>
            <button
              type="button"
              className={`mode-btn ${mode === "logs" ? "active" : ""}`}
              onClick={() => setMode("logs")}
            >
              <span>📋 Logs</span>
            </button>
          </div>

          {/* Clear Records Button */}
          <button
            type="button"
            className="clear-records-btn"
            onClick={() => setShowClearModal(true)}
            title="Clear all generated simulation videos, thumbnails, and preview clips"
          >
            <span>🗑️</span>
            <span>Clear Records</span>
          </button>

          {/* Local Worker Concurrency Indicator */}
          {catalog?.worker_concurrency ? (
            <div
              className="worker-badge tooltip"
              title="Locally configured thread pool executor capacity"
            >
              <span className="active-dot" />
              <span>
                Workers: <b style={{ color: "var(--accent)" }}>{catalog.worker_concurrency.max_workers}</b>
              </span>
              <div className="dttip" style={{ width: "260px" }}>
                <b>Local Job Concurrency</b>
                <div style={{ marginTop: "4px", lineHeight: "1.4" }}>
                  Currently running with <b>{catalog.worker_concurrency.max_workers} default workers</b> (queue limit: {catalog.worker_concurrency.max_queue_size}).
                </div>
                <div className="hint" style={{ marginTop: "6px" }}>
                  💡 To increase workers locally, start the backend with:
                  <div className="mono" style={{ marginTop: "4px", color: "var(--accent-hover)", background: "var(--bg)", padding: "3px 6px", borderRadius: "4px" }}>
                    SIM_MAX_WORKERS=4 ./run.sh
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {/* Theme / Appearance Switcher */}
          <div className="theme-selector">
            <button
              type="button"
              className={`theme-btn ${theme === "dark" ? "active" : ""}`}
              onClick={() => changeTheme("dark")}
              title="Dark Mode"
            >
              <span>🌙 Dark</span>
            </button>
            <button
              type="button"
              className={`theme-btn ${theme === "light" ? "active" : ""}`}
              onClick={() => changeTheme("light")}
              title="Light Mode"
            >
              <span>☀️ Light</span>
            </button>
            <button
              type="button"
              className={`theme-btn ${theme === "midnight" ? "active" : ""}`}
              onClick={() => changeTheme("midnight")}
              title="Midnight (OLED) Mode"
            >
              <span>🌌 Midnight</span>
            </button>
          </div>
        </div>
      </div>

      {/* Local-Only Execution Notice Banner */}
      <div className="local-notice">
        <div>
          <span>🛡️ <b>Local Diagnostic Tool</b>: Designed strictly for local research and development benchmarking. Not intended or licensed for SaaS or commercial deployment.</span>
        </div>
        <span style={{ fontSize: "11px", opacity: 0.8, whiteSpace: "nowrap" }}>Local Execution Only</span>
      </div>

      {clearStatus ? (
        <div className="panel" style={{ marginBottom: "16px", padding: "10px 16px", borderLeft: "4px solid #10b981", color: "#6ee7b7", background: "rgba(16, 185, 129, 0.1)" }}>
          ✓ {clearStatus}
        </div>
      ) : null}

      {catalogErr ? (
        <div className="err" style={{ marginBottom: "16px" }}>
          Can&apos;t reach the backend: {catalogErr}. Start it with <span className="mono">uvicorn app.main:app --port 8000</span> in backend/.
        </div>
      ) : null}

      {/* Loading state — shown only before catalog arrives */}
      {!catalog && !catalogErr ? (
        <div className="panel" style={{ padding: "40px", textAlign: "center" }}>
          <span className="spin" style={{ width: 22, height: 22 }} />
          <span>Loading tracker catalog &amp; engines…</span>
        </div>
      ) : null}

      {/* LogsView — always mounted, hidden when not active.
          Keeps scroll position and loaded log lines intact. */}
      <div style={{ display: mode === "logs" ? "block" : "none" }}>
        <LogsView />
      </div>

      {/* SimulatorWorkspace — always mounted so running job state (job ID,
          progress, results) is never lost when switching to Logs tab and back. */}
      {catalog ? (
        <div style={{ display: mode !== "logs" ? "block" : "none" }}>
          <SimulatorWorkspace catalog={catalog} mode={mode} />
        </div>
      ) : null}

      {/* Clear Records Confirmation Modal */}
      {showClearModal ? (
        <div className="modal-backdrop" onClick={() => !clearing && setShowClearModal(false)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                <span>🗑️</span>
                <span>Clear All Generated Records</span>
              </h3>
              <button
                type="button"
                onClick={() => !clearing && setShowClearModal(false)}
                style={{ background: "transparent", border: "none", color: "var(--muted)", cursor: "pointer", fontSize: "16px" }}
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <p style={{ margin: "0 0 12px", color: "var(--text)", lineHeight: 1.5 }}>
                Are you sure you want to delete and purge all generated simulation records?
              </p>
              <div style={{ background: "var(--bg-3)", padding: "12px", borderRadius: "8px", fontSize: "12.5px", color: "var(--muted)", border: "1px solid var(--line)", marginBottom: "14px" }}>
                <div>• All benchmark simulation video clips (<span className="mono">*.mp4</span>)</div>
                <div>• All failure event frame thumbnails (<span className="mono">*.jpg</span>)</div>
                <div>• All synthetic scenario preview videos</div>
                <div>• Any temporary video uploads</div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <label style={{ fontSize: "12px", fontWeight: 600, color: "var(--text)" }}>
                  Target Worker Node:
                </label>
                <select
                  value={selectedClearWorker}
                  onChange={(e) => setSelectedClearWorker(e.target.value)}
                  disabled={clearing}
                  style={{
                    padding: "8px 10px",
                    borderRadius: "6px",
                    background: "var(--bg-2)",
                    border: "1px solid var(--line)",
                    color: "var(--text)",
                    fontSize: "13px",
                    fontWeight: 600,
                  }}
                >
                  <option value="worker-n">⚡ worker-n (Active Node)</option>
                </select>
              </div>
              {clearError ? (
                <div className="err" style={{ marginTop: "12px" }}>
                  {clearError}
                </div>
              ) : null}
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn ghost"
                style={{ padding: "8px 14px", fontSize: "13px" }}
                onClick={() => setShowClearModal(false)}
                disabled={clearing}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-danger"
                onClick={handleClearAllRecords}
                disabled={clearing}
              >
                {clearing ? "Clearing Records…" : "Yes, Delete All Records"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Unified Simulator Workspace (Standard & Production Modes)           */
/* ------------------------------------------------------------------ */

function SimulatorWorkspace({ catalog, mode }: { catalog: Catalog; mode: "standard" | "production" | "logs" }) {
  const isProduction = mode === "production";
  const presets = isProduction ? PRODUCTION_PRESETS : STANDARD_PRESETS;
  const initialPresetKey = isProduction ? "cctv_surveillance" : "crossing";

  const [preset, setPreset] = useState<string>(initialPresetKey);
  const [scenario, setScenario] = useState<ParamValues>({
    ...SCENARIO_DEFAULTS,
    ...(presets[initialPresetKey]?.scenario || {}),
  });
  const [detection, setDetection] = useState<ParamValues>({});
  const [selected, setSelected] = useState<Record<string, ParamValues | null>>({});
  const [preview, setPreview] = useState<ScenarioMeta | null>(null);
  const [running, setRunning] = useState<JobStatus | null>(null);
  const [activeWorker, setActiveWorker] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<boolean>(false);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [history, setHistory] = useState<SimulationResult[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [showAdvancedDet, setShowAdvancedDet] = useState(false);

  // Stable ref so event handlers always see the current jobId without stale closure
  const activeJobIdRef = useRef<string | null>(null);
  activeJobIdRef.current = activeJobId;

  // Sync preset defaults when mode switches
  useEffect(() => {
    const defaultKey = isProduction ? "cctv_surveillance" : "crossing";
    const p = presets[defaultKey];
    setPreset(defaultKey);
    if (p) {
      setScenario({ ...SCENARIO_DEFAULTS, ...p.scenario });
      if (p.detection) {
        setDetection((d) => ({ ...d, ...p.detection }));
      }
    }
  }, [mode, isProduction]);

  // Load tracker catalog defaults
  useEffect(() => {
    const det: ParamValues = {};
    [...catalog.detector_params, ...catalog.scenario_detection_params].forEach((p) => {
      det[p.key] = p.default;
    });
    setDetection(det);

    const defaults: Record<string, ParamValues | null> = {};
    // By default, select standard top performers: ByteTrack, BoT-SORT, OC-SORT
    catalog.trackers
      .filter((t) => t.available)
      .forEach((t) => {
        if (["bytetrack", "botsort", "ocsort"].includes(t.id)) {
          defaults[t.id] = { ...(catalog.defaults[t.id] || {}) };
        }
      });
    setSelected(defaults);
  }, [catalog]);

  // ── On mount: restore any active job this tab had before a tab-switch / focus loss ──
  useEffect(() => {
    const saved = getActiveJob();
    if (!saved) return;
    // Job IDs are unique per-tab via jobStorage, so this is always this user's job
    setActiveJobId(saved.jobId);
    if (saved.worker) setActiveWorker(saved.worker);
    // Reattach polling — updates progress bar and fetches result when done
    pollUntilDone(saved.jobId, setRunning)
      .then((job) => {
        clearActiveJob();
        setActiveJobId(null);
        if (job.status === "done" && job.result) {
          const res = job.result as SimulationResult;
          setResult(res);
          setHistory((h) => [res, ...h].slice(0, 4));
        } else if (job.status === "cancelled") {
          setErr("Simulation was cancelled.");
        } else if (job.status === "error") {
          setErr(job.error || "Simulation job encountered an error.");
        }
      })
      .catch(() => { clearActiveJob(); setActiveJobId(null); })
      .finally(() => { setRunning(null); setCancelling(false); });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount only

  // ── beforeunload: warn user if a simulation is running and cancel on confirm ──
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (!activeJobIdRef.current) return;
      // Show browser confirmation dialog
      e.preventDefault();
      // Most browsers ignore custom messages, but setting returnValue triggers the dialog
      e.returnValue = "A tracker simulation is currently running. Leaving or refreshing will cancel the background worker. Are you sure?";
      // Fire cancel via sendBeacon (guaranteed delivery even during unload)
      sendBeaconCancel(activeJobIdRef.current);
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  // ── visibilitychange: re-sync progress when user returns to this browser tab ──
  useEffect(() => {
    const handler = () => {
      if (document.visibilityState !== "visible") return;
      // If we have a running job, kick an immediate poll to refresh UI
      const jid = activeJobIdRef.current;
      if (!jid) return;
      import("@/lib/api").then(({ getJob }) => {
        getJob(jid).then(setRunning).catch(() => {});
      });
    };
    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, []);

  const applyPreset = (key: string) => {
    setPreset(key);
    const p = presets[key];
    if (p) {
      setScenario({ ...SCENARIO_DEFAULTS, ...p.scenario });
      if (p.detection) {
        setDetection((d) => ({ ...d, ...p.detection }));
      }
    }
  };

  const handleCancel = useCallback(async () => {
    const jid = activeJobIdRef.current;
    if (!jid || cancelling) return;
    setCancelling(true);
    try {
      await cancelJob(jid);
    } catch {
      // Cancellation errors are non-fatal; polling will detect the state change
    }
  }, [cancelling]);

  const run = async () => {
    setErr(null);
    setResult(null);
    setActiveWorker(null);
    setActiveJobId(null);
    setCancelling(false);
    const chosen = catalog.trackers.filter((t) => selected[t.id]);
    if (!chosen.length) return setErr("Select at least one tracker to simulate.");
    const unavailable = chosen.filter((t) => !t.available);
    if (unavailable.length) {
      return setErr(
        `"${unavailable.map((t) => t.name).join(", ")}" is not available in this build — choose an active tracker.`
      );
    }

    try {
      const { job_id, worker } = await startSimulation({
        scenario,
        detection,
        trackers: chosen.map((t) => ({ tracker_id: t.id, params: selected[t.id] || {} })),
      });
      // Persist the active job to browser storage before polling begins so that
      // switching tabs or background-ing the page never loses the job reference.
      saveActiveJob({
        jobId: job_id,
        worker: worker || null,
        mode: isProduction ? "production" : "standard",
        startedAt: Date.now(),
        scenario,
        trackerIds: chosen.map((t) => t.id),
      });
      setActiveJobId(job_id);
      if (worker) setActiveWorker(worker);

      const job = await pollUntilDone(job_id, setRunning);
      clearActiveJob();
      setActiveJobId(null);
      if (job.status === "cancelled") {
        setErr("Simulation was cancelled.");
      } else if (job.status === "error") {
        throw new Error(job.error || "Simulation job encountered an error.");
      } else {
        const res = job.result as SimulationResult;
        setResult(res);
        setHistory((h) => [res, ...h].slice(0, 4));
      }
    } catch (e) {
      clearActiveJob();
      setActiveJobId(null);
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("429") || msg.toLowerCase().includes("too many candidates")) {
        setErr(
          "⚠️ Too many candidates, please wait. It is running on free version.\n\n" +
          "curl -fsSL https://raw.githubusercontent.com/nomi181472/trackers/main/docker-compose.yml -o docker-compose.yml && docker compose pull && docker compose up -d\n\n" +
          "you can also run in your local"
        );
      } else {
        setErr(msg);
      }
    }
    setRunning(null);
    setCancelling(false);
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
      {/* LEFT COLUMN: Controls & Configuration */}
      <div>
        {/* Production Mode Informational Banner */}
        {isProduction ? (
          <div className="prod-banner">
            <h4>
              <span>🚀 Production Benchmarking Mode</span>
            </h4>
            <p>
              Profile trackers against realistic deployment environments and verify Service Level Agreements (SLAs).
              Monitors FPS throughput, latency budgets, and false-positive resilience.
            </p>
            <div className="sla-grid">
              <div className="sla-card">
                <div className="sla-label">Target Throughput</div>
                <div className="sla-target">≥ {Number(scenario.fps || 15)} FPS</div>
              </div>
              <div className="sla-card">
                <div className="sla-label">Latency Ceiling</div>
                <div className="sla-target">≤ 35 ms</div>
              </div>
              <div className="sla-card">
                <div className="sla-label">Tracking Accuracy</div>
                <div className="sla-target">MOTA ≥ 0.70</div>
              </div>
            </div>
          </div>
        ) : null}

        {/* 1. Presets Section */}
        <div className="panel">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
            <h3 style={{ margin: 0 }}>
              {isProduction ? "1. Production Environments" : "1. Failure Presets"}
            </h3>
            <span style={{ fontSize: "11px", color: "var(--muted)" }}>
              {Object.keys(presets).length} profiles
            </span>
          </div>

          <div className="presets">
            {Object.entries(presets).map(([k, v]) => (
              <div
                key={k}
                className={`preset ${preset === k ? "active" : ""}`}
                onClick={() => applyPreset(k)}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "3px" }}>
                  <b>{v.label}</b>
                  {v.badge ? (
                    <span
                      style={{
                        fontSize: "9.5px",
                        padding: "1px 5px",
                        borderRadius: "4px",
                        background: "rgba(255,255,255,0.08)",
                        color: "var(--muted)",
                        fontWeight: 600,
                      }}
                    >
                      {v.badge}
                    </span>
                  ) : null}
                </div>
                <div>{v.desc}</div>
              </div>
            ))}
          </div>

          {/* Scenario Knobs */}
          <div style={{ borderTop: "1px solid var(--line)", paddingTop: "12px", marginTop: "4px" }}>
            <SelectRow
              label="Simulated Object"
              value={String(scenario.object_type || "person")}
              options={[
                { value: "person", label: "🏃 Person (Pedestrian / Skeleton)" },
                { value: "car", label: "🚗 Vehicle (Automotive)" },
                { value: "ball", label: "⚽ Circle (Geometric Baseline)" },
              ]}
              hint="Choose target entity geometry and motion physics."
              onChange={(v) => setScenario((s) => ({ ...s, object_type: v }))}
            />

            <SliderRow
              label="Scene Seed"
              min={0}
              max={99}
              step={1}
              value={Number(scenario.seed)}
              hint="Changes pseudo-random initial placement and trajectories."
              onChange={(v) => setScenario((s) => ({ ...s, seed: v }))}
            />
            <SliderRow
              label="Object Count"
              min={1}
              max={7}
              step={1}
              value={Number(scenario.num_objects)}
              hint="Total concurrent targets to track in the scene."
              onChange={(v) => setScenario((s) => ({ ...s, num_objects: v }))}
            />
            <SliderRow
              label="Duration"
              min={2}
              max={15}
              step={1}
              value={Number(scenario.duration_seconds)}
              unit="s"
              hint="Length of simulated scenario."
              onChange={(v) => setScenario((s) => ({ ...s, duration_seconds: v }))}
            />
            <SliderRow
              label="Frame Rate"
              min={10}
              max={30}
              step={1}
              value={Number(scenario.fps)}
              unit="fps"
              hint="Video sampling rate. Lower FPS stresses motion extrapolation."
              onChange={(v) => setScenario((s) => ({ ...s, fps: v }))}
            />

            <BoolRow
              label="Crossing Objects"
              text="Targets cross paths at the center, creating ambiguous bounding box overlaps."
              value={Boolean(scenario.crossing)}
              onChange={(v) => setScenario((s) => ({ ...s, crossing: v }))}
            />
            <BoolRow
              label="Occlusion Barrier"
              text="Targets pass behind a physical obstacle where detections vanish."
              value={Boolean(scenario.occlusion)}
              onChange={(v) => setScenario((s) => ({ ...s, occlusion: v }))}
            />
            {scenario.occlusion ? (
              <SliderRow
                label="Barrier Width"
                min={30}
                max={140}
                step={2}
                value={Number(scenario.occluder_width)}
                unit="px"
                hint="Wider barriers keep objects hidden longer, challenging track memory."
                onChange={(v) => setScenario((s) => ({ ...s, occluder_width: v }))}
              />
            ) : null}

            <BoolRow
              label="Camera Shake"
              text="Simulates handheld camera jitter or drone vibration (GMC challenge)."
              value={Boolean(scenario.camera_shake)}
              onChange={(v) => setScenario((s) => ({ ...s, camera_shake: v }))}
            />
            {scenario.camera_shake ? (
              <SliderRow
                label="Shake Amplitude"
                min={2}
                max={30}
                step={1}
                value={Number(scenario.shake_px)}
                unit="px"
                onChange={(v) => setScenario((s) => ({ ...s, shake_px: v }))}
              />
            ) : null}

            <BoolRow
              label="Motion Blur"
              text="Simulates fast camera panning or exposure lag, degrading detector confidence."
              value={Boolean(scenario.blur)}
              onChange={(v) => setScenario((s) => ({ ...s, blur: v }))}
            />
            {scenario.blur ? (
              <SliderRow
                label="Blur Kernel Sigma"
                min={1}
                max={7}
                step={0.5}
                value={Number(scenario.blur_sigma)}
                onChange={(v) => setScenario((s) => ({ ...s, blur_sigma: v }))}
              />
            ) : null}

            <BoolRow
              label="Identical Appearance"
              text="All objects share the exact same visual cues (tests motion-only tracking)."
              value={Boolean(scenario.similar_colors)}
              onChange={(v) => setScenario((s) => ({ ...s, similar_colors: v }))}
            />

            <div style={{ display: "flex", gap: "8px", marginTop: "10px" }}>
              <button
                type="button"
                className="btn ghost"
                style={{ width: "100%", fontSize: "12px", padding: "8px" }}
                onClick={previewIt}
                disabled={running !== null}
              >
                👁️ Preview Ground Truth Scene
              </button>
            </div>

            {preview ? (
              <div style={{ marginTop: 12 }}>
                <video
                  src={mediaUrl(preview.preview_data_url || preview.preview_url)}
                  controls
                  muted
                  loop
                  style={{ width: "100%", borderRadius: 8, border: "1px solid var(--line)" }}
                />
                <small style={{ display: "block", marginTop: "4px" }}>
                  {preview.frames} frames · {preview.width}×{preview.height} @ {preview.fps} FPS · Ground-truth verified
                </small>
              </div>
            ) : null}
          </div>
        </div>

        {/* 2. Detector Settings */}
        <div className="panel" style={{ marginTop: 16 }}>
          <h3>2. Upstream Detector Properties</h3>
          <small style={{ display: "block", marginBottom: 10, color: "var(--muted)" }}>
            The detector generates raw bounding box inputs for the tracker algorithms.
          </small>

          {catalog.detector_params.map((p) => (
            <ParamControl
              key={p.key}
              p={p}
              value={detection[p.key] ?? p.default}
              onChange={(v) => setDetection((d) => ({ ...d, [p.key]: v }))}
            />
          ))}

          <div className="collapse-head" onClick={() => setShowAdvancedDet((s) => !s)}>
            <span style={{ transform: `rotate(${showAdvancedDet ? 90 : 0}deg)`, display: "inline-block" }}>▸</span>
            {showAdvancedDet ? "Hide" : "Show"} Synthetic Detector Imperfections ({catalog.scenario_detection_params.length} knobs)
          </div>

          {showAdvancedDet &&
            catalog.scenario_detection_params.map((p) => (
              <ParamControl
                key={p.key}
                p={p}
                value={detection[p.key] ?? p.default}
                onChange={(v) => setDetection((d) => ({ ...d, [p.key]: v }))}
              />
            ))}
        </div>

        {/* 3. Tracker Selection */}
        <div className="panel" style={{ marginTop: 16 }}>
          <h3>3. Trackers Under Test</h3>
          <small style={{ display: "block", marginBottom: 10, color: "var(--muted)" }}>
            Select one or multiple algorithms to run head-to-head under identical conditions.
          </small>
          <TrackerPicker catalog={catalog} selected={selected} setSelected={setSelected} />
        </div>

        {err ? (
          <div className="err" style={{ marginTop: 14, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {err}
          </div>
        ) : null}

        {/* Execution Button */}
        <button
          className="btn"
          style={{ width: "100%", marginTop: 16, padding: "13px 20px", fontSize: "14px" }}
          onClick={run}
          disabled={running !== null}
        >
          {running ? "Simulating Trackers…" : isProduction ? "▶ Run Production Benchmark" : "▶ Run Diagnostic Simulation"}
        </button>

        {running ? (
          <div style={{ marginTop: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px", flexWrap: "wrap", gap: "6px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                {activeWorker ? (
                  <div
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "6px",
                      padding: "4px 8px",
                      borderRadius: "6px",
                      fontSize: "12px",
                      fontWeight: 600,
                      color: "#93c5fd",
                      background: "rgba(59, 130, 246, 0.15)",
                      border: "1px solid rgba(59, 130, 246, 0.3)",
                    }}
                  >
                    <span>⚡</span>
                    <span>{activeWorker}</span>
                  </div>
                ) : null}
                {activeJobId ? (
                  <span
                    className="mono"
                    style={{
                      fontSize: "11px",
                      padding: "3px 7px",
                      borderRadius: "4px",
                      background: "var(--bg-3)",
                      border: "1px solid var(--line)",
                      color: "var(--muted)",
                    }}
                    title={`Your active simulation job ID: ${activeJobId}`}
                  >
                    Job: {activeJobId.slice(0, 12)}…
                  </span>
                ) : null}
              </div>

              <button
                type="button"
                className="btn-danger"
                style={{ padding: "4px 10px", fontSize: "11.5px", borderRadius: "5px" }}
                onClick={handleCancel}
                disabled={cancelling}
              >
                {cancelling ? "Cancelling…" : "⏹️ Cancel"}
              </button>
            </div>

            <div className="progress">
              <div style={{ width: `${Math.round((running.progress || 0) * 100)}%` }} />
            </div>
            <small style={{ display: "block", marginTop: 6, color: "var(--muted)" }}>
              <span className="spin" /> <span className="mono">{running.message}</span> ({Math.round((running.progress || 0) * 100)}%)
            </small>
          </div>
        ) : null}
      </div>

      {/* RIGHT COLUMN: Results & Benchmarks */}
      <div>
        {result ? (
          <div>
            {isProduction ? (
              <ProductionSLAOverview result={result} targetFps={Number(scenario.fps || 15)} />
            ) : null}
            <ResultsView result={result} />
          </div>
        ) : null}

        {history.length > 1 ? <CompareTable history={history} /> : null}

        {!result && !history.length ? (
          <div className="panel" style={{ textAlign: "center", padding: "80px 24px", color: "var(--muted)" }}>
            <div style={{ fontSize: 44, marginBottom: 12 }}>⚡</div>
            <h3 style={{ color: "var(--text)", fontSize: "16px", marginBottom: "6px" }}>
              {isProduction ? "Production Tracker Benchmark Ready" : "Tracker Failure Simulator Ready"}
            </h3>
            <p style={{ maxWidth: "480px", margin: "0 auto", fontSize: "13px", lineHeight: "1.6" }}>
              {isProduction
                ? "Select a production environment (CCTV, Traffic, Retail, Edge Robotics), configure targets, and hit Run Production Benchmark to profile throughput, SLA compliance, and failure telemetry."
                : "Choose a failure stress scenario (occlusion, crossings, motion blur), pick algorithms to benchmark, and hit Run Simulation to diagnose exactly where trackers fail."}
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Production SLA Overview Component                                  */
/* ------------------------------------------------------------------ */

function ProductionSLAOverview({ result, targetFps }: { result: SimulationResult; targetFps: number }) {
  const maxLatencyThreshold = 35; // ms
  const minMotaThreshold = 0.65;

  return (
    <div className="panel" style={{ marginBottom: 16, borderColor: "rgba(139, 92, 246, 0.4)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <h3 style={{ margin: 0, color: "#d8b4fe" }}>
          🚀 Production SLA Verification Report
        </h3>
        <span className="mode-pill mode-pill-prod">Production Qualified</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 }}>
        {result.results.map((r) => {
          const m = r.metrics;
          const fps = m.fps ?? 0;
          const latency = m.avg_time_ms ?? 0;
          const mota = m.mota ?? 0;

          const fpsPass = fps >= targetFps;
          const latencyPass = latency <= maxLatencyThreshold;
          const motaPass = mota >= minMotaThreshold;
          const allPass = fpsPass && latencyPass && motaPass;

          return (
            <div key={r.tracker_id} className="prod-score-card">
              <div>
                <div style={{ fontWeight: 700, fontSize: "13.5px", color: "var(--text)" }}>{r.name}</div>
                <div style={{ fontSize: "11px", color: "var(--muted)", marginTop: "2px" }}>
                  {fps.toFixed(1)} FPS · {latency.toFixed(1)} ms/frame · MOTA {(mota * 100).toFixed(0)}%
                </div>
              </div>
              <span className={`sla-badge ${allPass ? "pass" : fpsPass || motaPass ? "warn" : "fail"}`}>
                {allPass ? "✓ SLA Pass" : fpsPass ? "⚠️ Degraded" : "✗ SLA Fail"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}