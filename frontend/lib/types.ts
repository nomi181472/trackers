export type ParamType = "number" | "int" | "bool" | "select";

export interface HyperParam {
  key: string;
  label: string;
  type: ParamType;
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  tooltip?: string;
  hint?: string;
  options?: Record<string, string>;
}

export interface TrackerMeta {
  id: string;
  name: string;
  engine: string;
  mode: "multi" | "single";
  tagline: string;
  description: string;
  strengths: string[];
  failure_modes: string[];
  params: HyperParam[];
  available: boolean;
}

export interface WorkerConcurrencyStats {
  max_workers: number;
  max_queue_size: number;
  running_jobs: number;
  queued_jobs: number;
}

export interface Catalog {
  trackers: TrackerMeta[];
  detector_params: HyperParam[];
  scenario_detection_params: HyperParam[];
  defaults: Record<string, Record<string, number | boolean | string>>;
  worker_concurrency?: WorkerConcurrencyStats;
}

export interface ScenarioMeta {
  fps: number;
  width: number;
  height: number;
  frames: number;
  num_objects: number;
  crossing: boolean;
  occlusion: boolean;
  blur: boolean;
  camera_shake: boolean;
  similar_colors: boolean;
  occluder_box?: number[] | null;
  seed: number;
  preview_url?: string;
  preview_data_url?: string;
  preview_codec?: string;
}

export interface EventItem {
  frame: number;
  type: string;
  text: string;
  severity: "critical" | "warning" | "informational";
  gt_ids: number[];
  track_ids: number[];
  fix: string | null;
  blame: "tracker" | "detector";
}

export interface Metrics {
  mota?: number;
  motp?: number;
  idf1?: number;
  idp?: number;
  idr?: number;
  idsw?: number;
  fp?: number;
  fn?: number;
  mt?: number;
  ml?: number;
  pt?: number;
  accuracy?: number;
  correct_frames?: number;
  total_visible?: number;
  lost_frames?: number;
  longest_correct_run?: number;
  gt_total?: number;
  gt_objects?: number;
  frames?: number;
  avg_time_ms?: number;
  fps?: number;
  latencies?: number[];
  motmetrics?: {
    mota: number;
    motp: number;
    idf1: number;
    idp: number;
    idr: number;
    idsw: number;
    fp: number;
    fn: number;
    mt: number;
    ml: number;
    pt: number;
  };
}

export interface ReportSection {
  name: string;
  text?: string;
  bullets?: string[];
}

export interface Report {
  tracker_id: string;
  name: string;
  tagline: string;
  grade: string;
  verdict: string;
  sections: ReportSection[];
  failed: boolean;
}

export interface FrameSummaryItem {
  frame: number;
  tracks: number;
  gt: number;
  matched: number;
  fp: number;
  fn: number;
  idsw: number;
  events?: string[];
}

export interface TrackerResult {
  tracker_id: string;
  name: string;
  tagline: string;
  mode: string;
  metrics: Metrics;
  events: EventItem[];
  frame_summary?: FrameSummaryItem[];
  report: Report;
  video_url: string;
  video_data_url?: string;
  codec: string;
  thumbnails?: { frame: number; type: string; severity: string; url: string; data_url?: string }[];
  error?: string;
}

export interface SimulationResult {
  scenario: ScenarioMeta;
  detection: Record<string, number | boolean | string>;
  trackers: string[];
  results: TrackerResult[];
  chart_url?: string;
}

export interface JobStatus {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message?: string;
  error?: string;
  result?: SimulationResult;
}

export interface ParamValues {
  [key: string]: number | boolean | string;
}

export interface ParamGroup {
  [key: string]: ParamValues;
}

export interface LogFileInfo {
  filename: string;
  date: string;
  size_bytes: number;
  modified_at: number;
}

export interface LogPage {
  file: string | null;
  lines: string[];
  next_cursor: number | null;
  total_lines: number;
  start_line: number;
  end_line: number;
}