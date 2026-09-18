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
  engine: "ultralytics" | "custom" | "opencv";
  mode: "multi" | "single";
  tagline: string;
  description: string;
  strengths: string[];
  failure_modes: string[];
  params: HyperParam[];
  available: boolean;
}

export interface Catalog {
  trackers: TrackerMeta[];
  detector_params: HyperParam[];
  scenario_detection_params: HyperParam[];
  defaults: Record<string, Record<string, number | boolean | string>>;
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

export interface TrackerResult {
  tracker_id: string;
  name: string;
  tagline: string;
  mode: string;
  metrics: Metrics;
  events: EventItem[];
  report: Report;
  video_url: string;
  codec: string;
  thumbnails?: { frame: number; type: string; severity: string; url: string }[];
  error?: string;
}

export interface RealResult {
  results: TrackerResult[];
}

export interface SimulationResult {
  scenario: ScenarioMeta;
  detection: Record<string, number | boolean | string>;
  trackers: string[];
  results: TrackerResult[];
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