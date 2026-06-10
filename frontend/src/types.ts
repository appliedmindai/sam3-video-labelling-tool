export interface ObjectClass {
  id: number;
  name: string;
  color: string;
}

export interface TrackedObject {
  obj_id: number;
  class_id: number;
}

export interface SessionState {
  classes: ObjectClass[];
  objects: TrackedObject[];
  bbox_padding?: Record<number, Record<number, BboxPadding>>;
  /** Monotonic counter for lost-update guard on PUT /state (#80). */
  version?: number;
}

export interface ClickPoint {
  x: number;
  y: number;
  label: number;
  frameIdx: number;
  objId: number;
}

export interface KeyframePrompt {
  type: "click" | "box";
  points?: number[][];
  labels?: number[];
  box?: [number, number, number, number];
}

export interface MaskResult {
  rle: { counts: string; size: number[] };
  bbox: [number, number, number, number];
  area: number;
  source_keyframe: number | null;
  confidence?: number;
}

export interface ConfidenceWarning {
  frame_idx: number;
  obj_id: number;
  confidence: number;
}

export interface FrameResult {
  frame_idx: number;
  masks: Record<number, MaskResult>;
  source_keyframe?: number | null;
}

export interface VideoInfo {
  width: number;
  height: number;
  duration: number;
  fps: number;
}

export interface UploadResult {
  session_id: string;
  video_name: string;
  duplicate?: boolean;
}

export interface BboxPadding {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

export interface TextInstance {
  obj_id: number;
  rle: { counts: string; size: number[] };
  bbox: [number, number, number, number];
  area: number;
  confidence: number;
}

export interface TextSegmentResult {
  frame_idx: number;
  text: string;
  instances: TextInstance[];
}

export interface SessionSummary {
  session_id: string;
  frame_count: number;
  original_name: string;
  fps: number;
  video_info: VideoInfo;
  class_count: number;
  object_count: number;
  updated_at: string;
  disk_size: number;
  live: boolean;
}

export interface PropagationStatus {
  status: string;
  start_frame?: number;
  reverse?: boolean;
  frames_processed?: number;
}

export interface ServiceStatus {
  phase: "idle" | "extracting" | "initializing" | "ready" | "error";
  session_id: string | null;
  video_name: string | null;
  progress: number;
  error: string | null;
  frame_count: number | null;
  propagation: PropagationStatus | null;
  boot_id: string;
}
