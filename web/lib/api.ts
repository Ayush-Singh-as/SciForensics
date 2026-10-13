/**
 * Typed client for the SciForensics API, plus the client-side re-scoring that
 * makes the threshold sliders honest.
 *
 * The types here mirror `sciforensics.types` deliberately field-for-field. They
 * are hand-written rather than generated because the generated shape would hide
 * the one property that matters most in this UI: `calibrated`. Everything that
 * renders a score has to consult it, so it is non-optional on the result.
 */

const BASE = (process.env.SCIFORENSICS_API ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export type Verdict = "likely_manipulated" | "suspicious" | "inconclusive" | "clean";

export const VERDICT_LABEL: Record<Verdict, string> = {
  likely_manipulated: "Likely reused or manipulated",
  suspicious: "Suspicious — warrants expert review",
  inconclusive: "Inconclusive",
  clean: "No evidence of reuse found",
};

export interface Bands {
  likely_manipulated: number;
  suspicious: number;
  inconclusive: number;
}

export interface AffineDecomposition {
  model: "full" | "similarity";
  rotation_deg: number;
  scale_x: number;
  scale_y: number;
  shear_deg: number;
  translation: [number, number];
  determinant: number;
  flip: boolean;
  anisotropic: boolean;
  sheared: boolean;
}

export interface GeometryEvidence {
  verified: boolean;
  rejection_reason: string;
  method: string;
  inlier_count: number;
  inlier_ratio: number;
  distinct_inliers: number;
  inlier_spread: number;
  reproj_rms: number | null;
  condition_number: number | null;
  transform: AffineDecomposition | null;
  matched_area_fraction: number;
}

export interface KeypointEvidence {
  detector: string;
  detected_left: number;
  detected_right: number;
  /** Counts after the difference-of-Gaussians ROI filter -- what matching saw. */
  kept_left: number;
  kept_right: number;
  /** ROI restriction was dropped to keep a side above the per-side floor. */
  roi_abandoned_left: boolean;
  roi_abandoned_right: boolean;
  enhancement_scale: number;
}

export interface MatchEvidence {
  matcher: string;
  raw: number;
  ratio_passed: number;
  good: number;
  distinct_left: number;
  distinct_right: number;
  mutual_nn: boolean;
  ratio: number;
}

export interface Contribution {
  name: string;
  value: number;
  weight: number;
  contribution: number;
  description: string;
}

export interface ImageMeta {
  path: string;
  sha256: string;
  width: number;
  height: number;
  analysed_at: [number, number] | null;
}

export interface ScanResult {
  left: ImageMeta;
  right: ImageMeta;
  global_evidence: {
    distance: number;
    similarity: number;
    distance_threshold: number;
    local_trigger_distance: number;
    is_match: boolean;
    triggered_local: boolean;
    embedding_dim: number;
  };
  keypoints: KeypointEvidence | null;
  matches: MatchEvidence | null;
  geometry: GeometryEvidence | null;
  verdict: Verdict;
  confidence: number;
  calibrated: boolean;
  contributions: Contribution[];
  notes: string[];
  warnings: string[];
  timings: Record<string, number>;
}

export interface CopyMoveRegion {
  source_box: [number, number, number, number];
  target_box: [number, number, number, number];
  offset: [number, number];
  rotation_deg: number;
  scale: number;
  flip: boolean;
  inlier_count: number;
  area_px: number;
}

export interface CopyMoveResult {
  image: ImageMeta;
  evidence: {
    detected: boolean;
    cluster_count: number;
    regions: CopyMoveRegion[];
    self_matches: number;
    keypoints: number;
  };
  verdict: Verdict;
  confidence: number;
  calibrated: boolean;
  contributions: Contribution[];
  notes: string[];
  warnings: string[];
  timings: Record<string, number>;
}

/**
 * One rendered overlay, as the API describes it.
 *
 * `caption` and `trust` are computed server-side in `report/overlays.py` and
 * reused by both the PDF template and this client, so the demo cannot label an
 * overlay differently from the report of the same analysis.
 */
export interface OverlayRef {
  name: string;
  /** Filename for the asset route. */
  file: string;
  caption: string;
  trust: "raw" | "indicative" | "verified" | "rejected";
}

export interface JobPayload<T> {
  job_id: string;
  result: T;
  overlays: OverlayRef[];
  warnings: string[];
  report_format: "pdf" | "html";
  viewer: ViewerData;
}

export interface ViewerData {
  aspects: Record<string, number>;
  matches: { left: [number, number]; right: [number, number]; inlier: boolean }[];
  hulls: { left?: [number, number][]; right?: [number, number][] };
}

export interface ServerConfig {
  global_match: {
    distance_threshold: number;
    local_trigger_distance: number;
    similarity_temperature: number;
  };
  geometry: { min_inliers: number; min_inlier_ratio: number };
  bands: Bands;
  calibrated: boolean;
}

export interface Example {
  id: string;
  label: string;
  note: string;
  kind: "positive" | "negative";
  left: string;
  right: string | null;
  mode: "compare" | "cmfd";
}

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    // The API returns a uniform `{error, status}` shape; fall back to the status
    // text only if the body is not JSON (a proxy error page, say).
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.error ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export async function getConfig(): Promise<ServerConfig> {
  return unwrap(await fetch(`${BASE}/v1/config`, { cache: "no-store" }));
}

export async function getExamples(): Promise<{ examples: Example[] }> {
  return unwrap(await fetch(`${BASE}/v1/examples`, { cache: "no-store" }));
}

export type Progress = (stage: string) => void;

async function analysis<T>(kind: string, form: FormData, onProgress?: Progress): Promise<JobPayload<T>> {
  const started = await unwrap<{ job_id: string }>(
    await fetch(`${BASE}/v1/${kind}?background=true`, { method: "POST", body: form }),
  );
  let socket: WebSocket | undefined;
  try {
    socket = new WebSocket(`${BASE.replace(/^http/, "ws")}/v1/jobs/${started.job_id}/progress`);
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.stage) onProgress?.(message.stage);
    };
  } catch { /* Polling also works when a proxy cannot forward WebSockets. */ }
  try {
    const deadline = Date.now() + 10 * 60 * 1000;
    while (Date.now() < deadline) {
      const job = await unwrap<JobPayload<T> & { status: string; stage: string; error?: string }>(
        await fetch(`${BASE}/v1/jobs/${started.job_id}`, { cache: "no-store" }),
      );
      onProgress?.(job.stage);
      if (job.status === "failed") throw new Error(job.error ?? "Analysis failed");
      if (job.status === "complete") return { ...job, job_id: started.job_id };
      await new Promise((resolve) => setTimeout(resolve, 350));
    }
    throw new Error("Analysis timed out. The API may still be working; try again shortly.");
  } finally { socket?.close(); }
}

export async function compare(left: File, right: File, onProgress?: Progress): Promise<JobPayload<ScanResult>> {
  const form = new FormData();
  form.append("left", left);
  form.append("right", right);
  return analysis("compare", form, onProgress);
}

export async function cmfd(image: File, onProgress?: Progress): Promise<JobPayload<CopyMoveResult>> {
  const form = new FormData();
  form.append("image", image);
  return analysis("cmfd", form, onProgress);
}

export async function exampleFile(path: string, name: string): Promise<File> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) throw new Error("Could not load the example image");
  const blob = await response.blob();
  return new File([blob], `${name}.${blob.type.includes("jpeg") ? "jpg" : "png"}`, { type: blob.type });
}

/** Preview a changed embedding threshold from cached distance and contributions.
 * Local evidence stays fixed. This is a rule-score preview, never a probability.
 */
export function rescore(result: ScanResult, threshold: number, temperature: number) {
  const similarity = 1 / (1 + Math.exp((result.global_evidence.distance - threshold) / temperature));
  const contributions = result.contributions.map((item) => item.name === "embedding_similarity"
    ? { ...item, value: similarity - .5, contribution: item.weight * (similarity - .5) }
    : item);
  const logit = contributions.reduce((sum, item) => sum + item.contribution, 0);
  return { confidence: 1 / (1 + Math.exp(-logit)), contributions };
}

export function assetUrl(jobId: string, name: string): string {
  return `${BASE}/v1/jobs/${jobId}/assets/${encodeURIComponent(name)}`;
}

export function reportUrl(jobId: string): string {
  return `${BASE}/v1/jobs/${jobId}/report`;
}

/**
 * Re-band a score against user-moved thresholds.
 *
 * This is the whole point of returning structured evidence: the sliders re-band
 * a score the backend already computed, with no round trip and no re-analysis.
 * That is not merely a latency win — re-running the pipeline on every slider
 * tick would let the *evidence* move while the user believes they are only
 * moving a threshold.
 *
 * Ordering matters: bands must be tested from strongest down, since the
 * thresholds are lower bounds and a score above `likely_manipulated` is also
 * above `suspicious`.
 */
export function bandFor(confidence: number, bands: Bands): Verdict {
  if (confidence >= bands.likely_manipulated) return "likely_manipulated";
  if (confidence >= bands.suspicious) return "suspicious";
  if (confidence >= bands.inconclusive) return "inconclusive";
  return "clean";
}

export function verdictTone(verdict: Verdict): "alert" | "warn" | "ok" | "mute" {
  switch (verdict) {
    case "likely_manipulated":
      return "alert";
    case "suspicious":
      return "warn";
    case "clean":
      return "ok";
    default:
      return "mute";
  }
}

export { ApiError };
