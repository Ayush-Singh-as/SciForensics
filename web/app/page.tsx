"use client";

/**
 * The demo.
 *
 * Layout is a working surface plus an evidence rail, not a landing page: the
 * images take the space and the numbers sit beside them where they can be read
 * against each other. There is no hero, no feature grid and no call to action,
 * because the audience is someone who wants to check an image, not be sold to.
 *
 * State lives here rather than in a store. There is exactly one analysis in
 * flight at a time and one result on screen; a reducer or a context provider for
 * that would be scaffolding for a requirement this page does not have.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { Viewer } from "@/components/Viewer";
import { Dropzone } from "@/components/Dropzone";
import {
  Contributions,
  GeometryCard,
  MatchFunnel,
  ThresholdControls,
  Verdict,
} from "@/components/Evidence";
import {
  type Bands,
  type CopyMoveResult,
  type Example,
  type JobPayload,
  type ScanResult,
  type ServerConfig,
  assetUrl,
  bandFor,
  cmfd,
  compare,
  getConfig,
  getExamples,
  reportUrl,
  exampleFile,
  rescore,
} from "@/lib/api";

type Mode = "compare" | "cmfd";
type Job =
  | { kind: "compare"; payload: JobPayload<ScanResult> }
  | { kind: "cmfd"; payload: JobPayload<CopyMoveResult> };

export default function Page() {
  const [mode, setMode] = useState<Mode>("compare");
  const [left, setLeft] = useState<File | null>(null);
  const [right, setRight] = useState<File | null>(null);
  const [single, setSingle] = useState<File | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);

  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [bands, setBands] = useState<Bands | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [stage, setStage] = useState("queued");
  const [threshold, setThreshold] = useState<number | null>(null);

  // Server config is the source of truth for the band defaults, so the sliders
  // start where the backend actually sits rather than at numbers duplicated in
  // the frontend (bug 10's failure mode, in a new place).
  useEffect(() => {
    getConfig()
      .then((value) => {
        setConfig(value);
        setBands(value.bands);
        setThreshold(value.global_match.distance_threshold);
      })
      .catch(() => setError("Cannot reach the API. Is `sciforensics serve` running?"));
    getExamples()
      .then((value) => setExamples(value.examples))
      .catch(() => undefined);
  }, []);

  const run = useCallback(async () => {
    setBusy(true);
    setError(null);
    setJob(null);
    setStage("queued");
    try {
      if (mode === "compare") {
        if (!left || !right) return;
        setJob({ kind: "compare", payload: await compare(left, right, setStage) });
      } else {
        if (!single) return;
        setJob({ kind: "cmfd", payload: await cmfd(single, setStage) });
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }, [mode, left, right, single]);

  const ready = mode === "compare" ? Boolean(left && right) : Boolean(single);

  const preview = useMemo(() => {
    if (!job) return null;
    if (job.kind === "compare" && threshold !== null && config && !job.payload.result.calibrated) {
      return rescore(job.payload.result, threshold, config.global_match.similarity_temperature);
    }
    return { confidence: job.payload.result.confidence, contributions: job.payload.result.contributions };
  }, [job, threshold, config]);
  const verdict = preview && bands ? bandFor(preview.confidence, bands) : null;
  const previewChanged = Boolean(job?.kind === "compare" && threshold !== job.payload.result.global_evidence.distance_threshold);

  async function useExample(example: Example) {
    setBusy(true); setError(null); setJob(null); setStage("loading example");
    setMode(example.mode);
    try {
      const a = await exampleFile(example.left, `${example.id}-left`);
      if (example.mode === "compare" && example.right) {
        const b = await exampleFile(example.right, `${example.id}-right`);
        setLeft(a); setRight(b);
        setJob({ kind: "compare", payload: await compare(a, b, setStage) });
      } else {
        setSingle(a);
        setJob({ kind: "cmfd", payload: await cmfd(a, setStage) });
      }
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Example failed"); }
    finally { setBusy(false); }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="mark">SciForensics</span>
          <span className="rev">v0.2-dev</span>
        </div>
        <nav>
          {(["compare", "cmfd"] as Mode[]).map((value) => (
            <button
              key={value}
              aria-current={mode === value}
              disabled={busy}
              onClick={() => {
                setMode(value);
                setJob(null);
                setError(null);
              }}
            >
              {value === "compare" ? "Compare pair" : "Copy-move"}
            </button>
          ))}
        </nav>
      </header>

      <div className="main">
        {/* --------------------------------------------------------- stage */}
        <div className="stage">
          {error && (
            <p className="notice alert" style={{ marginBottom: 14 }}>
              {error}
            </p>
          )}

          {!job && !busy && (
            <div className="empty">
              <strong style={{ color: "var(--text-dim)", fontSize: 14 }}>
                No analysis loaded
              </strong>
              <span style={{ fontSize: 12.5, maxWidth: 420 }}>
                {mode === "compare"
                  ? "Load two figures to test for reuse under rotation, scaling, reflection or partial overlap."
                  : "Load one figure to search it for duplicated regions."}
              </span>
            </div>
          )}

          {busy && (
            <div className="empty">
              <span className="spinner" />
              <span style={{ fontSize: 12.5 }}>
                {stage === "report" ? "Rendering report…" : `Analysing: ${stage.replace(/_/g, " ")}…`}
              </span>
            </div>
          )}

          {job && (
            <>
              <Viewer key={job.payload.job_id} jobId={job.payload.job_id} overlays={job.payload.overlays}
                data={job.payload.viewer} single={job.kind === "cmfd"} />
              {job.payload.warnings.length > 0 && (
                <div className="panel">
                  <header>
                    <h2>Warnings</h2>
                  </header>
                  <div className="body">
                    {job.payload.warnings.map((warning) => (
                      <p className="notice" key={warning} style={{ marginTop: 0 }}>
                        {warning}
                      </p>
                    ))}
                  </div>
                </div>
              )}

              {job.kind === "cmfd" && job.payload.result.evidence.regions.length > 0 && (
                <div className="panel">
                  <header>
                    <h2>
                      Cloned regions ({job.payload.result.evidence.regions.length} of{" "}
                      {job.payload.result.evidence.cluster_count} candidate clusters)
                    </h2>
                  </header>
                  <div className="body">
                    <table className="kv">
                      <tbody>
                        {job.payload.result.evidence.regions.map((region, index) => (
                          <tr key={index}>
                            <td className="wrap">
                              Region {index + 1}
                              {region.flip ? " · mirrored" : ""}
                            </td>
                            <td>
                              {region.rotation_deg >= 0 ? "+" : ""}
                              {region.rotation_deg.toFixed(1)}° · ×{region.scale.toFixed(2)} ·{" "}
                              {region.inlier_count} inliers
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ---------------------------------------------------------- rail */}
        <aside className="rail">
          <div className="panel">
            <header>
              <h3>Input</h3>
            </header>
            <div className="body">
              {mode === "compare" ? (
                <div style={{ display: "grid", gap: 9 }}>
                  <Dropzone label="Figure A" file={left} onFile={setLeft} />
                  <Dropzone label="Figure B" file={right} onFile={setRight} />
                </div>
              ) : (
                <Dropzone label="Figure" file={single} onFile={setSingle} />
              )}

              <button
                className="btn primary"
                style={{ width: "100%", marginTop: 11 }}
                disabled={!ready || busy}
                onClick={run}
              >
                {busy ? (
                  <>
                    <span className="spinner" /> Analysing
                  </>
                ) : (
                  "Run analysis"
                )}
              </button>

              {job && (
                <a
                  className="btn"
                  href={reportUrl(job.payload.job_id)}
                  target="_blank"
                  rel="noreferrer"
                  style={{ width: "100%", marginTop: 7, textDecoration: "none" }}
                >
                  Download {job.payload.report_format === "pdf" ? "PDF" : "HTML"} report
                </a>
              )}
            </div>
          </div>

          {job && verdict && bands && config && (
            <>
              <div className="panel">
                <header>
                  <h3>Finding</h3>
                </header>
                <Verdict
                  verdict={verdict}
                  confidence={preview?.confidence ?? job.payload.result.confidence}
                  calibrated={job.payload.result.calibrated}
                  notes={previewChanged ? ["Threshold preview: evidence is unchanged; the downloaded report records the original server settings."] : job.payload.result.notes}
                />
              </div>

              <div className="panel">
                <header>
                  <h3>Thresholds</h3>
                </header>
                <div className="body">
                  {job.kind === "compare" && threshold !== null && !job.payload.result.calibrated && <div style={{ marginBottom: 16 }}>
                    <label htmlFor="match-threshold">Embedding distance boundary <output className="num">{threshold.toFixed(2)}</output></label>
                    <input id="match-threshold" className="slider" type="range" min={0.1} max={5} step={0.05} value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
                    <p className="caveat">Re-scores the cached embedding distance and contribution bars. Local geometry stays fixed.</p>
                  </div>}
                  <ThresholdControls
                    bands={bands}
                    onChange={setBands}
                    onReset={() => { setBands(config.bands); setThreshold(config.global_match.distance_threshold); }}
                  />
                </div>
              </div>

              {job.kind === "compare" && job.payload.result.matches && (
                <div className="panel">
                  <header>
                    <h3>Matching</h3>
                  </header>
                  <div className="body">
                    <MatchFunnel
                      matches={job.payload.result.matches}
                      keypoints={job.payload.result.keypoints}
                    />
                  </div>
                </div>
              )}

              {job.kind === "compare" && job.payload.result.geometry && (
                <div className="panel">
                  <header>
                    <h3>Geometry</h3>
                  </header>
                  <div className="body">
                    <GeometryCard geometry={job.payload.result.geometry} />
                  </div>
                </div>
              )}

              {job.payload.result.contributions.length > 0 && (
                <div className="panel">
                  <header>
                    <h3>Score drivers</h3>
                  </header>
                  <div className="body">
                    <Contributions contributions={preview?.contributions ?? job.payload.result.contributions} />
                  </div>
                </div>
              )}
            </>
          )}

          {examples.length > 0 && (
            <div className="panel">
              <header>
                <h3>Examples</h3>
              </header>
              <div className="body">
                <div className="examples">
                  {examples.map((example) => (
                    <div className="example" key={example.id}>
                      <div className="title">
                        {example.label}
                        <span className={`badge ${example.kind === "negative" ? "ok" : "alert"}`}>
                          {example.kind === "negative" ? "control" : "positive"}
                        </span>
                      </div>
                      <div className="note">{example.note}</div>
                      <button className="btn" disabled={busy} onClick={() => useExample(example)} aria-label={`Run ${example.label} example`}>Run example</button>
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: 11, color: "var(--text-faint)", margin: "10px 0 0" }}>
                  Negative controls are listed deliberately: a gallery of only true positives is how
                  a false-positive rate goes unmeasured.
                </p>
              </div>
            </div>
          )}
        </aside>
      </div>

      <footer className="footnote">
        SciForensics performs <strong>automated screening</strong>. A finding is evidence that two
        images share content under a transform — not a determination of intent or misconduct.
        Duplicate imagery has legitimate explanations, including shared controls and properly
        attributed republication. Any finding requires confirmation by a qualified reviewer with
        access to the original data, and absence of a finding is not proof of originality.
      </footer>
    </div>
  );
}
