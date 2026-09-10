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
  const [overlay, setOverlay] = useState<string | null>(null);

  // Server config is the source of truth for the band defaults, so the sliders
  // start where the backend actually sits rather than at numbers duplicated in
  // the frontend (bug 10's failure mode, in a new place).
  useEffect(() => {
    getConfig()
      .then((value) => {
        setConfig(value);
        setBands(value.bands);
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
    setOverlay(null);
    try {
      if (mode === "compare") {
        if (!left || !right) return;
        setJob({ kind: "compare", payload: await compare(left, right) });
      } else {
        if (!single) return;
        setJob({ kind: "cmfd", payload: await cmfd(single) });
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }, [mode, left, right, single]);

  const ready = mode === "compare" ? Boolean(left && right) : Boolean(single);

  // The re-banded verdict. Recomputed from the cached confidence whenever a
  // slider moves — no request, so the evidence cannot shift underneath it.
  const verdict = useMemo(() => {
    if (!job || !bands) return null;
    return bandFor(job.payload.result.confidence, bands);
  }, [job, bands]);

  const overlays = job?.payload.overlays ?? [];
  const active = overlay ?? overlays.find((name) => name.startsWith("matches")) ?? overlays[0];

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
                Running embedding, keypoint and geometry stages…
              </span>
            </div>
          )}

          {job && active && (
            <>
              <div className="panel">
                <header>
                  <h2>Evidence</h2>
                  <div className="toggles" style={{ marginLeft: "auto" }}>
                    {overlays.map((name) => (
                      <button
                        key={name}
                        className="toggle"
                        aria-pressed={active === name}
                        onClick={() => setOverlay(name)}
                      >
                        {name.replace(/\.png$/, "").replace(/_/g, " ")}
                      </button>
                    ))}
                  </div>
                </header>
                <div className="body">
                  <figure className="viewer" style={{ margin: 0 }}>
                    <img src={assetUrl(job.payload.job_id, active)} alt="Forensic overlay" />
                  </figure>
                </div>
              </div>

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
                  Download report
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
                  confidence={job.payload.result.confidence}
                  calibrated={job.payload.result.calibrated}
                  notes={job.payload.result.notes}
                />
              </div>

              <div className="panel">
                <header>
                  <h3>Thresholds</h3>
                </header>
                <div className="body">
                  <ThresholdControls
                    bands={bands}
                    onChange={setBands}
                    onReset={() => setBands(config.bands)}
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
                    <Contributions contributions={job.payload.result.contributions} />
                  </div>
                </div>
              )}
            </>
          )}

          {!job && examples.length > 0 && (
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
