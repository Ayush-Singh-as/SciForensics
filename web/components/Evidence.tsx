"use client";

/**
 * Evidence presentation components.
 *
 * Each of these exists to make one of the fixed bugs legible rather than to
 * decorate a number:
 *
 *   - `Verdict`      — carries the uncalibrated caveat inline, never hidden.
 *   - `MatchFunnel`  — raw → ratio → mutual-NN with distinct counts (bug 4).
 *   - `TransformCard`— rotation/scale/**flip** (bugs 2 and 3).
 *   - `Contributions`— which signal drove the score, signed.
 *
 * All presentational: they compute no forensic quantity, so every figure shown
 * traces back to a field on the result the backend returned.
 */

import {
  type Bands,
  type Contribution,
  type GeometryEvidence,
  type MatchEvidence,
  VERDICT_LABEL,
  type Verdict as VerdictName,
  verdictTone,
} from "@/lib/api";

export function Verdict({
  verdict,
  confidence,
  calibrated,
  notes,
}: {
  verdict: VerdictName;
  confidence: number;
  calibrated: boolean;
  notes: string[];
}) {
  const tone = verdictTone(verdict);
  const colour = `var(--${tone === "mute" ? "text-faint" : tone})`;

  return (
    <div className="verdict" data-band={verdict}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="label">{VERDICT_LABEL[verdict]}</div>
          <div style={{ marginTop: 8 }}>
            <div className="meter" aria-hidden="true">
              <span style={{ width: `${Math.round(confidence * 100)}%`, background: colour }} />
            </div>
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="score" style={{ color: colour }}>
            {confidence.toFixed(3)}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-faint)", letterSpacing: "0.07em" }}>
            {calibrated ? "PROBABILITY" : "SCORE"}
          </div>
        </div>
      </div>

      {/*
        Rendered whenever no calibrator is fitted, which is every run until
        stage B5. Not collapsible and not a tooltip: a reader who takes 0.87 for
        an 87% likelihood has been misled by the interface, not by the model.
      */}
      {!calibrated && (
        <p className="caveat">
          <strong style={{ color: "var(--text-dim)" }}>Uncalibrated.</strong> This is a monotone
          ranking value, not a probability — it orders pairs by strength of evidence but does not
          estimate a likelihood. Do not read it as “{Math.round(confidence * 100)}% chance of
          manipulation”.
        </p>
      )}

      {notes.length > 0 && (
        <ul style={{ margin: "9px 0 0", paddingLeft: 17, fontSize: 11.5, color: "var(--warn)" }}>
          {notes.map((note) => (
            <li key={note} style={{ marginBottom: 2 }}>
              {note}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * The matching funnel.
 *
 * Bug 4 in one picture. On the `mountains` pair the prototype reported 231
 * "good matches" and 121 "inliers" while one side held 2,000 keypoints and the
 * other 12 — every match landing on at most 12 distinct points, and the verdict
 * read "Strong evidence". Showing the stages proportionally, with the distinct
 * counts beside them, is what turns that from a flattering number into a
 * visible collapse.
 */
export function MatchFunnel({
  matches,
  keypoints,
}: {
  matches: MatchEvidence;
  keypoints: { left: number; right: number; detector: string } | null;
}) {
  const injective =
    matches.good === matches.distinct_left && matches.good === matches.distinct_right;
  const widest = Math.max(matches.raw, 1);

  const steps = [
    { name: "Candidates", count: matches.raw, flag: false },
    { name: "Ratio test", count: matches.ratio_passed, flag: false },
    {
      name: matches.mutual_nn ? "Mutual NN" : "Ratio only",
      count: matches.good,
      flag: false,
    },
  ];

  return (
    <div>
      {keypoints && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 12,
            color: "var(--text-dim)",
            marginBottom: 11,
            paddingBottom: 9,
            borderBottom: "1px solid var(--border)",
          }}
        >
          <span>Keypoints ({keypoints.detector})</span>
          <span className="num">
            {keypoints.left.toLocaleString()} / {keypoints.right.toLocaleString()}
          </span>
        </div>
      )}

      <div className="funnel">
        {steps.map((step) => (
          <div className="step" key={step.name} data-flag={step.flag ? "bad" : undefined}>
            <span className="name">{step.name}</span>
            <span className="track">
              <span className="fill" style={{ width: `${(step.count / widest) * 100}%` }} />
            </span>
            <span className="count">{step.count.toLocaleString()}</span>
          </div>
        ))}

        <div className="step" data-flag={injective ? undefined : "bad"}>
          <span className="name">Distinct used</span>
          <span className="track">
            <span
              className="fill"
              style={{
                width: `${(Math.min(matches.distinct_left, matches.distinct_right) / widest) * 100}%`,
              }}
            />
          </span>
          <span className="count">
            {matches.distinct_left}/{matches.distinct_right}
          </span>
        </div>
      </div>

      <p
        style={{
          fontSize: 11.5,
          marginTop: 10,
          marginBottom: 0,
          color: injective ? "var(--text-faint)" : "var(--alert)",
        }}
      >
        {injective ? (
          <>One-to-one correspondence; no many-to-one collapse.</>
        ) : (
          <>
            <strong>Not injective</strong> — matches collapse onto fewer distinct keypoints than
            there are matches. Inlier counts derived from this are inflated.
          </>
        )}
      </p>
    </div>
  );
}

/**
 * Geometry outcome, including the named refusal.
 *
 * "Not verified" and "inliers collapsed onto too few distinct keypoints" are
 * different findings, and surfacing the second is what separates this tool from
 * the prototype that reported a verified homography with scale 0.000.
 */
export function GeometryCard({ geometry }: { geometry: GeometryEvidence }) {
  const t = geometry.transform;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 11 }}>
        <span className={`badge ${geometry.verified ? "ok" : "warn"}`}>
          {geometry.verified ? "verified" : "refused"}
        </span>
        <span style={{ fontSize: 11.5, color: "var(--text-faint)" }}>via {geometry.method}</span>
      </div>

      {!geometry.verified && geometry.rejection_reason !== "none" && (
        <p className="notice" style={{ marginBottom: 11 }}>
          {geometry.rejection_reason.replace(/_/g, " ")}
        </p>
      )}

      <table className="kv">
        <tbody>
          <tr>
            <td>Inliers</td>
            <td>
              {geometry.inlier_count} ({geometry.inlier_ratio.toFixed(2)})
            </td>
          </tr>
          <tr>
            <td>Distinct inliers</td>
            <td>{geometry.distinct_inliers}</td>
          </tr>
          {geometry.reproj_rms !== null && (
            <tr>
              <td>Reprojection RMS</td>
              <td>{geometry.reproj_rms.toFixed(2)} px</td>
            </tr>
          )}
          <tr>
            <td>Inlier spread</td>
            <td>{geometry.inlier_spread.toFixed(3)}</td>
          </tr>
          <tr>
            <td>Matched area</td>
            <td>{geometry.matched_area_fraction.toFixed(3)}</td>
          </tr>
        </tbody>
      </table>

      {t && (
        <>
          <h3
            style={{
              fontSize: 10.5,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "var(--text-dim)",
              margin: "15px 0 7px",
            }}
          >
            Recovered transform
          </h3>
          <table className="kv">
            <tbody>
              <tr>
                <td>Rotation</td>
                <td>{t.rotation_deg >= 0 ? "+" : ""}{t.rotation_deg.toFixed(2)}°</td>
              </tr>
              <tr>
                <td>Scale</td>
                <td>{Math.sqrt(Math.abs(t.scale_x * t.scale_y)).toFixed(4)}</td>
              </tr>
              {t.anisotropic && (
                <tr>
                  <td>Scale (x, y)</td>
                  <td>
                    {t.scale_x.toFixed(3)}, {t.scale_y.toFixed(3)}
                  </td>
                </tr>
              )}
              {/*
                Bug 2's fix, made visible. `flip` was unreachable in the
                prototype: a similarity transform's determinant is a²+b², always
                positive, yet the code tested det < 0. Mirrored images reported
                "no" forever. It is only observable at all under the full 6-DOF
                model, which is why `model` is shown next to it.
              */}
              <tr>
                <td>Reflection</td>
                <td style={{ color: t.flip ? "var(--alert)" : undefined }}>
                  {t.flip ? "PRESENT" : "none"}
                </td>
              </tr>
              {t.sheared && (
                <tr>
                  <td>Shear</td>
                  <td>{t.shear_deg.toFixed(2)}°</td>
                </tr>
              )}
              <tr>
                <td>Model</td>
                <td>{t.model === "full" ? "full (6-DOF)" : "similarity (4-DOF)"}</td>
              </tr>
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

/** Signed contribution bars: which evidence pushed the score, and how far. */
export function Contributions({ contributions }: { contributions: Contribution[] }) {
  if (contributions.length === 0) return null;

  const ordered = [...contributions].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution),
  );
  // Normalise to the strongest signal so the chart stays readable whether that
  // is 0.4 or 4.0 log-odds; the numeric column carries the absolute value.
  const largest = Math.max(...ordered.map((c) => Math.abs(c.contribution)), 1e-6);

  return (
    <div className="contrib">
      {ordered.map((c) => {
        const width = (Math.abs(c.contribution) / largest) * 50;
        return (
          <div className="row" key={c.name} title={c.description || c.name}>
            <span className="name">{c.description || c.name}</span>
            <span className="axis">
              <span
                className={`bar ${c.contribution >= 0 ? "pos" : "neg"}`}
                style={{ width: `${width}%` }}
              />
            </span>
            <span className="val">
              {c.contribution >= 0 ? "+" : ""}
              {c.contribution.toFixed(2)}
            </span>
          </div>
        );
      })}
      <p style={{ fontSize: 11, color: "var(--text-faint)", margin: "7px 0 0" }}>
        Right (red) pushes toward manipulation, left (teal) away. Weights are hand-set, not fitted.
      </p>
    </div>
  );
}

/**
 * Threshold sliders.
 *
 * These re-band a cached score client-side. No request is made, which is what
 * keeps them honest: re-running the pipeline per tick would let the evidence
 * shift while the user believes they are only moving a threshold.
 */
export function ThresholdControls({
  bands,
  onChange,
  onReset,
}: {
  bands: Bands;
  onChange: (bands: Bands) => void;
  onReset: () => void;
}) {
  const rows: { key: keyof Bands; label: string }[] = [
    { key: "likely_manipulated", label: "Likely manipulated" },
    { key: "suspicious", label: "Suspicious" },
    { key: "inconclusive", label: "Inconclusive" },
  ];

  return (
    <div>
      {rows.map(({ key, label }) => (
        <div key={key} style={{ marginBottom: 9 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5 }}>
            <label htmlFor={`band-${key}`} style={{ color: "var(--text-dim)" }}>
              {label}
            </label>
            <span className="num">{bands[key].toFixed(2)}</span>
          </div>
          <input
            id={`band-${key}`}
            className="slider"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={bands[key]}
            onChange={(event) =>
              onChange({ ...bands, [key]: Number.parseFloat(event.target.value) })
            }
          />
        </div>
      ))}
      <button className="btn" onClick={onReset} style={{ width: "100%", marginTop: 3 }}>
        Reset to server defaults
      </button>
      <p style={{ fontSize: 11, color: "var(--text-faint)", margin: "9px 0 0" }}>
        Re-bands the cached score locally. The evidence is not re-analysed, so moving a threshold
        cannot change the measurements above.
      </p>
    </div>
  );
}
