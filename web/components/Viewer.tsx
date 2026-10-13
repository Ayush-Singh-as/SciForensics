"use client";

import { useRef, useState } from "react";
import { assetUrl, type OverlayRef, type ViewerData } from "@/lib/api";

/** Both panes and their SVG evidence use one normalized coordinate frame. */
export function Viewer({ jobId, overlays, data, single = false }: {
  jobId: string; overlays: OverlayRef[]; data: ViewerData; single?: boolean;
}) {
  const [layer, setLayer] = useState("input");
  const [lines, setLines] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const sides = single ? ["input"] : ["left", "right"];
  function overlayFor(side: string) {
    return overlays.find((item) => item.name === (layer === "input" || layer === "region" ? `thumb_${side}` : `${layer}_${side}`))
      ?? overlays.find((item) => item.name === `thumb_${side}`);
  }
  const layers = [
    { key: "input", label: "Inputs", available: true },
    { key: "attribution", label: "Heatmaps", available: overlays.some((o) => o.name.startsWith("attribution_")) },
    { key: "keypoints", label: "Keypoints", available: overlays.some((o) => o.name.startsWith("keypoints_")) },
    { key: "region", label: "Verified regions", available: Object.keys(data.hulls).length > 0 },
    { key: "clones", label: "Clone mask", available: single && overlays.some((o) => o.name === "copymove") },
  ];
  const transform = `translate(${pan.x} ${pan.y}) translate(.5 .5) scale(${zoom}) translate(-.5 -.5)`;
  function frame(side: string) {
    const ratio = data.aspects[side] ?? 1;
    const w = Math.min(1, ratio), h = Math.min(1, 1 / ratio);
    return { x: (1 - w) / 2, y: (1 - h) / 2, w, h };
  }
  const coordinate = (point: [number, number], right: boolean) => ({
    x: ((frame(right ? "right" : "left").x + point[0] * frame(right ? "right" : "left").w - .5) * zoom + .5 + pan.x) * .5 + (right ? .5 : 0),
    y: (frame(right ? "right" : "left").y + point[1] * frame(right ? "right" : "left").h - .5) * zoom + .5 + pan.y,
  });
  return <div className="panel evidence-viewer">
    <header><h2>Inspect evidence</h2><span className="rev">{single ? "Single image" : "Synced panes"}</span></header>
    <div className="viewer-toolbar">
      {layers.filter((item) => item.available).map((item) => <button key={item.key} className="toggle" aria-pressed={layer === item.key} onClick={() => setLayer(item.key)}>{item.label}</button>)}
      {!single && <label><input type="checkbox" checked={lines} onChange={(event) => setLines(event.target.checked)} /> Match lines</label>}
      <button className="btn" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(1, value / 1.3))}>−</button>
      <output aria-label="Zoom level">{zoom.toFixed(1)}×</output>
      <button className="btn" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(8, value * 1.3))}>+</button>
      <button className="btn" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>Fit images</button>
    </div>
    <div className={`paired-view ${single ? "single-view" : ""}`}>
      {sides.map((side) => {
        const overlay = layer === "clones" ? overlays.find((o) => o.name === "copymove") : overlayFor(side);
        const hull = data.hulls[side as "left" | "right"];
        const f = frame(side);
        return <div className="image-pane" key={side} data-testid={`pane-${side}`}>
          <div className="pane-label">{side === "left" ? "Figure A" : side === "right" ? "Figure B" : "Figure"}</div>
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" role="img" aria-label={`Evidence ${side}`}
            onWheel={(event) => { setZoom((value) => Math.max(1, Math.min(8, value * (event.deltaY < 0 ? 1.12 : .89)))); }}
            onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); drag.current = { x: event.clientX, y: event.clientY, px: pan.x, py: pan.y }; }}
            onPointerMove={(event) => { if (!drag.current) return; const bounds = event.currentTarget.getBoundingClientRect(); setPan({ x: drag.current.px + (event.clientX - drag.current.x) / bounds.width, y: drag.current.py + (event.clientY - drag.current.y) / bounds.height }); }}
            onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
            <g transform={transform} data-testid={`transform-${side}`}>
              {overlay && <image href={assetUrl(jobId, overlay.file)} x={f.x} y={f.y} width={f.w} height={f.h} preserveAspectRatio="none" />}
              {layer === "region" && hull && <polygon points={hull.map((p) => `${f.x + p[0] * f.w},${f.y + p[1] * f.h}`).join(" ")} fill="#ffac3530" stroke="#ffac35" strokeWidth=".004" />}
            </g>
          </svg>
        </div>;
      })}
      {!single && lines && <svg className="match-lines" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Correspondence lines">
        {data.matches.map((match, index) => {
          const a = coordinate(match.left, false), b = coordinate(match.right, true);
          if ([a.x, a.y, b.x, b.y].some((value) => value < 0 || value > 1) || a.x > .5 || b.x < .5) return null;
          return <line key={index} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={match.inlier ? "#72dcbb" : "#ff7777"} strokeWidth=".0012" opacity={match.inlier ? .6 : .25} />;
        })}
      </svg>}
    </div>
    <p className="viewer-help">Drag either image to pan; scroll or use +/− to zoom. {layer === "attribution" ? "Heatmaps show network attention, not proof of reuse." : "Green lines are geometric inliers; red lines are rejected candidates."}</p>
  </div>;
}
