"""Small, normalized drawing payloads for the browser's SVG viewer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from sciforensics.pipeline import CopyMoveAnalysis, PairAnalysis


def viewer_payload(analysis: PairAnalysis | CopyMoveAnalysis, *, limit: int) -> dict[str, Any]:
    if isinstance(analysis, CopyMoveAnalysis):
        h, w = analysis.image.shape
        return {"matches": [], "hulls": {}, "aspects": {"input": w / h}}
    lh, lw = analysis.left.shape
    rh, rw = analysis.right.shape
    aspects = {"left": lw / lh, "right": rw / rh}
    corr = analysis.correspondences
    if corr is None:
        return {"matches": [], "hulls": {}, "aspects": aspects}
    src = corr.src / np.array([lw, lh])
    dst = corr.dst / np.array([rw, rh])
    mask = (
        analysis.verification.inlier_mask
        if analysis.verification is not None
        else np.zeros(len(corr), dtype=bool)
    )
    # Prefer verified inliers; retain a sample of rejected matches for inspection.
    indices = sorted(range(len(corr)), key=lambda i: not bool(mask[i]))[:limit]
    matches = [
        {"left": src[i].tolist(), "right": dst[i].tolist(), "inlier": bool(mask[i])}
        for i in indices
    ]
    hulls = {}
    if analysis.verification is not None and analysis.verification.evidence.verified:
        for side, points in (("left", src[mask]), ("right", dst[mask])):
            if len(points) >= 3:
                hulls[side] = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).tolist()
    return {"matches": matches, "hulls": hulls, "aspects": aspects}
