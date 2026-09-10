"""Raster overlays: the images a reader actually looks at.

Four overlays, in ascending order of how much a reader should trust them:

``thumbnail``
    The input, scaled. No claim at all.
``attribution``
    Grad-CAM, blended. A claim about what the *embedding* attended to, which is
    not the same as evidence of reuse -- and after bug 5 it is at least
    symmetric between the two branches. Labelled as indicative, never as proof.
``keypoints`` / ``matches``
    Correspondences, drawn with the consensus mask so inliers and outliers are
    visually distinguishable.
``hull``
    The convex hull of *verified* inliers, warped between images. This is the
    honest overlay: it exists only when geometry passed every gate, and it shows
    the region whose correspondence survived MAGSAC.

**Rejections are drawn too, and that is deliberate.** The ``Verification``
docstring makes the argument: on the ``mountains`` pair the prototype claimed
121 inliers, and *showing* 121 lines converging onto a dozen points is a more
convincing account of ``DEGENERATE_CORRESPONDENCES`` than the sentence is. So
match lines are rendered on rejection as well, with the reason stamped on the
image -- what changes is the label and the colour, never the suppression of the
evidence.

Everything here works in **analysis space** and is scaled to output size once,
at the end. Coordinates that reach a caption are original-image pixels, because
that is the frame the user handed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from sciforensics.config import ReportConfig
from sciforensics.io.images import LoadedImage
from sciforensics.runtime import get_logger

if TYPE_CHECKING:
    # Every one of these is annotation-only, and `from __future__ import
    # annotations` means none is needed at runtime. Importing them eagerly drags
    # torch (via global_match.embed) and scikit-learn (via local_match.copymove)
    # into a module that is otherwise pure OpenCV -- so drawing a box would
    # require the full model stack even with no attribution to render.
    from sciforensics.global_match.embed import Attribution
    from sciforensics.local_match.copymove import CopyMoveDetection
    from sciforensics.local_match.geometry import Verification
    from sciforensics.local_match.keypoints import Detection
    from sciforensics.local_match.matcher import Correspondences

_log = get_logger(__name__)

# BGR. Chosen for colour-blind separability (blue/orange rather than red/green)
# and to stay legible over both bright-field and fluorescence panels.
_INLIER = (60, 180, 75)
_OUTLIER = (60, 60, 220)
_HULL = (255, 160, 40)
_SOURCE = (255, 160, 40)
_TARGET = (80, 200, 255)


@dataclass(frozen=True)
class Overlay:
    """One rendered raster, its filename, and how much it may be trusted.

    ``trust`` drives presentation, not content: the template renders an
    ``indicative`` overlay under a caveat and a ``verified`` one without. Keeping
    it on the object means a new overlay cannot be added without answering the
    question.
    """

    name: str
    image: np.ndarray
    caption: str
    trust: str  # "raw" | "indicative" | "verified" | "rejected"

    def write(self, directory: Path, *, max_px: int) -> Path:
        path = directory / f"{self.name}.png"
        cv2.imwrite(str(path), _fit(self.image, max_px))
        return path


def _fit(image: np.ndarray, max_px: int) -> np.ndarray:
    """Downscale so the longest edge is ``max_px``. Never upscales."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_px:
        return image
    scale = max_px / float(longest)
    out: np.ndarray = cv2.resize(
        image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )
    return out


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _stack(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, int]:
    """Place two images side by side on one canvas, returning the x offset.

    Heights are padded rather than resized: rescaling one panel to match the
    other would misrepresent the relative scale that :func:`decompose_affine`
    just measured.
    """
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    height = max(lh, rh)
    canvas = np.zeros((height, lw + rw, 3), dtype=np.uint8)
    canvas[:lh, :lw] = _to_bgr(left)
    canvas[:rh, lw:] = _to_bgr(right)
    return canvas, lw


def _label(image: np.ndarray, text: str, *, colour: tuple[int, int, int]) -> None:
    """Stamp a caption band at the top-left.

    Bug 7 was text written past the edge of a fixed canvas. Here the band is
    sized from the measured text extent, and the *HTML* carries the prose -- this
    is a short tag, not a paragraph, so it cannot overflow.
    """
    if not text:
        return
    scale = max(0.45, min(1.0, image.shape[1] / 1400.0))
    thickness = max(1, round(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    pad = round(8 * scale)
    cv2.rectangle(image, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (24, 24, 24), -1)
    cv2.putText(
        image,
        text,
        (pad, th + pad),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def thumbnail(image: LoadedImage, *, side: str) -> Overlay:
    return Overlay(
        name=f"thumb_{side}",
        image=_to_bgr(image.bgr),
        caption=f"Input {side} — {image.meta.width}x{image.meta.height} px",
        trust="raw",
    )


def attribution_overlay(
    image: LoadedImage,
    attribution: Attribution,
    *,
    side: str,
    colormap: str,
    alpha: float,
) -> Overlay | None:
    """Grad-CAM blended onto the analysis image, or ``None`` if it explains nothing.

    A degenerate (constant) map is dropped rather than drawn. Rendering a flat
    map as though it localised something is precisely what the prototype did on
    every right-hand panel, and a missing overlay is more honest than a blue
    rectangle.
    """
    if attribution.is_degenerate:
        _log.info("attribution map for %s is constant; not rendering an overlay", side)
        return None

    heat = attribution.to_source()
    base = _to_bgr(image.bgr)
    if heat.shape[:2] != base.shape[:2]:
        heat = cv2.resize(heat, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Imported at call time for the same reason the type is: an attribution
    # overlay is the only thing here that needs the model stack.
    from sciforensics.global_match.embed import blend_overlay

    blended = blend_overlay(base, heat, colormap=colormap, alpha=alpha)
    suppressed = attribution.zero_fraction
    return Overlay(
        name=f"attribution_{side}",
        image=blended,
        caption=(
            f"Embedding attribution ({attribution.stage}, pooling={attribution.pooling}) — "
            f"{suppressed:.1%} of cells at zero. Indicative of where the network looked; "
            "not evidence of reuse."
        ),
        trust="indicative",
    )


def keypoint_overlay(image: LoadedImage, detection: Detection, *, side: str) -> Overlay:
    canvas = _to_bgr(image.bgr)
    for point in detection.points:
        x, y = round(float(point[0])), round(float(point[1]))
        cv2.circle(canvas, (x, y), 3, _SOURCE, 1, cv2.LINE_AA)
    return Overlay(
        name=f"keypoints_{side}",
        image=canvas,
        caption=f"{detection.count} keypoints ({detection.detector})",
        trust="raw",
    )


def match_overlay(
    left: LoadedImage,
    right: LoadedImage,
    correspondences: Correspondences,
    verification: Verification | None,
    cfg: ReportConfig,
) -> Overlay:
    """Side-by-side correspondence lines, inliers distinguished from outliers.

    Drawn whether or not verification passed -- see the module docstring. The
    caption is what carries the distinction, so a rejected pair cannot be
    mistaken for a verified one while still showing why it was rejected.
    """
    canvas, offset = _stack(left.bgr, right.bgr)

    mask = (
        verification.inlier_mask
        if verification is not None and len(verification.inlier_mask) == len(correspondences)
        else np.zeros(len(correspondences), dtype=bool)
    )
    verified = verification is not None and verification.verified

    # Draw outliers first so inliers land on top, and cap the count: 2,000 lines
    # is an opaque wash that hides the very structure the overlay exists to show.
    order = np.concatenate([np.flatnonzero(~mask), np.flatnonzero(mask)])
    if cfg.max_match_lines and len(order) > cfg.max_match_lines:
        keep_out = np.flatnonzero(~mask)[: cfg.max_match_lines // 2]
        keep_in = np.flatnonzero(mask)[: cfg.max_match_lines - len(keep_out)]
        order = np.concatenate([keep_out, keep_in])

    for index in order:
        src = correspondences.src[index]
        dst = correspondences.dst[index]
        is_inlier = bool(mask[index])
        colour = _INLIER if is_inlier else _OUTLIER
        p0 = (round(float(src[0])), round(float(src[1])))
        p1 = (round(float(dst[0])) + offset, round(float(dst[1])))
        if cfg.draw_match_lines:
            cv2.line(canvas, p0, p1, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, colour, -1, cv2.LINE_AA)

    if cfg.draw_inlier_hull and verified:
        _draw_hull(canvas, correspondences, mask, offset)

    inliers = int(mask.sum())
    if verified:
        caption = f"{len(correspondences)} correspondences, {inliers} geometric inliers (green)"
        trust = "verified"
    else:
        reason = (
            verification.evidence.rejection_reason.value if verification is not None else "not run"
        )
        caption = (
            f"{len(correspondences)} correspondences; verification refused ({reason}). "
            f"{inliers} met the estimator's consensus but did not survive the gates — "
            "shown so the failure is visible, not as evidence."
        )
        trust = "rejected"

    _label(canvas, "VERIFIED" if verified else "REJECTED", colour=_INLIER if verified else _OUTLIER)
    return Overlay(name="matches", image=canvas, caption=caption, trust=trust)


def _draw_hull(
    canvas: np.ndarray,
    correspondences: Correspondences,
    mask: np.ndarray,
    offset: int,
) -> None:
    """Outline the convex hull of verified inliers on both panels.

    Drawn from the *inlier points themselves* rather than by warping a rectangle
    through the homography: warping would show where the model says the region
    goes, which is a claim about the model. The hull of the surviving points is a
    claim about the data.

    Takes only the mask, not the ``Verification`` — the caller has already
    decided the fit was verified, and passing the evidence object as well would
    imply this function re-checks it.
    """
    if not mask.any():
        return
    for points, dx in ((correspondences.src[mask], 0), (correspondences.dst[mask], offset)):
        if len(points) < 3:
            continue
        hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
        hull = hull.astype(np.int32)
        hull[:, :, 0] += dx
        cv2.polylines(canvas, [hull], True, _HULL, 2, cv2.LINE_AA)


def copy_move_overlay(image: LoadedImage, detection: CopyMoveDetection) -> Overlay:
    """Cloned lobes, one colour pair per region, with the union mask tinted.

    Bug 6's fix is what makes this drawable at all: the prototype fitted a single
    global transform, so it could never render more than one region however many
    clones were present.
    """
    canvas = _to_bgr(image.bgr)

    tint = np.zeros_like(canvas)
    for mask in detection.masks:
        resized = mask
        if resized.shape[:2] != canvas.shape[:2]:
            resized = cv2.resize(
                mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        tint[resized > 0] = _TARGET
    if detection.masks:
        canvas = cv2.addWeighted(canvas, 0.75, tint, 0.25, 0.0)

    # Boxes are in ORIGINAL pixels but the canvas is analysis space, so scale.
    scale = detection.scale
    for index, region in enumerate(detection.evidence.regions, start=1):
        for box, colour in ((region.source_box, _SOURCE), (region.target_box, _TARGET)):
            x, y, w, h = (round(v * scale) for v in box)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index),
                (x + 4, y + 18),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                colour,
                1,
                cv2.LINE_AA,
            )
        sx, sy, sw, sh = (round(v * scale) for v in region.source_box)
        tx, ty, tw, th = (round(v * scale) for v in region.target_box)
        cv2.line(
            canvas,
            (sx + sw // 2, sy + sh // 2),
            (tx + tw // 2, ty + th // 2),
            _HULL,
            1,
            cv2.LINE_AA,
        )

    count = detection.region_count
    return Overlay(
        name="copymove",
        image=canvas,
        caption=(
            f"{count} cloned region{'s' if count != 1 else ''} "
            f"from {detection.evidence.cluster_count} candidate offset cluster(s)"
        ),
        trust="verified" if count else "raw",
    )
