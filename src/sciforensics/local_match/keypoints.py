"""Keypoint detection with contrast enhancement and region-of-interest focusing.

Scientific figure panels are a hostile case for keypoint detectors: they are
small, low-contrast, often greyscale, and frequently re-compressed. The
enhancement pipeline here (upscale, mild deblur, CLAHE) exists to recover usable
keypoints from them, and it is ported essentially intact from the prototype
because it worked.

Two defects are fixed.

**Bug 13 — the hardcoded ``/ 4.0``.** The legacy code detected keypoints on a 4x
upscaled image and then divided coordinates by the literal ``4.0`` in six
separate places, with the scale itself written as a default argument somewhere
else entirely. Changing the scale silently corrupted every reported coordinate.
Here the enhancement factor is carried on :class:`Detection` and applied exactly
once, in :meth:`_to_analysis_space`.

**Bug 4 (detection half) — asymmetric ROI collapse.** The legacy
``_filter_to_regions`` fell back to *unfiltered* keypoints when the ROI mask
retained fewer than 8, using a hardcoded threshold with no relation to anything.
The two sides therefore made the decision independently: on the ``mountains``
pair one side kept its ROI-filtered 12 keypoints while the other fell back to all
2000. Matching 2000 against 12 is what produced the phantom "231 matches".
Abandonment is now governed by ``local_match.roi.min_keypoints`` and, more
importantly, is *recorded* on the result so the asymmetry is visible in the
report instead of invisible. The separate hard floor below which matching is not
attempted at all lives in the pipeline, as
``local_match.min_keypoints_per_side``; conflating the two is what made every
small panel report ``too_few_keypoints`` for a shortfall that was really the
matcher's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from sciforensics.config import EnhanceConfig, LocalMatchConfig, RoiConfig
from sciforensics.io.images import INTERPOLATION
from sciforensics.local_match.geometry import rescale_box
from sciforensics.runtime import get_logger
from sciforensics.types import BBox

__all__ = [
    "Detection",
    "KeypointDetector",
    "OrbDetector",
    "build_detector",
    "dog_regions",
    "enhance",
    "sobel_magnitude",
]

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# enhancement
# ---------------------------------------------------------------------------
def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """Normalised Sobel gradient magnitude as ``uint8``.

    Ported from the prototype unchanged apart from typing: used to bias
    region-of-interest detection toward structural edges rather than flat
    background.
    """
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(dx, dy)
    peak = float(magnitude.max())
    if peak > 0:
        magnitude *= 255.0 / peak
    return magnitude.astype(np.uint8)


def enhance(gray: np.ndarray, cfg: EnhanceConfig) -> np.ndarray:
    """Upscale and locally equalise a greyscale panel to expose keypoints.

    Upscaling before detection is not cosmetic: ORB's FAST corner test operates
    on a fixed pixel radius, so on a 200 px panel there is physically not enough
    room for the descriptor patch. The mild Gaussian afterwards suppresses the
    cubic interpolation's ringing, which FAST would otherwise fire on.
    """
    if not cfg.enabled or cfg.scale == 1.0:
        return gray

    interpolation = INTERPOLATION[cfg.interpolation]
    enlarged = cv2.resize(gray, None, fx=cfg.scale, fy=cfg.scale, interpolation=interpolation)
    blurred = cv2.GaussianBlur(enlarged, (0, 0), 0.8)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_grid, cfg.clahe_tile_grid),
    )
    return clahe.apply(blurred)


def dog_regions(gray: np.ndarray, cfg: RoiConfig) -> list[BBox]:
    """Difference-of-Gaussians blob regions, as ``(x, y, w, h)`` boxes.

    The threshold is applied to a band-pass response computed in float ``[0, 1]``
    units, so ``roi.dog_threshold`` is a physically meaningful contrast level.
    The legacy implementation passed a ``uint8`` 0-255 Sobel image to
    ``skimage.feature.blob_dog(threshold=0.08)``, where 0.08 on a 0-255 scale
    accepts essentially every response -- which is why the old ROI boxes bore
    little relation to actual image structure.

    Falls back to Otsu thresholding when the fixed threshold finds nothing, so a
    uniformly low-contrast panel still yields regions rather than silently
    degrading to whole-image detection.
    """
    if not cfg.enabled:
        return []

    normalised = gray.astype(np.float32) / 255.0
    low = cv2.GaussianBlur(normalised, (0, 0), cfg.dog_sigma_low)
    high = cv2.GaussianBlur(normalised, (0, 0), cfg.dog_sigma_high)
    response = np.abs(low - high)

    mask: np.ndarray = (response > cfg.dog_threshold).astype(np.uint8) * 255
    if not mask.any():
        peak = float(response.max())
        scaled = (response * (255.0 / peak)).astype(np.uint8) if peak > 0 else mask
        _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _log.debug("DoG threshold %.3f found nothing; fell back to Otsu", cfg.dog_threshold)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if cfg.dilate_px > 0:
        # Dilate so the box comfortably contains the descriptor patch around
        # each blob, not just the blob's own extent.
        size = 2 * cfg.dilate_px + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in (cv2.boundingRect(c) for c in contours)
        if w * h >= cfg.min_area
    ]
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes[: cfg.max_regions]


def _mask_from_boxes(shape: tuple[int, int], boxes: list[BBox]) -> np.ndarray:
    """Rasterise boxes into a binary mask.

    Passing a mask to ``detectAndCompute`` is better than detecting everywhere
    and filtering afterwards: ORB distributes its ``nfeatures`` budget across the
    masked area, so a restricted region yields densely-sampled keypoints rather
    than whatever survives of a global budget spent mostly on background.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y, w, h in boxes:
        mask[y : y + h, x : x + w] = 255
    return mask


# ---------------------------------------------------------------------------
# detections
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    """Keypoints and descriptors for one image, in **analysis space**.

    Coordinates have already been divided by :attr:`enhancement_scale`, once, at
    the single point where that conversion belongs. Nothing downstream needs to
    know that detection happened on an upscaled image, which is precisely the
    property the legacy code lacked.
    """

    points: np.ndarray  # (N, 2) float32
    descriptors: np.ndarray | None  # (N, D)
    sizes: np.ndarray  # (N,) float32
    responses: np.ndarray  # (N,) float32
    #: Keypoint orientation in degrees, or ``-1`` where the detector does not
    #: produce one. Required rather than defaulted: copy-move clustering uses the
    #: *difference* in orientation between two keypoints as its rotation estimate,
    #: and a detector silently supplying no angles would put every pair at the
    #: cluster centre of that dimension -- which looks exactly like "no rotation"
    #: instead of like "no measurement". Backends that genuinely have no
    #: orientation (dense matchers) must say so with ``-1``.
    angles: np.ndarray  # (N,) float32 degrees
    detector: str
    enhancement_scale: float
    #: Keypoints found before the ROI restriction.
    detected: int
    #: ROI boxes, in analysis space, for the report overlay.
    roi_boxes: tuple[BBox, ...] = ()
    #: ROI restriction was dropped because it would have left too few keypoints.
    roi_abandoned: bool = False
    warnings: tuple[str, ...] = field(default=())

    @property
    def count(self) -> int:
        return len(self.points)

    def keypoints(self) -> list[cv2.KeyPoint]:
        """Rebuild ``cv2.KeyPoint`` objects for drawing helpers."""
        return [
            cv2.KeyPoint(
                x=float(pt[0]),
                y=float(pt[1]),
                size=float(size),
                angle=float(angle),
                response=float(resp),
            )
            for pt, size, angle, resp in zip(
                self.points, self.sizes, self.angles, self.responses, strict=True
            )
        ]


@runtime_checkable
class KeypointDetector(Protocol):
    """Interface every detector backend satisfies.

    Keeping this a ``Protocol`` is what lets stage B3 drop in DISK/SuperPoint
    without the pipeline learning anything about them: they produce float
    descriptors and need a different matcher norm, both of which are properties
    of the detector object rather than branches at the call site.
    """

    name: str
    #: ``cv2.NORM_HAMMING`` for binary descriptors, ``cv2.NORM_L2`` for float.
    norm: int

    def detect(self, gray: np.ndarray) -> Detection: ...


class OrbDetector:
    """ORB detection over enhanced, optionally ROI-restricted input.

    Always available and CPU-only, so it stays the default and the fallback.
    """

    name = "orb"
    norm = cv2.NORM_HAMMING

    def __init__(self, cfg: LocalMatchConfig) -> None:
        self._cfg = cfg
        # `cv2.ORB.create` rather than the older `cv2.ORB_create` alias: it is the
        # form OpenCV 5 actually types, and it has been available since OpenCV 3,
        # so this costs no backwards compatibility.
        self._orb = cv2.ORB.create(
            nfeatures=cfg.max_features,
            scaleFactor=cfg.orb.scale_factor,
            nlevels=cfg.orb.n_levels,
            edgeThreshold=cfg.orb.edge_threshold,
            patchSize=cfg.orb.patch_size,
            fastThreshold=cfg.orb.fast_threshold,
        )

    def detect(self, gray: np.ndarray) -> Detection:
        cfg = self._cfg
        scale = cfg.enhance.scale if cfg.enhance.enabled else 1.0
        work = enhance(gray, cfg.enhance)
        warnings: list[str] = []

        boxes = dog_regions(sobel_magnitude(work), cfg.roi) if cfg.roi.enabled else []

        # Detect globally first so `detected` reflects what the image actually
        # offers, independent of the ROI decision. That count is what makes an
        # asymmetric outcome diagnosable after the fact.
        global_kps, global_desc = self._orb.detectAndCompute(work, None)
        detected = len(global_kps)

        keypoints, descriptors = global_kps, global_desc
        roi_abandoned = False

        if boxes:
            mask = _mask_from_boxes(work.shape[:2], boxes)
            roi_kps, roi_desc = self._orb.detectAndCompute(work, mask)
            if len(roi_kps) >= cfg.roi.min_keypoints:
                keypoints, descriptors = roi_kps, roi_desc
            else:
                # The legacy threshold here was a bare `< 8`, decided per side
                # with no record. Both facts were the bug.
                roi_abandoned = True
                warnings.append(
                    f"region-of-interest filtering would have left {len(roi_kps)} keypoints "
                    f"(floor is {cfg.roi.min_keypoints}); using unrestricted detection "
                    f"for this image"
                )
                _log.debug(warnings[-1])

        if not keypoints:
            return Detection(
                points=np.zeros((0, 2), np.float32),
                descriptors=None,
                sizes=np.zeros((0,), np.float32),
                responses=np.zeros((0,), np.float32),
                angles=np.zeros((0,), np.float32),
                detector=self.name,
                enhancement_scale=scale,
                detected=detected,
                roi_boxes=tuple(rescale_box(b, scale=scale) for b in boxes),
                roi_abandoned=roi_abandoned,
                warnings=tuple(warnings),
            )

        return self._to_analysis_space(
            keypoints,
            descriptors,
            scale=scale,
            detected=detected,
            boxes=boxes,
            roi_abandoned=roi_abandoned,
            warnings=warnings,
        )

    def _to_analysis_space(
        self,
        keypoints: Sequence[cv2.KeyPoint],
        descriptors: np.ndarray | None,
        *,
        scale: float,
        detected: int,
        boxes: list[BBox],
        roi_abandoned: bool,
        warnings: list[str],
    ) -> Detection:
        """The single place enhancement-space coordinates become analysis-space.

        Every coordinate and every keypoint size divides by ``scale`` here and
        nowhere else. This is the structural fix for bug 13.

        Orientation and response deliberately do *not* divide: an angle and a
        corner score are not lengths, and scaling them would be the same class of
        mistake as failing to scale the ones that are.
        """
        inv = 1.0 / scale
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32) * inv
        sizes = np.array([kp.size for kp in keypoints], dtype=np.float32) * inv
        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
        angles = np.array([kp.angle for kp in keypoints], dtype=np.float32)

        return Detection(
            points=np.ascontiguousarray(points),
            descriptors=descriptors,
            sizes=sizes,
            responses=responses,
            angles=angles,
            detector=self.name,
            enhancement_scale=scale,
            detected=detected,
            roi_boxes=tuple(rescale_box(b, scale=scale) for b in boxes),
            roi_abandoned=roi_abandoned,
            warnings=tuple(warnings),
        )


def build_detector(cfg: LocalMatchConfig, *, device: str = "cpu") -> KeypointDetector:
    """Instantiate the configured detector backend.

    ``disk`` arrived in stage B3. ``superpoint`` is still unimplemented, and
    requesting it is an explicit error rather than a silent downgrade to ORB:
    a benchmark row labelled "SuperPoint" that actually ran ORB would be worse
    than no row.
    """
    if cfg.detector == "orb":
        return OrbDetector(cfg)
    if cfg.detector == "disk":
        # Imported here, not at module scope: kornia and torch are an optional
        # extra, and `orb` must keep working without them.
        from sciforensics.local_match.learned import DiskDetector

        return DiskDetector(cfg, device=device)
    raise NotImplementedError(f"detector {cfg.detector!r} is not available yet. Use orb or disk.")
