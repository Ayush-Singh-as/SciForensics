"""Learned keypoints and matching: DISK + LightGlue via kornia. Stage B3.

**Why this exists.** ORB is a binary-descriptor detector built for speed, and it
fails on two cases this project must handle:

*Signal degradation.* On the three ``*_ex2_degraded.jpg`` pairs (JPEG q25 +
Gaussian noise) ORB yields 6, 7 and 2 good matches — **0 of 3 detected**. A
manipulated figure that has merely been recompressed for publication is exactly
the realistic case, so 0/3 is not an edge case, it is the common one.

*Reflection.* ORB descriptors are not mirror-invariant. Verified during Stage A:
a 45 degree, 1.2x, horizontal-mirror pair with 2,000 keypoints per side produced 45
matches and only 5 inliers, so geometry abstained. The flip *decode* is correct
(bug 2 is fixed and unit-tested at ``det = -1.440``), but no real image pair
could demonstrate it because ORB never supplies enough correspondences to fit
the transform. DISK's learned descriptors do not share that blind spot.

**No silent downgrade.** ``kornia`` is an optional dependency; if it is missing,
constructing these raises with an install hint. A benchmark row labelled
"LightGlue" that actually ran ORB would be worse than no row at all, which is
why :func:`build_detector` never falls back.

**Protocols, not branches.** These satisfy the existing
:class:`~sciforensics.local_match.keypoints.KeypointDetector` and
:class:`~sciforensics.local_match.matcher.Matcher` protocols, so nothing in
``pipeline.py`` learns that they exist. The two properties the rest of the
pipeline relies on are preserved deliberately:

* Coordinates are in **analysis space** — divided by the enhancement scale once,
  at the same single point ORB does it (bug 13).
* Correspondences are **injective**. LightGlue is a joint matcher and already
  produces at most one partner per keypoint, so ``MatchEvidence.is_injective``
  holds without a separate mutual-NN pass — and bug 4's degeneracy gates in
  ``geometry.py`` still run regardless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from sciforensics.config import LocalMatchConfig
from sciforensics.local_match.keypoints import Detection, dog_regions, enhance
from sciforensics.local_match.matcher import Correspondences
from sciforensics.runtime import get_logger
from sciforensics.types import MatchEvidence

# `torch` is a hard dependency of the package, so it is imported at module
# scope; only `kornia` is optional, and only that import is guarded.
_log = get_logger(__name__)

_INSTALL_HINT = 'install the learned-matching extra: pip install -e ".[match]"'


class LearnedBackendError(RuntimeError):
    """Raised when a learned backend is requested but cannot be constructed."""


def _require_kornia() -> Any:
    try:
        import kornia
    except ImportError as exc:
        raise LearnedBackendError(f"kornia is not installed ({exc}); {_INSTALL_HINT}") from exc
    return kornia


def _to_tensor(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    """``(H, W)`` uint8 → ``(1, 3, H, W)`` float in ``[0, 1]``.

    DISK is trained on RGB, so a greyscale panel is replicated across channels
    rather than fed as one. Feeding a single channel silently changes the
    input distribution the weights were fitted on.
    """
    array = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array)[None, None].to(device)
    repeated: torch.Tensor = tensor.repeat(1, 3, 1, 1)
    return repeated


@dataclass
class DiskDetector:
    """DISK keypoints and 128-d float descriptors.

    Satisfies :class:`~sciforensics.local_match.keypoints.KeypointDetector`.
    ``norm`` is ``NORM_L2`` rather than ``NORM_HAMMING``: DISK descriptors are
    float, and matching them under a Hamming norm would be silently meaningless
    rather than an error. That the norm travels *on the detector* is what lets
    the matcher stay ignorant of which backend produced its input.
    """

    cfg: LocalMatchConfig
    device: str = "cpu"
    name: str = "disk"
    norm: int = cv2.NORM_L2

    def __post_init__(self) -> None:
        kornia = _require_kornia()
        self._torch_device = torch.device(self.device)
        try:
            self._model = kornia.feature.DISK.from_pretrained(
                "depth", device=self._torch_device
            ).eval()
        except Exception as exc:  # pragma: no cover - network/weights dependent
            raise LearnedBackendError(
                f"could not load DISK weights ({exc}). They download on first use; "
                "check network access or pre-populate the torch hub cache."
            ) from exc

    def detect(self, gray: np.ndarray) -> Detection:
        # Same enhancement path as ORB so the two are compared on equal footing
        # in the benchmark -- and so the scale bookkeeping is identical.
        working = enhance(gray, self.cfg.enhance)
        scale = self.cfg.enhance.scale if self.cfg.enhance.enabled else 1.0

        tensor = _to_tensor(working, self._torch_device)
        with torch.inference_mode():
            features = self._model(
                tensor,
                n=self.cfg.max_features,
                # DISK's encoder downsamples by 16; a non-divisible input is
                # padded rather than rejected.
                pad_if_not_divisible=True,
            )[0]

        points = features.keypoints.detach().cpu().numpy().astype(np.float32)
        descriptors = features.descriptors.detach().cpu().numpy().astype(np.float32)
        responses = features.detection_scores.detach().cpu().numpy().astype(np.float32)

        detected = len(points)
        if detected == 0:
            _log.warning("DISK found no keypoints")
            return self._empty(scale)

        # Back to analysis space, once, here -- the single conversion point that
        # bug 13 established.
        points = points / scale

        roi_boxes: tuple[tuple[int, int, int, int], ...] = ()
        abandoned = False
        if self.cfg.roi.enabled:
            points, descriptors, responses, roi_boxes, abandoned = self._restrict(
                gray, points, descriptors, responses, scale
            )

        return Detection(
            points=points,
            descriptors=descriptors,
            # DISK produces neither a scale nor an orientation. `-1` for angles
            # is the documented contract for "no measurement": copy-move
            # clustering uses orientation *differences*, and silently supplying
            # 0 would make every pair look like "no rotation" instead of like
            # "not measured".
            sizes=np.full(len(points), float(self.cfg.orb.patch_size), dtype=np.float32),
            responses=responses,
            angles=np.full(len(points), -1.0, dtype=np.float32),
            detector=self.name,
            enhancement_scale=scale,
            detected=detected,
            roi_boxes=roi_boxes,
            roi_abandoned=abandoned,
        )

    def _restrict(
        self,
        gray: np.ndarray,
        points: np.ndarray,
        descriptors: np.ndarray,
        responses: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...], bool]:
        """Keep keypoints inside DoG regions, unless that drops below the floor.

        Identical policy to ORB's, including the abandonment rule: the legacy
        code's hard ``< 8`` fallback is what allowed one side to collapse to 12
        keypoints while the other kept 2,000 (bug 4).
        """
        boxes = dog_regions(enhance(gray, self.cfg.enhance), self.cfg.roi)
        if not boxes:
            return points, descriptors, responses, (), True

        analysis_boxes = tuple(
            (
                round(x / scale),
                round(y / scale),
                max(1, round(w / scale)),
                max(1, round(h / scale)),
            )
            for x, y, w, h in boxes
        )

        keep = np.zeros(len(points), dtype=bool)
        for x, y, w, h in analysis_boxes:
            inside = (
                (points[:, 0] >= x)
                & (points[:, 0] < x + w)
                & (points[:, 1] >= y)
                & (points[:, 1] < y + h)
            )
            keep |= inside

        if int(keep.sum()) < self.cfg.roi.min_keypoints:
            _log.info(
                "ROI filtering would leave %d keypoints (floor is %d); "
                "using unrestricted detection",
                int(keep.sum()),
                self.cfg.roi.min_keypoints,
            )
            return points, descriptors, responses, analysis_boxes, True

        return points[keep], descriptors[keep], responses[keep], analysis_boxes, False

    def _empty(self, scale: float) -> Detection:
        return Detection(
            points=np.zeros((0, 2), dtype=np.float32),
            descriptors=np.zeros((0, 128), dtype=np.float32),
            sizes=np.zeros(0, dtype=np.float32),
            responses=np.zeros(0, dtype=np.float32),
            angles=np.zeros(0, dtype=np.float32),
            detector=self.name,
            enhancement_scale=scale,
            detected=0,
        )


@dataclass
class LightGlueMatcher:
    """LightGlue: a *joint* matcher over both keypoint sets at once.

    Unlike nearest-neighbour matching it reasons about both images together, so
    it resolves ambiguous texture by global consistency rather than by a ratio
    threshold. It also emits at most one partner per keypoint, so the
    correspondence set is **injective by construction** — the property bug 4's
    fix had to enforce manually for brute-force matching.

    Injective does not mean *correct*: the degeneracy gates in
    :mod:`sciforensics.local_match.geometry` (distinct-correspondence count,
    inlier spread, reprojection RMS, condition number) still run, because a
    confident matcher can still agree on a degenerate configuration.
    """

    cfg: LocalMatchConfig
    device: str = "cpu"
    name: str = "lightglue"
    feature: str = "disk"

    def __post_init__(self) -> None:
        kornia = _require_kornia()
        self._torch_device = torch.device(self.device)
        try:
            self._matcher = (
                kornia.feature.LightGlueMatcher(self.feature).to(self._torch_device).eval()
            )
        except Exception as exc:  # pragma: no cover - weights dependent
            raise LearnedBackendError(
                f"could not load LightGlue weights for {self.feature!r} ({exc})."
            ) from exc

    def match(self, left: Detection, right: Detection) -> Correspondences:
        if left.descriptors is None or right.descriptors is None:
            return self._empty(left, right, "one side has no descriptors")
        if left.count == 0 or right.count == 0:
            return self._empty(left, right, "one side has no keypoints")

        lafs_left = self._lafs(left)
        lafs_right = self._lafs(right)
        desc_left = torch.from_numpy(left.descriptors).to(self._torch_device)
        desc_right = torch.from_numpy(right.descriptors).to(self._torch_device)

        with torch.inference_mode():
            _, indices = self._matcher(
                desc_left,
                desc_right,
                lafs_left,
                lafs_right,
            )

        pairs = indices.detach().cpu().numpy()
        if pairs.size == 0:
            return self._empty(left, right, "LightGlue returned no matches")

        src_index = pairs[:, 0].astype(np.int32)
        dst_index = pairs[:, 1].astype(np.int32)
        src = left.points[src_index]
        dst = right.points[dst_index]

        # LightGlue reports a confidence, not a descriptor distance. Recording
        # zeros would imply a perfect match everywhere, so the true descriptor
        # distance is computed for the surviving pairs -- it feeds the report and
        # keeps `Correspondences.distances` meaning one thing across backends.
        distances = np.linalg.norm(
            left.descriptors[src_index] - right.descriptors[dst_index], axis=1
        ).astype(np.float32)

        evidence = MatchEvidence(
            matcher=self.name,
            # LightGlue does not expose a candidate count; the honest figure for
            # "pairs considered" is the smaller keypoint set, since that bounds
            # any injective assignment.
            raw=min(left.count, right.count),
            ratio_passed=len(src_index),
            good=len(src_index),
            distinct_left=len(np.unique(src_index)),
            distinct_right=len(np.unique(dst_index)),
            # Injective by construction rather than by a mutual-NN pass.
            mutual_nn=True,
            ratio=self.cfg.nn_ratio,
        )
        return Correspondences(
            src=src.astype(np.float32),
            dst=dst.astype(np.float32),
            src_index=src_index,
            dst_index=dst_index,
            distances=distances,
            evidence=evidence,
        )

    def _lafs(self, detection: Detection) -> torch.Tensor:
        """Local affine frames, which is how kornia represents keypoints.

        DISK supplies no scale or orientation, so the frames are axis-aligned at
        a fixed radius. LightGlue's positional encoding uses the *centres*; the
        frame geometry beyond that is not information DISK provides, and
        inventing an orientation here would be fabricating a measurement.
        """
        kornia = _require_kornia()
        points = torch.from_numpy(detection.points).to(self._torch_device)[None]
        scales = torch.full(
            (1, detection.count, 1, 1),
            float(self.cfg.orb.patch_size) / 2.0,
            device=self._torch_device,
        )
        # Batched ``(B, N, 2, 3)``. kornia's LAF check requires the batch
        # dimension even though ``LightGlueMatcher`` handles a single pair, so
        # indexing it away here raised a ShapeError inside ``get_laf_center``.
        lafs: torch.Tensor = kornia.feature.laf_from_center_scale_ori(points, scales)
        return lafs

    def _empty(self, left: Detection, right: Detection, why: str) -> Correspondences:
        _log.info("no correspondences: %s", why)
        return Correspondences(
            src=np.zeros((0, 2), dtype=np.float32),
            dst=np.zeros((0, 2), dtype=np.float32),
            src_index=np.zeros(0, dtype=np.int32),
            dst_index=np.zeros(0, dtype=np.int32),
            distances=np.zeros(0, dtype=np.float32),
            evidence=MatchEvidence(
                matcher=self.name,
                raw=min(left.count, right.count),
                ratio_passed=0,
                good=0,
                distinct_left=0,
                distinct_right=0,
                mutual_nn=True,
                ratio=self.cfg.nn_ratio,
            ),
        )
