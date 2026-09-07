"""Multi-region copy-move forgery detection.

**Bug 6 — the prototype could only ever report one cloned region.** It collected
self-matches, fit a *single* ``estimateAffinePartial2D`` over all of them, and
returned a list called ``regions``. The list could never hold more than one entry,
because one global transform is one region by construction. Worse, a figure with
two independent clones produces two contradictory offsets, and a single robust fit
resolves that by treating one of them as outliers -- so the second clone was not
merely unreported, it was actively suppressed by the estimator.

The fix is to find the *modes* before fitting anything. Correspondences belonging
to one clone share a transform, so they land in one tight cluster of the
``(dx, dy, log s, theta)`` space; independent clones land in different clusters.
DBSCAN over that space recovers each mode, and each mode then gets its own
MAGSAC++ fit. Region count becomes an output rather than a constant.

Three further defects sit in the same code path.

**The ``qx < tx`` direction hack.** Self-matching is symmetric -- ``(i, j)`` and
``(j, i)`` are one clone relationship seen from both ends -- so duplicates must be
collapsed. The legacy code did that by keeping only pairs whose query point was
left of its train point. For a clone offset purely vertically, ``qx == tx`` for
every correspondence and the test kept none of them: a vertical copy-paste, the
single most natural thing to do in an image editor, was invisible. Deduplication
now happens on index order in :func:`~sciforensics.local_match.self_match`, which
has no geometric blind spot.

**Index canonicalisation splits a clone in half (found here, not in the audit).**
Deduplicating on ``i < j`` is correct for *removing duplicates* but it does not
give the surviving offsets a consistent direction. ORB orders keypoints by
response, and two lobes of the same content have near-identical response profiles,
so their indices interleave: some correspondences end up with the lower index in
lobe A and some in lobe B. The offset ``p_j - p_i`` is then ``+d`` for the first
group and ``-d`` for the second, and DBSCAN -- correctly, given what it was handed
-- reports two clusters. One clone becomes two regions, each with roughly half the
support, which is enough to push both under ``min_cluster_inliers`` and lose the
clone entirely. :func:`canonical_orientation` fixes the direction geometrically
before clustering. It is the legacy hack's intent done properly: lexicographic on
``(dx, dy)``, so the ``dx == 0`` case that defeated ``qx < tx`` falls through to a
tiebreak instead of to a coin flip.

**Rotation wraps at 180 degrees.** ``theta`` is an angular difference, so clustering
it as a plain scalar puts ``+179`` and ``-179`` degrees 358 degrees apart when they
are 2 degrees apart. That is not a hypothetical: a 180-degree rotation is a
one-click operation and it lands exactly on the discontinuity, where ORB's own
angle noise straddles the wrap and shatters the cluster. :func:`offset_features`
encodes the angle as ``(cos theta, sin theta)`` scaled by its configured weight,
which makes the distance ``2 w sin(dtheta / 2)`` -- equal to ``w * dtheta`` for
small differences, so the configured ``eps`` keeps its meaning, and continuous
everywhere, so the wrap cannot fragment anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from sciforensics.config import AffineConfig, ClusterConfig, CopyMoveConfig
from sciforensics.local_match.geometry import (
    decompose_affine,
    distinct_count,
    estimate_affine,
    reprojection_errors,
    rescale_box,
)
from sciforensics.local_match.keypoints import Detection
from sciforensics.local_match.matcher import Correspondences, self_match
from sciforensics.runtime import get_logger
from sciforensics.types import BBox, CopyMoveEvidence, CopyMoveRegion, MatchEvidence

__all__ = [
    "CopyMoveDetection",
    "canonical_orientation",
    "cluster_offsets",
    "detect_copy_move",
    "offset_features",
    "orient_correspondences",
]

_log = get_logger(__name__)

#: Floor on keypoint size before taking a log ratio. Real detectors never report
#: zero -- ORB's smallest patch is 31 px -- but a hand-built ``Detection`` can,
#: and ``log(0)`` would poison the whole feature matrix with ``-inf``.
_MIN_SIZE = 1e-6


@dataclass(frozen=True)
class CopyMoveDetection:
    """Reportable evidence plus the arrays an overlay needs to draw it.

    The split mirrors the rest of the pipeline: :attr:`evidence` is the pydantic
    record that goes into the report and the JSON, and the numpy members are the
    working data the renderer consumes and then discards. Keeping masks out of the
    evidence model is what lets a result be serialised without deciding where its
    mask files live -- the report layer writes them and fills in
    :attr:`~sciforensics.types.CopyMoveRegion.mask_path`.
    """

    evidence: CopyMoveEvidence
    #: Self-matches, reoriented so every offset points the same way. This is the
    #: array :attr:`labels` indexes, *not* the one ``self_match`` returned.
    correspondences: Correspondences
    #: Per-correspondence region index, ``-1`` for a correspondence that belongs
    #: to no reported region. Indexes ``evidence.regions``, not DBSCAN's own
    #: labels: clusters that failed verification or fell outside ``max_regions``
    #: are folded back into ``-1``, so drawing by label cannot show a region that
    #: the report does not mention.
    labels: np.ndarray
    #: One mask per reported region, parallel to ``evidence.regions``, covering
    #: *both* lobes. ``uint8`` in ``{0, 255}``, in **analysis** space -- masks are
    #: rasters, so unlike the boxes they are not converted to original pixels;
    #: the caller resizes once, at draw time.
    masks: tuple[np.ndarray, ...]
    #: Analysis-space ``(height, width)`` the masks are drawn on.
    shape: tuple[int, int]
    #: Analysis-space pixels per original pixel, as used to convert the boxes.
    scale: float

    @property
    def region_count(self) -> int:
        return len(self.evidence.regions)


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------
def canonical_orientation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Which correspondences point the wrong way and must be swapped.

    Returns a boolean mask over correspondences. A pair is "wrong way" when its
    offset falls in the negative half-plane under lexicographic ordering on
    ``(dx, dy)``; swapping those puts every offset of a given clone on the same
    side, which is what makes them cluster.

    The ordering is lexicographic rather than a bare ``dx < 0`` precisely because
    of the legacy bug: ``dx`` alone leaves purely vertical offsets undecided, and
    whatever the comparison then returns is a property of floating-point tie
    behaviour rather than of the image. With the ``dy`` tiebreak, every non-zero
    offset has exactly one canonical direction. The zero offset has none, but
    ``copy_move.min_spatial_separation`` has already excluded it -- a keypoint
    matching itself is not a clone.

    Note the direction is *arbitrary* (down-and-right by construction), not
    forensic: copy-move gives no way to tell an original from its paste. It only
    has to be consistent.
    """
    offsets = np.asarray(dst, dtype=np.float64).reshape(-1, 2) - np.asarray(
        src, dtype=np.float64
    ).reshape(-1, 2)
    dx, dy = offsets[:, 0], offsets[:, 1]
    swap: np.ndarray = (dx < 0.0) | ((dx == 0.0) & (dy < 0.0))
    return swap


def orient_correspondences(matches: Correspondences) -> Correspondences:
    """Reverse the correspondences :func:`canonical_orientation` marks.

    Recomputes ``distinct_left``/``distinct_right`` rather than carrying them
    over, because swapping moves points between the two sides: the counts describe
    the arrays as they now are, and a stale pair of them would misreport exactly
    the many-to-one collapse those fields exist to expose.
    """
    swap = canonical_orientation(matches.src, matches.dst)
    if not swap.any():
        return matches

    pick = swap[:, None]
    src = np.ascontiguousarray(np.where(pick, matches.dst, matches.src))
    dst = np.ascontiguousarray(np.where(pick, matches.src, matches.dst))
    src_index = np.where(swap, matches.dst_index, matches.src_index)
    dst_index = np.where(swap, matches.src_index, matches.dst_index)

    previous = matches.evidence
    return Correspondences(
        src=src,
        dst=dst,
        src_index=src_index,
        dst_index=dst_index,
        distances=matches.distances,
        evidence=MatchEvidence(
            matcher=previous.matcher,
            raw=previous.raw,
            ratio_passed=previous.ratio_passed,
            good=previous.good,
            distinct_left=distinct_count(src),
            distinct_right=distinct_count(dst),
            mutual_nn=previous.mutual_nn,
            ratio=previous.ratio,
        ),
    )


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def offset_features(
    matches: Correspondences,
    detection: Detection,
    cfg: ClusterConfig,
    *,
    diagonal: float,
) -> np.ndarray:
    """Build the clustering feature matrix, one row per correspondence.

    Five columns for four configured weights. ``feature_weights`` is
    ``(dx, dy, log_scale, theta)`` as a reader of ``default.yaml`` expects, but the
    angle occupies two columns internally -- see the module docstring for why a
    scalar angle cannot be clustered.

    Translation is normalised by the image diagonal so ``eps`` is a fraction of
    image size and means the same thing on a thumbnail and on a full-page figure.
    Scale enters as a log ratio so that doubling and halving are equidistant from
    unity, which a plain ratio would not give.

    Per-keypoint scale and rotation come from the detector's own ``sizes`` and
    ``angles``. For a losslessly pasted region those agree exactly between the two
    lobes, so the columns are near-zero and cost nothing; they earn their weight on
    a clone that was resized or rotated, where the offset alone would still cluster
    but would not distinguish two clones sharing a displacement.

    **A per-match scale or rotation estimate is only usable if its resolution is
    finer than the tolerance it is clustered under**, and for ORB the scale column
    is not. ORB's keypoint ``size`` is ``patch_size * scale_factor ** octave``, so
    a ratio between two of them is always an exact power of ``orb.scale_factor``
    and ``log_scale`` is quantised at ``log(1.2) = 0.182`` -- 2.3x the shipped
    ``eps`` of 0.08. The column can therefore only report "same octave" or "hard
    split", never "approximately equal". Measured on a 1.25x clone: the 14
    correspondences at the true offset spread over ``{-2, 0, +1, +2} x log(1.2)``,
    a weighted span of 0.365 at ``w = 0.5``, which sent every one of them to noise
    and yielded zero regions. ``default.yaml`` consequently ships
    ``feature_weights[2] = 0.0`` for ORB. The arithmetic here stays general so a
    continuous-scale detector (SIFT, or DISK/SuperPoint in stage B3) can enable it
    by config alone -- the weight is read, not ignored, so this is not a dead
    setting.

    The rotation column has no such defect: ORB's orientation is an
    intensity-centroid angle, continuous in the content, and it is load-bearing at
    its configured weight.
    """
    weights = np.asarray(cfg.feature_weights, dtype=np.float64)
    span = diagonal if diagonal > 1e-9 else 1.0

    offsets = matches.dst.astype(np.float64) - matches.src.astype(np.float64)

    sizes = np.asarray(detection.sizes, dtype=np.float64)
    src_size = np.maximum(sizes[matches.src_index], _MIN_SIZE)
    dst_size = np.maximum(sizes[matches.dst_index], _MIN_SIZE)
    log_scale = np.log(dst_size / src_size)

    angles = np.asarray(detection.angles, dtype=np.float64)
    src_angle = angles[matches.src_index]
    dst_angle = angles[matches.dst_index]
    # A detector that reports no orientation uses -1. Treating that as a zero
    # difference is the only option that does not invent a measurement, and it is
    # recorded here rather than hidden: the column then carries no information for
    # those pairs instead of carrying noise.
    measured = (src_angle >= 0.0) & (dst_angle >= 0.0)
    theta = np.where(measured, np.radians(dst_angle - src_angle), 0.0)

    features = np.empty((len(matches), 5), dtype=np.float64)
    features[:, 0] = weights[0] * offsets[:, 0] / span
    features[:, 1] = weights[1] * offsets[:, 1] / span
    features[:, 2] = weights[2] * log_scale
    features[:, 3] = weights[3] * np.cos(theta)
    features[:, 4] = weights[3] * np.sin(theta)
    return features


def cluster_offsets(features: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """DBSCAN over the feature matrix; returns labels with ``-1`` for noise.

    DBSCAN rather than k-means because the number of clones is exactly what is
    being asked, not something to supply in advance, and because most
    correspondences in a real figure are noise -- repeated texture, gel background,
    axis ticks -- which k-means would be forced to assign to some cluster.
    """
    if len(features) < cfg.min_samples:
        return np.full(len(features), -1, dtype=np.int32)
    labels = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples, metric="euclidean").fit_predict(
        features
    )
    return np.asarray(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def _lobe_mask(
    points: np.ndarray, sizes: np.ndarray, shape: tuple[int, int], close_px: int
) -> np.ndarray:
    """Rasterise one lobe of a region as a filled disk per verified keypoint.

    A convex hull of the inliers would be the other obvious choice and is what
    much of the keypoint-CMFD literature does, but it over-claims badly on a
    concave or L-shaped paste, and over-claimed area is precisely what inflates an
    IoU score against ground truth without detecting anything more. Disks plus a
    morphological close approximate the hull where inliers are dense and stay
    honest where they are sparse.

    The disk radius is the keypoint's own support radius, floored at
    ``mask_close_px`` -- reusing the closing radius rather than introducing a
    second magic number, since it already encodes "the distance at which nearby
    evidence is one region".
    """
    height, width = shape
    mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    for (x, y), size in zip(points, sizes, strict=True):
        radius = max(1, round(max(float(size) * 0.5, float(close_px))))
        cv2.circle(mask, (round(float(x)), round(float(y))), radius, 255, -1)

    if close_px > 0:
        extent = 2 * close_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extent, extent))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _mask_box(mask: np.ndarray) -> BBox:
    x, y, w, h = cv2.boundingRect(mask)
    return (int(x), int(y), int(w), int(h))


# ---------------------------------------------------------------------------
# per-cluster verification
# ---------------------------------------------------------------------------
def _build_region(
    matches: Correspondences,
    detection: Detection,
    member: np.ndarray,
    cfg: CopyMoveConfig,
    affine: AffineConfig,
    *,
    shape: tuple[int, int],
    scale: float,
) -> tuple[CopyMoveRegion, np.ndarray, np.ndarray] | None:
    """Verify one cluster and describe it, or return ``None`` if it does not hold.

    Returns ``(region, union_mask, inlier_selection)``, where the selection is a
    boolean mask over *all* correspondences so the caller can label them.

    Clustering answers "do these correspondences share a transform?" only
    approximately -- it works in a normalised feature space, on per-keypoint scale
    and angle estimates that are themselves noisy. Fitting the transform properly
    is what turns a candidate into evidence, and it is also the second filter: a
    cluster of coincidentally-similar offsets across unrelated texture will not
    survive a MAGSAC++ fit at a 4 px threshold.
    """
    src = matches.src[member]
    dst = matches.dst[member]
    if len(src) < cfg.min_cluster_inliers:
        return None

    matrix, mask = estimate_affine(src, dst, affine)
    if matrix is None or mask is None:
        return None

    inliers = np.asarray(mask, dtype=bool).ravel()
    inlier_count = int(inliers.sum())
    if inlier_count < cfg.min_cluster_inliers:
        return None

    src_in = src[inliers]
    dst_in = dst[inliers]
    # Bug 4's lesson applies per cluster: rows of a correspondence table are not
    # independent constraints. g2NN deliberately lets one keypoint match several
    # partners, so a keypoint can appear more than once here, and without this the
    # gate could be satisfied by eight rows over three locations.
    if min(distinct_count(src_in), distinct_count(dst_in)) < cfg.min_cluster_inliers:
        _log.debug(
            "cluster rejected: %d inlier rows span only %d/%d distinct locations",
            inlier_count,
            distinct_count(src_in),
            distinct_count(dst_in),
        )
        return None

    # The median rather than the affine's translation column: for a rotated clone
    # the translation is measured from the origin, not from the content, so it can
    # be thousands of pixels away from anything a reader would call the offset.
    # Computed in analysis space, where `min_spatial_separation` is defined, and
    # converted to original pixels only for the report below.
    displacement = np.median(dst_in.astype(np.float64) - src_in.astype(np.float64), axis=0)

    # An *object* matching itself is not a clone, for the same reason a keypoint
    # matching itself is not one. `min_spatial_separation` says so already, but it
    # is enforced per correspondence, and a rotationally symmetric object defeats
    # a pairwise test: a filled disc of radius r maps onto itself under a 180
    # degree rotation about its own centre, so each rim point matches the point
    # across the diameter, 2r apart. At r=17 that is 34px -- every match clears a
    # 32px floor individually while the region they form is one disc. The rim
    # displacements disagree in direction, so their median collapses (measured:
    # 18.3px, against ~34px for the individual matches), and that collapse is the
    # signature. A real clone displaces every point the same way, so its median
    # equals the individual magnitudes.
    #
    # This removes nothing the detector could otherwise find: a clone whose true
    # offset is below the floor has *every* correspondence suppressed pairwise, so
    # it never reaches clustering. The gate is implied by the one in
    # `self_match`, not an additional threshold on top of it.
    separation = float(np.hypot(*displacement))
    if separation < cfg.min_spatial_separation:
        _log.debug(
            "cluster rejected: %d inlier rows span a median displacement of only %.1fpx "
            "(min_spatial_separation=%.1f); the two lobes are the same object",
            inlier_count,
            separation,
            cfg.min_spatial_separation,
        )
        return None

    decomposition = decompose_affine(
        matrix,
        model=affine.model,
        anisotropy_tolerance=affine.anisotropy_tolerance,
        shear_tolerance_deg=affine.shear_tolerance_deg,
    )

    homogeneous = np.vstack([matrix, [0.0, 0.0, 1.0]])
    errors = reprojection_errors(homogeneous, src_in, dst_in)
    reproj_rms = float(np.sqrt(np.mean(np.square(errors))))

    selection = np.zeros(len(matches), dtype=bool)
    selection[np.flatnonzero(member)[inliers]] = True

    src_mask = _lobe_mask(
        src_in, detection.sizes[matches.src_index[selection]], shape, cfg.mask_close_px
    )
    dst_mask = _lobe_mask(
        dst_in, detection.sizes[matches.dst_index[selection]], shape, cfg.mask_close_px
    )
    union: np.ndarray = cv2.bitwise_or(src_mask, dst_mask)

    region = CopyMoveRegion(
        source_box=rescale_box(_mask_box(src_mask), scale=scale),
        target_box=rescale_box(_mask_box(dst_mask), scale=scale),
        offset=(float(displacement[0] / scale), float(displacement[1] / scale)),
        rotation_deg=decomposition.rotation_deg,
        scale=decomposition.scale,
        flip=decomposition.flip,
        inlier_count=inlier_count,
        reproj_rms=reproj_rms,
        area_px=round(float(np.count_nonzero(union)) / (scale * scale)),
    )
    return region, union, selection


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def detect_copy_move(
    detection: Detection,
    cfg: CopyMoveConfig,
    *,
    affine: AffineConfig,
    shape: tuple[int, int],
    scale: float = 1.0,
    norm: int = cv2.NORM_HAMMING,
) -> CopyMoveDetection:
    """Find every cloned region in one image.

    Follows the coordinate convention :func:`sciforensics.local_match.verify`
    establishes: matching, clustering, fitting and masking all happen in analysis
    space, and only the *reported* boxes, offsets and areas are converted back to
    original-image pixels. Numbers in a report have to live in the frame of the
    image the reader is looking at; intermediate geometry does not.

    Parameters
    ----------
    detection
        Keypoints for the image, in analysis space.
    cfg
        ``copy_move`` settings. ``cfg.enabled`` is *not* consulted here -- a
        detector that returned "nothing found" because it was switched off would
        be indistinguishable from one that looked and found nothing, which is the
        class of silent-fallback bug this refactor exists to remove. The caller
        skips the call.
    affine
        Base affine-fitting settings. ``copy_move.reproj_threshold`` overrides
        ``affine.reproj_threshold`` for the per-cluster fits, because a clone is
        held to a tighter tolerance than a cross-image match: it is the *same*
        pixels, so a loose threshold buys nothing but false positives.
    shape
        Analysis-space ``(height, width)``, used for the mask raster and for the
        diagonal that normalises the offset features.
    scale
        Analysis-space pixels per original pixel, i.e.
        ``LoadedImage.analysis_scale``.
    norm
        Descriptor norm, ``cv2.NORM_HAMMING`` for ORB.
    """
    height, width = shape
    diagonal = math.hypot(float(width), float(height))
    matches = orient_correspondences(self_match(detection, cfg, norm=norm))

    if len(matches) == 0:
        return _empty(matches, detection, shape=shape, scale=scale)

    features = offset_features(matches, detection, cfg.cluster, diagonal=diagonal)
    labels = cluster_offsets(features, cfg.cluster)
    cluster_ids = [int(label) for label in np.unique(labels) if label >= 0]
    if not cluster_ids:
        return _empty(matches, detection, shape=shape, scale=scale)

    # `model_copy` rather than a mutated dict: `cfg.reproj_threshold` is already
    # validated as a positive float by `CopyMoveConfig`, so nothing is skipped,
    # and the result stays typed as an `AffineConfig`.
    cluster_affine = affine.model_copy(update={"reproj_threshold": cfg.reproj_threshold})

    verified: list[tuple[CopyMoveRegion, np.ndarray, np.ndarray]] = []
    for cluster_id in cluster_ids:
        built = _build_region(
            matches,
            detection,
            labels == cluster_id,
            cfg,
            cluster_affine,
            shape=shape,
            scale=scale,
        )
        if built is not None:
            verified.append(built)

    # Strongest evidence first, area breaking ties, so the ordering is total and a
    # rerun cannot reshuffle a report. Truncation is logged rather than silent:
    # "8 regions" when 11 were verified is a materially different claim.
    verified.sort(key=lambda item: (-item[0].inlier_count, -item[0].area_px))
    if len(verified) > cfg.max_regions:
        _log.info(
            "%d verified clone regions exceed copy_move.max_regions=%d; reporting the strongest",
            len(verified),
            cfg.max_regions,
        )
        verified = verified[: cfg.max_regions]

    region_labels = np.full(len(matches), -1, dtype=np.int32)
    for index, (_, _, selection) in enumerate(verified):
        region_labels[selection] = index

    regions = tuple(region for region, _, _ in verified)
    return CopyMoveDetection(
        evidence=CopyMoveEvidence(
            detected=bool(regions),
            cluster_count=len(cluster_ids),
            regions=regions,
            self_matches=len(matches),
            keypoints=detection.count,
        ),
        correspondences=matches,
        labels=region_labels,
        masks=tuple(mask for _, mask, _ in verified),
        shape=shape,
        scale=scale,
    )


def _empty(
    matches: Correspondences,
    detection: Detection,
    *,
    shape: tuple[int, int],
    scale: float,
) -> CopyMoveDetection:
    """No regions, but still a full record of what was looked at.

    ``self_matches`` and ``keypoints`` are populated even here: "no clones found"
    and "there were nine keypoints to work with" are different findings, and only
    the second explains itself.
    """
    return CopyMoveDetection(
        evidence=CopyMoveEvidence(
            detected=False,
            cluster_count=0,
            regions=(),
            self_matches=len(matches),
            keypoints=detection.count,
        ),
        correspondences=matches,
        labels=np.full(len(matches), -1, dtype=np.int32),
        masks=(),
        shape=shape,
        scale=scale,
    )
