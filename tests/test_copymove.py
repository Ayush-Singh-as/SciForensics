"""Copy-move detection: multi-region output and the four defects around it.

Bug 6 is the headline -- the legacy detector fit one global transform over all
self-matches, so ``regions`` could never hold more than one entry -- but three
smaller defects lived in the same code path, and each has a test here that fails
against the legacy behaviour:

* ``test_vertical_clone_is_found`` covers the ``qx < tx`` direction hack, which
  discarded every correspondence of a purely vertical paste.
* ``test_offsets_all_point_the_same_way`` covers the index-canonicalisation sign
  split, which halved a single clone's support across two clusters.
* ``test_opposite_angles_cluster_together`` covers the 180-degree rotation wrap.

The end-to-end assertions are deliberately loose on *counts* and tight on
*content*. Over-segmenting one clone into two regions is a known, documented
limitation (see :func:`detect_copy_move`), and pinning exact region counts on
synthetic texture would encode ORB's response ordering as if it were a
specification. What the tests pin instead is that the reported geometry describes
the manipulation that was actually applied.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sciforensics.config import ClusterConfig, LocalMatchConfig, Settings
from sciforensics.local_match.copymove import (
    CopyMoveDetection,
    canonical_orientation,
    cluster_offsets,
    detect_copy_move,
    offset_features,
    orient_correspondences,
)
from sciforensics.local_match.geometry import distinct_count
from sciforensics.local_match.keypoints import Detection, OrbDetector
from sciforensics.local_match.matcher import Correspondences
from sciforensics.types import MatchEvidence
from tests.helpers import SEED, clone_patch, localised_texture, textured_image

# Analysis-space diagonal of the default 320x400 fixture, used wherever a test
# calls `offset_features` directly.
DIAGONAL = float(np.hypot(400.0, 320.0))


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _detector(cfg: Settings) -> OrbDetector:
    """ORB with the region-of-interest stage switched off.

    The fixtures are textured edge to edge, so DoG proposes one region covering
    the whole frame and the ROI stage can only add noise to what these tests are
    actually about. :mod:`test_keypoints` covers ROI behaviour on its own.
    """
    local: LocalMatchConfig = cfg.local_match.model_copy(
        update={"roi": cfg.local_match.roi.model_copy(update={"enabled": False})}
    )
    return OrbDetector(local)


def _run(image: np.ndarray, cfg: Settings, *, scale: float = 1.0) -> CopyMoveDetection:
    detection = _detector(cfg).detect(image)
    return detect_copy_move(
        detection,
        cfg.copy_move,
        affine=cfg.geometry.affine,
        shape=image.shape[:2],
        scale=scale,
    )


def _synthetic(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    sizes: np.ndarray | None = None,
    angles: np.ndarray | None = None,
) -> tuple[Correspondences, Detection]:
    """Hand-built correspondences plus the Detection their indices point into.

    Feature-level tests need to state a geometry exactly. Deriving one from a real
    image would mean asserting against whatever ORB happened to find, which is the
    opposite of a unit test.
    """
    src = np.asarray(src, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    n = len(src)
    index = np.arange(n, dtype=np.int32)

    detection = Detection(
        points=np.vstack([src, dst]),
        descriptors=None,
        sizes=np.full(2 * n, 31.0, np.float32) if sizes is None else np.asarray(sizes, np.float32),
        responses=np.full(2 * n, 0.01, np.float32),
        angles=np.zeros(2 * n, np.float32) if angles is None else np.asarray(angles, np.float32),
        detector="synthetic",
        enhancement_scale=1.0,
        detected=2 * n,
    )
    matches = Correspondences(
        src=src,
        dst=dst,
        src_index=index,
        dst_index=index + n,
        distances=np.zeros(n, np.float32),
        evidence=MatchEvidence(
            matcher="synthetic",
            raw=n,
            ratio_passed=n,
            good=n,
            # Measured from the arrays, not asserted: a builder that hardcoded
            # ``n`` here would silently contradict its own data on any fixture
            # with repeated points, which is precisely the many-to-one case these
            # counts exist to expose.
            distinct_left=distinct_count(src),
            distinct_right=distinct_count(dst),
            mutual_nn=True,
            ratio=0.8,
        ),
    )
    return matches, detection


def _weights(cfg: Settings, weights: tuple[float, float, float, float]) -> ClusterConfig:
    return cfg.copy_move.cluster.model_copy(update={"feature_weights": weights})


def _pairs(matches: Correspondences) -> set[tuple[int, int]]:
    """Unordered index pairs, so a reoriented set can be compared to its input."""
    return {
        (int(min(a, b)), int(max(a, b)))
        for a, b in zip(matches.src_index, matches.dst_index, strict=True)
    }


# ---------------------------------------------------------------------------
# canonical orientation -- the sign-split defect
# ---------------------------------------------------------------------------
def test_offsets_all_point_the_same_way() -> None:
    """A clone whose correspondences arrive in mixed directions is unified.

    This is the defect that DBSCAN cannot see past: half the offsets at ``+d`` and
    half at ``-d`` are two clusters by any honest metric, each with half the
    support, which is enough to push both below ``min_cluster_inliers``.
    """
    offset = np.array([40.0, -25.0], dtype=np.float32)
    base = np.array([[100.0, 200.0], [110.0, 205.0], [120.0, 210.0], [130.0, 215.0]], np.float32)

    # Rows 0 and 2 stated forwards, rows 1 and 3 backwards -- the interleaving ORB's
    # response ordering actually produces.
    src = np.vstack([base[0], base[1] + offset, base[2], base[3] + offset])
    dst = np.vstack([base[0] + offset, base[1], base[2] + offset, base[3]])

    matches, _ = _synthetic(src, dst)
    oriented = orient_correspondences(matches)

    offsets = oriented.dst - oriented.src
    assert np.allclose(offsets, offsets[0]), "every offset should now point one way"
    assert offsets[0][0] > 0.0, "canonical direction is the positive half-plane"


def test_orientation_preserves_the_pair_set() -> None:
    """Reorienting reverses correspondences; it must never add or drop one."""
    rng = np.random.default_rng(SEED)
    src = rng.uniform(0.0, 300.0, size=(30, 2)).astype(np.float32)
    dst = rng.uniform(0.0, 300.0, size=(30, 2)).astype(np.float32)

    matches, _ = _synthetic(src, dst)
    oriented = orient_correspondences(matches)

    assert len(oriented) == len(matches)
    assert _pairs(oriented) == _pairs(matches)


def test_orientation_is_idempotent() -> None:
    """Once canonical, reorienting changes nothing -- so the fix cannot oscillate."""
    rng = np.random.default_rng(SEED)
    src = rng.uniform(0.0, 300.0, size=(40, 2)).astype(np.float32)
    dst = rng.uniform(0.0, 300.0, size=(40, 2)).astype(np.float32)

    once = orient_correspondences(_synthetic(src, dst)[0])
    twice = orient_correspondences(once)

    assert np.array_equal(once.src, twice.src)
    assert np.array_equal(once.dst, twice.dst)


def test_purely_vertical_offsets_survive_orientation() -> None:
    """``dx == 0`` gets a tiebreak, not a coin flip.

    The legacy ``qx < tx`` test rejected every correspondence of a vertical clone
    because no pair satisfies a strict inequality when both x-coordinates are
    equal. Lexicographic ordering resolves the tie on ``dy`` instead of discarding
    the evidence.
    """
    src = np.array([[100.0, 50.0], [100.0, 220.0], [140.0, 60.0]], np.float32)
    dst = np.array([[100.0, 220.0], [100.0, 50.0], [140.0, 230.0]], np.float32)

    swap = canonical_orientation(src, dst)
    assert swap.tolist() == [False, True, False]

    oriented = orient_correspondences(_synthetic(src, dst)[0])
    dy = (oriented.dst - oriented.src)[:, 1]
    assert (dy > 0).all(), "vertical offsets should all point downward, none dropped"


def test_orientation_recomputes_distinct_counts() -> None:
    """Swapping moves points between sides, so carried-over counts would lie.

    ``distinct_left``/``distinct_right`` exist to expose many-to-one collapse
    (bug 4). Here one point on the left matches three on the right, and after
    reorientation two of those correspondences reverse -- so the collapse now sits
    on the *right*. A stale count would report the asymmetry backwards.
    """
    hub = np.array([200.0, 200.0], np.float32)
    src = np.tile(hub, (3, 1))
    dst = np.array([[150.0, 200.0], [140.0, 200.0], [130.0, 200.0]], np.float32)

    matches, _ = _synthetic(src, dst)
    assert matches.evidence.distinct_left == 1
    assert matches.evidence.distinct_right == 3

    # Every offset points left, so every correspondence reverses and the collapse
    # moves wholesale from one side to the other.
    oriented = orient_correspondences(matches)
    assert oriented.evidence.distinct_left == 3
    assert oriented.evidence.distinct_right == 1
    # The upstream funnel is a property of the matcher, not of reorientation.
    assert oriented.evidence.good == matches.evidence.good


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def test_translation_is_normalised_by_the_diagonal(cfg: Settings) -> None:
    """``eps`` is a fraction of image size, so it means the same at any resolution."""
    src = np.array([[10.0, 20.0], [30.0, 40.0]], np.float32)
    dst = src + np.array([32.0, 40.0], np.float32)
    matches, detection = _synthetic(src, dst)

    features = offset_features(matches, detection, cfg.copy_move.cluster, diagonal=DIAGONAL)

    weights = cfg.copy_move.cluster.feature_weights
    assert features[:, 0] == pytest.approx(weights[0] * 32.0 / DIAGONAL)
    assert features[:, 1] == pytest.approx(weights[1] * 40.0 / DIAGONAL)


def test_a_zero_diagonal_does_not_divide_by_zero(cfg: Settings) -> None:
    """A degenerate shape must not produce ``inf`` features and poison DBSCAN."""
    src = np.array([[0.0, 0.0], [1.0, 1.0]], np.float32)
    matches, detection = _synthetic(src, src + 5.0)

    features = offset_features(matches, detection, cfg.copy_move.cluster, diagonal=0.0)

    assert np.isfinite(features).all()


def test_opposite_angles_cluster_together(cfg: Settings) -> None:
    """A 179-degree and a -179-degree difference are 2 degrees apart, not 358.

    The 180-degree rotation wrap. ``cv2.KeyPoint.angle`` lives in ``[0, 360)``, so
    a *difference* of two of them spans ``(-360, 360)`` and a true 180-degree clone
    straddles the discontinuity: whichever lobe holds the numerically smaller angle
    decides the sign, and canonical orientation is geometric, so both signs occur
    within one clone. Row 0 below is ``180 - 1`` and row 1 is ``1 - 180``. As plain
    scalars they differ by 358 -- vastly beyond any usable ``eps`` -- so a one-click
    rotation landed exactly where the clustering shattered. The ``(cos, sin)``
    encoding makes the distance ``2 w sin(dtheta / 2)``, continuous through the wrap.
    """
    src = np.array([[100.0, 100.0], [101.0, 101.0]], np.float32)
    dst = src + np.array([50.0, 50.0], np.float32)

    # (src0, src1, dst0, dst1): row 0 measures +179 degrees, row 1 measures -179.
    angles = np.array([1.0, 180.0, 180.0, 1.0], np.float32)
    matches, detection = _synthetic(src, dst, angles=angles)

    features = offset_features(matches, detection, cfg.copy_move.cluster, diagonal=DIAGONAL)

    separation = float(np.linalg.norm(features[0] - features[1]))
    assert separation < cfg.copy_move.cluster.eps, (
        f"+179 and -179 degrees sit {separation:.4f} apart, beyond eps="
        f"{cfg.copy_move.cluster.eps}; the angle is being clustered as a scalar"
    )


def test_different_rotations_stay_distinguishable(cfg: Settings) -> None:
    """The wrap fix must not flatten genuinely different rotations together."""
    src = np.array([[100.0, 100.0], [101.0, 101.0]], np.float32)
    dst = src + np.array([50.0, 50.0], np.float32)

    angles = np.array([0.0, 0.0, 0.0, 180.0], np.float32)
    matches, detection = _synthetic(src, dst, angles=angles)

    features = offset_features(matches, detection, cfg.copy_move.cluster, diagonal=DIAGONAL)

    separation = float(np.linalg.norm(features[0] - features[1]))
    assert separation > cfg.copy_move.cluster.eps


def test_unmeasured_angles_do_not_invent_a_rotation(cfg: Settings) -> None:
    """A detector reporting ``-1`` must land at "no information", not "no rotation".

    Both rows below have unmeasured angles, so both sit at the same point in the
    angular columns -- but that point is reached by an explicit branch rather than
    by treating ``-1`` as a degree value, which would place them 0 degrees apart
    while claiming a measurement was made.
    """
    src = np.array([[100.0, 100.0], [101.0, 101.0]], np.float32)
    dst = src + np.array([50.0, 50.0], np.float32)

    angles = np.array([-1.0, -1.0, -1.0, -1.0], np.float32)
    matches, detection = _synthetic(src, dst, angles=angles)

    features = offset_features(matches, detection, cfg.copy_move.cluster, diagonal=DIAGONAL)
    weights = cfg.copy_move.cluster.feature_weights

    assert features[:, 3] == pytest.approx(weights[3])  # cos(0)
    assert features[:, 4] == pytest.approx(0.0)  # sin(0)


def test_log_scale_is_symmetric_in_the_ratio(cfg: Settings) -> None:
    """Doubling and halving are equidistant from unity; a plain ratio is not."""
    src = np.array([[100.0, 100.0], [200.0, 100.0]], np.float32)
    dst = src + np.array([40.0, 40.0], np.float32)

    # Rows are (src, src, dst, dst): row 0 grows 2x, row 1 shrinks 2x.
    sizes = np.array([16.0, 32.0, 32.0, 16.0], np.float32)
    matches, detection = _synthetic(src, dst, sizes=sizes)

    cluster = _weights(cfg, (1.0, 1.0, 1.0, 0.0))
    features = offset_features(matches, detection, cluster, diagonal=DIAGONAL)

    assert features[0, 2] == pytest.approx(-features[1, 2])
    assert abs(features[0, 2]) == pytest.approx(np.log(2.0))


def test_zero_keypoint_size_does_not_produce_negative_infinity(cfg: Settings) -> None:
    """``log(0)`` would make one bad row poison the entire feature matrix."""
    src = np.array([[100.0, 100.0], [200.0, 100.0]], np.float32)
    matches, detection = _synthetic(src, src + 40.0, sizes=np.zeros(4, np.float32))

    cluster = _weights(cfg, (1.0, 1.0, 1.0, 0.0))
    features = offset_features(matches, detection, cluster, diagonal=DIAGONAL)

    assert np.isfinite(features).all()


def test_weights_scale_each_axis_independently(cfg: Settings) -> None:
    """Every configured weight reaches the feature it names.

    A weight that silently did nothing would be exactly the dead-config defect
    bug 10 names, and the ``log_scale`` column shipping at ``0.0`` makes that
    easy to miss -- so the mapping is asserted rather than assumed.
    """
    src = np.array([[100.0, 100.0], [150.0, 150.0]], np.float32)
    dst = src + np.array([30.0, 60.0], np.float32)
    sizes = np.array([16.0, 16.0, 32.0, 32.0], np.float32)
    # 45 degrees so that neither the cos nor the sin column sits at zero, which
    # would make the non-triviality check below vacuous for that column.
    angles = np.array([0.0, 0.0, 45.0, 45.0], np.float32)
    matches, detection = _synthetic(src, dst, sizes=sizes, angles=angles)

    unit = offset_features(
        matches, detection, _weights(cfg, (1.0, 1.0, 1.0, 1.0)), diagonal=DIAGONAL
    )
    doubled = offset_features(
        matches, detection, _weights(cfg, (2.0, 2.0, 2.0, 2.0)), diagonal=DIAGONAL
    )

    assert doubled == pytest.approx(2.0 * unit)
    assert (np.abs(unit).max(axis=0) > 1e-6).all(), "a column at zero proves nothing"


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def test_two_offsets_give_two_clusters(cfg: Settings) -> None:
    rng = np.random.default_rng(SEED)
    anchors = rng.uniform(20.0, 280.0, size=(20, 2)).astype(np.float32)
    first = np.array([60.0, 20.0], np.float32)
    second = np.array([-15.0, 90.0], np.float32)

    src = np.vstack([anchors, anchors])
    dst = np.vstack([anchors + first, anchors + second])
    matches, detection = _synthetic(src, dst)

    features = offset_features(matches, detection, cfg.copy_move.cluster, diagonal=DIAGONAL)
    labels = cluster_offsets(features, cfg.copy_move.cluster)

    assert len({int(label) for label in labels if label >= 0}) == 2
    assert len(set(labels[:20].tolist())) == 1
    assert len(set(labels[20:].tolist())) == 1
    assert labels[0] != labels[20]


def test_scattered_offsets_are_noise(cfg: Settings) -> None:
    """No shared transform means no cluster -- the precision half of the design."""
    rng = np.random.default_rng(SEED)
    src = rng.uniform(20.0, 300.0, size=(40, 2)).astype(np.float32)
    dst = rng.uniform(20.0, 300.0, size=(40, 2)).astype(np.float32)
    matches, detection = _synthetic(src, dst)

    features = offset_features(matches, detection, cfg.copy_move.cluster, diagonal=DIAGONAL)
    labels = cluster_offsets(features, cfg.copy_move.cluster)

    assert (labels == -1).all()


def test_too_few_correspondences_cluster_to_nothing(cfg: Settings) -> None:
    """Below ``min_samples`` there is nothing to cluster, and no crash either."""
    n = cfg.copy_move.cluster.min_samples - 1
    src = np.arange(2 * n, dtype=np.float32).reshape(n, 2) * 10.0
    matches, detection = _synthetic(src, src + 50.0)

    features = offset_features(matches, detection, cfg.copy_move.cluster, diagonal=DIAGONAL)
    labels = cluster_offsets(features, cfg.copy_move.cluster)

    assert labels.shape == (n,)
    assert (labels == -1).all()


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def _covers(box: tuple[int, int, int, int], point: tuple[int, int], *, slack: int = 30) -> bool:
    x, y, w, h = box
    px, py = point
    return (x - slack) <= px <= (x + w + slack) and (y - slack) <= py <= (y + h + slack)


def test_single_clone_is_localised(cfg: Settings) -> None:
    """The core case: one pasted patch, found, with both lobes bounded."""
    image = clone_patch(textured_image(), (40, 40), (190, 210))
    result = _run(image, cfg)

    assert result.evidence.detected
    assert result.region_count >= 1

    region = result.evidence.regions[0]
    boxes = {region.source_box, region.target_box}
    assert any(_covers(box, (85, 85)) for box in boxes), "source lobe not bounded"
    assert any(_covers(box, (235, 255)) for box in boxes), "target lobe not bounded"
    assert region.offset == pytest.approx((150.0, 170.0), abs=8.0)
    assert region.scale == pytest.approx(1.0, abs=0.1)
    assert region.rotation_deg == pytest.approx(0.0, abs=3.0)
    assert not region.flip


def test_vertical_clone_is_found(cfg: Settings) -> None:
    """Bug 6's companion defect: ``qx < tx`` made a vertical paste invisible.

    Every correspondence of this clone has ``dx == 0``, so the legacy strict
    inequality kept none of them and the detector reported a clean image.
    """
    image = clone_patch(textured_image(), (40, 40), (40, 210))
    result = _run(image, cfg)

    assert result.evidence.detected, "a purely vertical clone must not be discarded"
    region = result.evidence.regions[0]
    assert region.offset[0] == pytest.approx(0.0, abs=6.0)
    assert region.offset[1] == pytest.approx(170.0, abs=8.0)


def test_two_clones_are_reported_separately(cfg: Settings) -> None:
    """**The bug 6 regression.** One global fit could only ever return one region.

    The two pastes have deliberately *different* displacements -- ``(0, 170)`` and
    ``(-100, 170)``. Two patches sharing a displacement genuinely are one affine
    transform and should merge; it is independent transforms that the legacy
    single-fit design could not represent, and whose minority it actively
    suppressed as outliers.
    """
    image = clone_patch(textured_image(), (30, 30), (30, 200), size=70)
    image = clone_patch(image, (250, 40), (150, 210), size=70)
    result = _run(image, cfg)

    assert result.region_count >= 2, (
        f"expected both clones, got {result.region_count} region(s) from "
        f"{result.evidence.cluster_count} cluster(s)"
    )

    offsets = [region.offset for region in result.evidence.regions]
    assert any(abs(dx) < 12.0 and abs(dy - 170.0) < 12.0 for dx, dy in offsets), (
        f"vertical clone missing from {offsets}"
    )
    assert any(abs(abs(dx) - 100.0) < 14.0 and abs(abs(dy) - 170.0) < 14.0 for dx, dy in offsets), (
        f"diagonal clone missing from {offsets}"
    )


def test_unmanipulated_image_reports_no_regions(cfg: Settings) -> None:
    """The negative control. Without one, recall numbers mean nothing."""
    result = _run(textured_image(), cfg)

    assert not result.evidence.detected
    assert result.region_count == 0
    assert result.masks == ()
    assert (result.labels == -1).all()


def test_repeated_similar_texture_is_not_a_clone(cfg: Settings) -> None:
    """Two clusters of near-identical blobs are a hard negative, not a forgery.

    :func:`localised_texture` draws 14 random disks around each centre from the
    same distribution, so its two clusters genuinely resemble each other without
    either being a copy. This is the false-positive mode that matters in practice
    -- a western blot is repeated bands on empty film -- and it is asserted as a
    *bound* rather than as zero, because at the shipped
    ``min_cluster_inliers`` this fixture does yield weak regions.
    """
    result = _run(localised_texture(), cfg)

    for region in result.evidence.regions:
        assert region.inlier_count < 20, (
            "coincidental texture should not produce strongly-supported regions; "
            f"got {region.inlier_count} inliers"
        )


def test_rotated_clone_recovers_its_angle(cfg: Settings) -> None:
    """A 180-degree paste is reported as such rather than shattered by the wrap.

    The sign is negated relative to ``cv2.getRotationMatrix2D`` -- see
    :func:`tests.helpers.clone_patch` -- and 180 is symmetric under negation,
    which is exactly why it is the interesting angle here.
    """
    image = clone_patch(textured_image(), (40, 40), (200, 200), size=100, rotation_deg=180.0)
    result = _run(image, cfg)

    assert result.evidence.detected
    angles = [abs(region.rotation_deg) for region in result.evidence.regions]
    assert any(angle > 170.0 for angle in angles), (
        f"a 180-degree clone should be reported near +/-180, got {angles}"
    )


def test_rescaled_clone_survives_octave_quantised_keypoint_sizes(cfg: Settings) -> None:
    """The shipped weights find a 1.25x clone; weighting ``log_scale`` loses it.

    ORB's keypoint ``size`` is octave-quantised, so ``log(size ratio)`` can only be
    a multiple of ``log(orb.scale_factor) = 0.182`` -- already 2.3x the shipped
    ``eps`` of 0.08. On this image the correspondences at the true offset spread
    over four distinct octave ratios, a weighted span of 0.365 at ``w = 0.5``, and
    DBSCAN sends every one of them to noise.

    Both halves are asserted. The first is the behaviour users get; the second
    records *why* the default is what it is, so that raising the weight for a
    continuous-scale detector in stage B3 is a deliberate act with a failing test
    to explain the trade-off.
    """
    image = clone_patch(textured_image(), (40, 40), (200, 200), size=100, scale=1.25)
    detection = _detector(cfg).detect(image)
    shape = image.shape[:2]

    assert cfg.copy_move.cluster.feature_weights[2] == 0.0, "default should not weight log_scale"
    shipped = detect_copy_move(detection, cfg.copy_move, affine=cfg.geometry.affine, shape=shape)
    assert shipped.evidence.detected, "a 1.25x clone must be found with the shipped weights"

    weighted = cfg.copy_move.model_copy(update={"cluster": _weights(cfg, (1.0, 1.0, 0.5, 0.5))})
    degraded = detect_copy_move(detection, weighted, affine=cfg.geometry.affine, shape=shape)
    assert degraded.region_count < shipped.region_count, (
        "weighting an octave-quantised feature against a continuous eps should cost "
        "recall; if this now passes, re-derive the default in configs/default.yaml"
    )


def test_boxes_and_offsets_are_reported_in_original_pixels(cfg: Settings) -> None:
    """Analysis space is an implementation detail; the report is not.

    Halving ``scale`` means analysis ran at half resolution, so every reported
    length doubles and every area quadruples. Masks are exempt: they are rasters
    in analysis space, resized once by the caller at draw time.
    """
    image = clone_patch(textured_image(), (40, 40), (190, 210))
    full = _run(image, cfg)
    half = _run(image, cfg, scale=0.5)

    assert full.evidence.detected and half.evidence.detected
    a, b = full.evidence.regions[0], half.evidence.regions[0]

    assert b.offset == pytest.approx((a.offset[0] * 2.0, a.offset[1] * 2.0), rel=0.02)
    assert b.source_box == pytest.approx(tuple(v * 2 for v in a.source_box), abs=2)
    assert b.area_px == pytest.approx(a.area_px * 4, rel=0.02)
    # Geometry is frame-invariant: a clone is the same shape at any resolution.
    assert b.scale == pytest.approx(a.scale, rel=0.02)
    assert b.rotation_deg == pytest.approx(a.rotation_deg, abs=0.5)
    assert half.masks[0].shape == full.masks[0].shape == image.shape[:2]


def test_masks_are_parallel_to_regions_and_cover_both_lobes(cfg: Settings) -> None:
    image = clone_patch(textured_image(), (40, 40), (190, 210))
    result = _run(image, cfg)

    assert len(result.masks) == result.region_count
    for mask, region in zip(result.masks, result.evidence.regions, strict=True):
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) <= {0, 255}

        components, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(components) >= 2, "the mask should cover both lobes, not just one"
        assert np.count_nonzero(mask) > 0
        # The reported area is the union of both lobes, converted to original px.
        assert region.area_px == pytest.approx(np.count_nonzero(mask), rel=0.02)


def test_labels_index_reported_regions(cfg: Settings) -> None:
    """Labels address ``evidence.regions``, not DBSCAN's own cluster ids.

    Clusters that failed verification fold back to ``-1``, so an overlay drawn by
    label cannot display a region the report does not mention.
    """
    image = clone_patch(textured_image(), (40, 40), (190, 210))
    result = _run(image, cfg)

    assert len(result.labels) == len(result.correspondences)
    assigned = {int(label) for label in result.labels if label >= 0}
    assert assigned == set(range(result.region_count))

    for index, region in enumerate(result.evidence.regions):
        assert int((result.labels == index).sum()) == region.inlier_count


def test_regions_are_ordered_by_strength(cfg: Settings) -> None:
    image = clone_patch(textured_image(), (30, 30), (30, 200), size=70)
    image = clone_patch(image, (250, 40), (150, 210), size=70)
    result = _run(image, cfg)

    keys = [(-r.inlier_count, -r.area_px) for r in result.evidence.regions]
    assert keys == sorted(keys)


def test_max_regions_truncates_to_the_strongest(cfg: Settings) -> None:
    image = clone_patch(textured_image(), (30, 30), (30, 200), size=70)
    image = clone_patch(image, (250, 40), (150, 210), size=70)
    detection = _detector(cfg).detect(image)

    uncapped = detect_copy_move(
        detection, cfg.copy_move, affine=cfg.geometry.affine, shape=image.shape[:2]
    )
    assert uncapped.region_count >= 2, "fixture must produce enough regions to truncate"

    capped_cfg = cfg.copy_move.model_copy(update={"max_regions": 1})
    capped = detect_copy_move(
        detection, capped_cfg, affine=cfg.geometry.affine, shape=image.shape[:2]
    )

    assert capped.region_count == 1
    assert len(capped.masks) == 1
    assert capped.evidence.regions[0] == uncapped.evidence.regions[0]
    # The candidate count is unaffected: truncation is a reporting cap, not a
    # change to what was searched.
    assert capped.evidence.cluster_count == uncapped.evidence.cluster_count


def test_evidence_is_populated_even_with_no_regions(cfg: Settings) -> None:
    """ "Nothing found" and "nothing to look at" are different findings."""
    result = _run(textured_image(), cfg)

    assert not result.evidence.detected
    assert result.evidence.keypoints > 0
    assert result.evidence.self_matches >= 0
    assert result.shape == (320, 400)
    assert result.scale == 1.0


def test_an_empty_detection_is_handled(cfg: Settings) -> None:
    """A flat image yields no keypoints; that must be a clean report, not a crash."""
    flat = np.full((120, 160), 128, dtype=np.uint8)
    result = _run(flat, cfg)

    assert not result.evidence.detected
    assert result.evidence.self_matches == 0
    assert result.region_count == 0
    assert result.labels.shape == (0,)


def test_detection_is_deterministic(cfg: Settings) -> None:
    """Same input, same evidence. A flaky forensic result is not a result."""
    image = clone_patch(textured_image(), (40, 40), (190, 210))

    first = _run(image, cfg)
    second = _run(image, cfg)

    assert first.evidence == second.evidence
    assert np.array_equal(first.labels, second.labels)
    for a, b in zip(first.masks, second.masks, strict=True):
        assert np.array_equal(a, b)
