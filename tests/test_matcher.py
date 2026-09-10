"""Matching: the mutual-nearest-neighbour constraint and what it structurally prevents.

The regression at the centre of this module is bug 4. The legacy pipeline reported
"Keypoints: 2000 vs 12 / 231 good matches / 107 inliers / Strong evidence" on a
pair whose evidence was entirely a counting artefact. Two independent defects
combined to produce it -- an ROI filter that collapsed one side, and a matcher with
no symmetry constraint to notice. :mod:`test_keypoints` covers the first; this
module covers the second, and asserts that the matcher alone is sufficient to
prevent the pathology even if a collapse somehow reaches it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sciforensics.config import (
    ConfigError,
    CopyMoveConfig,
    LocalMatchConfig,
    Settings,
    load_config,
)
from sciforensics.local_match.keypoints import Detection
from sciforensics.local_match.matcher import (
    BruteForceMatcher,
    _g2nn_accept,
    build_matcher,
    keypoint_evidence,
    self_match,
)

DESCRIPTOR_BYTES = 32  # ORB: 256 bits


def _detection(points: np.ndarray, descriptors: np.ndarray) -> Detection:
    """Wrap raw arrays as a Detection, bypassing actual image detection.

    The matcher only reads ``points`` and ``descriptors``, and hand-built
    descriptors let these tests state a Hamming geometry exactly instead of hoping
    a synthetic image happens to produce one.
    """
    n = len(points)
    return Detection(
        points=np.asarray(points, dtype=np.float32).reshape(-1, 2),
        descriptors=np.asarray(descriptors, dtype=np.uint8),
        sizes=np.full(n, 31.0, dtype=np.float32),
        responses=np.full(n, 0.01, dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        detector="synthetic",
        enhancement_scale=1.0,
        detected=n,
    )


def _bits(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 256, size=(n, DESCRIPTOR_BYTES), dtype=np.uint8)


def _flip_bits(descriptor: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Perturb a descriptor by exactly ``count`` bit flips."""
    out = descriptor.copy()
    positions = rng.choice(DESCRIPTOR_BYTES * 8, size=count, replace=False)
    for position in positions:
        out[position // 8] ^= np.uint8(1 << (position % 8))
    return out


def _spread_points(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(10.0, 390.0, size=(n, 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# the core invariant
# ---------------------------------------------------------------------------
def test_mutual_nn_makes_the_correspondence_set_index_injective(cfg: Settings) -> None:
    """No left index and no right index may appear twice.

    This is not a statistical property, it is structural: if ``(i, j)`` and
    ``(i', j)`` both survive, then ``j``'s unique best partner is simultaneously
    ``i`` and ``i'``, so ``i == i'``. The test asserts the guarantee rather than
    the incidence, which is why it can be exact.
    """
    rng = np.random.default_rng(101)
    left_desc = _bits(rng, 400)
    right_desc = np.vstack([_flip_bits(d, 6, rng) for d in left_desc[:250]])

    left = _detection(_spread_points(400, rng), left_desc)
    right = _detection(_spread_points(250, rng), right_desc)

    matches = BruteForceMatcher(cfg.local_match).match(left, right)

    assert len(np.unique(matches.src_index)) == len(matches.src_index)
    assert len(np.unique(matches.dst_index)) == len(matches.dst_index)
    assert matches.evidence.mutual_nn is True


def test_asymmetric_keypoint_counts_cannot_manufacture_matches(cfg: Settings) -> None:
    """The bug-4 regression, reproduced exactly and then defused.

    2000 left descriptors are each a 5-bit perturbation of one of only 12 right
    "anchors". Every left descriptor therefore passes Lowe's ratio test
    emphatically -- nearest anchor at Hamming ~5, second-nearest at ~128, ratio
    ~0.04 against a 0.8 threshold -- because the ratio test asks only whether a
    descriptor's best match is *distinctive*, never whether the partner is
    already taken. Ratio-only yields ~2000 matches over 12 distinct points, which
    is precisely the legacy report. Mutual NN yields at most 12: one per anchor.
    """
    rng = np.random.default_rng(202)
    anchors = _bits(rng, 12)
    left_desc = np.vstack([_flip_bits(anchors[i % 12], 5, rng) for i in range(2000)])

    left = _detection(_spread_points(2000, rng), left_desc)
    right = _detection(_spread_points(12, rng), anchors)

    ratio_only = BruteForceMatcher(cfg.local_match.model_copy(update={"mutual_nn": False})).match(
        left, right
    )
    mutual = BruteForceMatcher(cfg.local_match).match(left, right)

    # The legacy shape: an enormous match count over a dozen distinct partners.
    assert ratio_only.count > 1500
    assert len(np.unique(ratio_only.dst_index)) == 12
    assert ratio_only.count > 100 * len(np.unique(ratio_only.dst_index))

    # The fix: one correspondence per right keypoint, maximum.
    assert mutual.count <= 12
    assert len(np.unique(mutual.dst_index)) == mutual.count
    assert keypoint_evidence(left, right).asymmetry == pytest.approx(2000 / 12, rel=1e-6)


def test_ratio_test_and_mutual_nn_reject_different_failures(cfg: Settings) -> None:
    """Both filters are needed; neither subsumes the other.

    The ratio test removes *ambiguous* descriptors (repeated texture, where the
    best and second-best partners are equally good). Mutual NN removes
    *asymmetric* ones (many-to-one collapse, where the partner prefers somebody
    else). Keeping only one filter leaves the other failure mode wide open, which
    is why the config exposes both and this test pins the difference.
    """
    rng = np.random.default_rng(303)
    anchors = _bits(rng, 12)
    left_desc = np.vstack([_flip_bits(anchors[i % 12], 5, rng) for i in range(600)])
    left = _detection(_spread_points(600, rng), left_desc)
    right = _detection(_spread_points(12, rng), anchors)

    ratio_only = BruteForceMatcher(cfg.local_match.model_copy(update={"mutual_nn": False})).match(
        left, right
    )
    both = BruteForceMatcher(cfg.local_match).match(left, right)

    # Every one of these passed the ratio test, so the ratio test is not what
    # saved us here -- the mutual constraint is.
    assert ratio_only.count > both.count
    assert both.count <= 12


def test_disabling_the_ratio_test_is_not_equivalent_to_disabling_mutual_nn(
    cfg: Settings,
) -> None:
    """Ambiguous-but-symmetric descriptors survive mutual NN and need the ratio test.

    Two right descriptors are placed equidistant from a left one. The pair is
    perfectly symmetric -- the left descriptor's best match reciprocates -- so
    mutual NN admits it, while the ratio test correctly calls it ambiguous.
    """
    rng = np.random.default_rng(404)
    base = _bits(rng, 40)
    # Each left descriptor gets a near-twin and a second, equally-close right
    # descriptor: nearest and second-nearest are both ~10 bits away.
    right_desc = np.vstack(
        [np.vstack([_flip_bits(d, 10, rng), _flip_bits(d, 11, rng)]) for d in base]
    )
    left = _detection(_spread_points(40, rng), base)
    right = _detection(_spread_points(80, rng), right_desc)

    permissive = cfg.local_match.model_copy(update={"nn_ratio": 1.0})
    strict = cfg.local_match.model_copy(update={"nn_ratio": 0.5})

    assert BruteForceMatcher(strict).match(left, right).count < (
        BruteForceMatcher(permissive).match(left, right).count
    )


def test_matches_carry_coordinates_consistent_with_their_indices(cfg: Settings) -> None:
    """``src``/``dst`` must be the points the indices name -- geometry depends on it."""
    rng = np.random.default_rng(505)
    left_desc = _bits(rng, 120)
    right_desc = np.vstack([_flip_bits(d, 4, rng) for d in left_desc])
    left_pts, right_pts = _spread_points(120, rng), _spread_points(120, rng)

    left = _detection(left_pts, left_desc)
    right = _detection(right_pts, right_desc)
    matches = BruteForceMatcher(cfg.local_match).match(left, right)

    assert matches.count > 0
    assert np.array_equal(matches.src, left_pts[matches.src_index])
    assert np.array_equal(matches.dst, right_pts[matches.dst_index])
    assert len(matches.distances) == matches.count


def test_empty_and_single_sided_inputs_return_empty_not_raise(cfg: Settings) -> None:
    rng = np.random.default_rng(606)
    populated = _detection(_spread_points(50, rng), _bits(rng, 50))
    empty = _detection(np.zeros((0, 2), np.float32), np.zeros((0, DESCRIPTOR_BYTES), np.uint8))

    for left, right in ((populated, empty), (empty, populated), (empty, empty)):
        matches = BruteForceMatcher(cfg.local_match).match(left, right)
        assert matches.count == 0
        assert matches.src.shape == (0, 2)


# ---------------------------------------------------------------------------
# copy-move self-matching
# ---------------------------------------------------------------------------
def test_g2nn_reduces_to_lowes_test_for_a_single_clone() -> None:
    """One partner then a gap: accept exactly one, same as the 2-NN ratio test."""
    assert _g2nn_accept([2.0, 100.0, 110.0, 120.0], 0.8) == 1


def test_g2nn_accepts_every_copy_of_a_multiply_cloned_region() -> None:
    """Three mutually-close partners followed by a gap: accept all three.

    The plain 2-NN ratio test scores ``d1 / d2 = 2.0 / 2.4 = 0.83`` here, fails
    its own 0.8 threshold, and therefore rejects *all three* genuine partners --
    the triple clone becomes invisible rather than merely under-reported. g2NN
    instead finds the real discontinuity, between the third partner and the first
    unrelated descriptor.
    """
    assert _g2nn_accept([2.0, 2.4, 2.8, 200.0, 210.0], 0.8) == 3
    # And the 2-NN view of the same neighbourhood, for contrast.
    assert 2.0 >= 0.8 * 2.4, "premise: Lowe's test rejects this neighbourhood outright"


def test_g2nn_treats_exact_duplicates_as_a_continuing_run() -> None:
    """Losslessly cloned regions give leading zeros; 0/0 is not a gap.

    This is the common case for a copy-paste with no resampling, and reading
    ``0 / 0`` as "distinctive" or as "ambiguous" both give the wrong answer. The
    informative comparison is the last zero against the first non-zero.
    """
    assert _g2nn_accept([0.0, 0.0, 0.0, 150.0, 160.0], 0.8) == 3
    assert _g2nn_accept([0.0, 140.0], 0.8) == 1


def test_g2nn_rejects_uniformly_ambiguous_neighbourhoods() -> None:
    """No gap anywhere means flat or repetitive texture, not a clone."""
    assert _g2nn_accept([90.0, 95.0, 100.0, 104.0, 108.0], 0.8) == 0
    assert _g2nn_accept([0.0, 0.0, 0.0, 0.0], 0.8) == 0
    assert _g2nn_accept([50.0], 0.8) == 0


def test_g2nn_bounded_by_the_neighbour_budget() -> None:
    """``knn`` caps discoverable clone multiplicity, and the cap is honest.

    With five equal-distance partners but only four neighbours fetched, there is
    no visible gap at all, so nothing is accepted rather than an arbitrary subset.
    """
    assert _g2nn_accept([1.0, 1.0, 1.0, 1.0], 0.8) == 0


def test_self_match_suppresses_the_trivial_and_the_adjacent(cfg: Settings) -> None:
    """Every keypoint is its own nearest neighbour; that is not a clone.

    Nor is its immediate neighbour on the same physical corner. Without the
    separation floor, a smooth image self-matches everywhere and the "clone"
    count measures keypoint density.
    """
    rng = np.random.default_rng(707)
    # Twelve descriptors, each present twice: once at a location and once 200 px
    # away. Only the far copy is a genuine clone relationship.
    base = _bits(rng, 12)
    descriptors = np.vstack([base, base])
    origins = rng.uniform(20.0, 180.0, size=(12, 2)).astype(np.float32)
    points = np.vstack([origins, origins + np.array([0.0, 200.0], np.float32)])

    detection = _detection(points, descriptors)
    matches = self_match(detection, cfg.copy_move)

    assert matches.count > 0
    # No self-pairs, and no pair closer than the configured floor.
    assert not np.any(matches.src_index == matches.dst_index)
    separations = np.linalg.norm(matches.dst - matches.src, axis=1)
    assert separations.min() >= cfg.copy_move.min_spatial_separation


def test_self_match_finds_purely_vertical_clone_offsets(cfg: Settings) -> None:
    """Bug 6's other half: the legacy ``qx < tx`` ordering dropped these entirely.

    Canonicalising a symmetric relation by comparing x-coordinates discards every
    pair whose x-coordinates are equal -- that is, every clone displaced straight
    down. Canonicalising on the index instead is total.
    """
    rng = np.random.default_rng(808)
    base = _bits(rng, 20)
    descriptors = np.vstack([base, base])
    origins = np.stack(
        [np.full(20, 150.0, np.float32), np.linspace(20.0, 120.0, 20, dtype=np.float32)], axis=1
    )
    # Identical x for every pair: the exact case the ordering hack lost.
    points = np.vstack([origins, origins + np.array([0.0, 180.0], np.float32)])

    matches = self_match(_detection(points, descriptors), cfg.copy_move)

    assert matches.count > 0
    assert np.allclose(matches.src[:, 0], matches.dst[:, 0])


def test_self_match_deduplicates_the_symmetric_pair(cfg: Settings) -> None:
    """``(i, j)`` and ``(j, i)`` are one clone relationship seen from both ends."""
    rng = np.random.default_rng(909)
    base = _bits(rng, 16)
    descriptors = np.vstack([base, base])
    origins = rng.uniform(20.0, 150.0, size=(16, 2)).astype(np.float32)
    points = np.vstack([origins, origins + np.array([200.0, 100.0], np.float32)])

    matches = self_match(_detection(points, descriptors), cfg.copy_move)

    pairs = {(int(i), int(j)) for i, j in zip(matches.src_index, matches.dst_index, strict=True)}
    assert all(i < j for i, j in pairs), "pairs must be canonicalised on index order"
    assert len(pairs) == len(matches.src_index), "duplicates survived"


def test_self_match_knn_allows_more_than_one_clone_per_keypoint(cfg: Settings) -> None:
    """``knn > 2`` is what makes a thrice-cloned region findable at all.

    With k=2, one keypoint can report at most one partner after the trivial
    self-match is dropped, so a region pasted three times is invisible beyond its
    first copy. This is why ``copy_move.knn`` defaults to 6.
    """
    rng = np.random.default_rng(1010)
    base = _bits(rng, 10)
    # Four copies of every descriptor, well separated.
    offsets = [(0.0, 0.0), (200.0, 0.0), (0.0, 200.0), (200.0, 200.0)]
    origins = rng.uniform(20.0, 120.0, size=(10, 2)).astype(np.float32)
    points = np.vstack([origins + np.array(o, np.float32) for o in offsets])
    descriptors = np.vstack([base] * 4)

    detection = _detection(points, descriptors)
    generous = self_match(detection, cfg.copy_move)
    minimal = self_match(detection, cfg.copy_move.model_copy(update={"knn": 2}))

    assert generous.count > minimal.count
    # Six pairs per group of four copies is the complete set; k=2 cannot reach it.
    assert generous.count > 10


def test_self_match_config_floor_is_respected_exactly() -> None:
    """A zero separation floor is legal and must not be silently overridden."""
    cfg = CopyMoveConfig.model_validate(
        {
            "enabled": True,
            "max_features": 500,
            "nn_ratio": 0.9,
            "min_spatial_separation": 0.0,
            "knn": 3,
            "cluster": {"eps": 0.08, "min_samples": 4, "feature_weights": [1.0, 1.0, 0.5, 0.5]},
            "min_cluster_inliers": 8,
            "reproj_threshold": 4.0,
            "mask_close_px": 9,
            "max_regions": 8,
        }
    )
    rng = np.random.default_rng(1111)
    base = _bits(rng, 8)
    points = np.vstack(
        [
            rng.uniform(20.0, 60.0, size=(8, 2)).astype(np.float32),
            rng.uniform(20.0, 60.0, size=(8, 2)).astype(np.float32),
        ]
    )
    matches = self_match(_detection(points, np.vstack([base, base])), cfg)
    # Only the trivial i == i match is suppressed; nearby pairs now survive.
    assert not np.any(matches.src_index == matches.dst_index)


# ---------------------------------------------------------------------------
# evidence reporting
# ---------------------------------------------------------------------------
def test_keypoint_evidence_reports_asymmetry_in_the_direction_that_hurts() -> None:
    """The ratio must exceed 1 whichever side is starved, or the gate is one-sided."""
    rng = np.random.default_rng(1212)
    big = _detection(_spread_points(2000, rng), _bits(rng, 2000))
    small = _detection(_spread_points(12, rng), _bits(rng, 12))

    assert keypoint_evidence(big, small).asymmetry == pytest.approx(2000 / 12)
    assert keypoint_evidence(small, big).asymmetry == pytest.approx(2000 / 12)
    assert keypoint_evidence(big, big).asymmetry == pytest.approx(1.0)


def test_keypoint_evidence_records_roi_abandonment_per_side() -> None:
    """Which side dropped its ROI is a fact about the run, not a debug detail.

    The legacy code made this decision silently and per side, so a report could
    not distinguish "both images were analysed the same way" from "one was
    restricted to twelve keypoints and the other was not".
    """
    rng = np.random.default_rng(1313)
    plain = _detection(_spread_points(100, rng), _bits(rng, 100))
    abandoned = Detection(
        points=_spread_points(100, rng),
        descriptors=_bits(rng, 100),
        sizes=np.full(100, 31.0, np.float32),
        responses=np.full(100, 0.01, np.float32),
        angles=np.zeros(100, np.float32),
        detector="synthetic",
        enhancement_scale=4.0,
        detected=100,
        roi_abandoned=True,
        warnings=("roi left too few keypoints",),
    )

    evidence = keypoint_evidence(plain, abandoned)
    assert evidence.roi_abandoned_left is False
    assert evidence.roi_abandoned_right is True
    assert evidence.kept_left == evidence.kept_right  # same count, different provenance


def test_build_matcher_returns_the_brute_force_backend_by_default(cfg: Settings) -> None:
    """`lightglue` landed in stage B3 and is covered by `tests/test_learned.py`.

    The no-silent-fallback principle is still asserted -- on the detector side,
    where `superpoint` remains unimplemented. `Literal` typing on
    `LocalMatchConfig.matcher` means an unknown *name* is rejected by config
    validation before a builder ever sees it, so there is no third matcher left
    to assert a NotImplementedError against here.
    """
    assert isinstance(build_matcher(cfg.local_match), BruteForceMatcher)

    # An unknown matcher name cannot reach a builder at all: `Literal` typing on
    # `LocalMatchConfig.matcher` rejects it during config validation, which is
    # the earlier and better place to fail.
    with pytest.raises(ConfigError, match="matcher"):
        load_config(overrides=["local_match.matcher=not-a-matcher"], use_env=False)


def test_matcher_uses_hamming_for_binary_descriptors(cfg: Settings) -> None:
    """ORB descriptors are bit strings; L2 on them is meaningless.

    Asserted behaviourally: an exact 3-bit perturbation must be recovered as the
    nearest neighbour, which only holds under a Hamming metric.
    """
    rng = np.random.default_rng(1414)
    left_desc = _bits(rng, 60)
    right_desc = np.vstack([_flip_bits(d, 3, rng) for d in left_desc])
    left = _detection(_spread_points(60, rng), left_desc)
    right = _detection(_spread_points(60, rng), right_desc)

    matches = BruteForceMatcher(cfg.local_match, norm=cv2.NORM_HAMMING).match(left, right)

    assert matches.count > 50
    # Index i should pair with index i, since that is the descriptor it came from.
    assert np.mean(matches.src_index == matches.dst_index) > 0.95
    assert matches.distances.max() <= 3.0


def test_local_match_config_rejects_a_ratio_above_one() -> None:
    """``nn_ratio > 1`` accepts every match, which is not a configuration."""
    with pytest.raises(ValueError):
        LocalMatchConfig.model_validate(
            {
                "detector": "orb",
                "matcher": "mutual_nn",
                "max_features": 2000,
                "min_keypoints_per_side": 100,
                "nn_ratio": 1.5,
                "mutual_nn": True,
                "orb": {
                    "scale_factor": 1.2,
                    "n_levels": 8,
                    "edge_threshold": 31,
                    "patch_size": 31,
                    "fast_threshold": 10,
                },
                "enhance": {
                    "enabled": True,
                    "scale": 4.0,
                    "interpolation": "cubic",
                    "clahe_clip_limit": 2.0,
                    "clahe_tile_grid": 8,
                },
                "roi": {
                    "enabled": True,
                    "dog_sigma_low": 1.0,
                    "dog_sigma_high": 2.0,
                    "dog_threshold": 0.02,
                    "min_area": 16,
                    "max_regions": 64,
                    "dilate_px": 8,
                    "min_keypoints": 100,
                },
            }
        )
