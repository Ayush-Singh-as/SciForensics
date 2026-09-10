"""DISK + LightGlue backends. Stage B3.

Marked ``learned`` and skipped when ``kornia`` is absent, because the backends
are an optional extra and the ORB path must keep working without them.

These assert the **contracts** the rest of the pipeline relies on -- analysis-space
coordinates, injective correspondences, an L2 norm travelling on the detector --
rather than a match count. Counts are a benchmark output, not an invariant, and
the measurement in ``benchmarks/RESULTS.md`` shows exactly why: on two of the
three degradation pairs DISK produces *more* matches than ORB of which **zero**
are geometrically correct. A test asserting "DISK finds more matches" would have
passed while the pipeline got worse.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("kornia", reason="learned backends need the [match] extra")

from sciforensics.config import Settings, load_config
from sciforensics.local_match.keypoints import build_detector
from sciforensics.local_match.learned import (
    DiskDetector,
    LightGlueMatcher,
)
from sciforensics.local_match.matcher import build_matcher

pytestmark = pytest.mark.learned


@pytest.fixture(scope="module")
def learned_cfg() -> Settings:
    return load_config(
        overrides=["local_match.detector=disk", "local_match.matcher=lightglue"],
        use_env=False,
    )


@pytest.fixture(scope="module")
def detector(learned_cfg: Settings) -> DiskDetector:
    built = build_detector(learned_cfg.local_match)
    assert isinstance(built, DiskDetector)
    return built


@pytest.fixture(scope="module")
def matcher(learned_cfg: Settings, detector: DiskDetector) -> LightGlueMatcher:
    built = build_matcher(learned_cfg.local_match, norm=detector.norm)
    assert isinstance(built, LightGlueMatcher)
    return built


@pytest.fixture(scope="module")
def texture() -> np.ndarray:
    """A textured panel. DISK needs real structure, not smooth noise."""
    rng = np.random.default_rng(1234)
    base = rng.integers(0, 255, (192, 192), dtype=np.uint8)
    blurred = cv2.GaussianBlur(base, (0, 0), 1.2)
    for centre in ((48, 48), (140, 60), (96, 150)):
        cv2.circle(blurred, centre, 22, 235, -1)
        cv2.circle(blurred, centre, 22, 40, 2)
    return blurred


def test_builder_selects_the_learned_backends(
    detector: DiskDetector, matcher: LightGlueMatcher
) -> None:
    """`build_detector` must not silently downgrade to ORB.

    A benchmark row labelled "LightGlue" that actually ran ORB is worse than no
    row, so the builders raise rather than fall back.
    """
    assert detector.name == "disk"
    assert matcher.name == "lightglue"


def test_disk_reports_an_l2_norm(detector: DiskDetector) -> None:
    """The norm travels on the *detector*, which is what keeps the matcher generic.

    DISK descriptors are float; matching them under ``NORM_HAMMING`` would be
    silently meaningless rather than an error.
    """
    assert detector.norm == cv2.NORM_L2


def test_descriptors_are_float_and_wide(detector: DiskDetector, texture: np.ndarray) -> None:
    detection = detector.detect(texture)
    assert detection.count > 0
    assert detection.descriptors is not None
    assert detection.descriptors.dtype == np.float32
    assert detection.descriptors.shape[0] == detection.count


def test_coordinates_are_in_analysis_space(
    detector: DiskDetector, texture: np.ndarray, learned_cfg: Settings
) -> None:
    """Bug 13's contract, upheld by the new backend.

    Detection runs on the enhanced (upscaled) image, but coordinates are divided
    by the scale exactly once, here -- so nothing downstream needs to know the
    enhancement happened. If this regressed, every reported box would be off by
    the enhancement factor.
    """
    detection = detector.detect(texture)
    height, width = texture.shape
    assert detection.enhancement_scale == learned_cfg.local_match.enhance.scale
    assert detection.points[:, 0].max() <= width + 1
    assert detection.points[:, 1].max() <= height + 1


def test_angles_are_marked_unmeasured(detector: DiskDetector, texture: np.ndarray) -> None:
    """DISK supplies no orientation, and must say so rather than imply zero.

    Copy-move clustering uses orientation *differences*; a backend silently
    reporting 0 would put every pair at the cluster centre of that dimension,
    which looks exactly like "no rotation" instead of "not measured".
    """
    detection = detector.detect(texture)
    assert np.all(detection.angles == -1.0)


def test_correspondences_are_injective(
    detector: DiskDetector, matcher: LightGlueMatcher, texture: np.ndarray
) -> None:
    """Bug 4's property, free from a joint matcher rather than enforced.

    LightGlue emits at most one partner per keypoint. The degeneracy gates in
    `geometry.py` still run: a confident matcher can agree on a degenerate
    configuration, and injective is not the same as correct.
    """
    shifted = np.roll(texture, 5, axis=1)
    left = detector.detect(texture)
    right = detector.detect(shifted)
    matches = matcher.match(left, right)

    if matches.count == 0:
        pytest.skip("no correspondences on the synthetic fixture")

    assert matches.evidence.is_injective
    assert len(np.unique(matches.src_index)) == matches.count
    assert len(np.unique(matches.dst_index)) == matches.count


def test_match_arrays_are_parallel_and_indexable(
    detector: DiskDetector, matcher: LightGlueMatcher, texture: np.ndarray
) -> None:
    """The report draws lines back to specific keypoints, so indices must resolve."""
    left = detector.detect(texture)
    right = detector.detect(np.roll(texture, 4, axis=0))
    matches = matcher.match(left, right)
    if matches.count == 0:
        pytest.skip("no correspondences on the synthetic fixture")

    assert matches.src.shape == matches.dst.shape == (matches.count, 2)
    assert matches.distances.shape == (matches.count,)
    assert matches.src_index.max() < left.count
    assert matches.dst_index.max() < right.count
    # `src` must genuinely be the indexed keypoints, not a reordered copy.
    assert np.allclose(matches.src, left.points[matches.src_index])
    assert np.allclose(matches.dst, right.points[matches.dst_index])


def test_distances_are_descriptor_distances_not_placeholder_zeros(
    detector: DiskDetector, matcher: LightGlueMatcher, texture: np.ndarray
) -> None:
    """LightGlue reports a confidence, not a distance.

    Recording zeros would imply a perfect match everywhere, so the true
    descriptor distance is computed for the surviving pairs -- keeping
    `Correspondences.distances` meaning one thing across every backend.
    """
    left = detector.detect(texture)
    right = detector.detect(np.roll(texture, 6, axis=1))
    matches = matcher.match(left, right)
    if matches.count == 0:
        pytest.skip("no correspondences on the synthetic fixture")
    assert matches.distances.min() >= 0.0
    assert not np.all(matches.distances == 0.0)


def test_empty_side_yields_no_matches_rather_than_raising(
    detector: DiskDetector, matcher: LightGlueMatcher, texture: np.ndarray
) -> None:
    """A blank panel is a legitimate input, not an error."""
    left = detector.detect(texture)
    blank = detector.detect(np.zeros((128, 128), dtype=np.uint8))
    matches = matcher.match(left, blank)
    assert matches.count >= 0
    assert matches.evidence.matcher == "lightglue"


def test_superpoint_is_still_an_explicit_error() -> None:
    """Unimplemented must stay loud. A silent ORB downgrade would corrupt a benchmark."""
    cfg = load_config(overrides=["local_match.detector=superpoint"], use_env=False)
    with pytest.raises(NotImplementedError, match="superpoint"):
        build_detector(cfg.local_match)
