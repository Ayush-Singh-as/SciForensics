"""Detection: coordinate spaces, ROI abandonment, and what gets recorded.

Two audited defects live in this layer. Bug 13 was a hardcoded ``/ 4.0`` repeated
in six places while the scale itself was a configurable default elsewhere, so
changing the scale silently corrupted every reported coordinate. Bug 4's first
half was an ROI filter with a bare ``< 8`` fallback, decided per side and never
recorded, which let one image keep 12 region-filtered keypoints while the other
kept 2000 -- an asymmetry no downstream stage could see, let alone diagnose.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from sciforensics.config import LocalMatchConfig, Settings
from sciforensics.local_match.keypoints import (
    OrbDetector,
    build_detector,
    dog_regions,
    enhance,
    sobel_magnitude,
)
from tests.helpers import localised_texture, textured_image

HEIGHT, WIDTH = 320, 400


def _cfg(cfg: Settings, **overrides: object) -> LocalMatchConfig:
    return cfg.local_match.model_copy(update=overrides)


def _with_enhance(cfg: Settings, **overrides: object) -> LocalMatchConfig:
    return cfg.local_match.model_copy(
        update={"enhance": cfg.local_match.enhance.model_copy(update=overrides)}
    )


def _with_roi(cfg: Settings, **overrides: object) -> LocalMatchConfig:
    return cfg.local_match.model_copy(
        update={"roi": cfg.local_match.roi.model_copy(update=overrides)}
    )


def _coverage(boxes: list[tuple[int, int, int, int]]) -> float:
    """Fraction of the frame the boxes cover, counting overlap once."""
    return float(_box_mask(boxes).mean())


def _structure_recall(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float:
    """Fraction of the image's structured pixels that the boxes retain.

    The property an ROI must satisfy is that it does not *exclude* evidence, and
    that is a question about content, not about box geometry. Asserting instead
    that a cluster's centre lies inside some box conflates the two: a proposal can
    fragment a cluster into several boxes that surround its centroid without
    containing it, which is a perfectly good ROI and a failing assertion.
    """
    structured = image > (int(image.min()) + int(image.max())) // 2
    if not structured.any():
        return 1.0
    return float((structured & _box_mask(boxes)).sum() / structured.sum())


def _box_mask(boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for x, y, w, h in boxes:
        mask[y : y + h, x : x + w] = True
    return mask


# ---------------------------------------------------------------------------
# bug 13: one conversion, in one place
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0, 6.0])
def test_coordinates_stay_in_analysis_space_at_every_enhancement_scale(
    cfg: Settings, texture: np.ndarray, scale: float
) -> None:
    """Detection happens on an upscaled image; nothing outside reports that space.

    This is the invariant bug 13 violated. Because the conversion was open-coded
    at each use site while the scale lived in a default argument, a scale change
    left some coordinates converted and others not -- and both kinds looked
    plausible. Sweeping the scale is the only test that would have caught it: at
    the shipped value of 4.0 the hardcoded ``/ 4.0`` was accidentally correct.
    """
    detection = OrbDetector(_with_enhance(cfg, scale=scale, enabled=True)).detect(texture)

    assert detection.count > 0
    assert detection.points[:, 0].max() <= WIDTH, "x escaped the analysis frame"
    assert detection.points[:, 1].max() <= HEIGHT, "y escaped the analysis frame"
    assert detection.points.min() >= 0.0
    assert detection.enhancement_scale == (scale if scale != 1.0 else 1.0)


def test_keypoint_sizes_are_converted_with_their_coordinates(
    cfg: Settings, texture: np.ndarray
) -> None:
    """A size in enhancement pixels beside a coordinate in analysis pixels is a bug.

    Sizes are what a report draws its keypoint circles from, so an unconverted
    size produces an overlay whose markers are four times too large -- visibly
    wrong, but only if someone looks.

    ORB's finest pyramid level always reports exactly ``patch_size``, which makes
    the smallest size a fixed, known quantity rather than a distribution: 31 px at
    detection resolution, so 31/4 in analysis space once converted. Asserting the
    exact value is what distinguishes "converted" from "happens to be smaller".
    """
    patch = float(cfg.local_match.orb.patch_size)
    plain = OrbDetector(_with_enhance(cfg, enabled=False)).detect(texture)
    upscaled = OrbDetector(_with_enhance(cfg, scale=4.0, enabled=True)).detect(texture)

    assert plain.count > 0 and upscaled.count > 0
    assert plain.sizes.min() == pytest.approx(patch)
    assert upscaled.sizes.min() == pytest.approx(patch / 4.0)  # 31.0 if unconverted


def test_roi_boxes_are_reported_in_analysis_space_too(cfg: Settings, texture: np.ndarray) -> None:
    """The overlay boxes go through the same conversion as the keypoints."""
    detection = OrbDetector(_with_enhance(cfg, scale=4.0, enabled=True)).detect(texture)

    assert detection.roi_boxes
    for x, y, w, h in detection.roi_boxes:
        assert 0 <= x <= WIDTH and 0 <= y <= HEIGHT
        assert x + w <= WIDTH + 1 and y + h <= HEIGHT + 1


def test_enhance_upscales_by_exactly_the_configured_factor(
    cfg: Settings, texture: np.ndarray
) -> None:
    enlarged = enhance(texture, cfg.local_match.enhance.model_copy(update={"scale": 3.0}))
    assert enlarged.shape == (HEIGHT * 3, WIDTH * 3)


def test_enhance_is_a_no_op_when_disabled(cfg: Settings, texture: np.ndarray) -> None:
    """Returning the input unchanged is what makes ``scale`` safe to ignore."""
    untouched = enhance(texture, cfg.local_match.enhance.model_copy(update={"enabled": False}))
    assert np.array_equal(untouched, texture)


# ---------------------------------------------------------------------------
# bug 4, first half: the ROI filter must not be allowed to collapse a side
# ---------------------------------------------------------------------------
def test_roi_is_abandoned_rather_than_starving_a_side(cfg: Settings, texture: np.ndarray) -> None:
    """An ROI that would leave too few keypoints is dropped, and the drop is recorded.

    A deliberately strict ROI keeps a couple of hundred keypoints out of two
    thousand; setting ``roi.min_keypoints`` above what it can supply is what a
    genuinely low-contrast panel does on its own. The legacy floor was a bare
    ``< 8``, decided per side with no record -- low enough that "12 keypoints"
    passed, and silent enough that nothing downstream could tell.
    """
    strict_roi = _with_roi(cfg, dog_threshold=0.9, dilate_px=0, min_area=4096, max_regions=1)
    restricted = OrbDetector(strict_roi).detect(texture)
    assert 0 < restricted.count < restricted.detected, "premise: the strict ROI narrows detection"

    floor = restricted.count + 1
    demanding = strict_roi.model_copy(
        update={"roi": strict_roi.roi.model_copy(update={"min_keypoints": floor})}
    )
    detection = OrbDetector(demanding).detect(texture)

    assert detection.roi_abandoned is True
    assert detection.count == detection.detected, "abandonment falls back to the whole image"
    assert detection.count >= floor
    assert detection.warnings, "an abandoned ROI must say so in the evidence"
    assert str(restricted.count) in detection.warnings[0]
    assert str(floor) in detection.warnings[0]
    # The boxes are still reported: knowing *which* region was proposed and then
    # rejected is the diagnostic, and the overlay should show it.
    assert detection.roi_boxes


def test_detected_count_is_measured_before_the_roi_decision(
    cfg: Settings, texture: np.ndarray
) -> None:
    """``detected`` describes the image; ``count`` describes the analysis.

    Keeping them separate is what makes an asymmetric run diagnosable afterwards.
    With the ROI applied, ``count <= detected``; the gap is the ROI's effect, and
    it is now visible rather than inferred.
    """
    unrestricted = OrbDetector(_with_roi(cfg, enabled=False)).detect(texture)
    restricted = OrbDetector(_with_roi(cfg, enabled=True)).detect(texture)

    assert unrestricted.detected == unrestricted.count
    assert unrestricted.roi_boxes == ()
    assert restricted.detected == unrestricted.detected, "detection is ROI-independent"
    assert restricted.count <= restricted.detected


def test_two_differently_contrasted_images_stay_comparably_sampled(cfg: Settings) -> None:
    """The regression at pair level: neither side may collapse relative to the other.

    A bright, high-contrast panel against a dim, low-contrast one is exactly the
    input that produced the 2000-vs-12 report. Both must now land within a modest
    factor of each other, because the floor is enforced per side.
    """
    bright = textured_image(seed=11)
    dim = (textured_image(seed=11).astype(np.float32) * 0.25 + 20).astype(np.uint8)

    detector = OrbDetector(cfg.local_match)
    left, right = detector.detect(bright), detector.detect(dim)

    assert left.count >= cfg.local_match.roi.min_keypoints
    assert right.count >= cfg.local_match.roi.min_keypoints
    asymmetry = max(left.count, right.count) / min(left.count, right.count)
    assert asymmetry < 4.0, f"sides sampled {asymmetry:.1f}x apart"


def test_an_unsatisfiable_floor_degrades_rather_than_raising(cfg: Settings) -> None:
    """When no configuration can meet the floor, report the shortfall and continue.

    The complement of the test above: there the ROI was the problem and dropping
    it was the fix, here the image itself cannot supply the demanded keypoints. A
    sparse panel against the largest ``roi.min_keypoints`` the config permits --
    ``max_features``, since :class:`LocalMatchConfig` rejects anything higher as
    unsatisfiable by construction -- so this is a state a real user can actually
    reach.

    Best-effort detection plus a warning is the right answer. Refusing outright
    would make an unlucky threshold look like a crash, and succeeding silently is
    what produced the 12-keypoint report.
    """
    sparse = localised_texture()
    demanding = _with_roi(cfg, min_keypoints=cfg.local_match.max_features)
    detection = OrbDetector(demanding).detect(sparse)

    assert 0 < detection.count < cfg.local_match.max_features, "premise: the panel is sparse"
    assert detection.roi_abandoned is True
    assert detection.count == detection.detected
    assert detection.warnings
    assert str(cfg.local_match.max_features) in detection.warnings[0]


# ---------------------------------------------------------------------------
# region proposal
# ---------------------------------------------------------------------------
def test_dog_regions_localise_structure_and_decline_to_narrow_a_flat_field(
    cfg: Settings,
) -> None:
    """The threshold is in float [0, 1] contrast units, so it means something.

    The legacy call passed a ``uint8`` 0-255 image to ``blob_dog(threshold=0.08)``,
    where 0.08 on that scale accepts essentially every response -- the boxes were
    not so much wrong as uninformative.

    Two different correct answers here, and the distinction is the point. A panel
    with localised structure yields one region per cluster, each bounding it and
    excluding the background. A flat panel yields one region covering everything:
    there is no structure to prefer, and excluding parts of an image for no reason
    would throw away evidence. Returning the whole frame is a refusal to narrow,
    not a failure.
    """
    centres = ((80, 75), (290, 235))
    image = localised_texture(centres)
    localised = dog_regions(sobel_magnitude(image), cfg.local_match.roi)
    flat = dog_regions(np.full((HEIGHT, WIDTH), 128, dtype=np.uint8), cfg.local_match.roi)

    assert len(flat) == 1
    assert _coverage(flat) == pytest.approx(1.0, abs=0.02)

    assert len(localised) == len(centres), "one region per cluster"
    # High recall on a small area is the whole point: keep the evidence, drop the
    # background. Either number alone is trivially satisfiable.
    assert _structure_recall(image, localised) > 0.95
    assert _coverage(localised) < 0.4


def test_dog_regions_fall_back_to_otsu_rather_than_returning_nothing(
    cfg: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """A faint panel still gets *localised* regions, not zero and not everything.

    Zero regions would silently mean whole-image detection with no record of why,
    and a whole-frame region would mean the ROI stage did nothing while appearing
    to work. Otsu picks its threshold from the histogram it is given, so it adapts
    to a panel whose entire dynamic range sits below the fixed threshold.

    The log assertion pins the premise. Whether the fixed threshold found nothing
    is a fact about the internal band-pass response, and the fallback's own debug
    line is the one honest way to observe it -- recomputing the DoG here would just
    restate the implementation and pass even if the branch never ran.

    The recall bar is lower than the clean path's 0.95 because it should be: this
    input has been compressed to roughly a dozen grey levels, so the faintest blob
    edges sit below any histogram-derived threshold. Enrichment is what carries the
    test -- boxes covering a sixth of the frame that hold four-fifths of the
    structure are selective *for* content, which a merely-small or merely-large
    proposal would not be.
    """
    image = localised_texture()
    faint = (image.astype(np.float32) * 0.05 + 10).astype(np.uint8)
    strict = cfg.local_match.roi.model_copy(update={"dog_threshold": 0.95})

    with caplog.at_level(logging.DEBUG, logger="sciforensics.local_match.keypoints"):
        boxes = dog_regions(sobel_magnitude(faint), strict)
    assert "fell back to Otsu" in caplog.text, "premise: the fixed threshold found nothing"

    recall, coverage = _structure_recall(image, boxes), _coverage(boxes)
    assert boxes, "Otsu fallback produced no regions"
    assert coverage < 0.4, "the fallback returned essentially the whole frame"
    assert recall > 0.7, "the fallback dropped real structure"
    assert recall > 3.0 * coverage, "the regions are small but not aimed at anything"


def test_dog_regions_respects_max_regions_and_min_area(cfg: Settings) -> None:
    structured = sobel_magnitude(textured_image())
    capped = cfg.local_match.roi.model_copy(update={"max_regions": 3, "min_area": 64})
    boxes = dog_regions(structured, capped)

    assert len(boxes) <= 3
    assert all(w * h >= 64 for _, _, w, h in boxes)
    # Sorted largest-first, so a cap keeps the most substantial regions.
    areas = [w * h for _, _, w, h in boxes]
    assert areas == sorted(areas, reverse=True)


def test_dog_regions_disabled_returns_nothing(cfg: Settings) -> None:
    disabled = cfg.local_match.roi.model_copy(update={"enabled": False})
    assert dog_regions(sobel_magnitude(textured_image()), disabled) == []


def test_sobel_magnitude_is_normalised_and_typed(cfg: Settings) -> None:
    magnitude = sobel_magnitude(textured_image())
    assert magnitude.dtype == np.uint8
    assert magnitude.max() == 255, "an unnormalised magnitude makes the DoG threshold meaningless"


def test_sobel_magnitude_survives_a_constant_image() -> None:
    """A zero-gradient input must not divide by its own zero peak."""
    magnitude = sobel_magnitude(np.full((32, 32), 200, dtype=np.uint8))
    assert magnitude.max() == 0


# ---------------------------------------------------------------------------
# detection output contract
# ---------------------------------------------------------------------------
def test_detection_arrays_are_mutually_consistent(cfg: Settings, texture: np.ndarray) -> None:
    detection = OrbDetector(cfg.local_match).detect(texture)

    assert detection.descriptors is not None
    n = detection.count
    assert detection.points.shape == (n, 2)
    assert detection.sizes.shape == (n,)
    assert detection.responses.shape == (n,)
    assert len(detection.descriptors) == n
    assert detection.points.dtype == np.float32
    assert detection.descriptors.dtype == np.uint8
    assert len(detection.keypoints()) == n


def test_featureless_image_yields_an_empty_detection_not_an_error(
    cfg: Settings,
) -> None:
    """A blank panel is a legitimate input, and it has no keypoints."""
    detection = OrbDetector(cfg.local_match).detect(np.full((HEIGHT, WIDTH), 128, dtype=np.uint8))

    assert detection.count == 0
    assert detection.points.shape == (0, 2)
    assert detection.descriptors is None
    assert detection.detected == 0


def test_detection_is_deterministic(cfg: Settings, texture: np.ndarray) -> None:
    """Identical input, identical evidence -- the property the whole tool rests on."""
    detector = OrbDetector(cfg.local_match)
    first, second = detector.detect(texture), detector.detect(texture)

    assert np.array_equal(first.points, second.points)
    assert first.descriptors is not None and second.descriptors is not None
    assert np.array_equal(first.descriptors, second.descriptors)


def test_build_detector_refuses_a_backend_it_cannot_run(cfg: Settings) -> None:
    """A benchmark row labelled "SuperPoint" that silently ran ORB is worse than none.

    `disk` landed in stage B3, so `superpoint` is the remaining unimplemented
    backend and carries this assertion. The principle is unchanged: an
    unavailable backend must raise rather than downgrade, because a mislabelled
    benchmark row is worse than a missing one.
    """
    assert isinstance(build_detector(cfg.local_match), OrbDetector)
    with pytest.raises(NotImplementedError, match="superpoint"):
        build_detector(_cfg(cfg, detector="superpoint"))


def test_detector_declares_the_norm_its_descriptors_need(cfg: Settings) -> None:
    """The matcher reads this off the detector rather than branching on its name.

    That is what lets a float-descriptor backend drop in later without the
    pipeline learning anything about it.
    """
    assert build_detector(cfg.local_match).norm == cv2.NORM_HAMMING
