"""Geometry: decomposition properties, the flip bug, degeneracy gates, coordinate spaces.

Every test here corresponds to a defect that shipped in the prototype, or to an
invariant whose violation would let one back in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sciforensics.config import AffineConfig, GeometryConfig, Settings
from sciforensics.local_match.geometry import (
    compose_affine,
    decompose_affine,
    distinct_count,
    estimate_affine,
    rescale_affine,
    rescale_box,
    rescale_homography,
    verify,
)
from sciforensics.types import RejectionReason
from tests.helpers import affine_matrix, apply_affine

DIAG = math.hypot(400, 320)


def _grid(n: int = 200, *, width: float = 400.0, height: float = 320.0) -> np.ndarray:
    """A well-spread point cloud on a jittered lattice."""
    generator = np.random.default_rng(7)
    side = math.ceil(math.sqrt(n))
    xs = np.linspace(10.0, width - 10.0, side)
    ys = np.linspace(10.0, height - 10.0, side)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)[:n]
    return (grid + generator.uniform(-1.5, 1.5, grid.shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# decomposition: properties
# ---------------------------------------------------------------------------
def test_decompose_compose_is_an_exact_roundtrip() -> None:
    """``compose(decompose(A)) == A`` for arbitrary well-conditioned ``A``.

    This property is why the decomposition is a single QR factorisation rather
    than the mix of SVD scales and QR angles it started as: a mixed report cannot
    be recomposed, so there is no way to test that its components describe the
    matrix they came from.
    """
    generator = np.random.default_rng(11)
    worst = 0.0
    for _ in range(2000):
        matrix = generator.uniform(-3.0, 3.0, (2, 3))
        if abs(np.linalg.det(matrix[:, :2])) < 1e-3:
            continue  # near-singular: decomposition is legitimately undefined
        recomposed = compose_affine(decompose_affine(matrix))
        worst = max(worst, float(np.abs(recomposed - matrix).max()))
    assert worst < 1e-9, f"round-trip error {worst:.3e}"


@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        ({"rotation_deg": 30.0, "scale_x": 1.5}, {"rotation_deg": 30.0, "scale": 1.5}),
        ({"flip": True}, {"flip": True, "rotation_deg": 0.0}),
        # A horizontal mirror is a vertical mirror composed with a 180-degree
        # rotation. The decomposition fixes the vertical convention, so this is
        # the documented -- and only self-consistent -- way it can report.
        ({"flip": True, "rotation_deg": 180.0}, {"flip": True, "rotation_deg": 180.0}),
        ({"rotation_deg": 45.0, "scale_x": 1.2, "flip": True}, {"flip": True, "scale": 1.2}),
        ({"scale_x": 2.0, "scale_y": 1.0}, {"anisotropic": True, "flip": False}),
        ({"shear_deg": 15.0}, {"shear_deg": 15.0, "sheared": True}),
        # A 180-degree rotation is NOT a reflection: det = +1. The mountains pair
        # is exactly this case, and calling it a flip would be wrong.
        ({"rotation_deg": 180.0}, {"flip": False, "scale": 1.0}),
    ],
)
def test_known_transforms_are_recovered(kwargs: dict, expect: dict) -> None:
    decomposition = decompose_affine(affine_matrix(**kwargs))
    for field, value in expect.items():
        actual = getattr(decomposition, field)
        if isinstance(value, bool):
            assert actual is value, f"{field}: expected {value}, got {actual}"
        else:
            assert actual == pytest.approx(value, abs=1e-6), f"{field}: got {actual}"


def test_flip_is_exactly_negative_determinant() -> None:
    """``flip`` must be convention-independent even though ``rotation`` is not."""
    generator = np.random.default_rng(3)
    for _ in range(500):
        matrix = generator.uniform(-2.0, 2.0, (2, 3))
        determinant = float(np.linalg.det(matrix[:, :2]))
        if abs(determinant) < 1e-6:
            continue
        assert decompose_affine(matrix).flip is (determinant < 0.0)


def test_collapsed_linear_part_reports_zeros_not_confident_nonsense() -> None:
    """The legacy failure mode was ``scale: 0.000`` presented as a real number."""
    decomposition = decompose_affine(np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 7.0]]))
    assert decomposition.scale_x == 0.0
    assert decomposition.scale_y == 0.0
    assert decomposition.translation == (5.0, 7.0)


# ---------------------------------------------------------------------------
# bug 2: flip detection was mathematically unreachable
# ---------------------------------------------------------------------------
def test_similarity_model_cannot_represent_a_mirror_but_full_model_can() -> None:
    """The decisive A/B for bug 2, on a *true* mirror, both ways.

    ``estimateAffinePartial2D`` returns ``[[a, -b, tx], [b, a, ty]]``, whose
    determinant is ``a**2 + b**2 > 0``. Fitting it to mirrored correspondences
    cannot yield ``det < 0``, so the legacy pipeline's ``flip`` flag was dead code
    for every input it ever saw. The same points through ``estimateAffine2D``
    recover the reflection exactly.
    """
    src = _grid(200)
    dst = (src * np.array([1.0, -1.0], dtype=np.float32) + np.array([0.0, 500.0])).astype(
        np.float32
    )

    def affine_cfg(model: str) -> AffineConfig:
        return AffineConfig(
            model=model,  # type: ignore[arg-type]
            method="magsac",
            reproj_threshold=4.0,
            anisotropy_tolerance=0.05,
            shear_tolerance_deg=2.0,
        )

    similarity, _ = estimate_affine(src, dst, affine_cfg("similarity"))
    full, _ = estimate_affine(src, dst, affine_cfg("full"))
    assert similarity is not None and full is not None

    similarity_decomposition = decompose_affine(similarity, model="similarity")
    full_decomposition = decompose_affine(full, model="full")

    assert similarity_decomposition.determinant > 0.0
    assert similarity_decomposition.flip is False, "the legacy bug: a mirror read as no-flip"

    assert full_decomposition.determinant < 0.0
    assert full_decomposition.flip is True, "the fix: the mirror is detected"
    assert full_decomposition.scale == pytest.approx(1.0, abs=1e-3)


def test_config_refuses_the_flip_blind_model(cfg: Settings) -> None:
    """Selecting ``similarity`` is a configuration error, not a silent downgrade.

    The legacy pipeline used the 4-DOF model *and* tested ``det < 0``, so it
    answered "flip: no" to every mirrored image it was ever shown. Making that
    combination unconstructible is the structural fix; the arithmetic fix above is
    only half of it.
    """
    payload = cfg.model_dump(mode="json")
    payload["geometry"]["affine"]["model"] = "similarity"
    with pytest.raises(ValueError, match="cannot represent a reflection"):
        Settings.model_validate(payload)


# ---------------------------------------------------------------------------
# coordinate spaces
# ---------------------------------------------------------------------------
def test_rescale_homography_removes_the_downscale_artifact() -> None:
    """Analysis-space numbers are not reportable numbers.

    Two images capped to the same ``max_dimension`` from different original sizes
    get different scale factors, and a homography fitted between the downscaled
    pair carries their ratio as a spurious scale. This is the mountains case:
    3840x2400 (scale 0.5333) against 2343x1421 (scale 0.8741) manufactures a
    1.639x "enlargement" out of an unchanged image.
    """
    src_scale, dst_scale = 0.5333333333, 0.8740930431
    identity = np.eye(3)
    analysis = np.diag([dst_scale / src_scale, dst_scale / src_scale, 1.0])

    # In analysis space this looks like a 1.639x enlargement...
    assert decompose_affine(analysis[:2, :]).scale == pytest.approx(1.639, abs=1e-3)
    # ...and in original space it is the identity, which is the truth.
    recovered = rescale_homography(analysis, src_scale=src_scale, dst_scale=dst_scale)
    assert np.abs(recovered - identity).max() < 1e-9


def test_rescale_is_consistent_with_transforming_points() -> None:
    """``H_original`` must map original points the way ``H_analysis`` maps scaled ones."""
    src_scale, dst_scale = 0.4, 0.9
    analysis = affine_matrix(rotation_deg=20.0, scale_x=1.3, translation=(15.0, -8.0))
    original = rescale_affine(analysis, src_scale=src_scale, dst_scale=dst_scale)

    points_original = _grid(50).astype(np.float64)
    via_original = apply_affine(original, points_original)
    via_analysis = apply_affine(analysis, points_original * src_scale) / dst_scale
    assert np.abs(via_original - via_analysis).max() < 1e-9


def test_rescale_is_identity_when_scales_are_one() -> None:
    matrix = affine_matrix(rotation_deg=13.0, scale_x=0.8, translation=(3.0, 4.0))
    assert np.abs(rescale_affine(matrix, src_scale=1.0, dst_scale=1.0) - matrix).max() == 0.0


def test_rescale_box_divides_by_the_scale() -> None:
    """``p_fine = scale * p_coarse``, so converting to the coarser frame divides."""
    assert rescale_box((40, 60, 100, 80), scale=0.5) == (80, 120, 200, 160)
    assert rescale_box((40, 60, 100, 80), scale=2.0) == (20, 30, 50, 40)
    assert rescale_box((40, 60, 100, 80), scale=1.0) == (40, 60, 100, 80)


def test_rescale_box_never_collapses_a_region_to_zero_extent() -> None:
    """A region that exists is not zero pixels wide.

    Rounding a 1 px box down at a large scale factor would delete it from every
    overlay while leaving it in the evidence table -- a box the reader is told
    about but cannot see.
    """
    x, y, w, h = rescale_box((10, 10, 1, 1), scale=50.0)
    assert (w, h) == (1, 1)
    assert (x, y) == (0, 0)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def test_healthy_pair_verifies_and_reports_the_true_transform(cfg: Settings) -> None:
    src = _grid(200)
    matrix = affine_matrix(rotation_deg=-2.0, scale_x=1.01, translation=(4.0, -3.0))
    dst = apply_affine(matrix, src).astype(np.float32)

    evidence = verify(src, dst, cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)

    assert evidence.verified is True
    assert evidence.rejection_reason is RejectionReason.NONE
    assert evidence.transform is not None
    assert evidence.transform.rotation_deg == pytest.approx(-2.0, abs=0.3)
    assert evidence.transform.scale == pytest.approx(1.01, abs=0.02)
    assert evidence.transform.flip is False
    assert evidence.reproj_rms is not None and evidence.reproj_rms < 1.0
    assert evidence.inlier_ratio > 0.9
    assert len(evidence.hull_left) >= 3


def test_condition_number_ignores_translation_magnitude(cfg: Settings) -> None:
    """A large translation is not a degeneracy.

    Computed on the raw pixel-space matrix, the condition number tracks the
    translation (the mountains 180-degree rotation scored 9.6e6 against a 1e7
    gate, on a fit with 0.46 px RMS). Normalising both frames first makes the
    metric measure geometry instead of image size.
    """
    src = _grid(200)
    near = apply_affine(affine_matrix(translation=(2.0, 2.0)), src).astype(np.float32)
    far = apply_affine(affine_matrix(rotation_deg=180.0, translation=(4000.0, 2500.0)), src).astype(
        np.float32
    )

    near_evidence = verify(src, near, cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)
    far_evidence = verify(src, far, cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)

    assert near_evidence.verified is True
    assert far_evidence.verified is True, far_evidence.rejection_reason.value
    assert far_evidence.condition_number is not None
    assert far_evidence.condition_number < 100.0
    assert far_evidence.transform is not None
    assert far_evidence.transform.flip is False  # 180 degrees is a rotation
    assert far_evidence.transform.scale == pytest.approx(1.0, abs=1e-3)


def test_many_to_one_correspondences_are_rejected_as_degenerate(cfg: Settings) -> None:
    """Bug 4's geometry half: inlier *rows* are not geometric constraints.

    Reproduces the legacy shape -- hundreds of matches collapsing onto a handful
    of distinct points -- and requires the specific structural diagnosis rather
    than a vague "low inlier ratio".
    """
    anchors_src = np.array([[40, 40], [360, 45], [200, 160], [50, 280]], dtype=np.float32)
    anchors_dst = anchors_src + np.array([12.0, -7.0], dtype=np.float32)
    src = np.repeat(anchors_src, 60, axis=0)
    dst = np.repeat(anchors_dst, 60, axis=0)

    evidence = verify(src, dst, cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)

    assert evidence.verified is False
    assert evidence.rejection_reason is RejectionReason.DEGENERATE_CORRESPONDENCES
    assert evidence.distinct_inliers <= 4
    assert evidence.inlier_count > evidence.distinct_inliers  # the legacy shape exactly
    assert "many-to-one" in evidence.rejection_reason.explanation


def test_clustered_inliers_are_rejected_for_insufficient_spread(cfg: Settings) -> None:
    generator = np.random.default_rng(5)
    src = (np.array([180.0, 150.0]) + generator.uniform(-12, 12, (80, 2))).astype(np.float32)
    dst = apply_affine(affine_matrix(translation=(9.0, 4.0)), src).astype(np.float32)

    evidence = verify(src, dst, cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)

    assert evidence.verified is False
    assert evidence.rejection_reason is RejectionReason.INSUFFICIENT_SPREAD
    assert evidence.inlier_spread < cfg.geometry.min_inlier_spread


def test_near_collinear_inliers_are_rejected(cfg: Settings) -> None:
    """Points on a line do not determine a homography, however tightly they fit.

    Deliberately *near*-collinear rather than exactly collinear: OpenCV refuses
    the exactly-degenerate case itself and returns ``estimation_failed``, which
    would leave this gate untested.
    """
    generator = np.random.default_rng(9)
    xs = np.linspace(20.0, 380.0, 120)
    src = np.stack([xs, 0.9 * xs + generator.uniform(-0.6, 0.6, xs.size)], axis=1).astype(
        np.float32
    )
    dst = apply_affine(affine_matrix(translation=(6.0, 5.0)), src).astype(np.float32)

    evidence = verify(src, dst, cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)

    assert evidence.verified is False
    assert evidence.rejection_reason in {
        RejectionReason.COLLINEAR_INLIERS,
        RejectionReason.INSUFFICIENT_SPREAD,
        RejectionReason.ESTIMATION_FAILED,
    }


def test_unrelated_point_sets_do_not_verify(cfg: Settings) -> None:
    """The negative control the legacy demo never ran."""
    generator = np.random.default_rng(13)
    src = generator.uniform(0, 400, (200, 2)).astype(np.float32)
    dst = generator.uniform(0, 400, (200, 2)).astype(np.float32)

    evidence = verify(src, dst, cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)

    assert evidence.verified is False


def test_too_few_matches_short_circuits(cfg: Settings) -> None:
    """Below 4 correspondences a homography is unsolvable, not merely unreliable.

    Nothing is measured, so nothing is claimed: the metrics stay at their
    unmeasured defaults rather than reporting a confident zero.
    """
    src = _grid(5)
    evidence = verify(src, src, cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)
    assert evidence.verified is False
    assert evidence.rejection_reason is RejectionReason.TOO_FEW_MATCHES
    assert evidence.inlier_count == 0
    assert evidence.transform is None
    assert evidence.homography is None
    assert evidence.reproj_rms is None


def test_mismatched_correspondence_counts_raise(cfg: Settings) -> None:
    with pytest.raises(ValueError, match="correspondence count mismatch"):
        verify(_grid(20), _grid(19), cfg.geometry, src_diagonal=DIAG, dst_diagonal=DIAG)


def test_rejection_reasons_all_have_explanations() -> None:
    for reason in RejectionReason:
        assert reason.explanation
        assert reason.explanation[0].isupper()


# ---------------------------------------------------------------------------
# distinct counting
# ---------------------------------------------------------------------------
def test_distinct_count_merges_subpixel_duplicates() -> None:
    """Two detections of one physical corner are one constraint, not two."""
    base = np.array([[10.0, 10.0], [50.0, 80.0]], dtype=np.float32)
    jittered = np.vstack([base, base + 0.1])
    assert distinct_count(jittered) == 2
    assert distinct_count(np.vstack([base, base + 5.0])) == 4
    assert distinct_count(np.zeros((0, 2), np.float32)) == 0


def test_geometry_config_rejects_unsolvable_thresholds() -> None:
    """A homography needs 4 correspondences; fewer is unsolvable, not lenient."""
    with pytest.raises(ValueError):
        GeometryConfig.model_validate(
            {
                "method": "magsac",
                "reproj_threshold": 4.0,
                "max_iters": 1000,
                "confidence": 0.999,
                "min_matches": 3,
                "min_inliers": 15,
                "min_inlier_ratio": 0.2,
                "min_distinct_inliers": 12,
                "min_inlier_spread": 0.1,
                "max_reproj_rms": 5.0,
                "max_condition_number": 1e3,
                "affine": {
                    "model": "full",
                    "method": "magsac",
                    "reproj_threshold": 4.0,
                    "anisotropy_tolerance": 0.05,
                    "shear_tolerance_deg": 2.0,
                },
            }
        )
