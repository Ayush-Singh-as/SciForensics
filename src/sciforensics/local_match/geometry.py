"""Robust geometric verification and interpretable transform decomposition.

Three of the audited defects are fixed here, and they are the three that made the
legacy pipeline actively misleading rather than merely imprecise.

**Bug 2 — flip detection was mathematically unreachable.** The legacy code fit
``cv2.estimateAffinePartial2D``, a 4-DOF *similarity* transform whose matrix is
``[[a, -b, tx], [b, a, ty]]``. Its determinant is ``a**2 + b**2``, which is
strictly positive for any non-degenerate fit. It then tested ``det < 0`` to
report a flip. That branch could never be taken, for any input, ever: the
pipeline reported "Flip detected: no" for every mirrored image it was ever
shown. :func:`estimate_affine` uses ``cv2.estimateAffine2D`` (6 DOF), where a
reflection is representable and ``det < 0`` genuinely detects it.

**Bug 3 — degenerate fits were reported as successes.** Because a similarity
transform cannot express a mirror, fitting one to mirrored correspondences
collapses. On the ``mountains`` pair the legacy output was ``scale 0.000`` with a
1635 px translation -- physically meaningless numbers, presented with three
decimal places and no caveat.

**Bug 4 — many-to-one matches inflated the inlier count.** With no mutual-nearest
-neighbour constraint, 2000 keypoints on the left could all match the same 12 on
the right. The legacy scanner turned that into "231 good matches, 107 inliers,
Strong evidence". 107 inlier *rows* over 12 unique points is 12 independent
constraints. :func:`verify` counts distinct correspondences, measures their
spatial spread, checks reprojection RMS and rejects ill-conditioned models --
and records *which* gate failed via :class:`~sciforensics.types.RejectionReason`
so a rejection is explicable rather than a shrug.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from sciforensics.config import AffineConfig, GeometryConfig
from sciforensics.runtime import get_logger
from sciforensics.types import AffineDecomposition, BBox, GeometryEvidence, RejectionReason

__all__ = [
    "Verification",
    "compose_affine",
    "convex_hull",
    "decompose_affine",
    "distinct_count",
    "estimate_affine",
    "method_flag",
    "polygon_area",
    "reprojection_errors",
    "rescale_affine",
    "rescale_box",
    "rescale_homography",
    "verify",
    "verify_with_mask",
]

_log = get_logger(__name__)

#: Robust-estimator flags. MAGSAC++ is the default because it is markedly less
#: sensitive to the inlier threshold than plain RANSAC -- which matters when the
#: threshold is a user-facing config value someone will inevitably mis-set.
_METHODS: dict[str, int] = {
    "magsac": getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
    "ransac": cv2.RANSAC,
    "lmeds": cv2.LMEDS,
}

#: Correspondences are deduplicated on a half-pixel grid. Sub-pixel jitter from
#: the enhancement upscale would otherwise make two detections of the same
#: physical corner count as two independent constraints.
_DEDUP_QUANTUM = 2.0  # 1/0.5 px

#: Ratio of second to first singular value of the centred inlier cloud below
#: which the points are treated as collinear. Four collinear points do not
#: determine a homography, however tightly they fit one.
_COLLINEARITY_RATIO = 0.02


def method_flag(name: str) -> int:
    """Map a config method name to its OpenCV flag."""
    try:
        return _METHODS[name]
    except KeyError:  # pragma: no cover - config Literal prevents this
        raise ValueError(
            f"unknown estimator {name!r}; expected one of {sorted(_METHODS)}"
        ) from None


def _as_points(array: np.ndarray) -> np.ndarray:
    """Normalise any accepted point layout to a contiguous ``(N, 2)`` float32."""
    pts = np.asarray(array, dtype=np.float32).reshape(-1, 2)
    return np.ascontiguousarray(pts)


# ---------------------------------------------------------------------------
# coordinate spaces
# ---------------------------------------------------------------------------
def rescale_homography(homography: np.ndarray, *, src_scale: float, dst_scale: float) -> np.ndarray:
    """Re-express an analysis-space homography in original-image coordinates.

    Analysis runs on images capped at ``image.max_dimension``, and the two inputs
    of a pair are almost never capped by the same factor -- a 3840x2400 image and
    a 2343x1421 one both land at 2048 wide, at scales 0.533 and 0.874. A
    homography fitted between those two *analysis* images therefore carries a
    spurious ``0.874 / 0.533 = 1.639`` scale factor that has nothing to do with
    the manipulation.

    That is not a rounding detail, it is a wrong answer of exactly the kind this
    refactor exists to eliminate: on the ``mountains`` pair the true scale is
    1.0, and reporting "scale 1.64" would tell a reader the image had been
    enlarged 64% when it had not. Numbers in the report must live in the frame of
    the images the reader is looking at.

    Given ``p_a = s * p_o`` for each side::

        p_o_dst = S_dst^-1 @ H_analysis @ S_src @ p_o_src

    with ``S = diag(s, s, 1)``.
    """
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    if src_scale == dst_scale == 1.0:
        return matrix
    s_src = np.diag([src_scale, src_scale, 1.0])
    s_dst_inv = np.diag([1.0 / dst_scale, 1.0 / dst_scale, 1.0])
    rescaled = s_dst_inv @ matrix @ s_src
    # Renormalise so H[2, 2] == 1; the conjugation above leaves it at 1 already
    # for affine-like matrices but not in general, and downstream readers (and
    # the condition-number check) assume the normalised form.
    if abs(rescaled[2, 2]) > 1e-12:
        rescaled = rescaled / rescaled[2, 2]
    return rescaled


def rescale_affine(affine: np.ndarray, *, src_scale: float, dst_scale: float) -> np.ndarray:
    """:func:`rescale_homography` for a ``2x3`` affine matrix.

    The linear part is unchanged in *ratio* terms but the translation is not, so
    this cannot be done by scaling the translation column alone.
    """
    matrix = np.asarray(affine, dtype=np.float64).reshape(2, 3)
    if src_scale == dst_scale == 1.0:
        return matrix
    homogeneous = np.vstack([matrix, [0.0, 0.0, 1.0]])
    return rescale_homography(homogeneous, src_scale=src_scale, dst_scale=dst_scale)[:2, :]


def rescale_box(box: BBox, *, scale: float) -> BBox:
    """The box member of the rescale family: divide ``(x, y, w, h)`` by ``scale``.

    Frame-agnostic on purpose. The convention throughout is
    ``p_fine = scale * p_coarse`` -- enhancement space is ``enhance.scale`` times
    analysis space, and analysis space is ``LoadedImage.analysis_scale`` times
    original space -- so both conversions are this one division, and both callers
    get the same rounding.

    Width and height are floored at 1 rather than allowed to round to 0: a region
    that exists is not zero pixels wide, and a zero-area box would silently
    disappear from an overlay.
    """
    inv = 1.0 / scale
    x, y, w, h = box
    return (round(x * inv), round(y * inv), max(1, round(w * inv)), max(1, round(h * inv)))


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------
def decompose_affine(
    matrix: np.ndarray,
    *,
    model: Literal["full", "similarity"] = "full",
    anisotropy_tolerance: float = 0.05,
    shear_tolerance_deg: float = 2.0,
) -> AffineDecomposition:
    """Decompose a ``2x3`` affine matrix into interpretable components.

    Uses a single QR-style factorisation of the linear part ``A``::

        A = R(theta) @ [[sx,  m ],
                        [0,   sy]]

    so the map reads as "scale (and shear) in image axes, then rotate". Being one
    factorisation rather than two means the reported components recompose exactly
    into ``A`` -- which is a property the test suite asserts against random
    matrices, and which a mixed SVD/QR report cannot offer.

    Derivation: with ``A = [[a, c], [b, d]]``, taking ``sx = hypot(a, b)`` and
    ``theta = atan2(b, a)`` makes ``R(theta)^-1 A`` upper triangular with
    ``sy = det(A) / sx`` and ``m = (a*c + b*d) / sx``.

    Flip convention
    ---------------
    Since ``sx > 0`` always, ``sign(sy) == sign(det A)``: the reflection lands in
    ``sy`` automatically, giving a **vertical** (``y → -y``) convention for free.
    ``(flip, rotation)`` is not a unique parameterisation -- mirroring vertically
    then rotating by ``theta`` equals mirroring horizontally then rotating by
    ``theta + 180`` -- so a convention has to be fixed, and this is ours.
    ``flip`` itself is convention-independent: it is just ``det(A) < 0``.

    ``scale_x`` and ``scale_y`` are reported as magnitudes; the sign lives in
    ``flip`` so that ``scale`` is always a meaningful positive number.

    Parameters
    ----------
    matrix
        ``2x3`` affine matrix, as returned by ``cv2.estimateAffine2D``.
    model
        Recorded on the result so a reader can tell whether ``flip`` was even
        observable: a ``"similarity"`` fit cannot represent a reflection.
    """
    affine = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    linear = affine[:, :2]
    tx, ty = float(affine[0, 2]), float(affine[1, 2])

    a, c = float(linear[0, 0]), float(linear[0, 1])
    b, d = float(linear[1, 0]), float(linear[1, 1])

    determinant = a * d - b * c
    flip = determinant < 0.0

    scale_x = math.hypot(a, b)
    if scale_x < 1e-12:
        # A collapsed transform: the first basis vector has no length, so
        # rotation and shear are undefined. The legacy code reported
        # "scale: 0.000" here with three decimal places and full confidence;
        # we return zeros and let the caller's gates reject the fit.
        _log.debug("degenerate linear part in affine decomposition: %s", linear.tolist())
        return AffineDecomposition(
            model=model,
            rotation_deg=0.0,
            scale_x=0.0,
            scale_y=0.0,
            shear_deg=0.0,
            translation=(tx, ty),
            determinant=determinant,
            flip=flip,
            anisotropic=False,
            sheared=False,
        )

    rotation_deg = math.degrees(math.atan2(b, a))
    signed_scale_y = determinant / scale_x
    scale_y = abs(signed_scale_y)
    shear_offset = (a * c + b * d) / scale_x
    shear_deg = math.degrees(math.atan2(shear_offset, scale_y)) if scale_y > 1e-12 else 0.0

    ratio = scale_x / scale_y if scale_y > 1e-12 else float("inf")
    anisotropic = abs(ratio - 1.0) > anisotropy_tolerance
    sheared = abs(shear_deg) > shear_tolerance_deg

    return AffineDecomposition(
        model=model,
        rotation_deg=rotation_deg,
        scale_x=scale_x,
        scale_y=scale_y,
        shear_deg=shear_deg,
        translation=(tx, ty),
        determinant=determinant,
        flip=flip,
        anisotropic=anisotropic,
        sheared=sheared,
    )


def compose_affine(decomposition: AffineDecomposition) -> np.ndarray:
    """Rebuild the ``2x3`` matrix a :class:`AffineDecomposition` describes.

    Exact inverse of :func:`decompose_affine`. Exists so the round-trip is an
    executable property rather than a claim in a docstring.
    """
    theta = math.radians(decomposition.rotation_deg)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    signed_scale_y = -decomposition.scale_y if decomposition.flip else decomposition.scale_y
    shear_offset = math.tan(math.radians(decomposition.shear_deg)) * decomposition.scale_y
    upper = np.array(
        [[decomposition.scale_x, shear_offset], [0.0, signed_scale_y]], dtype=np.float64
    )
    linear = rotation @ upper
    tx, ty = decomposition.translation
    return np.hstack([linear, np.array([[tx], [ty]], dtype=np.float64)])


def estimate_affine(
    src: np.ndarray,
    dst: np.ndarray,
    cfg: AffineConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit an affine transform mapping ``src`` onto ``dst``.

    Returns ``(matrix, inlier_mask)``, either of which may be ``None`` if the
    estimator failed.

    ``cfg.model == "full"`` uses ``cv2.estimateAffine2D`` (6 DOF: rotation,
    anisotropic scale, shear and reflection). ``"similarity"`` uses
    ``estimateAffinePartial2D`` (4 DOF) and is retained only so the benchmark can
    quantify what the legacy choice cost; it cannot detect a flip.
    """
    src_pts, dst_pts = _as_points(src), _as_points(dst)
    if len(src_pts) < 3 or len(src_pts) != len(dst_pts):
        return None, None

    estimator = cv2.estimateAffine2D if cfg.model == "full" else cv2.estimateAffinePartial2D
    flag = method_flag(cfg.method)

    try:
        matrix, mask = estimator(
            src_pts,
            dst_pts,
            method=flag,
            ransacReprojThreshold=cfg.reproj_threshold,
        )
    except cv2.error:
        # Not every OpenCV build accepts every USAC flag for the affine
        # estimators (unlike findHomography, where USAC support is universal).
        # Fall back rather than failing the whole comparison over an estimator
        # preference.
        if flag != cv2.RANSAC:
            _log.debug("affine estimator rejected method=%s; falling back to RANSAC", cfg.method)
            try:
                matrix, mask = estimator(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=cfg.reproj_threshold,
                )
            except cv2.error:
                _log.debug("affine estimation failed outright", exc_info=True)
                return None, None
        else:
            _log.debug("affine estimation failed outright", exc_info=True)
            return None, None

    if matrix is None:
        return None, None
    return np.asarray(matrix, dtype=np.float64), mask


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def distinct_count(points: np.ndarray) -> int:
    """Unique points on a half-pixel grid.

    This is the number that matters for bug 4. ``inlier_count`` counts *rows* of
    the correspondence table; this counts the independent geometric constraints
    those rows actually carry.
    """
    if len(points) == 0:
        return 0
    quantised = np.round(np.asarray(points, dtype=np.float64) * _DEDUP_QUANTUM).astype(np.int64)
    return len(np.unique(quantised, axis=0))


def _spread(points: np.ndarray, diagonal: float) -> float:
    """Bounding-box diagonal of a point cloud, as a fraction of the image's."""
    if len(points) < 2 or diagonal <= 0:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    extent = pts.max(axis=0) - pts.min(axis=0)
    return float(math.hypot(extent[0], extent[1]) / diagonal)


def _is_collinear(points: np.ndarray) -> bool:
    """Whether a point cloud has effectively no second dimension."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return True
    centred = pts - pts.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[0] < 1e-9:
        return True
    return bool(singular[1] / singular[0] < _COLLINEARITY_RATIO)


def reprojection_errors(homography: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Symmetric transfer error per correspondence, in pixels.

    Averages the forward (``src -> dst`` under ``H``) and backward
    (``dst -> src`` under ``H^-1``) distances. The one-directional error that is
    usually reported can be made small by a transform that squashes everything
    toward a point -- exactly the failure mode that produced the legacy
    ``scale 0.000`` result -- whereas the symmetric error cannot.
    """
    src_pts = _as_points(src).reshape(-1, 1, 2)
    dst_pts = _as_points(dst).reshape(-1, 1, 2)

    forward = cv2.perspectiveTransform(src_pts, homography).reshape(-1, 2)
    forward_err = np.linalg.norm(forward - dst_pts.reshape(-1, 2), axis=1)

    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return np.asarray(forward_err, dtype=np.float64)

    backward = cv2.perspectiveTransform(dst_pts, inverse).reshape(-1, 2)
    backward_err = np.linalg.norm(backward - src_pts.reshape(-1, 2), axis=1)
    return np.asarray(0.5 * (forward_err + backward_err), dtype=np.float64)


def _condition_number(homography: np.ndarray, *, src_diagonal: float, dst_diagonal: float) -> float:
    """Condition number of the homography in *normalised* coordinates.

    Taking the SVD of a raw pixel-space homography measures the wrong thing. Such
    a matrix mixes units: the linear block is O(1) while the translation column is
    O(image size), so the condition number scales with the translation magnitude
    rather than with any geometric degeneracy. The 180-degree rotation recovered
    from the ``mountains`` pair -- a healthy fit, 377 inliers at 0.46 px RMS --
    scores 9.6e6 purely because its translation is (3356, 2097). A gate on that
    number would reject large images for being large.

    Conditioning both coordinate frames to the unit scale first (Hartley-style)
    removes the units problem, leaving a number that responds to what actually
    matters: a transform that collapses one direction, or an extreme projective
    warp. Healthy fits land near 1-10; degenerate ones diverge.
    """
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    scale = matrix[2, 2]
    if abs(scale) > 1e-12:
        matrix = matrix / scale

    d_src = src_diagonal if src_diagonal > 1e-9 else 1.0
    d_dst = dst_diagonal if dst_diagonal > 1e-9 else 1.0
    # x_n = N x  and  y_n = M y  with N = diag(1/d_src, 1/d_src, 1), so the
    # normalised map is  H_n = M @ H @ N^-1.
    n_inv = np.diag([d_src, d_src, 1.0])
    m = np.diag([1.0 / d_dst, 1.0 / d_dst, 1.0])
    normalised = m @ matrix @ n_inv

    singular = np.linalg.svd(normalised, compute_uv=False)
    if singular[-1] < 1e-15:
        return float("inf")
    return float(singular[0] / singular[-1])


def convex_hull(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Convex hull of a point cloud, as a tuple of vertices."""
    pts = _as_points(points)
    if len(pts) < 3:
        return tuple((float(x), float(y)) for x, y in pts)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    return tuple((float(p[0][0]), float(p[0][1])) for p in hull)


def polygon_area(vertices: tuple[tuple[float, float], ...]) -> float:
    """Shoelace area of a simple polygon."""
    if len(vertices) < 3:
        return 0.0
    arr = np.asarray(vertices, dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Verification:
    """Reportable geometry evidence plus the per-correspondence consensus mask.

    The same split as :class:`~sciforensics.local_match.copymove.CopyMoveDetection`:
    :attr:`evidence` is the pydantic record that goes into the report JSON, and the
    numpy member is working data the renderer consumes and then discards. A boolean
    array of length *number of correspondences* has no business in a report schema,
    but the report needs it to draw which lines held and which did not.

    **The mask is populated on rejection too, and that is the point.** It marks the
    estimator's consensus set, which is a claim about a model that the gates may
    still have refused -- so it is only *evidence* when
    :attr:`~sciforensics.types.GeometryEvidence.verified` is true. Reading it
    otherwise would reinstate exactly the error this module exists to prevent.
    Keeping it is what lets a report *show* a rejection: on the ``mountains`` pair
    the legacy pipeline claimed 121 inliers, and 121 lines converging on a dozen
    points is a far more convincing account of
    :attr:`~sciforensics.types.RejectionReason.DEGENERATE_CORRESPONDENCES` than the
    sentence is.
    """

    evidence: GeometryEvidence
    #: ``(N,)`` bool over the input correspondences, in their original order.
    #: All-``False`` when no model was fitted at all.
    inlier_mask: np.ndarray

    @property
    def verified(self) -> bool:
        return self.evidence.verified


def _no_consensus(n: int) -> np.ndarray:
    """An all-``False`` mask, for rejections that happened before any fit.

    Length-correct rather than empty, so a caller can index its correspondence
    array by the mask unconditionally.
    """
    return np.zeros(n, dtype=bool)


def _rejected(
    reason: RejectionReason,
    method: str,
    inlier_mask: np.ndarray,
    **measured: object,
) -> Verification:
    """Build a failed :class:`Verification` that still carries its metrics.

    Populating the measurements on rejection is the point: it lets a report say
    "rejected: inliers spanned 3% of the diagonal, 10% required" instead of an
    unhelpful bare "not verified".
    """
    _log.debug("geometry rejected (%s): %s", reason.value, measured)
    evidence = GeometryEvidence(
        verified=False,
        rejection_reason=reason,
        method=method,
        # Each caller passes the subset of measurements its gate had available.
        # pydantic validates the values; mypy cannot see through the ``**``.
        **measured,  # type: ignore[arg-type]
    )
    return Verification(evidence=evidence, inlier_mask=inlier_mask)


def verify(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    cfg: GeometryConfig,
    *,
    src_diagonal: float,
    dst_diagonal: float,
    src_area: float | None = None,
    src_scale: float = 1.0,
    dst_scale: float = 1.0,
) -> GeometryEvidence:
    """:func:`verify_with_mask` for callers that only want the evidence record.

    The common case, and the one the test suite is written against. Use
    :func:`verify_with_mask` when you also need to know which correspondences the
    estimator agreed with -- the report and the web demo do.
    """
    return verify_with_mask(
        src_pts,
        dst_pts,
        cfg,
        src_diagonal=src_diagonal,
        dst_diagonal=dst_diagonal,
        src_area=src_area,
        src_scale=src_scale,
        dst_scale=dst_scale,
    ).evidence


def verify_with_mask(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    cfg: GeometryConfig,
    *,
    src_diagonal: float,
    dst_diagonal: float,
    src_area: float | None = None,
    src_scale: float = 1.0,
    dst_scale: float = 1.0,
) -> Verification:
    """Fit and validate a homography between two matched point sets.

    Every gate below exists because the legacy pipeline passed a case it should
    have rejected. Gates are ordered cheapest-first, and each records the
    measurement that tripped it.

    Parameters
    ----------
    src_pts, dst_pts
        Matched ``(N, 2)`` coordinates in *analysis* space, in correspondence
        order.
    src_diagonal, dst_diagonal
        Image diagonals, used to make the spread gate resolution-independent.
    src_area
        Left-image area in px², used for ``matched_area_fraction``. Derived from
        the diagonal assuming 4:3 if omitted.
    src_scale, dst_scale
        Original-to-analysis scale factors of the two images
        (:attr:`~sciforensics.io.images.LoadedImage.analysis_scale`). The fit and
        every gate run in analysis space, but the *reported* homography,
        decomposition and hulls are converted back to original-image pixels --
        see :func:`rescale_homography` for why this is not cosmetic.
    """
    src = _as_points(src_pts)
    dst = _as_points(dst_pts)
    method = cfg.method

    if len(src) != len(dst):
        raise ValueError(f"correspondence count mismatch: {len(src)} src vs {len(dst)} dst")

    n_matches = len(src)
    if n_matches < cfg.min_matches:
        return _rejected(RejectionReason.TOO_FEW_MATCHES, method, _no_consensus(n_matches))

    # --- fit ---------------------------------------------------------------
    try:
        homography, mask = cv2.findHomography(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            method_flag(cfg.method),
            cfg.reproj_threshold,
            maxIters=cfg.max_iters,
            confidence=cfg.confidence,
        )
    except cv2.error:
        _log.debug("findHomography raised", exc_info=True)
        return _rejected(RejectionReason.ESTIMATION_FAILED, method, _no_consensus(n_matches))

    if homography is None or mask is None:
        return _rejected(RejectionReason.ESTIMATION_FAILED, method, _no_consensus(n_matches))

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / n_matches
    src_in, dst_in = src[inlier_mask], dst[inlier_mask]

    # --- measurements (computed before gating so failures stay explicable) --
    distinct = min(distinct_count(src_in), distinct_count(dst_in))
    spread = min(_spread(src_in, src_diagonal), _spread(dst_in, dst_diagonal))
    condition = _condition_number(homography, src_diagonal=src_diagonal, dst_diagonal=dst_diagonal)
    errors = reprojection_errors(homography, src_in, dst_in) if inlier_count else np.array([])
    reproj_rms = float(np.sqrt(np.mean(errors**2))) if errors.size else None

    measured: dict[str, object] = {
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "distinct_inliers": distinct,
        "inlier_spread": spread,
        "reproj_rms": reproj_rms,
        "condition_number": condition if math.isfinite(condition) else None,
    }

    # --- gates -------------------------------------------------------------
    # Ordered by how specific the diagnosis is, not just by cost. When several
    # gates would fail, the most structural explanation is the most useful one
    # to report: "your matches are many-to-one" tells a caller what to fix,
    # whereas "the inlier ratio was low" is a symptom of it.
    if inlier_count < cfg.min_inliers:
        return _rejected(RejectionReason.TOO_FEW_INLIERS, method, inlier_mask, **measured)

    # Bug 4. The legacy pipeline accepted 107 inlier rows spread over only 12
    # unique keypoints and called it "Strong evidence". Rows are not
    # constraints; distinct correspondences are.
    if distinct < cfg.min_distinct_inliers:
        return _rejected(
            RejectionReason.DEGENERATE_CORRESPONDENCES, method, inlier_mask, **measured
        )

    if inlier_ratio < cfg.min_inlier_ratio:
        return _rejected(RejectionReason.LOW_INLIER_RATIO, method, inlier_mask, **measured)

    if spread < cfg.min_inlier_spread:
        return _rejected(RejectionReason.INSUFFICIENT_SPREAD, method, inlier_mask, **measured)

    if _is_collinear(src_in) or _is_collinear(dst_in):
        return _rejected(RejectionReason.COLLINEAR_INLIERS, method, inlier_mask, **measured)

    if reproj_rms is not None and reproj_rms > cfg.max_reproj_rms:
        return _rejected(RejectionReason.HIGH_REPROJECTION_ERROR, method, inlier_mask, **measured)

    if not math.isfinite(condition) or condition > cfg.max_condition_number:
        return _rejected(RejectionReason.ILL_CONDITIONED, method, inlier_mask, **measured)

    # --- accepted: decompose and describe ----------------------------------
    # Everything above ran in analysis space, which is correct: the gates are
    # either ratios (spread, inlier fraction) or are calibrated in analysis
    # pixels (reproj_threshold, max_reproj_rms). Everything reported from here
    # down is converted to ORIGINAL image pixels, because that is the frame the
    # reader's images live in.
    reported_h = rescale_homography(homography, src_scale=src_scale, dst_scale=dst_scale)

    transform: AffineDecomposition | None = None
    affine, _ = estimate_affine(src_in, dst_in, cfg.affine)
    if affine is not None:
        transform = decompose_affine(
            rescale_affine(affine, src_scale=src_scale, dst_scale=dst_scale),
            model=cfg.affine.model,
            anisotropy_tolerance=cfg.affine.anisotropy_tolerance,
            shear_tolerance_deg=cfg.affine.shear_tolerance_deg,
        )

    hull_left = convex_hull(src_in / src_scale)
    hull_right = convex_hull(dst_in / dst_scale)
    if src_area is None:
        # Diagonal d at 4:3 gives area 12/25 * d^2. Only used for a reported
        # fraction, never for a gate, so the assumption is harmless.
        src_area = 0.48 * src_diagonal**2
    # matched_area_fraction is a ratio, so it is scale-invariant; compute it in
    # analysis space where src_area was measured.
    matched_fraction = (
        min(1.0, polygon_area(convex_hull(src_in)) / src_area) if src_area > 0 else 0.0
    )

    return Verification(
        evidence=GeometryEvidence(
            verified=True,
            rejection_reason=RejectionReason.NONE,
            method=method,
            homography=tuple(tuple(float(v) for v in row) for row in reported_h),  # type: ignore[arg-type]
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            distinct_inliers=distinct,
            inlier_spread=spread,
            reproj_rms=reproj_rms,
            condition_number=condition,
            transform=transform,
            hull_left=hull_left,
            hull_right=hull_right,
            matched_area_fraction=matched_fraction,
        ),
        inlier_mask=inlier_mask,
    )
