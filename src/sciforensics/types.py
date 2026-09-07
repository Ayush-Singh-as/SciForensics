"""Result types shared by the CLI, the report renderer and the HTTP API.

These are pydantic models rather than plain dataclasses so that one definition
serves all three consumers: ``model_dump(mode="json")`` produces the report
JSON, the same classes annotate the FastAPI routes (giving the web demo a
generated OpenAPI schema for free), and the Jinja2 templates render attributes
off the very objects the pipeline produced.

The design principle throughout is **report the evidence, not just the
conclusion**. The pre-refactor pipeline emitted a hand-written verdict string
and a handful of numbers; when it was wrong -- and on the ``mountains`` pair it
was confidently, catastrophically wrong -- nothing in the output revealed why.
Every gate that can reject a result therefore records a machine-readable
:class:`RejectionReason` alongside the measurement that tripped it.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AffineDecomposition",
    "AuditBlock",
    "CopyMoveEvidence",
    "CopyMoveRegion",
    "CopyMoveResult",
    "EvidenceContribution",
    "GeometryEvidence",
    "GlobalEvidence",
    "ImageMeta",
    "KeypointEvidence",
    "MatchEvidence",
    "RejectionReason",
    "ScanResult",
    "Stage",
    "Verdict",
]

BBox = tuple[int, int, int, int]  # x, y, w, h; the coordinate frame is stated per field
Point = tuple[float, float]
Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Verdict(str, Enum):
    """Human-readable band derived from the fused confidence."""

    LIKELY_MANIPULATED = "likely_manipulated"
    SUSPICIOUS = "suspicious"
    INCONCLUSIVE = "inconclusive"
    CLEAN = "clean"

    @property
    def label(self) -> str:
        return {
            Verdict.LIKELY_MANIPULATED: "Likely reused or manipulated",
            Verdict.SUSPICIOUS: "Suspicious — warrants expert review",
            Verdict.INCONCLUSIVE: "Inconclusive",
            Verdict.CLEAN: "No evidence of reuse found",
        }[self]


class Stage(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    GEOMETRY = "geometry"
    COPY_MOVE = "copy_move"
    FUSION = "fusion"


class RejectionReason(str, Enum):
    """Why a geometric verification was refused.

    Each value maps to exactly one gate in
    :mod:`sciforensics.local_match.geometry`. Surfacing these is the difference
    between "not verified" and the legacy behaviour of reporting a *verified*
    homography with ``scale = 0.000`` and 107 inliers spread over 12 points.
    """

    NONE = "none"
    TOO_FEW_MATCHES = "too_few_matches"
    TOO_FEW_KEYPOINTS = "too_few_keypoints"
    ESTIMATION_FAILED = "estimation_failed"
    TOO_FEW_INLIERS = "too_few_inliers"
    LOW_INLIER_RATIO = "low_inlier_ratio"
    #: Many inlier rows collapsing onto few unique keypoints -- the degeneracy
    #: that manufactured the legacy pipeline's phantom 107 "inliers".
    DEGENERATE_CORRESPONDENCES = "degenerate_correspondences"
    #: Inliers confined to a small patch cannot constrain a global transform.
    INSUFFICIENT_SPREAD = "insufficient_spread"
    HIGH_REPROJECTION_ERROR = "high_reprojection_error"
    ILL_CONDITIONED = "ill_conditioned"
    #: Inliers are (near-)collinear, so the homography is underdetermined.
    COLLINEAR_INLIERS = "collinear_inliers"

    @property
    def explanation(self) -> str:
        return {
            RejectionReason.NONE: "Verification passed all gates.",
            RejectionReason.TOO_FEW_MATCHES: (
                "Not enough surviving correspondences to fit a transform."
            ),
            RejectionReason.TOO_FEW_KEYPOINTS: (
                "One or both images yielded too few keypoints to compare."
            ),
            RejectionReason.ESTIMATION_FAILED: "The robust estimator returned no model.",
            RejectionReason.TOO_FEW_INLIERS: (
                "Too few correspondences agreed with the fitted transform."
            ),
            RejectionReason.LOW_INLIER_RATIO: (
                "The fraction of agreeing correspondences was too low."
            ),
            RejectionReason.DEGENERATE_CORRESPONDENCES: (
                "Inliers collapsed onto too few distinct keypoints: many-to-one matches inflate "
                "the inlier count without adding independent geometric constraints."
            ),
            RejectionReason.INSUFFICIENT_SPREAD: (
                "Inliers were confined to a small region, which cannot constrain a global "
                "transform."
            ),
            RejectionReason.HIGH_REPROJECTION_ERROR: (
                "Inliers did not fit the transform tightly enough."
            ),
            RejectionReason.ILL_CONDITIONED: ("The estimated transform is numerically degenerate."),
            RejectionReason.COLLINEAR_INLIERS: (
                "Inliers were near-collinear, leaving the transform underdetermined."
            ),
        }[self]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
class ImageMeta(_Base):
    """Provenance and shape of one input, recorded for the audit trail."""

    path: Path
    sha256: str
    width: int
    height: int
    channels: int
    file_format: str | None = None
    size_bytes: int | None = None
    #: Set when :data:`ImageConfig.max_dimension` forced a downscale. All
    #: reported coordinates are mapped back to ``(width, height)`` regardless,
    #: but a reader deserves to know the analysis ran at reduced resolution.
    analysed_at: tuple[int, int] | None = None

    @property
    def was_downscaled(self) -> bool:
        return self.analysed_at is not None and self.analysed_at != (self.width, self.height)


# ---------------------------------------------------------------------------
# stage 1: global embedding
# ---------------------------------------------------------------------------
class GlobalEvidence(_Base):
    distance: float
    #: ``sigmoid((threshold - distance) / temperature)``, so ``0.5`` sits
    #: exactly on the decision boundary and the full ``(0, 1)`` range is
    #: reachable. The legacy ``sigmoid(1 - distance)`` saturated at 0.731.
    similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    distance_threshold: float
    local_trigger_distance: float
    is_match: bool
    triggered_local: bool
    embedding_dim: int
    #: ``None`` when attribution is disabled or the backbone offers no hook.
    attribution_left: Path | None = None
    attribution_right: Path | None = None


# ---------------------------------------------------------------------------
# stage 2: keypoints and matching
# ---------------------------------------------------------------------------
class KeypointEvidence(_Base):
    detector: str
    detected_left: int
    detected_right: int
    #: Counts after the difference-of-Gaussians region-of-interest filter.
    kept_left: int
    kept_right: int
    #: ``True`` when the ROI restriction was dropped because it would have
    #: pushed a side below ``min_keypoints_per_side``. The legacy code silently
    #: allowed 12-vs-2000 asymmetry here; recording it makes the asymmetry
    #: auditable instead of invisible.
    roi_abandoned_left: bool = False
    roi_abandoned_right: bool = False
    enhancement_scale: float = 1.0

    @property
    def asymmetry(self) -> float:
        """Ratio of the larger surviving keypoint set to the smaller.

        Values far from 1.0 mean the two sides were not comparably sampled, and
        any match count between them should be read with suspicion.
        """
        lo, hi = sorted((self.kept_left, self.kept_right))
        return float("inf") if lo == 0 else hi / lo


class MatchEvidence(_Base):
    matcher: str
    #: Candidate pairs before any filtering.
    raw: int
    #: Survivors of Lowe's ratio test.
    ratio_passed: int
    #: Survivors of the mutual-nearest-neighbour constraint. With mutual NN
    #: enforced this is by construction an injective set, so
    #: ``good == distinct_left == distinct_right``.
    good: int
    distinct_left: int
    distinct_right: int
    mutual_nn: bool
    ratio: float

    @property
    def is_injective(self) -> bool:
        return self.good == self.distinct_left == self.distinct_right


# ---------------------------------------------------------------------------
# stage 3: geometry
# ---------------------------------------------------------------------------
class AffineDecomposition(_Base):
    """SVD decomposition of the linear part of a fitted affine transform.

    Fitted with ``cv2.estimateAffine2D`` (6 DOF). The extra two degrees of
    freedom over a similarity transform are what make ``flip`` and
    ``anisotropic`` observable at all.
    """

    model: Literal["full", "similarity"]
    rotation_deg: float
    scale_x: float
    scale_y: float
    shear_deg: float
    translation: Point
    determinant: float
    #: ``determinant < 0``. Reachable only when ``model == "full"``.
    flip: bool
    anisotropic: bool
    sheared: bool

    @property
    def scale(self) -> float:
        """Geometric mean of the principal scales."""
        return float(abs(self.scale_x * self.scale_y) ** 0.5)


class GeometryEvidence(_Base):
    verified: bool
    rejection_reason: RejectionReason = RejectionReason.NONE
    method: str
    homography: Matrix3x3 | None = None
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    #: Unique keypoints participating in the inlier set. Compare against
    #: ``inlier_count``: a large gap means many-to-one matching.
    distinct_inliers: int = 0
    #: Bounding-box diagonal of the inlier cloud as a fraction of the image
    #: diagonal.
    inlier_spread: float = 0.0
    #: Root-mean-square symmetric reprojection error over the inlier set, px.
    reproj_rms: float | None = None
    condition_number: float | None = None
    transform: AffineDecomposition | None = None
    #: Convex hull of verified inliers in each image -- the honest overlay.
    hull_left: tuple[Point, ...] = ()
    hull_right: tuple[Point, ...] = ()
    matched_area_fraction: float = 0.0


# ---------------------------------------------------------------------------
# copy-move
# ---------------------------------------------------------------------------
class CopyMoveRegion(_Base):
    """One cloned region: a pair of lobes related by a single affine transform.

    Copy-move detection is symmetric and therefore cannot tell which lobe is the
    original and which is the paste -- both hold the same content, and nothing in
    the geometry distinguishes them. ``source_box`` and ``target_box`` are
    assigned deterministically by canonical offset direction (the target lies
    down-and-right of the source) so that repeated runs agree, and the names carry
    no forensic claim about direction.
    """

    #: Both boxes are in ORIGINAL image pixels.
    source_box: BBox
    target_box: BBox
    #: Median displacement from source to target lobe, original pixels.
    offset: Point
    rotation_deg: float
    scale: float
    flip: bool
    inlier_count: int
    reproj_rms: float | None = None
    #: Written by the report layer; the detector produces the mask in memory.
    mask_path: Path | None = None
    #: Area of the union of both lobes, original pixels.
    area_px: int = 0


class CopyMoveEvidence(_Base):
    detected: bool
    #: DBSCAN clusters over ``(dx, dy, log s, theta)``, *before* per-cluster
    #: geometric verification. Compare against ``len(regions)``: the gap is how
    #: many candidate offsets failed to hold up as an affine clone, and it is the
    #: same kind of funnel as ``MatchEvidence.raw`` -> ``good``. The legacy
    #: implementation fit a single global transform, so it could never report
    #: more than one region regardless of how many clones were present.
    cluster_count: int = 0
    regions: tuple[CopyMoveRegion, ...] = ()
    self_matches: int = 0
    keypoints: int = 0


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------
class EvidenceContribution(_Base):
    """One feature's signed push on the fused log-odds.

    Powers the per-evidence contribution bars in the web demo, so a reader can
    see *which* signal drove the score rather than trusting a bare number.
    """

    name: str
    value: float
    weight: float
    contribution: float
    description: str = ""


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
class AuditBlock(_Base):
    """Everything needed to reproduce a result, embedded in every report."""

    tool_version: str
    config_fingerprint: str
    timestamp_utc: str
    device: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    python_version: str | None = None
    torch_version: str | None = None
    opencv_version: str | None = None
    weights_sha256: str | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# top-level results
# ---------------------------------------------------------------------------
class ScanResult(_Base):
    """Outcome of comparing two images."""

    left: ImageMeta
    right: ImageMeta

    global_evidence: GlobalEvidence
    keypoints: KeypointEvidence | None = None
    matches: MatchEvidence | None = None
    geometry: GeometryEvidence | None = None

    verdict: Verdict
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    #: ``False`` until a calibrator is fitted (stage B5). Reports and the API
    #: must label an uncalibrated confidence as such: it is a monotone score,
    #: not a probability, and presenting it as the latter would be the same
    #: category of overclaiming the legacy verdict strings committed.
    calibrated: bool = False
    contributions: tuple[EvidenceContribution, ...] = ()
    #: Caveats the *verdict* carries, in prose, from
    #: :func:`sciforensics.fusion.rules._notes`. Distinct from :attr:`warnings`
    #: and rendered next to the confidence rather than in a footer: these say
    #: what a confident-looking number is hiding -- that only one stage actually
    #: looked, or that two stages disagreed -- and a reader who misses them
    #: misreads the result. ``warnings`` are operational facts about the run.
    notes: tuple[str, ...] = ()
    #: Which stages actually ran, for the report's methodology section.
    stages_run: tuple[Stage, ...] = ()
    #: Wall-clock seconds per stage.
    timings: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    artifacts: dict[str, Path] = Field(default_factory=dict)
    audit: AuditBlock

    @property
    def summary_line(self) -> str:
        qualifier = "" if self.calibrated else " (uncalibrated score)"
        return f"{self.verdict.label} — confidence {self.confidence:.2f}{qualifier}"


class CopyMoveResult(_Base):
    """Outcome of searching a single image for cloned regions."""

    image: ImageMeta
    evidence: CopyMoveEvidence
    verdict: Verdict
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    calibrated: bool = False
    #: Same shape as :attr:`ScanResult.contributions`, from the copy-move rule
    #: table, so one renderer draws the bars for both result types.
    contributions: tuple[EvidenceContribution, ...] = ()
    #: Always empty today -- the copy-move scorer has no cross-stage
    #: disagreement to report. Present so the two result types stay
    #: interchangeable to a template.
    notes: tuple[str, ...] = ()
    timings: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    artifacts: dict[str, Path] = Field(default_factory=dict)
    audit: AuditBlock

    @property
    def summary_line(self) -> str:
        n = len(self.evidence.regions)
        if not self.evidence.detected:
            return "No cloned regions found"
        return (
            f"{n} cloned region{'s' if n != 1 else ''} localized — confidence {self.confidence:.2f}"
        )
