"""Evidence to a bounded, named feature vector.

Every number that reaches a score passes through one of the two tables in this
module, and nothing else does. The legacy verdict was an ``if``/``elif`` chain over
four raw quantities, which meant there was no way to ask what a score was made of,
no way to fit a replacement for it, and no way to notice when a threshold edit
quietly changed what "strong evidence" meant.

Three properties of the tables are deliberate.

*Bounded.* Every feature lands inside a stated ``[lo, hi]``. Weights are therefore
directly comparable -- 2.4 really does outrank 1.2 -- which is what makes the
per-feature contribution bars in the report readable rather than decorative, and
what lets :mod:`sciforensics.fusion.rules` reason about its own ceiling.

*Defined when the local stage is absent.* The pipeline skips keypoints and
geometry entirely for a pair the embedder places far apart, so half of these
inputs are routinely ``None``. Absent evidence extracts to ``0.0`` -- neutral,
neither incriminating nor exculpatory -- so "we did not look" and "we looked and
found nothing" are different vectors rather than the same one.

*Gated on verification.* The strength features -- inlier count, inlier ratio,
reprojection tightness, matched area -- are read only from a **verified**
:class:`~sciforensics.types.GeometryEvidence`. A rejected fit still reports all of
those measurements (deliberately: they are what explains the rejection), and
feeding them in would let a fit that failed its gates raise the score in
proportion to how nearly it passed. They belong in the report, not in the score.

**Abstention is not refutation.** A refused verification means one of two quite
different things, and collapsing them is how a detector limitation comes to read as
an exoneration. ``TOO_FEW_MATCHES`` and ``TOO_FEW_KEYPOINTS`` mean the local stage
never had enough material to hold an opinion -- exactly what happens today on the
JPEG-degraded pairs, where ORB collapses to 2-7 matches on a manipulation that is
unquestionably present. Every other reason means there *was* material and it did
not survive the gates. The two go into separate features
(:data:`ABSTAINING_REASONS` selects the first kind) and the rule table weights only
the second, so an abstention leaves the verdict resting visibly on the embedding
rather than silently reading as "clean".
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from sciforensics.types import (
    CopyMoveEvidence,
    GeometryEvidence,
    GlobalEvidence,
    KeypointEvidence,
    MatchEvidence,
    RejectionReason,
)

__all__ = [
    "ABSTAINING_REASONS",
    "ASYMMETRY_FREE",
    "ASYMMETRY_SATURATED",
    "COPY_MOVE_FEATURES",
    "COPY_MOVE_FEATURE_NAMES",
    "FEATURES",
    "FEATURE_NAMES",
    "REFERENCE_INLIERS",
    "CopyMoveBundle",
    "EvidenceBundle",
    "Feature",
    "as_tuple",
    "extract",
    "extract_copy_move",
    "feature",
]


#: Rejection reasons that mean *the local stage could not form an opinion*, as
#: opposed to *it formed one and it was negative*. See the module docstring.
#:
#: ``ESTIMATION_FAILED`` is the debatable member. The estimator returning no model
#: from at least ``geometry.min_matches`` correspondences usually means those
#: correspondences were degenerate, which is closer to a refutation -- but it can
#: also be a numerical failure, and reading a refutation out of "the solver gave up"
#: claims more than was measured. It is classified as an abstention here, and
#: abstention is carried as its own feature rather than folded into the others, so
#: stage B5 can settle the question with data instead of judgement.
ABSTAINING_REASONS = frozenset(
    {
        RejectionReason.TOO_FEW_MATCHES,
        RejectionReason.TOO_FEW_KEYPOINTS,
        RejectionReason.ESTIMATION_FAILED,
    }
)

#: Inlier count at which the correspondence-count features saturate. Not a
#: threshold -- nothing is accepted or rejected by it -- but the scale on which
#: "how many correspondences agreed" stops adding confidence. A verified fit with
#: 100 geometrically distinct agreeing correspondences is not meaningfully less
#: certain than one with 400, and a log ramp to this reference keeps the feature
#: from tracking image resolution instead of evidence strength.
REFERENCE_INLIERS = 100

#: Keypoint-count asymmetry below which no penalty applies. Two panels of
#: different texture density routinely differ 2x, and that is unremarkable.
ASYMMETRY_FREE = 2.0

#: Asymmetry at which the penalty is full. ``local_match.min_keypoints_per_side``
#: now makes the legacy 2000-vs-12 collapse (167x) unreachable, so this ramp covers
#: the residual imbalance that the floor still permits.
ASYMMETRY_SATURATED = 20.0


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceBundle:
    """Everything the pair-comparison feature table reads.

    Assembled by the pipeline once the stages that ran have reported. The three
    optional members are ``None`` when their stage did not run at all, which is a
    distinct case from a stage that ran and produced nothing -- see the module
    docstring.
    """

    global_evidence: GlobalEvidence
    keypoints: KeypointEvidence | None = None
    matches: MatchEvidence | None = None
    geometry: GeometryEvidence | None = None
    #: ``geometry.max_reproj_rms`` from the active configuration, used only to
    #: normalise :func:`_reproj_tightness` onto ``[0, 1]``. Required rather than
    #: defaulted so this module cannot become a second home for a threshold that
    #: already lives in ``configs/default.yaml``.
    reproj_rms_limit: float = 5.0

    @property
    def verified(self) -> GeometryEvidence | None:
        """The geometry evidence only if it passed every gate, else ``None``.

        The accessor every strength feature goes through, so "read this only from a
        verified fit" is expressed once instead of repeated per feature.
        """
        geometry = self.geometry
        return geometry if geometry is not None and geometry.verified else None


@dataclass(frozen=True)
class CopyMoveBundle:
    """Everything the copy-move feature table reads.

    Separate from :class:`EvidenceBundle` because copy-move is a single-image
    question with no counterpart to compare against: there is no embedding
    distance, no keypoint asymmetry between sides, and no global transform. Sharing
    one table would mean most of it extracting to zero on every call.
    """

    evidence: CopyMoveEvidence
    #: Original-pixel area of the image searched, for :func:`_clone_area`.
    image_area: float
    #: ``copy_move.reproj_threshold`` from the active configuration. Copy-move
    #: verification is per cluster and uses its own budget, so this is a different
    #: number from :attr:`EvidenceBundle.reproj_rms_limit`.
    reproj_rms_limit: float = 4.0


BundleT = TypeVar("BundleT", EvidenceBundle, CopyMoveBundle)


@dataclass(frozen=True)
class Feature(Generic[BundleT]):
    """One named, bounded input to a score.

    ``description`` is reader-facing: it is copied onto
    :class:`~sciforensics.types.EvidenceContribution` and rendered beside the
    contribution bar, so it has to explain what the number means to someone who has
    not read this file.
    """

    name: str
    lo: float
    hi: float
    description: str
    extract: Callable[[BundleT], float]

    def __call__(self, bundle: BundleT) -> float:
        """Extract this feature, clamped to its declared range.

        Clamping here rather than trusting each extractor keeps the bound a property
        of the table -- which is what weight comparability and the ceiling argument
        in :mod:`sciforensics.fusion.rules` both rely on.
        """
        return _clip(self.extract(bundle), self.lo, self.hi)


def _clip(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        # An infinite feature would poison the whole log-odds sum. The only source
        # is KeypointEvidence.asymmetry with an empty side, which *means* maximum
        # asymmetry, so saturating at the bound is also the correct answer.
        return hi if value > 0 else lo
    return max(lo, min(hi, value))


def _ramp(value: float, start: float, end: float) -> float:
    """Linear 0-to-1 ramp between ``start`` and ``end``, unclamped.

    Callers do not clamp: :meth:`Feature.__call__` does it once for every feature.
    """
    if end <= start:
        return 1.0 if value >= end else 0.0
    return (value - start) / (end - start)


def _log_ramp(count: int, reference: int) -> float:
    """Diminishing-returns ramp on a count, reaching 1.0 at ``reference``."""
    return math.log1p(max(count, 0)) / math.log1p(reference)


# ---------------------------------------------------------------------------
# pair-comparison extractors
# ---------------------------------------------------------------------------
def _embedding_similarity(bundle: EvidenceBundle) -> float:
    """Centred embedding similarity: negative below the decision boundary.

    ``GlobalEvidence.similarity`` is already ``sigmoid((threshold - d) / T)``, so it
    is 0.5 exactly on the boundary the network was trained against. Centring it is
    what makes the feature signed, and therefore what lets a confidently dissimilar
    pair push the score *down* rather than merely fail to push it up.
    """
    return bundle.global_evidence.similarity - 0.5


def _geometry_verified(bundle: EvidenceBundle) -> float:
    return 1.0 if bundle.verified is not None else 0.0


def _geometry_refuted(bundle: EvidenceBundle) -> float:
    """1.0 when a fit was attempted on adequate material and failed its gates."""
    geometry = bundle.geometry
    if geometry is None or geometry.verified:
        return 0.0
    return 0.0 if _abstained(geometry) else 1.0


def _geometry_abstained(bundle: EvidenceBundle) -> float:
    """1.0 when the local stage ran but had too little to work with.

    Weighted zero by the rule table on purpose. It is carried anyway because the
    reader needs to see *why* a verdict rests on the embedding alone, and because
    stage B5 needs the column in order to fit a weight for it.
    """
    geometry = bundle.geometry
    if geometry is None or geometry.verified:
        return 0.0
    return 1.0 if _abstained(geometry) else 0.0


def _abstained(geometry: GeometryEvidence) -> bool:
    """Whether a rejection means "could not tell" rather than "did not hold up".

    A rejection with no reason recorded should be unreachable -- ``geometry.verify``
    names a reason on every failure path -- but reading it as an abstention is the
    conservative choice if one ever appears, since it claims nothing.
    """
    return (
        geometry.rejection_reason in ABSTAINING_REASONS
        or geometry.rejection_reason is RejectionReason.NONE
    )


def _inlier_strength(bundle: EvidenceBundle) -> float:
    """Agreeing-correspondence count on a log ramp to :data:`REFERENCE_INLIERS`.

    ``distinct_inliers`` rather than ``inlier_count``: rows are not constraints.
    That distinction is the whole of bug 4, and using the raw count here would
    reintroduce it at the scoring layer after the geometry gate had already rejected
    it at the fitting layer.
    """
    geometry = bundle.verified
    return 0.0 if geometry is None else _log_ramp(geometry.distinct_inliers, REFERENCE_INLIERS)


def _inlier_ratio(bundle: EvidenceBundle) -> float:
    geometry = bundle.verified
    return 0.0 if geometry is None else geometry.inlier_ratio


def _reproj_tightness(bundle: EvidenceBundle) -> float:
    """How far inside the reprojection budget the fit landed, ``0`` at the limit.

    Expressed as headroom rather than as error so that, like every other feature
    here, larger means stronger evidence.
    """
    geometry = bundle.verified
    if geometry is None or geometry.reproj_rms is None:
        return 0.0
    return 1.0 - _ramp(geometry.reproj_rms, 0.0, bundle.reproj_rms_limit)


def _matched_area(bundle: EvidenceBundle) -> float:
    """Fraction of the left frame covered by the verified inlier hull.

    A transform verified over 2% of the image and one verified over 60% are not the
    same claim, and the legacy pipeline could not tell them apart.
    """
    geometry = bundle.verified
    return 0.0 if geometry is None else geometry.matched_area_fraction


def _transform_manipulated(bundle: EvidenceBundle) -> float:
    """Share of {flip, anisotropic scale, shear} present in the fitted transform.

    These separate *reuse* from *manipulated reuse*: an honest duplicate figure is a
    near-identity transform, whereas a mirrored or non-uniformly rescaled one had to
    be worked on. Reachable at all only because bug 2 is fixed -- under the legacy
    similarity model ``det = a^2 + b^2 > 0`` always, so ``flip`` was unreportable
    and this feature would have been permanently understated.
    """
    geometry = bundle.verified
    if geometry is None or geometry.transform is None:
        return 0.0
    flags = (geometry.transform.flip, geometry.transform.anisotropic, geometry.transform.sheared)
    return sum(flags) / len(flags)


def _keypoint_asymmetry(bundle: EvidenceBundle) -> float:
    """Penalty ramp on how unevenly the two sides were sampled.

    Not a measure of manipulation -- a measure of how much to trust the match count
    that follows. It is the surviving trace of bug 4's mechanism: whatever matches
    between 2000 keypoints and 12 says more about the sampling than about the images.
    """
    keypoints = bundle.keypoints
    if keypoints is None:
        return 0.0
    asymmetry = keypoints.asymmetry
    if not math.isfinite(asymmetry):
        return 1.0
    return _ramp(
        math.log10(max(asymmetry, 1.0)),
        math.log10(ASYMMETRY_FREE),
        math.log10(ASYMMETRY_SATURATED),
    )


def _non_injective(bundle: EvidenceBundle) -> float:
    """1.0 when the surviving matches are many-to-one.

    Unreachable while ``local_match.mutual_nn`` is on, where injectivity holds by
    construction. Kept because the setting is a setting: someone will turn it off to
    reproduce the legacy behaviour, and when they do the score should notice.
    """
    matches = bundle.matches
    if matches is None:
        return 0.0
    return 0.0 if matches.is_injective else 1.0


# ---------------------------------------------------------------------------
# copy-move extractors
# ---------------------------------------------------------------------------
def _clone_verified(bundle: CopyMoveBundle) -> float:
    """1.0 when at least one candidate cluster survived its own affine fit."""
    return 1.0 if bundle.evidence.regions else 0.0


def _clone_strength(bundle: CopyMoveBundle) -> float:
    """Total agreeing correspondences across reported regions, log-ramped."""
    total = sum(region.inlier_count for region in bundle.evidence.regions)
    return _log_ramp(total, REFERENCE_INLIERS)


def _clone_area(bundle: CopyMoveBundle) -> float:
    """Share of the image covered by cloned regions, both lobes counted.

    A 2%-of-frame clone and a 40% one are different findings. Lobes can overlap in
    principle, so this is an upper bound on true coverage rather than an exact
    measure; :meth:`Feature.__call__` clamps it at 1.0.
    """
    if bundle.image_area <= 0:
        return 0.0
    return sum(region.area_px for region in bundle.evidence.regions) / bundle.image_area


def _clone_tightness(bundle: CopyMoveBundle) -> float:
    """Mean per-region reprojection headroom, ``0`` at the budget.

    Mean rather than best: a second loosely-fitting region should not be hidden by a
    first tight one, since every reported region is a separate claim.
    """
    errors = [r.reproj_rms for r in bundle.evidence.regions if r.reproj_rms is not None]
    if not errors:
        return 0.0
    return 1.0 - _ramp(sum(errors) / len(errors), 0.0, bundle.reproj_rms_limit)


def _cluster_survival(bundle: CopyMoveBundle) -> float:
    """Share of DBSCAN candidate clusters that survived verification.

    The copy-move analogue of ``inlier_ratio``: a run where one of twelve candidate
    offsets verified looks much more like clustering noise than one where the single
    candidate found held up. Zero clusters gives 0.0, which is also what zero
    regions gives -- correctly, since neither found anything.
    """
    clusters = bundle.evidence.cluster_count
    if clusters <= 0:
        return 0.0
    return len(bundle.evidence.regions) / clusters


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------
#: The pair-comparison feature schema, in a fixed order.
#:
#: Order is part of the contract: :mod:`sciforensics.fusion.calibrate` matches a
#: fitted artifact's coefficient vector against :data:`FEATURE_NAMES` and refuses to
#: run if they disagree. Appending a feature is safe; reordering or renaming one
#: invalidates every calibrator fitted before the change, which is exactly what that
#: check exists to catch.
FEATURES: tuple[Feature[EvidenceBundle], ...] = (
    Feature(
        name="embedding_similarity",
        lo=-0.5,
        hi=0.5,
        description="Whole-image embedding similarity, relative to the trained decision boundary",
        extract=_embedding_similarity,
    ),
    Feature(
        name="geometry_verified",
        lo=0.0,
        hi=1.0,
        description="A transform between the two images survived every degeneracy gate",
        extract=_geometry_verified,
    ),
    Feature(
        name="inlier_strength",
        lo=0.0,
        hi=1.0,
        description="How many geometrically distinct correspondences agreed with that transform",
        extract=_inlier_strength,
    ),
    Feature(
        name="inlier_ratio",
        lo=0.0,
        hi=1.0,
        description="Share of candidate matches that agreed with the transform",
        extract=_inlier_ratio,
    ),
    Feature(
        name="reproj_tightness",
        lo=0.0,
        hi=1.0,
        description="How far inside the reprojection error budget the fit landed",
        extract=_reproj_tightness,
    ),
    Feature(
        name="matched_area",
        lo=0.0,
        hi=1.0,
        description="Fraction of the image covered by the verified region",
        extract=_matched_area,
    ),
    Feature(
        name="transform_manipulated",
        lo=0.0,
        hi=1.0,
        description="The transform involves a flip, an anisotropic rescale or a shear",
        extract=_transform_manipulated,
    ),
    Feature(
        name="geometry_refuted",
        lo=0.0,
        hi=1.0,
        description="A transform was fitted on adequate material and failed verification",
        extract=_geometry_refuted,
    ),
    Feature(
        name="geometry_abstained",
        lo=0.0,
        hi=1.0,
        description="The local stage found too little to test, so it reached no conclusion",
        extract=_geometry_abstained,
    ),
    Feature(
        name="keypoint_asymmetry",
        lo=0.0,
        hi=1.0,
        description="The two images were sampled very unevenly, so match counts are unreliable",
        extract=_keypoint_asymmetry,
    ),
    Feature(
        name="non_injective",
        lo=0.0,
        hi=1.0,
        description="Surviving matches are many-to-one, which inflates counts without evidence",
        extract=_non_injective,
    ),
)

#: The copy-move feature schema. Rule-scored only: stage B5 calibrates the pair
#: path, for which labelled negatives are abundant, and leaves this one uncalibrated
#: until B4 supplies per-pixel ground truth from the Polimi masks.
COPY_MOVE_FEATURES: tuple[Feature[CopyMoveBundle], ...] = (
    Feature(
        name="clone_verified",
        lo=0.0,
        hi=1.0,
        description="At least one duplicated region survived its own geometric fit",
        extract=_clone_verified,
    ),
    Feature(
        name="clone_strength",
        lo=0.0,
        hi=1.0,
        description="How many correspondences support the reported regions",
        extract=_clone_strength,
    ),
    Feature(
        name="clone_area",
        lo=0.0,
        hi=1.0,
        description="Fraction of the image occupied by duplicated content",
        extract=_clone_area,
    ),
    Feature(
        name="clone_tightness",
        lo=0.0,
        hi=1.0,
        description="How precisely the duplicated lobes align",
        extract=_clone_tightness,
    ),
    Feature(
        name="cluster_survival",
        lo=0.0,
        hi=1.0,
        description="Share of candidate offsets that held up as real duplications",
        extract=_cluster_survival,
    ),
)

#: Feature names in :data:`FEATURES` order.
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)

#: Feature names in :data:`COPY_MOVE_FEATURES` order.
COPY_MOVE_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in COPY_MOVE_FEATURES)


def feature(name: str) -> Feature[EvidenceBundle] | Feature[CopyMoveBundle]:
    """Look one feature up by name across both tables.

    Raises ``KeyError`` naming every available feature, because the callers that
    reach for a feature by name -- a calibrator file, a ``--set`` override, a test --
    are exactly the places where a typo would otherwise be silently ignored.
    """
    for candidate in FEATURES:
        if candidate.name == name:
            return candidate
    for cm_candidate in COPY_MOVE_FEATURES:
        if cm_candidate.name == name:
            return cm_candidate
    available = ", ".join(FEATURE_NAMES + COPY_MOVE_FEATURE_NAMES)
    raise KeyError(f"unknown feature {name!r}; available: {available}")


def extract(bundle: EvidenceBundle) -> dict[str, float]:
    """Evaluate :data:`FEATURES`, in table order.

    A ``dict`` rather than an array because every downstream consumer -- the rule
    table, the contribution bars, the report JSON -- wants the names, and pairing a
    bare vector with a separate name list is how a calibrator ends up applying the
    right weights to the wrong columns.
    """
    return {f.name: f(bundle) for f in FEATURES}


def extract_copy_move(bundle: CopyMoveBundle) -> dict[str, float]:
    """Evaluate :data:`COPY_MOVE_FEATURES`, in table order."""
    return {f.name: f(bundle) for f in COPY_MOVE_FEATURES}


def as_tuple(values: dict[str, float]) -> tuple[float, ...]:
    """Order a pair-comparison feature mapping into a vector.

    The one place a named mapping becomes positional, which is where a calibrator's
    coefficients get lined up against it -- so a missing name raises here rather
    than shifting every subsequent column by one.
    """
    missing = [name for name in FEATURE_NAMES if name not in values]
    if missing:
        raise KeyError(f"feature vector is missing {', '.join(missing)}")
    return tuple(values[name] for name in FEATURE_NAMES)
