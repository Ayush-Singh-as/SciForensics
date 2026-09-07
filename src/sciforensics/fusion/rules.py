"""The uncalibrated rule model: features and weights to a verdict.

This replaces the legacy pipeline's four hard-coded verdict strings. It is a plain
additive log-odds model -- a prior plus a weighted sum of the bounded features in
:mod:`sciforensics.fusion.features`, squashed through a logistic -- chosen for three
properties that an ``if``/``elif`` chain does not have: every input's push on the
score is individually reportable, the whole thing is monotone in each feature by
construction, and it has the exact functional form that stage B5's fitted logistic
calibrator will drop into, so replacing it changes the numbers and nothing else.

**It is not a probability, and the code says so everywhere.** ``calibrated=False``
travels on every result, :attr:`~sciforensics.types.ScanResult.summary_line` prints
"(uncalibrated score)", and no weight here was fitted to data -- they encode a
reading of the evidence, argued below. Treating this output as a likelihood is
exactly the error the project is trying to stop making.

The weights encode one substantive commitment, stated in
:mod:`sciforensics.global_match`: *the global stage produces a score, never
evidence*. A high embedding similarity is a reason to look closer; the finding is a
geometrically verified correspondence set, and only the local stage can supply one.
So the table is built such that no pair can reach the ``likely_manipulated`` band on
embedding similarity alone -- in fact it cannot reach ``suspicious`` either. That is a
property of the numbers rather than a special case in the control flow: with
:data:`PRIOR_LOGIT` at -2.0 and the embedding weight at 3.0 over a feature bounded to
``[-0.5, 0.5]``, the best an unverified pair can do is ``sigmoid(-0.5) = 0.378``,
which lands just inside ``inconclusive``. ``tests/test_fusion.py`` searches the
feature space for a counterexample rather than trusting this paragraph.

Three consequences are worth stating plainly, because they are decisions rather than
accidents. Every number below is measured, either over the 15 manipulation pairs and
6 negative controls in ``inputs/`` or by evaluating the table at a stated point.

*In practice the embedding alone does not even reach ``inconclusive``, and that is a
fact about the checkpoint rather than about these weights.* Reaching the 0.378 ceiling
needs a similarity of 1.0; measured, this checkpoint's similarity on a true positive
tops out at 0.873, which scores **0.293** -- ``clean``. The same measurement shows why
no choice of weight fixes that: ``base_cell`` against its own JPEG-degraded copy
scores 0.230, *below* the 0.378 that the genuinely unrelated
``base_cell``/``base_cells_2`` pair scores. A real manipulation ranked beneath a
negative control is not a threshold problem, and any weight large enough to promote
the first would promote the second along with it. Stages B2 and B3 exist for this.
Until then, an unconfirmed screening hit is reported as what it is -- no evidence
found -- and :func:`_notes` attaches the reason the search may have failed, which is
the part of the output that keeps a false negative legible.

*A refuted verification pushes below the prior.* ``geometry_refuted`` carries -1.4, so
even the best-scoring true positive drops to **0.093** once its correspondences fail
the degeneracy gates. Matches that existed and did not survive are evidence against
reuse, not merely an absence of evidence for it -- which is the distinction the
abstention feature exists to protect (see :mod:`sciforensics.fusion.features`).

*A bare-minimum verification lands in ``suspicious``, not ``likely_manipulated``.* A
fit that passes every gate at its floor -- 15 inliers over 12 distinct points, 0.18
inlier ratio, 4.9 px reprojection error against a 5.0 px budget, 5% of the frame --
scores **0.809**. Crossing 0.85 needs corroboration from somewhere: a tighter fit,
more distinct correspondences, a larger verified area, or a transform that is itself a
manipulation. This is the one place the rule table is deliberately more conservative
than the gates it sits behind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sciforensics.config import BandsConfig
from sciforensics.fusion.features import (
    COPY_MOVE_FEATURES,
    FEATURES,
    CopyMoveBundle,
    EvidenceBundle,
    Feature,
    extract,
    extract_copy_move,
)
from sciforensics.types import EvidenceContribution, Verdict

__all__ = [
    "COPY_MOVE_PRIOR_LOGIT",
    "COPY_MOVE_WEIGHTS",
    "PRIOR_LOGIT",
    "WEIGHTS",
    "Fused",
    "band",
    "fuse",
    "fuse_copy_move",
    "logistic",
]


#: Log-odds of reuse before any evidence: ``sigmoid(-2.0) = 0.12``.
#:
#: A base rate, not a tuning knob. Most pairs a corpus-scale scan examines are
#: unrelated, so a model that starts at even odds would report a coin flip for every
#: pair it knows nothing about. It is also what makes the score's *floor* meaningful:
#: an unverified pair with a confidently dissimilar embedding reports 0.03, not 0.5.
PRIOR_LOGIT = -2.0

#: Per-feature weights, in log-odds per unit of feature. Because every feature is
#: bounded (see :mod:`sciforensics.fusion.features`), these are directly comparable
#: to each other and each one's maximum swing is ``weight * (hi - lo)``.
#:
#: The shape of the table, rather than any individual number, is the claim:
#:
#: * ``geometry_verified`` is the largest single positive term at 2.4, and the five
#:   features that can only be non-zero alongside it sum to 4.5 -- nearly double the
#:   verification term itself. Verified geometry *plus corroboration* is the only
#:   route to the top band; verification alone is not enough.
#: * ``embedding_similarity`` is weighted higher than any single geometry term but is
#:   bounded to ``[-0.5, 0.5]``, so its total swing is 3.0 against the 6.9 available to
#:   a fully corroborated verification. It is the strongest *screen* and a weak
#:   *proof*, which is what it is.
#: * ``geometry_abstained`` is exactly 0.0. Not an oversight -- see
#:   :func:`~sciforensics.fusion.features._geometry_abstained`.
#: * The two penalties are negative and, unlike the strength features, are *not*
#:   gated on verification: how much to trust a match count is a question that
#:   survives the fit succeeding. Together they are strong enough to matter -- a
#:   comfortably verified pair scoring 0.974 falls to 0.837, out of the top band, if
#:   its keypoints were 20x asymmetric and its matches many-to-one.
WEIGHTS: dict[str, float] = {
    "embedding_similarity": 3.0,
    "geometry_verified": 2.4,
    "inlier_strength": 1.2,
    "inlier_ratio": 1.0,
    "reproj_tightness": 0.8,
    "matched_area": 0.6,
    "transform_manipulated": 0.9,
    "geometry_refuted": -1.4,
    "geometry_abstained": 0.0,
    "keypoint_asymmetry": -1.0,
    "non_injective": -1.0,
}

#: Prior for the copy-move question, lower than :data:`PRIOR_LOGIT` at
#: ``sigmoid(-2.5) = 0.076``. Cloning within a single figure is rarer than reuse
#: between two figures a user chose to compare, and unlike the pair path there is no
#: cheap screening stage whose output could raise the baseline.
COPY_MOVE_PRIOR_LOGIT = -2.5

#: Copy-move weights. ``clone_verified`` dominates more here than
#: ``geometry_verified`` does on the pair path, because a surviving region has already
#: cleared a stricter bar: DBSCAN agreement on ``(dx, dy, log s, theta)``, a minimum
#: spatial separation so a keypoint cannot match itself, *and* its own affine fit.
COPY_MOVE_WEIGHTS: dict[str, float] = {
    "clone_verified": 3.2,
    "clone_strength": 1.2,
    "clone_area": 0.8,
    "clone_tightness": 0.6,
    "cluster_survival": 0.5,
}


def _check_table(
    features: tuple[Feature[EvidenceBundle], ...] | tuple[Feature[CopyMoveBundle], ...],
    weights: dict[str, float],
    label: str,
) -> None:
    """Fail at import if a weight table and its feature table have drifted apart.

    Adding a feature without a weight would otherwise silently score it as zero, and
    a weight for a feature that no longer exists would be silently ignored. Both are
    the kind of change that produces plausible numbers from a model that is no longer
    the model anyone reviewed.
    """
    named = {f.name for f in features}
    if missing := sorted(named - weights.keys()):
        raise RuntimeError(f"{label}: features without a weight: {', '.join(missing)}")
    if extra := sorted(weights.keys() - named):
        raise RuntimeError(f"{label}: weights for unknown features: {', '.join(extra)}")


_check_table(FEATURES, WEIGHTS, "fusion.rules.WEIGHTS")
_check_table(COPY_MOVE_FEATURES, COPY_MOVE_WEIGHTS, "fusion.rules.COPY_MOVE_WEIGHTS")


@dataclass(frozen=True)
class Fused:
    """A verdict, the number behind it, and the arithmetic that produced it."""

    verdict: Verdict
    confidence: float
    #: Signed log-odds pushes, prior first and then features ranked by magnitude.
    #: These sum to :attr:`logit` exactly, which is what makes the contribution bars
    #: in the report an audit rather than an illustration.
    contributions: tuple[EvidenceContribution, ...]
    #: Every feature that was evaluated, including the zeros. Retained so the API can
    #: re-score a cached result under different bands without re-running the
    #: pipeline, and so the report JSON records the input to the score and not only
    #: its output.
    features: dict[str, float] = field(default_factory=dict)
    logit: float = 0.0
    #: Disagreements between stages that a reader needs told about, in plain prose.
    notes: tuple[str, ...] = ()
    #: ``False`` for everything this module produces. Only a fitted calibrator from
    #: :mod:`sciforensics.fusion.calibrate` may set it.
    calibrated: bool = False


def logistic(logit: float) -> float:
    """Numerically stable ``1 / (1 + exp(-logit))``.

    The naive form overflows for a logit around -750, which the weights above cannot
    reach -- but ``sciforensics eval`` will fit and apply calibrators whose
    coefficients are unbounded, and a score that raises ``OverflowError`` on one row
    of a benchmark sweep is a worse failure than a saturated one.
    """
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    odds = math.exp(logit)
    return odds / (1.0 + odds)


def band(probability: float, bands: BandsConfig) -> Verdict:
    """Map a score onto its verdict band.

    Boundaries are inclusive at the bottom of each band, so a score exactly equal to
    ``bands.likely_manipulated`` reports that band. ``BandsConfig`` validates that
    the three thresholds strictly decrease, so the order of these tests is total.
    """
    if probability >= bands.likely_manipulated:
        return Verdict.LIKELY_MANIPULATED
    if probability >= bands.suspicious:
        return Verdict.SUSPICIOUS
    if probability >= bands.inconclusive:
        return Verdict.INCONCLUSIVE
    return Verdict.CLEAN


def _contributions(
    values: dict[str, float],
    weights: dict[str, float],
    descriptions: dict[str, str],
    prior: float,
) -> tuple[float, tuple[EvidenceContribution, ...]]:
    """Accumulate the log-odds from ``prior`` and describe each term.

    Only features with a non-zero *value* are reported. A zero feature contributed
    nothing and a bar of length zero for each of them would bury the ones that
    mattered -- but a non-zero feature with a zero weight *is* reported, because
    "the local stage abstained and that moved the score not at all" is something the
    reader needs to see rather than infer from an absence.
    """
    logit = prior
    reported: list[EvidenceContribution] = []
    for name, value in values.items():
        contribution = value * weights[name]
        logit += contribution
        if value != 0.0:
            reported.append(
                EvidenceContribution(
                    name=name,
                    value=value,
                    weight=weights[name],
                    contribution=contribution,
                    description=descriptions[name],
                )
            )
    reported.sort(key=lambda c: abs(c.contribution), reverse=True)
    return logit, tuple(reported)


def _prior(logit: float, description: str) -> EvidenceContribution:
    """The base rate as a contribution, so the reported terms sum to the logit."""
    return EvidenceContribution(
        name="prior",
        value=1.0,
        weight=logit,
        contribution=logit,
        description=description,
    )


def fuse(bundle: EvidenceBundle, bands: BandsConfig) -> Fused:
    """Score a pair comparison against the rule table.

    Parameters
    ----------
    bundle
        Evidence from however many stages ran; the missing ones extract to neutral.
    bands
        ``fusion.bands`` from the active configuration. Passed in rather than read
        from a module-level default so that the API's threshold sliders can re-band a
        cached score without touching global state.
    """
    values = extract(bundle)
    descriptions = {f.name: f.description for f in FEATURES}
    logit, contributions = _contributions(values, WEIGHTS, descriptions, PRIOR_LOGIT)
    prior = _prior(
        PRIOR_LOGIT, "Base rate: most image pairs are unrelated before evidence is weighed"
    )
    probability = logistic(logit)
    return Fused(
        verdict=band(probability, bands),
        confidence=probability,
        contributions=(prior, *contributions),
        features=values,
        logit=logit,
        notes=_notes(values, bundle),
        calibrated=False,
    )


def fuse_copy_move(bundle: CopyMoveBundle, bands: BandsConfig) -> Fused:
    """Score a copy-move search against the copy-move rule table.

    Same shape as :func:`fuse` and deliberately so -- one ``Fused`` type means the
    report renderer, the JSON schema and the contribution bars are written once --
    but a different prior, a different feature table and no calibrator seam.
    """
    values = extract_copy_move(bundle)
    descriptions = {f.name: f.description for f in COPY_MOVE_FEATURES}
    logit, contributions = _contributions(
        values, COPY_MOVE_WEIGHTS, descriptions, COPY_MOVE_PRIOR_LOGIT
    )
    prior = _prior(
        COPY_MOVE_PRIOR_LOGIT,
        "Base rate: duplication within a single figure is uncommon before evidence",
    )
    probability = logistic(logit)
    return Fused(
        verdict=band(probability, bands),
        confidence=probability,
        contributions=(prior, *contributions),
        features=values,
        logit=logit,
        calibrated=False,
    )


def _notes(values: dict[str, float], bundle: EvidenceBundle) -> tuple[str, ...]:
    """Prose for what a bare number hides.

    Two kinds of case. The genuine *disagreements* -- one stage says reuse and the
    other says no -- are triggered on the sign of the centred embedding feature, above
    zero meaning above the boundary the network was trained against, so no threshold is
    invented here that is not already in the configuration.

    An *abstention* is reported either way, and that asymmetry is deliberate. It is not
    a disagreement, it is a measurement that was never taken, so it qualifies the
    verdict regardless of which side the embedding came down on. The case where both
    stages come up empty is exactly the project's canonical false negative -- a
    JPEG-degraded pair where ORB collapses to a handful of matches on a manipulation
    that is unquestionably present -- and it scores 0.06, below the base rate, with a
    confident-looking ``clean`` beside it. Saying nothing there would be the single
    most misleading thing this module could do.

    The wording is checked against what the score actually reports. A refuted
    verification with a positive screening hit scores 0.09, which is ``clean``, so this
    says the geometry *contradicted* the screen and names the residual risk; it does
    not call the pair unresolved, because the verdict beside it does not.
    """
    notes: list[str] = []
    embedding_agrees = values["embedding_similarity"] > 0.0
    geometry = bundle.geometry

    if values["geometry_refuted"] and embedding_agrees and geometry is not None:
        notes.append(
            "The embedding placed these images on the same side of its decision boundary, but "
            f"geometric verification contradicted it: {geometry.rejection_reason.explanation} "
            "The geometry is the stronger signal and the verdict follows it. Note the residual "
            "risk: a real manipulation whose keypoints are destroyed by compression or blur can "
            "also land here."
        )
    if values["geometry_abstained"]:
        notes.append(
            (
                "The local stage found too little to test, so this verdict rests on the embedding "
                "alone -- which on this checkpoint cannot lift a pair above 'no evidence found' "
                "without corroboration. A weak-texture or heavily compressed panel produces this "
                "even when a manipulation is present."
            )
            if embedding_agrees
            else (
                "Neither stage found anything, but only one of them actually looked: the local "
                "stage had too little to test and reached no conclusion. Read this as an absence "
                "of evidence rather than evidence of absence. Heavy JPEG compression, noise or "
                "blur suppresses the keypoints this stage depends on, and it suppresses the "
                "embedding similarity at the same time -- so a genuinely manipulated pair can "
                "produce exactly this result."
            )
        )
    if values["geometry_verified"] and not embedding_agrees:
        notes.append(
            "Geometry verified a transform between these images even though the embedding scored "
            "them below its match threshold. The verified correspondences are the stronger "
            "evidence; the embedding is the stage that missed this."
        )
    return tuple(notes)
