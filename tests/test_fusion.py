"""Evidence fusion: the ceiling, the monotonicity, and the calibrator seam.

Two of these tests are searches rather than assertions, which is the point.

:mod:`sciforensics.fusion.rules` makes a structural claim in its own docstring -- *no
pair can reach* ``suspicious`` *on embedding similarity alone* -- and a docstring is
not a guard. :func:`test_no_unverified_pair_can_reach_suspicious` enumerates every
reachable corner of the unverified feature space and reports the maximum it found, so
a weight edit that breaks the claim fails here with the offending vector attached.
:func:`test_every_feature_moves_the_score_in_its_intended_direction` does the same for
the sign of each weight, against a direction table declared *in this file* rather than
read from ``WEIGHTS`` -- otherwise the test would agree with any sign the table
happened to hold.

The numeric pins elsewhere are regression guards, not specifications. What is
specified is the *band*: a bare-minimum verification must land in ``suspicious`` and
not ``likely_manipulated``, an abstention must not read as exoneration. The exact
probability beside each is pinned so that a weight change shows up as a failing test
rather than as a quiet shift in every benchmark number later.

One test needs the checkpoint. ``rules.py`` argues from a measurement -- that a real
manipulation (``base_cell`` vs its JPEG-degraded copy) scores *below* a genuine
negative control (``base_cell`` vs ``base_cells_2``) -- and that measurement is the
project's strongest argument for why stages B2/B3 exist. It is asserted against the
real weights in :func:`test_a_real_manipulation_ranks_below_a_negative_control`, under
the ``weights`` marker, so the claim cannot rot into folklore.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from sciforensics import fusion
from sciforensics.config import BandsConfig, Settings
from sciforensics.fusion import calibrate, rules
from sciforensics.fusion import features as features_module
from sciforensics.fusion.calibrate import CalibratorError, Isotonic, LogisticCalibrator
from sciforensics.fusion.features import (
    ABSTAINING_REASONS,
    COPY_MOVE_FEATURE_NAMES,
    COPY_MOVE_FEATURES,
    FEATURE_NAMES,
    FEATURES,
    REFERENCE_INLIERS,
    CopyMoveBundle,
    EvidenceBundle,
    as_tuple,
    extract,
    extract_copy_move,
    feature,
)
from sciforensics.fusion.rules import COPY_MOVE_WEIGHTS, PRIOR_LOGIT, WEIGHTS
from sciforensics.types import (
    AffineDecomposition,
    CopyMoveEvidence,
    CopyMoveRegion,
    GeometryEvidence,
    GlobalEvidence,
    KeypointEvidence,
    MatchEvidence,
    RejectionReason,
    Verdict,
)
from tests.helpers import SEED

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def global_evidence(similarity: float, *, triggered_local: bool = True) -> GlobalEvidence:
    """A ``GlobalEvidence`` carrying one similarity, since that is all fusion reads.

    ``distance`` is left at the threshold rather than derived from ``similarity``:
    the feature table reads only the calibrated similarity, and computing a
    consistent distance here would imply fusion cares about the raw L1 value.
    """
    return GlobalEvidence(
        distance=1.0,
        similarity=similarity,
        distance_threshold=1.0,
        local_trigger_distance=2.0,
        is_match=similarity >= 0.5,
        triggered_local=triggered_local,
        embedding_dim=128,
    )


def geometry(**overrides: Any) -> GeometryEvidence:
    """A verified fit sitting exactly on every gate floor in ``configs/default.yaml``.

    The defaults are the *bare minimum* deliberately: 15 inliers over 12 distinct
    points, a 0.18 inlier ratio and 4.9 px of reprojection error against a 5.0 px
    budget is the weakest fit ``geometry.verify`` will accept. Tests that want a
    comfortable fit override upward, which keeps the floor visible in the diff.
    """
    fields: dict[str, Any] = {
        "verified": True,
        "method": "magsac",
        "inlier_count": 15,
        "inlier_ratio": 0.18,
        "distinct_inliers": 12,
        "inlier_spread": 0.4,
        "reproj_rms": 4.9,
        "matched_area_fraction": 0.05,
    }
    fields.update(overrides)
    return GeometryEvidence(**fields)


def strong_geometry(**overrides: Any) -> GeometryEvidence:
    """A fit well clear of every gate: 60 distinct inliers, tight, over 40% of frame."""
    fields: dict[str, Any] = {
        "inlier_count": 60,
        "distinct_inliers": 60,
        "inlier_ratio": 0.6,
        "inlier_spread": 0.5,
        "reproj_rms": 1.5,
        "matched_area_fraction": 0.40,
    }
    fields.update(overrides)
    return geometry(**fields)


def manipulated_transform() -> AffineDecomposition:
    """A 45 degree, 1.2x, mirrored fit -- reuse that someone had to work at."""
    return AffineDecomposition(
        model="full",
        rotation_deg=45.0,
        scale_x=1.2,
        scale_y=0.9,
        shear_deg=0.0,
        translation=(0.0, 0.0),
        determinant=-1.08,
        flip=True,
        anisotropic=True,
        sheared=False,
    )


def keypoints(*, kept_left: int = 900, kept_right: int = 850) -> KeypointEvidence:
    return KeypointEvidence(
        detector="orb",
        detected_left=kept_left,
        detected_right=kept_right,
        kept_left=kept_left,
        kept_right=kept_right,
    )


def matches(*, injective: bool = True) -> MatchEvidence:
    """Match evidence, injective unless asked otherwise (mutual NN off)."""
    return MatchEvidence(
        matcher="bruteforce-hamming",
        raw=500,
        ratio_passed=200,
        good=100,
        distinct_left=100,
        distinct_right=100 if injective else 40,
        mutual_nn=injective,
        ratio=0.75,
    )


def clone_region(*, inliers: int = 40, reproj_rms: float | None = 1.2, area: int = 40_000) -> Any:
    return CopyMoveRegion(
        source_box=(10, 10, 110, 110),
        target_box=(200, 200, 300, 300),
        offset=(190.0, 190.0),
        rotation_deg=0.0,
        scale=1.0,
        flip=False,
        inlier_count=inliers,
        reproj_rms=reproj_rms,
        area_px=area,
    )


def copy_move_bundle(
    *,
    regions: tuple[CopyMoveRegion, ...] = (),
    cluster_count: int = 0,
    image_area: float = 640 * 480,
) -> CopyMoveBundle:
    return CopyMoveBundle(
        evidence=CopyMoveEvidence(
            detected=bool(regions),
            cluster_count=cluster_count,
            regions=regions,
            self_matches=120,
            keypoints=900,
        ),
        image_area=image_area,
    )


@pytest.fixture
def bands(cfg: Settings) -> BandsConfig:
    return cfg.fusion.bands


# ---------------------------------------------------------------------------
# the structural claim: screening cannot convict
# ---------------------------------------------------------------------------
def score_vector(values: dict[str, float], bands: BandsConfig) -> float:
    """Score a feature mapping directly, bypassing extraction.

    Used only by the search tests, which need to visit feature combinations no single
    bundle can express -- the point of a ceiling search is to cover the space the
    weights permit, not the subset today's extractors happen to produce.
    """
    logit = PRIOR_LOGIT + sum(values[name] * WEIGHTS[name] for name in FEATURE_NAMES)
    return rules.logistic(logit)


def unverified_corners() -> list[dict[str, float]]:
    """Every corner of the feature space reachable without a verified fit.

    The six strength features are zero by construction when ``geometry_verified`` is
    zero -- that gating is tested separately -- so the free axes are the embedding
    similarity, the two mutually exclusive rejection flags, and the two penalties.
    """
    corners: list[dict[str, float]] = []
    axes = itertools.product((-0.5, 0.0, 0.5), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    for embedding, refuted, abstained, asymmetry, non_injective in axes:
        if refuted and abstained:
            continue  # a rejection is one or the other, never both
        values = dict.fromkeys(FEATURE_NAMES, 0.0)
        values["embedding_similarity"] = embedding
        values["geometry_refuted"] = refuted
        values["geometry_abstained"] = abstained
        values["keypoint_asymmetry"] = asymmetry
        values["non_injective"] = non_injective
        corners.append(values)
    return corners


def test_no_unverified_pair_can_reach_suspicious(bands: BandsConfig) -> None:
    """The rule table's central structural claim, searched rather than asserted.

    The global stage produces a score, never evidence. If any combination of
    screening similarity and rejection flags could reach ``suspicious``, the pipeline
    would be able to escalate a pair no local stage ever corroborated -- which is the
    legacy behaviour this table replaces.
    """
    scored = [(score_vector(values, bands), values) for values in unverified_corners()]
    ceiling, worst = max(scored, key=lambda pair: pair[0])

    non_zero = {name: value for name, value in worst.items() if value}
    assert ceiling < bands.suspicious, f"unverified pair reached {ceiling:.4f} at {non_zero}"
    assert ceiling == pytest.approx(0.3775, abs=1e-4)
    assert rules.band(ceiling, bands) is Verdict.INCONCLUSIVE
    # The ceiling is reached by maximum similarity and nothing else, which is the
    # shape the claim depends on: no penalty or rejection flag can raise a score.
    assert non_zero == {"embedding_similarity": 0.5}


def test_the_ceiling_needs_a_similarity_the_checkpoint_never_produces(
    bands: BandsConfig,
) -> None:
    """Even the 0.3775 ceiling overstates what the shipped embedder can reach.

    Pinned because it is the honest version of the claim above and the reason
    :func:`sciforensics.fusion.rules._notes` carries the abstention caveat in prose
    instead of the verdict: reaching ``inconclusive`` unverified needs a similarity of
    1.0, and the best this checkpoint scores on a true positive is 0.873.
    """
    best_observed = 0.8725  # base_cells_1 vs base_cells_1_ex5_blackout, measured
    fused = rules.fuse(EvidenceBundle(global_evidence=global_evidence(best_observed)), bands)

    assert fused.confidence == pytest.approx(0.2927, abs=1e-4)
    assert fused.verdict is Verdict.CLEAN
    assert fused.confidence < bands.inconclusive


# ---------------------------------------------------------------------------
# monotonicity, against an independently declared direction table
# ---------------------------------------------------------------------------
#: Which way each feature is *intended* to move the score. Written out here rather
#: than derived from ``WEIGHTS`` on purpose: a test that reads the sign it is checking
#: passes whatever sign the table holds, including a flipped one. This is the
#: statement a reviewer should disagree with if they disagree with the model.
EXPECTED_DIRECTION: dict[str, str] = {
    "embedding_similarity": "up",
    "geometry_verified": "up",
    "inlier_strength": "up",
    "inlier_ratio": "up",
    "reproj_tightness": "up",
    "matched_area": "up",
    "transform_manipulated": "up",
    "geometry_refuted": "down",
    "geometry_abstained": "flat",
    "keypoint_asymmetry": "down",
    "non_injective": "down",
}


def test_the_direction_table_covers_the_schema() -> None:
    """A feature added without a stated direction would go untested in silence."""
    assert set(EXPECTED_DIRECTION) == set(FEATURE_NAMES)


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_every_feature_moves_the_score_in_its_intended_direction(
    name: str, bands: BandsConfig
) -> None:
    """Raising one feature from random starting points never moves the score wrongly.

    Random rather than a single fixed vector because the property being checked is
    *global* monotonicity: an additive log-odds model has it by construction, so what
    this really guards is that no weight's sign has been flipped and that no future
    non-additive term (an interaction, a hand-coded override) has quietly broken it.
    """
    generator = np.random.default_rng(SEED)
    bounds = {f.name: (f.lo, f.hi) for f in FEATURES}
    lo, hi = bounds[name]
    direction = EXPECTED_DIRECTION[name]

    for _ in range(2_000):
        base = {other: float(generator.uniform(*bounds[other])) for other in FEATURE_NAMES}
        low, high = sorted(float(generator.uniform(lo, hi)) for _ in range(2))
        if high - low < 1e-9:
            continue
        before = score_vector({**base, name: low}, bands)
        after = score_vector({**base, name: high}, bands)

        if direction == "up":
            assert after > before, f"{name}: {low:.3f}->{high:.3f} lowered the score"
        elif direction == "down":
            assert after < before, f"{name}: {low:.3f}->{high:.3f} raised the score"
        else:
            assert after == pytest.approx(before), f"{name} is weighted, not inert"


# ---------------------------------------------------------------------------
# band membership: what the score is actually for
# ---------------------------------------------------------------------------
def test_a_bare_minimum_verification_is_suspicious_not_convicted(bands: BandsConfig) -> None:
    """A fit that passes every gate at its floor must not reach the top band.

    This is the one place the rule table is deliberately stricter than the gates it
    sits behind: ``geometry.verify`` accepting a fit is a statement that the
    correspondences are not degenerate, not a statement that the finding is strong.
    Crossing 0.85 requires corroboration -- more distinct inliers, a tighter fit, more
    area, or a transform that is itself a manipulation.
    """
    bundle = EvidenceBundle(
        global_evidence=global_evidence(0.55),
        keypoints=keypoints(),
        matches=matches(),
        geometry=geometry(),
    )
    fused = rules.fuse(bundle, bands)

    assert fused.verdict is Verdict.SUSPICIOUS
    assert fused.confidence == pytest.approx(0.8089, abs=1e-4)
    assert fused.confidence < bands.likely_manipulated


def test_a_corroborated_verification_reaches_the_top_band(bands: BandsConfig) -> None:
    bundle = EvidenceBundle(
        global_evidence=global_evidence(0.7556),
        keypoints=keypoints(),
        matches=matches(),
        geometry=strong_geometry(),
    )
    fused = rules.fuse(bundle, bands)

    assert fused.verdict is Verdict.LIKELY_MANIPULATED
    assert fused.confidence == pytest.approx(0.9743, abs=1e-4)


def test_a_manipulated_transform_adds_to_a_verified_fit(bands: BandsConfig) -> None:
    """Bug 2's payoff. Under the legacy similarity model ``det = a^2 + b^2 > 0``, so
    ``flip`` was unreportable and this feature could never fire."""
    plain = strong_geometry()
    worked = strong_geometry(transform=manipulated_transform())

    def score(evidence: GeometryEvidence) -> float:
        bundle = EvidenceBundle(global_evidence=global_evidence(0.7556), geometry=evidence)
        return rules.fuse(bundle, bands).confidence

    assert score(worked) > score(plain)
    assert score(worked) == pytest.approx(0.9857, abs=1e-4)


def test_the_penalties_survive_a_successful_fit(bands: BandsConfig) -> None:
    """How much to trust a match count is a question the fit succeeding does not settle.

    Unlike the strength features, the two penalties are not gated on verification --
    so a comfortably verified pair still loses the top band if its keypoints were 20x
    asymmetric and its surviving matches were many-to-one. That combination is bug 4's
    signature, and the geometry gate rejecting it at the fitting layer is not a reason
    for the scoring layer to stop noticing.
    """
    clean = EvidenceBundle(
        global_evidence=global_evidence(0.7556),
        keypoints=keypoints(),
        matches=matches(),
        geometry=strong_geometry(),
    )
    degenerate = EvidenceBundle(
        global_evidence=global_evidence(0.7556),
        keypoints=keypoints(kept_left=2000, kept_right=100),
        matches=matches(injective=False),
        geometry=strong_geometry(),
    )

    assert rules.fuse(clean, bands).confidence == pytest.approx(0.9743, abs=1e-4)
    suspect = rules.fuse(degenerate, bands)
    assert suspect.confidence == pytest.approx(0.8369, abs=1e-4)
    assert suspect.verdict is Verdict.SUSPICIOUS


def test_band_boundaries_are_inclusive_at_the_bottom(bands: BandsConfig) -> None:
    assert rules.band(bands.likely_manipulated, bands) is Verdict.LIKELY_MANIPULATED
    assert rules.band(bands.suspicious, bands) is Verdict.SUSPICIOUS
    assert rules.band(bands.inconclusive, bands) is Verdict.INCONCLUSIVE
    assert rules.band(bands.inconclusive - 1e-9, bands) is Verdict.CLEAN
    assert rules.band(0.0, bands) is Verdict.CLEAN
    assert rules.band(1.0, bands) is Verdict.LIKELY_MANIPULATED


def test_logistic_saturates_instead_of_overflowing() -> None:
    """A fitted calibrator's coefficients are unbounded; a benchmark sweep must not
    die on one row of it."""
    assert rules.logistic(-800.0) == 0.0
    assert rules.logistic(800.0) == pytest.approx(1.0)
    assert rules.logistic(0.0) == pytest.approx(0.5)
    assert rules.logistic(PRIOR_LOGIT) == pytest.approx(0.1192, abs=1e-4)


# ---------------------------------------------------------------------------
# abstention is not refutation
# ---------------------------------------------------------------------------
def test_an_abstention_scores_exactly_as_if_the_local_stage_had_not_run(
    bands: BandsConfig,
) -> None:
    """``geometry_abstained`` carries weight 0.0, and that is the whole design.

    ORB collapsing to 2-7 matches on a JPEG-degraded pair is a fact about ORB. The
    score must not read it as evidence either way, which means an abstention has to be
    numerically identical to never having looked -- while still being *reported*, so
    the reader can see the verdict is resting on the embedding alone.
    """
    similarity = global_evidence(0.8725)
    abstained = rules.fuse(
        EvidenceBundle(
            global_evidence=similarity,
            geometry=geometry(verified=False, rejection_reason=RejectionReason.TOO_FEW_MATCHES),
        ),
        bands,
    )
    never_ran = rules.fuse(EvidenceBundle(global_evidence=similarity), bands)

    assert abstained.logit == pytest.approx(never_ran.logit)
    assert abstained.confidence == pytest.approx(0.2927, abs=1e-4)
    reported = {c.name for c in abstained.contributions}
    assert "geometry_abstained" in reported, "a zero weight is still a reportable fact"
    assert "geometry_abstained" not in {c.name for c in never_ran.contributions}


def test_a_refutation_pushes_below_the_prior(bands: BandsConfig) -> None:
    """Matches that existed and did not survive are evidence *against* reuse."""
    similarity = global_evidence(0.8725)
    refuted = rules.fuse(
        EvidenceBundle(
            global_evidence=similarity,
            geometry=geometry(
                verified=False,
                rejection_reason=RejectionReason.DEGENERATE_CORRESPONDENCES,
            ),
        ),
        bands,
    )

    assert refuted.confidence == pytest.approx(0.0926, abs=1e-4)
    assert refuted.logit < PRIOR_LOGIT
    assert refuted.verdict is Verdict.CLEAN


@pytest.mark.parametrize("reason", sorted(ABSTAINING_REASONS))
def test_abstaining_reasons_never_set_the_refutation_feature(reason: RejectionReason) -> None:
    values = extract(
        EvidenceBundle(
            global_evidence=global_evidence(0.6),
            geometry=geometry(verified=False, rejection_reason=reason),
        )
    )
    assert values["geometry_abstained"] == 1.0
    assert values["geometry_refuted"] == 0.0


@pytest.mark.parametrize(
    "reason",
    sorted(set(RejectionReason) - ABSTAINING_REASONS - {RejectionReason.NONE}),
)
def test_every_other_reason_is_a_refutation(reason: RejectionReason) -> None:
    values = extract(
        EvidenceBundle(
            global_evidence=global_evidence(0.6),
            geometry=geometry(verified=False, rejection_reason=reason),
        )
    )
    assert values["geometry_refuted"] == 1.0
    assert values["geometry_abstained"] == 0.0


def test_a_rejection_with_no_reason_recorded_is_read_as_an_abstention() -> None:
    """Unreachable through ``geometry.verify``, which names a reason on every failure
    path -- but reading it as an abstention claims nothing, which is the right default
    if one ever appears."""
    values = extract(
        EvidenceBundle(
            global_evidence=global_evidence(0.6),
            geometry=geometry(verified=False, rejection_reason=RejectionReason.NONE),
        )
    )
    assert values["geometry_abstained"] == 1.0
    assert values["geometry_refuted"] == 0.0


def test_absent_stages_are_neutral_not_exculpatory() -> None:
    """ "We did not look" and "we looked and found nothing" must be different vectors."""
    values = extract(EvidenceBundle(global_evidence=global_evidence(0.5)))

    assert values["geometry_refuted"] == 0.0
    assert values["geometry_abstained"] == 0.0
    assert values["keypoint_asymmetry"] == 0.0
    assert values["non_injective"] == 0.0
    assert set(values) == set(FEATURE_NAMES), "an absent stage still yields a full vector"


# ---------------------------------------------------------------------------
# gating: a failed fit must not score in proportion to how nearly it passed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "inlier_strength",
        "inlier_ratio",
        "reproj_tightness",
        "matched_area",
        "transform_manipulated",
    ],
)
def test_strength_features_are_zero_unless_the_fit_was_verified(name: str) -> None:
    """A rejected fit reports all of these measurements -- deliberately, since they are
    what explains the rejection -- and none of them may reach the score."""
    measured = strong_geometry(transform=manipulated_transform())
    rejected = measured.model_copy(
        update={
            "verified": False,
            "rejection_reason": RejectionReason.INSUFFICIENT_SPREAD,
        }
    )

    assert (
        extract(EvidenceBundle(global_evidence=global_evidence(0.6), geometry=measured))[name] > 0
    )
    assert (
        extract(EvidenceBundle(global_evidence=global_evidence(0.6), geometry=rejected))[name]
        == 0.0
    )


def test_inlier_strength_counts_distinct_points_not_matched_rows() -> None:
    """Bug 4 at the scoring layer: rows are not constraints.

    107 inlier rows collapsing onto 12 distinct keypoints is what manufactured the
    legacy pipeline's phantom verification. The geometry gate rejects that at the
    fitting layer -- but if this feature read ``inlier_count``, a fit that squeaked
    past the gate would still be scored as though every row were an independent
    constraint.
    """
    inflated = strong_geometry(inlier_count=400, distinct_inliers=20)
    honest = strong_geometry(inlier_count=20, distinct_inliers=20)

    def strength(evidence: GeometryEvidence) -> float:
        bundle = EvidenceBundle(global_evidence=global_evidence(0.6), geometry=evidence)
        return extract(bundle)["inlier_strength"]

    assert strength(inflated) == pytest.approx(strength(honest))
    assert strength(inflated) == pytest.approx(math.log1p(20) / math.log1p(REFERENCE_INLIERS))


def test_inlier_strength_saturates_at_the_reference_count() -> None:
    """Not a threshold -- the scale on which more correspondences stop adding
    confidence. Keeps the feature from tracking image resolution."""
    at_reference = strong_geometry(distinct_inliers=REFERENCE_INLIERS)
    far_beyond = strong_geometry(distinct_inliers=REFERENCE_INLIERS * 8)

    def strength(evidence: GeometryEvidence) -> float:
        return extract(EvidenceBundle(global_evidence=global_evidence(0.6), geometry=evidence))[
            "inlier_strength"
        ]

    assert strength(at_reference) == pytest.approx(1.0)
    assert strength(far_beyond) == pytest.approx(1.0), "clamped by the declared bound"


def test_reproj_tightness_is_headroom_and_hits_zero_at_the_budget() -> None:
    def tightness(rms: float | None, limit: float = 5.0) -> float:
        bundle = EvidenceBundle(
            global_evidence=global_evidence(0.6),
            geometry=strong_geometry(reproj_rms=rms),
            reproj_rms_limit=limit,
        )
        return extract(bundle)["reproj_tightness"]

    assert tightness(0.0) == pytest.approx(1.0)
    assert tightness(2.5) == pytest.approx(0.5)
    assert tightness(5.0) == pytest.approx(0.0)
    assert tightness(9.0) == pytest.approx(0.0), "clamped, never negative"
    assert tightness(None) == 0.0, "a fit with no error recorded claims no headroom"


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------
def test_an_empty_keypoint_side_saturates_the_penalty_instead_of_poisoning_the_sum(
    bands: BandsConfig,
) -> None:
    """``KeypointEvidence.asymmetry`` is ``inf`` when a side is empty, which *means*
    maximum asymmetry -- so saturating at the bound is both safe and correct."""
    bundle = EvidenceBundle(
        global_evidence=global_evidence(0.6), keypoints=keypoints(kept_left=900, kept_right=0)
    )
    values = extract(bundle)

    assert bundle.keypoints is not None
    assert math.isinf(bundle.keypoints.asymmetry)
    assert values["keypoint_asymmetry"] == 1.0
    assert math.isfinite(rules.fuse(bundle, bands).logit)


def test_keypoint_asymmetry_tolerates_ordinary_texture_differences() -> None:
    """Two panels of different texture density routinely differ 2x, and that is
    unremarkable -- the penalty ramp starts there, not at 1.0."""

    def penalty(left: int, right: int) -> float:
        bundle = EvidenceBundle(
            global_evidence=global_evidence(0.6),
            keypoints=keypoints(kept_left=left, kept_right=right),
        )
        return extract(bundle)["keypoint_asymmetry"]

    assert penalty(900, 900) == 0.0
    assert penalty(900, 450) == pytest.approx(0.0), "2x is free"
    assert penalty(900, 600) == 0.0, "below the free ramp, clamped"
    assert 0.0 < penalty(2000, 200) < 1.0
    assert penalty(2000, 100) == pytest.approx(1.0), "20x saturates"
    assert penalty(2000, 10) == pytest.approx(1.0), "clamped beyond saturation"


@pytest.mark.parametrize(
    "bundle",
    [
        EvidenceBundle(global_evidence=global_evidence(0.0)),
        EvidenceBundle(global_evidence=global_evidence(1.0)),
        EvidenceBundle(
            global_evidence=global_evidence(1.0),
            keypoints=keypoints(kept_left=5000, kept_right=1),
            matches=matches(injective=False),
            geometry=strong_geometry(
                distinct_inliers=10_000,
                inlier_ratio=1.0,
                reproj_rms=0.0,
                matched_area_fraction=1.5,
                transform=manipulated_transform(),
            ),
        ),
        EvidenceBundle(
            global_evidence=global_evidence(0.0),
            geometry=geometry(verified=False, rejection_reason=RejectionReason.ILL_CONDITIONED),
        ),
    ],
    ids=["dissimilar", "identical", "everything-maxed", "refuted"],
)
def test_every_feature_stays_inside_its_declared_bounds(bundle: EvidenceBundle) -> None:
    """Weight comparability and the ceiling argument both rest on this."""
    values = extract(bundle)
    for spec in FEATURES:
        assert spec.lo <= values[spec.name] <= spec.hi, f"{spec.name} escaped its bounds"


# ---------------------------------------------------------------------------
# the arithmetic is auditable
# ---------------------------------------------------------------------------
def test_reported_contributions_sum_to_the_logit(bands: BandsConfig) -> None:
    """What makes the report's contribution bars an audit rather than an illustration.

    If the prior were omitted from the list, or a feature silently dropped, the bars
    would still look plausible while no longer accounting for the score.
    """
    bundle = EvidenceBundle(
        global_evidence=global_evidence(0.7556),
        keypoints=keypoints(kept_left=2000, kept_right=200),
        matches=matches(injective=False),
        geometry=strong_geometry(transform=manipulated_transform()),
    )
    fused = rules.fuse(bundle, bands)

    assert sum(c.contribution for c in fused.contributions) == pytest.approx(fused.logit)
    assert rules.logistic(fused.logit) == pytest.approx(fused.confidence)
    assert fused.contributions[0].name == "prior"
    assert fused.contributions[0].contribution == PRIOR_LOGIT


def test_contributions_are_ranked_by_magnitude_and_omit_only_zero_values(
    bands: BandsConfig,
) -> None:
    bundle = EvidenceBundle(
        global_evidence=global_evidence(0.7556),
        keypoints=keypoints(),
        matches=matches(),
        geometry=strong_geometry(),
    )
    fused = rules.fuse(bundle, bands)
    reported = [c for c in fused.contributions if c.name != "prior"]

    magnitudes = [abs(c.contribution) for c in reported]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert {c.name for c in reported} == {
        name for name, value in fused.features.items() if value != 0.0
    }
    assert set(fused.features) == set(FEATURE_NAMES), "the zeros are retained for the JSON"


def test_the_rule_table_never_claims_to_be_calibrated(bands: BandsConfig) -> None:
    bundle = EvidenceBundle(global_evidence=global_evidence(0.9), geometry=strong_geometry())
    assert rules.fuse(bundle, bands).calibrated is False
    assert rules.fuse_copy_move(copy_move_bundle(), bands).calibrated is False


# ---------------------------------------------------------------------------
# the notes: what a bare number hides
# ---------------------------------------------------------------------------
def test_an_abstention_beside_a_screening_hit_says_so_in_prose(bands: BandsConfig) -> None:
    fused = rules.fuse(
        EvidenceBundle(
            global_evidence=global_evidence(0.8),
            geometry=geometry(verified=False, rejection_reason=RejectionReason.TOO_FEW_MATCHES),
        ),
        bands,
    )
    assert len(fused.notes) == 1
    assert "rests on the embedding" in fused.notes[0]


def test_an_abstention_is_reported_even_when_the_embedding_also_found_nothing(
    bands: BandsConfig,
) -> None:
    """The project's canonical false negative, and the one note that must never be
    silent.

    A JPEG-degraded pair suppresses the keypoints the local stage needs *and* the
    embedding similarity at the same time, so both stages come up empty and the score
    lands at 0.06 -- below the base rate -- with a confident-looking ``clean`` beside
    it. Nothing was actually tested. An abstention is a missing measurement rather
    than a disagreement, so unlike the other two notes it is not gated on which side
    the embedding came down on.
    """
    fused = rules.fuse(
        EvidenceBundle(
            global_evidence=global_evidence(0.2297),
            geometry=geometry(verified=False, rejection_reason=RejectionReason.TOO_FEW_MATCHES),
        ),
        bands,
    )

    assert fused.verdict is Verdict.CLEAN
    assert fused.logit < PRIOR_LOGIT
    assert len(fused.notes) == 1
    assert "absence of evidence rather than evidence of absence" in fused.notes[0]


def test_a_refutation_the_embedder_also_rejected_stays_quiet(bands: BandsConfig) -> None:
    """No note here, and that is right: both stages looked, both found nothing, and
    the local stage did form an opinion. There is nothing the number is hiding."""
    fused = rules.fuse(
        EvidenceBundle(
            global_evidence=global_evidence(0.2),
            geometry=geometry(
                verified=False, rejection_reason=RejectionReason.DEGENERATE_CORRESPONDENCES
            ),
        ),
        bands,
    )
    assert fused.notes == ()


def test_a_refutation_beside_a_screening_hit_quotes_the_gate_that_failed(
    bands: BandsConfig,
) -> None:
    reason = RejectionReason.DEGENERATE_CORRESPONDENCES
    fused = rules.fuse(
        EvidenceBundle(
            global_evidence=global_evidence(0.8),
            geometry=geometry(verified=False, rejection_reason=reason),
        ),
        bands,
    )
    assert len(fused.notes) == 1
    assert reason.explanation in fused.notes[0]


def test_a_verified_fit_the_embedder_missed_names_the_stage_that_missed_it(
    bands: BandsConfig,
) -> None:
    fused = rules.fuse(
        EvidenceBundle(global_evidence=global_evidence(0.2), geometry=strong_geometry()),
        bands,
    )
    assert len(fused.notes) == 1
    assert "the embedding is the stage that missed this" in fused.notes[0]


def test_stages_that_agree_produce_no_notes(bands: BandsConfig) -> None:
    agreed = rules.fuse(
        EvidenceBundle(global_evidence=global_evidence(0.9), geometry=strong_geometry()), bands
    )
    assert agreed.notes == ()


# ---------------------------------------------------------------------------
# copy-move
# ---------------------------------------------------------------------------
def test_a_clean_image_scores_below_the_pair_prior(bands: BandsConfig) -> None:
    """Cloning within one figure is rarer than reuse between two figures a user chose
    to compare, and there is no screening stage that could raise the baseline."""
    fused = rules.fuse_copy_move(copy_move_bundle(), bands)

    assert fused.confidence == pytest.approx(rules.logistic(rules.COPY_MOVE_PRIOR_LOGIT))
    assert fused.confidence < rules.logistic(PRIOR_LOGIT)
    assert fused.verdict is Verdict.CLEAN


def test_a_surviving_clone_region_convicts(bands: BandsConfig) -> None:
    fused = rules.fuse_copy_move(
        copy_move_bundle(regions=(clone_region(),), cluster_count=1), bands
    )
    assert fused.verdict is Verdict.LIKELY_MANIPULATED
    assert fused.confidence == pytest.approx(0.9364, abs=1e-4)


def test_one_cluster_surviving_out_of_many_scores_lower_than_one_out_of_one(
    bands: BandsConfig,
) -> None:
    """The copy-move analogue of ``inlier_ratio``: one of twelve candidate offsets
    verifying looks much more like clustering noise than the single candidate found
    holding up."""
    lucky = rules.fuse_copy_move(
        copy_move_bundle(regions=(clone_region(),), cluster_count=12), bands
    )
    clean_hit = rules.fuse_copy_move(
        copy_move_bundle(regions=(clone_region(),), cluster_count=1), bands
    )
    assert lucky.confidence < clean_hit.confidence


def test_clone_tightness_averages_rather_than_taking_the_best_region() -> None:
    """Every reported region is a separate claim, so a loose second one must not be
    hidden behind a tight first one."""
    tight_only = copy_move_bundle(regions=(clone_region(reproj_rms=0.5),), cluster_count=1)
    mixed = copy_move_bundle(
        regions=(clone_region(reproj_rms=0.5), clone_region(reproj_rms=3.9)), cluster_count=2
    )

    assert (
        extract_copy_move(tight_only)["clone_tightness"]
        > (extract_copy_move(mixed)["clone_tightness"])
    )


def test_a_region_with_no_reprojection_error_recorded_claims_no_tightness() -> None:
    bundle = copy_move_bundle(regions=(clone_region(reproj_rms=None),), cluster_count=1)
    assert extract_copy_move(bundle)["clone_tightness"] == 0.0


def test_clone_area_is_clamped_when_lobes_overlap() -> None:
    """Both lobes are counted, so overlapping lobes make this an upper bound on true
    coverage rather than an exact measure."""
    bundle = copy_move_bundle(
        regions=(clone_region(area=400_000), clone_region(area=400_000)),
        cluster_count=2,
        image_area=500_000,
    )
    assert extract_copy_move(bundle)["clone_area"] == 1.0


def test_a_zero_area_image_does_not_divide_by_zero() -> None:
    bundle = copy_move_bundle(regions=(clone_region(),), cluster_count=1, image_area=0.0)
    assert extract_copy_move(bundle)["clone_area"] == 0.0


# ---------------------------------------------------------------------------
# schema and table integrity
# ---------------------------------------------------------------------------
def test_a_feature_without_a_weight_fails_at_import() -> None:
    """Adding a feature and forgetting its weight would otherwise score it as zero:
    plausible numbers from a model nobody reviewed."""
    with pytest.raises(RuntimeError, match="features without a weight: geometry_abstained"):
        rules._check_table(FEATURES, {"embedding_similarity": 1.0}, "test")


def test_a_weight_for_a_nonexistent_feature_fails_at_import() -> None:
    with pytest.raises(RuntimeError, match="weights for unknown features: vibes"):
        rules._check_table(FEATURES, {**WEIGHTS, "vibes": 1.0}, "test")


def test_both_shipped_tables_agree_with_their_weights() -> None:
    assert set(WEIGHTS) == set(FEATURE_NAMES)
    assert set(COPY_MOVE_WEIGHTS) == set(COPY_MOVE_FEATURE_NAMES)


def test_feature_lookup_names_every_alternative_on_a_typo() -> None:
    """The callers that reach for a feature by name -- a calibrator file, a ``--set``
    override, a test -- are exactly where a typo would otherwise pass silently."""
    assert feature("inlier_ratio").name == "inlier_ratio"
    assert feature("clone_area").name == "clone_area"

    with pytest.raises(KeyError) as raised:
        feature("inlier_ration")
    message = str(raised.value)
    assert all(name in message for name in FEATURE_NAMES + COPY_MOVE_FEATURE_NAMES)


def test_as_tuple_follows_table_order_and_refuses_a_short_vector() -> None:
    """The one place a named mapping becomes positional, so a missing name has to
    raise rather than shift every subsequent column by one."""
    values = extract(EvidenceBundle(global_evidence=global_evidence(0.9)))
    assert as_tuple(values) == tuple(values[name] for name in FEATURE_NAMES)

    del values["inlier_ratio"]
    with pytest.raises(KeyError, match="missing inlier_ratio"):
        as_tuple(values)


def test_the_two_feature_tables_do_not_share_names() -> None:
    """``feature()`` searches both, so a shared name would resolve ambiguously."""
    assert not set(FEATURE_NAMES) & set(COPY_MOVE_FEATURE_NAMES)


def test_every_feature_description_is_reader_facing() -> None:
    """Descriptions are copied onto ``EvidenceContribution`` and rendered beside the
    bar, so they have to explain the number to someone who has not read the module."""
    for spec in FEATURES + COPY_MOVE_FEATURES:
        assert spec.description and spec.description[0].isupper()
        assert not spec.description.endswith("."), "rendered inline, not as a sentence"
        assert spec.lo < spec.hi


# ---------------------------------------------------------------------------
# the calibrator seam
# ---------------------------------------------------------------------------
def fitted_calibrator(**overrides: Any) -> LogisticCalibrator:
    """A plausible fitted artifact over the current schema."""
    fields: dict[str, Any] = {
        "features": FEATURE_NAMES,
        "coef": tuple(WEIGHTS[name] for name in FEATURE_NAMES),
        "intercept": PRIOR_LOGIT,
        "fitted_on": "benchmarks/splits/calibration-v1.json",
        "metrics": {"roc_auc": 0.94, "brier": 0.08},
    }
    fields.update(overrides)
    return LogisticCalibrator(**fields)


def calibrator_payload(**overrides: Any) -> dict[str, Any]:
    """A serialised calibrator as a mutable mapping, for the malformed-file tests.

    ``LogisticCalibrator.to_dict`` returns ``dict[str, object]`` -- correct for a
    serialisation boundary, awkward for a test that wants to corrupt one field. This
    rebuilds the same payload with the field types the tests need to edit.
    """
    payload: dict[str, Any] = dict(fitted_calibrator().to_dict())
    payload.update(overrides)
    return payload


def test_a_calibrator_round_trips_through_disk(tmp_path: Path) -> None:
    original = fitted_calibrator(isotonic=Isotonic(x=(0.0, 0.5, 1.0), y=(0.0, 0.7, 1.0)))
    path = original.save(tmp_path / "nested" / "calibrator.json")
    restored = calibrate.load(path)

    assert restored == original
    assert restored.fitted_on == "benchmarks/splits/calibration-v1.json"
    assert restored.metrics == {"roc_auc": 0.94, "brier": 0.08}


def test_a_calibrator_reproduces_the_rule_table_when_given_its_weights(
    bands: BandsConfig,
) -> None:
    """The two scorers are the same functional form on purpose: swapping the rule table
    for a fitted model must change the numbers and nothing about how one is read."""
    bundle = EvidenceBundle(
        global_evidence=global_evidence(0.7556),
        keypoints=keypoints(),
        matches=matches(),
        geometry=strong_geometry(),
    )
    ruled = rules.fuse(bundle, bands)
    calibrated = calibrate.apply(fitted_calibrator(), bundle, bands)

    assert calibrated.logit == pytest.approx(ruled.logit)
    assert calibrated.confidence == pytest.approx(ruled.confidence)
    assert calibrated.verdict is ruled.verdict
    assert calibrated.calibrated is True, "the only path by which this becomes true"
    assert calibrated.contributions[0].name == "prior"
    assert sum(c.contribution for c in calibrated.contributions) == pytest.approx(calibrated.logit)


def test_an_isotonic_step_is_disclosed_because_the_bars_no_longer_add_up(
    bands: BandsConfig,
) -> None:
    calibrator = fitted_calibrator(isotonic=Isotonic(x=(0.0, 1.0), y=(0.0, 0.5)))
    bundle = EvidenceBundle(global_evidence=global_evidence(0.9), geometry=strong_geometry())
    fused = calibrate.apply(calibrator, bundle, bands)

    assert fused.confidence == pytest.approx(rules.logistic(fused.logit) * 0.5, abs=1e-6)
    assert fused.notes and "isotonic" in fused.notes[0]
    assert rules.logistic(fused.logit) != pytest.approx(fused.confidence)


def test_a_reordered_schema_is_refused_and_the_diagnostic_says_so(tmp_path: Path) -> None:
    """The reason this module exists. Coefficients are positional: applied to a
    permuted schema they produce a plausible number from a model nobody fitted, and
    without this check that is not an error at all.
    """
    swapped = list(FEATURE_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(calibrator_payload(features=swapped)), encoding="utf-8")

    with pytest.raises(CalibratorError) as raised:
        calibrate.load(path)
    message = str(raised.value)
    assert "different feature schema" in message
    assert "different order" in message
    assert "Refit it" in message


def test_a_calibrator_fitted_before_a_feature_was_added_names_the_new_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stale.json"
    payload = calibrator_payload(
        features=list(FEATURE_NAMES)[:-1],
        coef=[WEIGHTS[name] for name in FEATURE_NAMES[:-1]],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        CalibratorError, match=f"features added since it was fitted: {FEATURE_NAMES[-1]}"
    ):
        calibrate.load(path)


def test_a_calibrator_expecting_a_removed_feature_names_it(tmp_path: Path) -> None:
    path = tmp_path / "ghost.json"
    payload = calibrator_payload(
        features=[*FEATURE_NAMES, "hash_distance"],
        coef=[*(WEIGHTS[name] for name in FEATURE_NAMES), 0.5],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CalibratorError, match="no longer exist: hash_distance"):
        calibrate.load(path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.update({"kind": "random_forest"}), "kind 'random_forest'"),
        (lambda p: p.update({"format_version": 99}), "format version 99"),
        (lambda p: p.pop("coef"), "missing 'coef'"),
        (lambda p: p.update({"coef": ["not a number"]}), "malformed coefficients"),
        (lambda p: p.update({"isotonic": {"x": [0.0, 1.0]}}), "malformed isotonic knots"),
    ],
    ids=["unknown-kind", "future-format", "missing-field", "bad-coef", "bad-isotonic"],
)
def test_a_malformed_calibrator_is_fatal_rather_than_a_silent_downgrade(
    tmp_path: Path, mutate: Any, match: str
) -> None:
    """``fusion.calibrator`` being set means someone asked for calibrated output.
    Falling back to the rule table would label the report with the wrong provenance."""
    payload = calibrator_payload()
    mutate(payload)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CalibratorError, match=match):
        calibrate.load(path)


def test_a_missing_or_unparseable_calibrator_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(CalibratorError, match="does not exist"):
        calibrate.load(tmp_path / "absent.json")

    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibratorError, match="could not read calibrator"):
        calibrate.load(garbage)

    array = tmp_path / "array.json"
    array.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CalibratorError, match="not a JSON object"):
        calibrate.load(array)


def test_a_coefficient_count_mismatch_is_caught_at_construction() -> None:
    with pytest.raises(CalibratorError, match="2 coefficients for 11 feature names"):
        fitted_calibrator(coef=(1.0, 2.0))


def test_a_calibrator_refuses_an_incomplete_feature_vector() -> None:
    values = extract(EvidenceBundle(global_evidence=global_evidence(0.9)))
    del values["non_injective"]

    with pytest.raises(CalibratorError, match="missing non_injective"):
        fitted_calibrator().logit(values)


# ---------------------------------------------------------------------------
# the package's own entry points
# ---------------------------------------------------------------------------
def test_load_calibrator_returns_nothing_when_the_rule_table_is_in_use(
    cfg: Settings,
) -> None:
    """``fusion.calibrator`` is unset in the shipped configuration, and ``None`` is how
    the pipeline learns to score against the rule table."""
    assert cfg.fusion.calibrator is None
    assert fusion.load_calibrator(cfg.fusion) is None


def test_load_calibrator_reads_a_configured_file_once(tmp_path: Path, cfg: Settings) -> None:
    path = fitted_calibrator().save(tmp_path / "calibrator.json")
    patched = cfg.fusion.model_copy(update={"calibrator": path})

    loaded = fusion.load_calibrator(patched)
    assert loaded == fitted_calibrator()


def test_a_configured_calibrator_that_is_missing_is_not_silently_downgraded(
    tmp_path: Path, cfg: Settings
) -> None:
    """The alternative -- falling back to the rule table -- would produce a report
    labelled ``calibrated`` from weights nobody fitted."""
    patched = cfg.fusion.model_copy(update={"calibrator": tmp_path / "absent.json"})

    with pytest.raises(CalibratorError, match="does not exist"):
        fusion.load_calibrator(patched)


def test_score_uses_the_rule_table_when_no_calibrator_is_passed(cfg: Settings) -> None:
    """Passing ``None`` while ``fusion.calibrator`` is set is not an error: it is how a
    caller deliberately scores against the rule table for comparison, and
    ``Fused.calibrated`` records which one ran."""
    bundle = EvidenceBundle(global_evidence=global_evidence(0.7556), geometry=strong_geometry())

    ruled = fusion.score(bundle, cfg.fusion)
    calibrated = fusion.score(bundle, cfg.fusion, calibrator=fitted_calibrator())

    assert ruled.calibrated is False
    assert calibrated.calibrated is True
    assert ruled.confidence == pytest.approx(calibrated.confidence), "same weights, same number"


def test_score_copy_move_is_always_rule_based(cfg: Settings) -> None:
    """B5 calibrates the pair path, where labelled negatives are abundant. Copy-move
    waits for B4's per-pixel ground truth."""
    fused = fusion.score_copy_move(
        copy_move_bundle(regions=(clone_region(),), cluster_count=1), cfg.fusion
    )
    assert fused.calibrated is False
    assert fused.verdict is Verdict.LIKELY_MANIPULATED


# ---------------------------------------------------------------------------
# defensive branches, documented rather than merely present
# ---------------------------------------------------------------------------
def test_clip_saturates_an_infinite_feature_at_whichever_bound_it_ran_toward() -> None:
    """An infinite feature would poison the whole log-odds sum. Only ``+inf`` is
    reachable today (an empty keypoint side), but a bound is not a bound if it holds in
    one direction only."""
    assert features_module._clip(math.inf, -0.5, 0.5) == 0.5
    assert features_module._clip(-math.inf, -0.5, 0.5) == -0.5
    assert features_module._clip(math.nan, -0.5, 0.5) == -0.5, "NaN is not > 0"


def test_a_degenerate_ramp_is_a_step_rather_than_a_division_by_zero() -> None:
    """``_ramp`` is called with configured limits, so a config that collapsed a range
    to a point must degrade to a step instead of raising."""
    assert features_module._ramp(4.0, 5.0, 5.0) == 0.0
    assert features_module._ramp(5.0, 5.0, 5.0) == 1.0
    assert features_module._ramp(6.0, 5.0, 5.0) == 1.0


# ---------------------------------------------------------------------------
# the isotonic map
# ---------------------------------------------------------------------------
def test_isotonic_interpolates_between_knots_and_clamps_outside_them() -> None:
    """Clamping rather than extrapolating: beyond the knots there was no data, and a
    linear extension would invent confidence exactly where the fit has none."""
    mapping = Isotonic(x=(0.2, 0.6, 0.8), y=(0.1, 0.5, 0.9))

    assert mapping(0.2) == pytest.approx(0.1)
    assert mapping(0.4) == pytest.approx(0.3)
    assert mapping(0.7) == pytest.approx(0.7)
    assert mapping(0.8) == pytest.approx(0.9)
    assert mapping(0.0) == pytest.approx(0.1), "clamped below"
    assert mapping(1.0) == pytest.approx(0.9), "clamped above"


def test_isotonic_tolerates_a_flat_segment() -> None:
    """Non-decreasing, not strictly increasing: a genuine fit routinely produces
    plateaus where a whole score range mapped to one calibrated value."""
    mapping = Isotonic(x=(0.0, 0.4, 0.8, 1.0), y=(0.0, 0.3, 0.3, 1.0))
    assert mapping(0.6) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("x", "y", "match"),
    [
        ((0.0, 1.0), (0.0,), "knots disagree"),
        ((0.5,), (0.5,), "at least two knots"),
        ((0.0, 0.0, 1.0), (0.0, 0.5, 1.0), "strictly increasing"),
        ((1.0, 0.0), (0.0, 1.0), "strictly increasing"),
        ((0.0, 0.5, 1.0), (0.0, 0.9, 0.4), "non-decreasing"),
        ((0.0, 1.0), (0.0, 1.5), r"probabilities in \[0, 1\]"),
    ],
    ids=["length", "too-few", "duplicate-x", "descending-x", "descending-y", "out-of-range"],
)
def test_a_hand_edited_isotonic_map_is_rejected(
    x: tuple[float, ...], y: tuple[float, ...], match: str
) -> None:
    """No isotonic fit produces any of these, so each one means a corrupted or
    hand-edited file rather than an unusual model."""
    with pytest.raises(CalibratorError, match=match):
        Isotonic(x=x, y=y)


# ---------------------------------------------------------------------------
# the measurement the rule table argues from
# ---------------------------------------------------------------------------
def read_asset(name: str) -> np.ndarray:
    path = Path(__file__).resolve().parents[1] / "inputs" / name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        pytest.skip(f"example asset {name} is not present")
    return image


@pytest.mark.weights
def test_a_real_manipulation_ranks_below_a_negative_control(
    cfg: Settings, weights_path: Path, bands: BandsConfig
) -> None:
    """Why no weight rescues the unverified case, and why stages B2/B3 exist.

    ``base_cell`` against its own JPEG-degraded copy is a manipulation by
    construction. It embeds *further* from the original than a genuinely unrelated
    image of different cells does, so any embedding weight large enough to promote the
    true positive out of ``clean`` promotes the negative control along with it. This
    is a fact about the checkpoint, not about the table -- which is exactly why it is
    measured here rather than argued in a docstring.
    """
    from sciforensics.global_match import Embedder

    embedder = Embedder(cfg.global_match, image=cfg.image, weights=weights_path, device="cpu")
    base = read_asset("base_cell.png")
    degraded = read_asset("base_cell_ex2_degraded.jpg")
    unrelated = read_asset("base_cells_2.png")

    manipulation = embedder.compare(base, degraded, attribution=False)
    control = embedder.compare(base, unrelated, attribution=False)

    assert manipulation.similarity == pytest.approx(0.2297, abs=0.01)
    assert control.similarity == pytest.approx(0.3781, abs=0.01)
    assert manipulation.similarity < control.similarity, "the inversion B2/B3 exist to fix"

    # The manipulation scores *below the base rate*: the embedding is confident enough
    # in the wrong direction to push it down. With the local stage abstaining -- which
    # is what ORB does on this pair -- nothing in the pipeline contradicts that, so the
    # note is the only thing standing between this number and a false exoneration.
    bundle = EvidenceBundle(
        global_evidence=global_evidence(manipulation.similarity),
        geometry=geometry(verified=False, rejection_reason=RejectionReason.TOO_FEW_MATCHES),
    )
    fused = rules.fuse(bundle, bands)

    assert fused.verdict is Verdict.CLEAN
    assert fused.confidence == pytest.approx(0.0567, abs=1e-3)
    assert fused.logit < PRIOR_LOGIT, "a real manipulation scored below the base rate"
    assert "absence of evidence rather than evidence of absence" in " ".join(fused.notes)
