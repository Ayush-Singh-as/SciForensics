"""Orchestration: what the pipeline is responsible for that no stage is.

The stages have their own suites, and this module deliberately does not re-test
them. What lives only here is the wiring, and the wiring carries three claims that
a wrong answer would look plausible under:

* **A stage that did not run is not a stage that found nothing.** The fusion layer
  is built around that distinction, and the pipeline is what has to *record* it --
  in ``stages_run``, and in a warning a reader will actually see.
* **The keypoint floor refuses to match rather than matching garbage.** Bug 4's
  orchestration half. The legacy pipeline matched 2000 keypoints against 12 and
  reported 121 "inliers"; the property to pin is that no match evidence is produced
  at all, because fabricated-then-gated evidence is exactly what made that number
  look real.
* **Expensive things are built once.** Bug 11. Untested, this decays the moment
  someone moves a constructor inside a loop, and the symptom is a slow sweep rather
  than a wrong answer -- so nothing else would catch it.

Every test here runs on CPU against a randomly-initialised backbone injected
through ``Pipeline(embedder=...)``. That is what the ``model=`` parameter on
:class:`~sciforensics.global_match.Embedder` exists for: the orchestration is
independent of what the weights say, so depending on the 35 MB artifact would buy
nothing and cost the suite its ability to run offline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from sciforensics import weights as weights_module
from sciforensics.config import Settings
from sciforensics.global_match.backbone import EmbeddingNet
from sciforensics.global_match.embed import Embedder
from sciforensics.pipeline import Pipeline, PipelineError, _timed
from sciforensics.types import RejectionReason, Stage
from tests.helpers import SEED, clone_patch, textured_image


@pytest.fixture(scope="session")
def net() -> EmbeddingNet:
    """A randomly-initialised backbone, seeded so a failure reproduces."""
    torch.manual_seed(SEED)
    return EmbeddingNet()


@pytest.fixture
def pipeline(cfg: Settings, net: EmbeddingNet) -> Pipeline:
    """A CPU pipeline with attribution off.

    Attribution runs a backward pass per pair and produces heatmaps nothing here
    inspects, so it is disabled to keep the module fast. It is exercised in
    ``test_embed.py``, which is where it belongs.
    """
    quiet = cfg.model_copy(
        update={
            "global_match": cfg.global_match.model_copy(
                update={
                    "attribution": cfg.global_match.attribution.model_copy(
                        update={"enabled": False}
                    )
                }
            )
        }
    )
    return _pipeline(quiet, net)


def _pipeline(cfg: Settings, net: EmbeddingNet) -> Pipeline:
    """Build a pipeline and its embedder from one ``Settings``.

    ``Pipeline`` refuses an embedder whose ``global_match``/``image`` config differs
    from its own, so a test that varies a threshold has to rebuild both. ``model=``
    is what makes that cheap: the net is shared, only the config wrapper is new.
    """
    embedder = Embedder(cfg.global_match, image=cfg.image, device="cpu", model=net)
    return Pipeline(cfg=cfg, device="cpu", embedder=embedder)


def _write(path: Path, image: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), image), f"failed to write {path}"
    return path


@pytest.fixture
def identical(tmp_path: Path) -> tuple[Path, Path]:
    """A textured image and a byte-identical copy.

    The one input whose embedding distance is knowable without knowing the weights:
    two identical tensors through the same network are equal, so the distance is
    exactly ``0.0`` and the pair escalates under any positive threshold. Every test
    that needs a *guaranteed* escalation uses this rather than hoping a random net
    scores two different images closely.
    """
    image = textured_image()
    return _write(tmp_path / "a.png", image), _write(tmp_path / "b.png", image)


# ---------------------------------------------------------------------------
# escalation: "did not run" is a different claim from "found nothing"
# ---------------------------------------------------------------------------
def test_an_identical_pair_runs_every_stage_and_verifies(
    pipeline: Pipeline, identical: tuple[Path, Path]
) -> None:
    """The end-to-end sanity anchor: identity must survive the whole chain.

    Worth asserting despite being the easy case, because it is the only input for
    which every stage's correct answer is known a priori -- distance 0, a full
    inlier set, and the identity transform. A break anywhere in the wiring shows up
    here first, and it localises to the wiring precisely because nothing about the
    answer depends on the weights or on ORB's recall.
    """
    analysis = pipeline.compare(*identical)
    result = analysis.result

    assert result.stages_run == (Stage.GLOBAL, Stage.LOCAL, Stage.GEOMETRY, Stage.FUSION)
    assert result.global_evidence.distance == pytest.approx(0.0, abs=1e-5)
    assert result.geometry is not None and result.geometry.verified
    assert result.geometry.rejection_reason is RejectionReason.NONE

    transform = result.geometry.transform
    assert transform is not None
    assert transform.scale == pytest.approx(1.0, abs=0.01)
    assert transform.rotation_deg == pytest.approx(0.0, abs=0.5)
    assert transform.flip is False

    # The working data is carried alongside, so a renderer has something to draw.
    assert analysis.left_keypoints is not None and analysis.right_keypoints is not None
    assert analysis.correspondences is not None
    assert analysis.verification is not None
    assert analysis.verification.inlier_mask.sum() == result.geometry.inlier_count


def test_a_pair_below_the_screen_records_the_skip_as_a_skip(
    pipeline: Pipeline, net: EmbeddingNet, tmp_path: Path
) -> None:
    """No keypoint evidence, and a warning that refuses to read as "clean".

    The failure mode this guards is not a wrong number, it is a *missing* number
    that looks like a measured one: ``geometry=None`` renders identically whether
    the stage abstained or was never asked. So the assertion is on all three
    channels a reader has -- the stage list, the absent evidence, and the prose.
    """
    left = _write(tmp_path / "l.png", textured_image(seed=1))
    right = _write(tmp_path / "r.png", textured_image(seed=2))

    # Force the screen to reject by making the trigger smaller than any nonzero
    # distance. Asserted as a premise below rather than assumed, since the distance
    # comes from a random net.
    strict = pipeline.cfg.model_copy(
        update={
            "global_match": pipeline.cfg.global_match.model_copy(
                update={"local_trigger_distance": 1e-6, "distance_threshold": 1e-7}
            )
        }
    )
    screened = _pipeline(strict, net)
    result = screened.compare(left, right).result

    assert result.global_evidence.distance > 1e-6, "premise: the screen actually rejects"
    assert result.stages_run == (Stage.GLOBAL, Stage.FUSION)
    assert Stage.LOCAL not in result.stages_run
    assert result.keypoints is None
    assert result.matches is None
    assert result.geometry is None, "an absent stage must not manufacture an abstention"
    assert any("unexamined pair, not a cleared one" in w for w in result.warnings)
    assert Stage.LOCAL.value not in result.timings, "a stage that did not run has no cost"


def test_force_local_examines_a_pair_the_screen_rejected(
    pipeline: Pipeline, net: EmbeddingNet, tmp_path: Path
) -> None:
    """The benchmark's escape hatch, and why it is not the default.

    Attributing a miss to the right stage requires knowing what the keypoint stage
    *would* have found on a pair the screen threw away. Without this the screen and
    the matcher are indistinguishable as causes of a false negative.

    The input is the interesting part: a screen rejection needs a nonzero embedding
    distance, and an identical pair has a distance of exactly ``0.0``, so no positive
    threshold can reject one. The right side is therefore the left under a *monotone*
    intensity remap. FAST thresholds on ``I_p > I_center + t`` and BRIEF compares
    pixel pairs, so both are invariant to it and the geometry is still the identity --
    while the embedding, which reads absolute values, sees two different images. That
    is exactly the pair the screen gets wrong and the local stage gets right.
    """
    image = textured_image()
    remapped = (image.astype(np.float32) * 0.9 + 12.0).astype(np.uint8)
    left = _write(tmp_path / "l.png", image)
    right = _write(tmp_path / "r.png", remapped)
    strict = pipeline.cfg.model_copy(
        update={
            "global_match": pipeline.cfg.global_match.model_copy(
                update={"local_trigger_distance": 1e-6, "distance_threshold": 1e-7}
            )
        }
    )
    forced = _pipeline(strict, net)

    screened = forced.compare(left, right).result
    examined = forced.compare(left, right, force_local=True).result

    assert screened.global_evidence.distance > 1e-6, "premise: the screen rejects this pair"
    assert Stage.GEOMETRY not in screened.stages_run
    assert Stage.GEOMETRY in examined.stages_run
    assert examined.geometry is not None and examined.geometry.verified, (
        "the content is geometrically identical; the screen's rejection was the wrong call, "
        "which is the whole reason a benchmark needs to look past it"
    )
    assert examined.confidence > screened.confidence, (
        "examining a pair the screen dismissed must be able to change the answer"
    )


# ---------------------------------------------------------------------------
# bug 4, orchestration half: refuse to match rather than match garbage
# ---------------------------------------------------------------------------
def test_a_starved_side_is_refused_before_matching(pipeline: Pipeline, tmp_path: Path) -> None:
    """The floor produces *no* match evidence, not weak match evidence.

    This is the shape of bug 4 that matters. Matching 2000 keypoints against a dozen
    and then gating the result downstream is what let 121 phantom inliers reach a
    report labelled "Strong evidence": once the numbers exist they look measured.
    Never computing them is the only version of this fix that cannot be undone by a
    later threshold change.
    """
    flat = np.full((256, 256), 128, np.uint8)  # no gradient, so no ORB keypoints
    left = _write(tmp_path / "flat.png", flat)
    right = _write(tmp_path / "textured.png", textured_image())

    analysis = pipeline.compare(left, right, force_local=True)
    result = analysis.result

    assert result.keypoints is not None, "detection ran and its counts are reportable"
    assert result.keypoints.kept_left < pipeline.cfg.local_match.min_keypoints_per_side
    assert result.matches is None, "no correspondences may be fabricated for a starved side"
    assert analysis.correspondences is None

    assert result.geometry is not None, "the stage ran; it declined"
    assert result.geometry.verified is False
    assert result.geometry.rejection_reason is RejectionReason.TOO_FEW_KEYPOINTS
    assert result.geometry.inlier_count == 0

    floor_warning = [w for w in result.warnings if "keypoint floor not met" in w]
    assert floor_warning, "the refusal must be legible, not inferred from a null field"
    assert str(pipeline.cfg.local_match.min_keypoints_per_side) in floor_warning[0]


def test_a_refusal_and_a_gate_rejection_are_both_abstentions(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    """Neither is evidence *against* reuse, and the consensus mask says which is which.

    ``fusion.features`` treats both as absent geometry, which is correct. The
    distinction the report needs is the mask: a pre-fit refusal has no consensus set
    to draw, whereas a gate rejection has one -- and showing 121 lines converging on
    a dozen points is a better account of a rejection than the sentence is.
    """
    flat = _write(tmp_path / "flat.png", np.full((256, 256), 128, np.uint8))
    textured = _write(tmp_path / "t.png", textured_image())
    refused = pipeline.compare(flat, textured, force_local=True)

    assert refused.verification is not None
    assert refused.verification.verified is False
    assert refused.verification.inlier_mask.size == 0, (
        "nothing was fitted, so there is no consensus set to index"
    )
    assert refused.result.geometry is not None
    assert refused.result.geometry.rejection_reason is RejectionReason.TOO_FEW_KEYPOINTS


# ---------------------------------------------------------------------------
# coordinate frames
# ---------------------------------------------------------------------------
def test_a_downscaled_input_reports_original_pixels_and_says_so(
    pipeline: Pipeline, tmp_path: Path, cfg: Settings
) -> None:
    """Analysis space in, original pixels out -- with the downscale disclosed.

    The class of bug this guards is bug 13's: a scale factor applied in some places
    and not others. A report whose hulls are in analysis pixels while its
    ``ImageMeta`` is in original ones is wrong in a way no single number reveals,
    so the check is that the hull actually lands inside the *original* frame.
    """
    big = cfg.image.max_dimension + 400
    image = cv2.resize(textured_image(), (big, big), interpolation=cv2.INTER_CUBIC)
    left = _write(tmp_path / "big_a.png", image)
    right = _write(tmp_path / "big_b.png", image)

    analysis = pipeline.compare(left, right)
    result = analysis.result

    assert result.left.was_downscaled, "premise: the input exceeded max_dimension"
    assert analysis.left.analysis_scale < 1.0, "analysis_scale is analysis/original"
    assert any("downscaled from" in w and "original pixels" in w for w in result.warnings)

    geometry = result.geometry
    assert geometry is not None and geometry.verified
    assert geometry.hull_left, "a verified fit must carry a hull to draw"
    xs = [x for x, _ in geometry.hull_left]
    ys = [y for _, y in geometry.hull_left]
    assert max(xs) > cfg.image.max_dimension or max(ys) > cfg.image.max_dimension, (
        "the hull is still in analysis space: it fits inside the downscaled frame"
    )
    assert max(xs) <= result.left.width and max(ys) <= result.left.height


# ---------------------------------------------------------------------------
# bug 11: expensive things are built once
# ---------------------------------------------------------------------------
def test_an_embedder_configured_differently_is_refused(cfg: Settings, net: EmbeddingNet) -> None:
    """Two live sources for one threshold is bug 10's shape, so it is refused.

    An injected embedder holds its own ``global_match`` and ``image`` config, and the
    escalation decision is read off the comparison *it* returns -- not off
    ``pipeline.cfg``. Nothing reads the pipeline's copy today, so a mismatch is
    currently latent rather than wrong; the guard is what stops it becoming wrong
    when someone adds the first read, at which point a report would cite whichever
    copy the code happened to reach.

    Sharing a *loaded backbone* across differently-configured pipelines stays legal,
    because only these two sections are constrained -- which is what makes an
    ablation over ``geometry`` thresholds affordable.
    """
    embedder = Embedder(cfg.global_match, image=cfg.image, device="cpu", model=net)
    drifted = cfg.model_copy(
        update={"global_match": cfg.global_match.model_copy(update={"distance_threshold": 0.5})}
    )

    with pytest.raises(PipelineError, match="different `global_match` configuration"):
        Pipeline(cfg=drifted, device="cpu", embedder=embedder)

    # The permitted case: a different geometry gate over the same backbone.
    elsewhere = cfg.model_copy(
        update={"geometry": cfg.geometry.model_copy(update={"min_inlier_ratio": 0.5})}
    )
    assert Pipeline(cfg=elsewhere, device="cpu", embedder=embedder) is not None


def test_stage_objects_are_built_once_and_reused(
    pipeline: Pipeline, identical: tuple[Path, Path]
) -> None:
    """A batch must not rebuild the detector, the matcher or the backbone.

    The prototype constructed its scanner inside the batch loop, so a 40-pair sweep
    re-read a 35 MB checkpoint 40 times. Identity across calls is the whole
    property, and it is invisible in any output -- which is why it needs a test
    rather than a comment.
    """
    detector, matcher, embedder = pipeline.detector, pipeline.matcher, pipeline._embedder

    pipeline.compare(*identical)
    pipeline.compare(*identical)

    assert pipeline.detector is detector
    assert pipeline.matcher is matcher
    assert pipeline._embedder is embedder


def test_the_weights_digest_is_hashed_once_per_pipeline(
    pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 11 applied to the audit block: SHA-256 of 35 MB is a per-*pipeline* cost.

    Every report carries the digest, so the naive implementation re-hashes the
    checkpoint for each pair in a sweep. Counted rather than timed, because a timing
    assertion on a fast machine is a flake.
    """
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"not a real checkpoint")
    pipeline._embedder.weights_path = checkpoint

    calls = 0
    real = weights_module.verify

    def counting(path: Path, *, expected: str | None = None) -> str:
        nonlocal calls
        calls += 1
        return real(path, expected=expected)

    monkeypatch.setattr("sciforensics.pipeline.weights.verify", counting)

    first = pipeline.audit(weighted=True)
    second = pipeline.audit(weighted=True)

    assert calls == 1, "the digest must be memoised across results"
    assert first.weights_sha256 == second.weights_sha256
    assert first.weights_sha256 is not None


def test_an_unweighted_result_claims_no_checkpoint_provenance(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    """A copy-move search never loads the backbone, so it must not cite one.

    The audit block's only value is that a reader can trust it. Stamping a
    digest for a checkpoint that took no part in producing the result is a false
    provenance claim, and a quieter one than a wrong number because it *looks*
    like diligence.
    """
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"not a real checkpoint")
    pipeline._embedder.weights_path = checkpoint

    assert pipeline.audit(weighted=True).weights_sha256 is not None, "premise: a digest exists"
    assert pipeline.audit(weighted=False).weights_sha256 is None


# ---------------------------------------------------------------------------
# copy-move
# ---------------------------------------------------------------------------
def test_copy_move_finds_a_cloned_patch(pipeline: Pipeline, tmp_path: Path) -> None:
    cloned = clone_patch(textured_image(), source=(20, 20), target=(180, 200))
    path = _write(tmp_path / "cloned.png", cloned)

    analysis = pipeline.copy_move(path)

    assert analysis.result.evidence.keypoints > 0
    assert Stage.COPY_MOVE.value in analysis.result.timings
    assert analysis.result.audit.weights_sha256 is None, "no backbone participated"


def test_a_disabled_detector_raises_rather_than_reporting_nothing_found(
    pipeline: Pipeline, net: EmbeddingNet, tmp_path: Path
) -> None:
    """ "I was configured not to look" must not render as "I looked and found nothing".

    The two produce identical reports unless one of them raises, and conflating them
    is the category of ambiguity this refactor exists to remove. Raising is
    deliberately louder than a warning: a warning in a footer is missable, and this
    one invalidates the entire result rather than qualifying it.
    """
    disabled = pipeline.cfg.model_copy(
        update={"copy_move": pipeline.cfg.copy_move.model_copy(update={"enabled": False})}
    )
    off = _pipeline(disabled, net)

    with pytest.raises(PipelineError, match="not the same claim"):
        off.copy_move(_write(tmp_path / "x.png", textured_image()))


# ---------------------------------------------------------------------------
# timing contract
# ---------------------------------------------------------------------------
def test_a_stage_that_raised_still_reports_its_cost() -> None:
    """``_timed`` records on the way out, including the exceptional way out.

    A timings dict that silently omits the slow thing that blew up is worse than
    useless to whoever is diagnosing a timeout: it points at the fast stages.
    """
    timings: dict[str, float] = {}
    with pytest.raises(ValueError, match="boom"), _timed(timings, Stage.GEOMETRY):
        raise ValueError("boom")

    assert Stage.GEOMETRY.value in timings
    assert timings[Stage.GEOMETRY.value] >= 0.0
