"""Embedding inference, attribution geometry, and the bugs that made the old maps lie.

Bug 5 is *demonstrated* here, not asserted. The tests run the legacy pooling mode and
measure the branch asymmetry it produces, so the regression guard fails if the fix is
reverted rather than if a docstring is edited. That the annihilation turns out to be
architectural -- present with a randomly initialised network, no checkpoint needed -- is
what lets the demonstration run in CI.

Two things about the weights-free tests are worth knowing before adding more of them.

*A random-init network barely discriminates.* Two unrelated textures embed 0.00013 apart
on an L1 norm of 2.02 -- a relative separation of 0.007%. Any test that asserts one pair
is farther apart than another therefore needs the real checkpoint and the ``weights``
marker; :func:`test_a_random_backbone_barely_discriminates` pins that so the trap is
documented rather than rediscovered.

*Which branch dies is not fixed.* Under ``relu`` the suppressed side depends on the
weights, the stage and the input -- over eight torch seeds it landed right five times and
left three. So the invariant to assert is that *one* branch is suppressed far more than
the other, never that it is the right one. Same reason the tests measure
:attr:`~sciforensics.global_match.Attribution.zero_fraction` rather than
:attr:`~sciforensics.global_match.Attribution.is_degenerate`: at ``block4`` the map is
100% zero on the repository's asset pairs but 98.4% zero on a synthetic pair, and a
degeneracy check calls the second one healthy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from sciforensics.config import STAGE_STRIDES, Settings
from sciforensics.global_match import (
    Attribution,
    AttributionUnavailableError,
    EmbeddedImage,
    Embedder,
    EmbeddingNet,
    PairComparison,
    blend_overlay,
    colourise,
    distance,
    grad_cam,
    preprocess,
)
from sciforensics.global_match.embed import PIXEL_MEAN, PIXEL_STD, resolve_colormap
from sciforensics.io import Letterbox, Squash
from tests.helpers import SEED, textured_image


# Fraction of cells a healthy map reports as zero: the argmin, sent to 0 by the rescale
# to [0, 1]. Exactly one cell, so the figure is 1/grid**2 and nothing about it is
# approximate.
def normalisation_floor(grid_size: int) -> float:
    return 1.0 / (grid_size * grid_size)


def build(cfg: Settings, /, **attribution: object) -> Embedder:
    """An Embedder on a fixed random backbone, with ``attribution`` fields patched.

    No checkpoint is read: ``model=`` bypasses weight resolution entirely, which is
    what keeps the bug-5 demonstration runnable in CI. The torch seed is fixed because
    the *side* the suppression lands on depends on it (see the module docstring) and a
    test that reported a different side per run would look flaky rather than informative.
    """
    torch.manual_seed(0)
    patched = cfg.global_match.model_copy(
        update={"attribution": cfg.global_match.attribution.model_copy(update=attribution)}
    )
    return Embedder(patched, image=cfg.image, device="cpu", model=EmbeddingNet())


@pytest.fixture
def pair() -> tuple[np.ndarray, np.ndarray]:
    """Two different textured images, 320x400 -- deliberately not square."""
    return textured_image(seed=SEED), textured_image(seed=SEED + 1)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def test_preprocess_maps_the_byte_range_onto_the_trained_interval() -> None:
    """The checkpoint's first BatchNorm was fitted to (x/255 - 0.5) / 0.5."""
    black, _ = preprocess(np.zeros((32, 48), dtype=np.uint8), embed_size=128)
    white, _ = preprocess(np.full((32, 48), 255, dtype=np.uint8), embed_size=128)

    assert black.min() == pytest.approx(-PIXEL_MEAN / PIXEL_STD)
    assert white.max() == pytest.approx((1.0 - PIXEL_MEAN) / PIXEL_STD)
    assert black.shape == (1, 1, 128, 128)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("squash", Squash), ("letterbox", Letterbox)],
)
def test_preprocess_returns_the_fit_that_produced_the_canvas(mode: str, expected: type) -> None:
    """Bug 8's structural fix: the forward transform is handed back, not re-derived."""
    gray = textured_image(size=(320, 400))
    tensor, fit = preprocess(gray, embed_size=128, mode=mode)  # type: ignore[arg-type]

    assert isinstance(fit, expected)
    assert tensor.shape == (1, 1, 128, 128)
    assert (fit.src_w, fit.src_h) == (400, 320)
    assert fit.out_size == 128


def test_letterboxing_normalises_the_pad_into_the_input_range() -> None:
    """A padded corner is ``-1.0``, which is why letterboxing this checkpoint hurts.

    Not a defect in :func:`~sciforensics.io.images.letterbox` -- the record of *why*
    ``image.embed_resize`` ships as ``squash``. Zero padding lands at the extreme end of
    the trained input range, and the network has never seen it.
    """
    gray = textured_image(size=(200, 400))
    tensor, fit = preprocess(gray, embed_size=128, mode="letterbox", pad_value=0)

    assert isinstance(fit, Letterbox)
    assert fit.pad_y > 0, "a 2:1 image must be padded vertically"
    assert tensor[0, 0, 0, 0].item() == pytest.approx(-PIXEL_MEAN / PIXEL_STD)


def test_preprocess_rejects_a_colour_image() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        preprocess(np.zeros((16, 16, 3), dtype=np.uint8), embed_size=128)


def test_preprocess_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="squash or letterbox"):
        preprocess(
            textured_image(size=(64, 96)),
            embed_size=128,
            mode="stretch",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# bug 5: the branch asymmetry, measured
# ---------------------------------------------------------------------------
def test_the_legacy_pooling_suppresses_one_branch_far_more_than_the_other(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """Bug 5, reproduced. This is the test the fix has to keep passing.

    Asserted as a ratio between the two branches' zeroed fractions, because that is what
    held across every configuration measured: 64x at ``block4``, 3-4x at ``block3``, and
    on this fixture a full annihilation. Not asserted as "the right branch dies" -- which
    side loses depends on the weights and the input.
    """
    left, right = pair
    result = build(cfg, stage="block4", gradient_pooling="relu").compare(left, right)

    left_map, right_map = result.attribution_left, result.attribution_right
    assert left_map is not None and right_map is not None

    dead, live = sorted((left_map.zero_fraction, right_map.zero_fraction), reverse=True)
    assert dead >= 0.9, f"expected one branch mostly annihilated, got {dead:.3f}"
    assert live <= 2 * normalisation_floor(left_map.grid_size)
    assert dead / live > 10


@pytest.mark.parametrize("pooling", ["abs", "signed"])
def test_the_fixed_pooling_modes_treat_both_branches_identically(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray], pooling: str
) -> None:
    """The other half of the demonstration: with the fix, the asymmetry is gone.

    Both surviving modes zero exactly one cell per branch -- the argmin the rescale to
    ``[0, 1]`` sends to zero -- so the comparison is exact rather than approximate, and
    ``signed`` is included to record that it does *not* reproduce bug 5 despite pooling
    the signed gradient.
    """
    left, right = pair
    result = build(cfg, stage="block4", gradient_pooling=pooling).compare(left, right)

    left_map, right_map = result.attribution_left, result.attribution_right
    assert left_map is not None and right_map is not None

    floor = normalisation_floor(left_map.grid_size)
    assert left_map.zero_fraction == pytest.approx(floor)
    assert right_map.zero_fraction == pytest.approx(floor)
    assert not left_map.is_degenerate
    assert not right_map.is_degenerate


def test_the_two_branches_get_different_maps(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """Symmetric treatment must not mean identical output.

    Guards the other way a fix could go wrong: pooling ``|gradient|`` makes the *weights*
    similar between branches, and if the maps came out identical the overlay would be
    decorative. They differ because each branch's activations are its own.
    """
    left, right = pair
    result = build(cfg, gradient_pooling="abs").compare(left, right)

    left_map, right_map = result.attribution_left, result.attribution_right
    assert left_map is not None and right_map is not None
    assert not np.allclose(left_map.grid, right_map.grid)


def test_selecting_the_legacy_pooling_warns_at_construction(
    cfg: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Warned once per Embedder, not once per pair -- construction is the rare event."""
    with caplog.at_level(logging.WARNING, logger="sciforensics.global_match.embed"):
        build(cfg, gradient_pooling="relu")

    assert "bug 5" in caplog.text
    assert "'relu'" in caplog.text


def test_the_signed_mode_is_not_described_as_bug_5(
    cfg: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """``signed`` warns about a different problem, because it *has* a different problem.

    Measured, it zeroes both branches identically; telling its user they have a dead
    branch would send them looking for something that is not there.
    """
    with caplog.at_level(logging.WARNING, logger="sciforensics.global_match.embed"):
        build(cfg, gradient_pooling="signed")

    assert caplog.records, "a non-default pooling mode must still be flagged"
    # The message may well mention bug 5 -- it names it to say this mode is *not* it.
    # What must not happen is a user being told they have a dead branch they do not have.
    assert "does not annihilate" in caplog.text
    assert "cancel" in caplog.text


def test_the_default_pooling_is_silent(cfg: Settings, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="sciforensics.global_match.embed"):
        build(cfg, gradient_pooling="abs")

    assert not caplog.records


def test_a_suppressed_branch_is_named_per_pair(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray], caplog: pytest.LogCaptureFixture
) -> None:
    """The per-pair warning must fire on the pair whose heatmap is unusable.

    Keyed on ``zero_fraction`` rather than ``is_degenerate`` precisely so that a map
    which is 98.4% zero -- one surviving cell, equally useless -- is still reported.
    """
    left, right = pair
    embedder = build(cfg, stage="block4", gradient_pooling="relu")

    with caplog.at_level(logging.WARNING, logger="sciforensics.global_match.embed"):
        embedder.compare(left, right)

    suppression = [r for r in caplog.records if "% zero at block4" in r.getMessage()]
    assert len(suppression) == 1
    assert "should not be rendered" in suppression[0].getMessage()


def test_a_healthy_pair_is_not_flagged(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray], caplog: pytest.LogCaptureFixture
) -> None:
    left, right = pair
    embedder = build(cfg, gradient_pooling="abs")

    with caplog.at_level(logging.WARNING, logger="sciforensics.global_match.embed"):
        embedder.compare(left, right)

    assert not caplog.records


# ---------------------------------------------------------------------------
# attribution: availability and error paths
# ---------------------------------------------------------------------------
def test_attribution_from_a_no_grad_capture_is_refused() -> None:
    """The prototype returned an un-overlaid copy here; this raises instead.

    An attribution that could not be computed, rendered as a heatmap with no salient
    region, is indistinguishable from a genuine finding of "nothing stands out".
    """
    model = EmbeddingNet()
    tensor = torch.zeros(1, 1, 128, 128)

    with torch.no_grad():
        captured = model.forward_to_stage(tensor, "block3")

    assert not captured.differentiable
    with pytest.raises(AttributionUnavailableError, match="no_grad"):
        grad_cam(captured)


def test_attribution_before_backward_is_refused() -> None:
    """Differentiable but ungradient-ed: a distinct failure with a distinct message."""
    model = EmbeddingNet()
    tensor = torch.zeros(1, 1, 128, 128, requires_grad=True)
    captured = model.forward_to_stage(tensor, "block3")

    assert captured.differentiable
    with pytest.raises(AttributionUnavailableError, match="no gradient yet"):
        grad_cam(captured)


def test_an_unknown_pooling_mode_is_rejected() -> None:
    model = EmbeddingNet()
    tensor = torch.zeros(1, 1, 128, 128, requires_grad=True)
    captured = model.forward_to_stage(tensor, "block3")
    distance(captured.embedding, torch.zeros_like(captured.embedding)).sum().backward()

    with pytest.raises(ValueError, match="unknown gradient_pooling"):
        grad_cam(captured, pooling="softmax")


def test_a_constant_map_is_flagged_rather_than_divided_by_zero() -> None:
    """Normalising a flat map would be a division by zero; it reports instead."""
    flat = Attribution(
        grid=np.zeros((16, 16), dtype=np.float32),
        stage="block3",
        pooling="relu",
        fit=Squash.compute(400, 320, 128),
    )

    assert flat.is_degenerate
    assert flat.zero_fraction == 1.0
    assert np.all(flat.to_source() == 0.0)


# ---------------------------------------------------------------------------
# attribution: geometry (bug 8)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["squash", "letterbox"])
def test_the_map_projects_onto_the_source_pixel_grid(cfg: Settings, mode: str) -> None:
    """Whatever the fit, ``to_source`` lands on the shape the caller handed in."""
    gray = textured_image(size=(320, 400))
    patched = cfg.image.model_copy(update={"embed_resize": mode})
    torch.manual_seed(0)
    embedder = Embedder(cfg.global_match, image=patched, device="cpu", model=EmbeddingNet())

    result = embedder.compare(gray, textured_image(size=(320, 400), seed=SEED + 1))
    left_map = result.attribution_left
    assert left_map is not None

    assert left_map.grid_size == cfg.image.embed_size // STAGE_STRIDES[left_map.stage]
    assert left_map.to_canvas().shape == (cfg.image.embed_size, cfg.image.embed_size)
    assert left_map.to_source().shape == gray.shape
    assert float(left_map.to_source().min()) >= 0.0
    assert float(left_map.to_source().max()) <= 1.0


def test_letterboxed_attribution_discards_what_fell_on_the_padding() -> None:
    """Attribution over padding is not attribution over the image.

    The sharp half of bug 8's geometry. A hot cell in the top-left of the grid lands
    entirely inside a wide image's letterbox padding, so unpadding must drop it; under
    ``squash`` there is no padding and the same cell survives. Skipping the unpad would
    instead smear that padding into the source map's top edge.
    """
    grid = np.zeros((16, 16), dtype=np.float32)
    grid[0, 0] = 1.0

    box = Letterbox.compute(400, 100, 128)
    assert box.pad_y >= 8, "a 4:1 image must pad enough to swallow the first grid row"

    padded = Attribution(grid=grid, stage="block3", pooling="abs", fit=box)
    squashed = Attribution(
        grid=grid, stage="block3", pooling="abs", fit=Squash.compute(400, 100, 128)
    )

    assert padded.to_canvas().max() == pytest.approx(1.0, abs=0.05)
    assert padded.to_source().max() < 0.05, "padding-only attribution must not survive"
    assert squashed.to_source().max() == pytest.approx(1.0, abs=0.05)


def test_a_canvas_map_of_the_wrong_size_is_refused() -> None:
    box = Letterbox.compute(400, 320, 128)
    with pytest.raises(ValueError, match="128x128 canvas map"):
        box.project_to_source(np.zeros((64, 64), dtype=np.float32))


# ---------------------------------------------------------------------------
# overlay rendering
# ---------------------------------------------------------------------------
def test_the_overlay_refuses_a_heatmap_of_the_wrong_shape() -> None:
    """Silently resizing here is how an overlay comes to point at the wrong pixels."""
    base = np.zeros((320, 400, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"heatmap is 128x128 but the image is 400x320"):
        blend_overlay(base, np.zeros((128, 128), dtype=np.float32))


def test_a_zero_alpha_overlay_is_the_original_image() -> None:
    base = cv2.cvtColor(textured_image(size=(64, 96)), cv2.COLOR_GRAY2BGR)
    heat = np.linspace(0, 1, 64 * 96, dtype=np.float32).reshape(64, 96)

    assert np.array_equal(blend_overlay(base, heat, alpha=0.0), base)
    assert not np.array_equal(blend_overlay(base, heat, alpha=0.45), base)


def test_jet_puts_red_at_the_hot_end() -> None:
    """Sanity-checks the direction of the colour ramp, in BGR."""
    ramp = np.array([[0.0, 1.0]], dtype=np.float32)
    coloured = colourise(ramp, colormap="JET")

    cold, hot = coloured[0, 0], coloured[0, 1]
    assert int(hot[2]) > int(cold[2]), "hot end should carry more red"
    assert int(cold[0]) > int(hot[0]), "cold end should carry more blue"


def test_an_unknown_colormap_lists_the_alternatives() -> None:
    with pytest.raises(ValueError, match=r"available: .*JET"):
        resolve_colormap("viridis_r")


# ---------------------------------------------------------------------------
# scoring and thresholds (bugs 9 and 10)
# ---------------------------------------------------------------------------
def test_an_image_against_itself_beats_the_legacy_ceiling(cfg: Settings) -> None:
    """Bug 9: ``sigmoid(1 - d)`` capped every score in the old reports at 0.7311.

    Nothing could exceed it, not even an image against itself, so a "similarity" column
    that never reached 0.74 was reporting the formula's limit rather than the evidence.
    """
    embedder = build(cfg)
    gray = textured_image()
    embedded = embedder.embed(gray)

    assert embedder.distance_to(embedded, embedder.embed(gray)) == 0.0
    assert embedder.score(0.0) > 0.75
    assert embedder.score(0.0) == pytest.approx(0.8808, abs=1e-4)


def test_similarity_falls_as_distance_grows(cfg: Settings) -> None:
    embedder = build(cfg)
    scores = [embedder.score(d) for d in (0.0, 0.5, 1.0, 2.0, 4.0)]

    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s < 1.0 for s in scores)


def test_the_threshold_sits_where_the_loss_put_it(cfg: Settings) -> None:
    """``d = 1.0`` is the trained decision boundary, so it must score exactly 0.5."""
    assert cfg.global_match.distance_threshold == 1.0
    assert build(cfg).score(1.0) == pytest.approx(0.5)


def make_comparison(cfg: Settings, dist: float) -> PairComparison:
    """A comparison at an arbitrary distance, without running the network."""
    embedded = EmbeddedImage(embedding=torch.zeros(1, 128), fit=Squash.compute(400, 320, 128))
    gm = cfg.global_match
    return PairComparison(
        left=embedded,
        right=embedded,
        distance=dist,
        similarity=0.0,
        distance_threshold=gm.distance_threshold,
        local_trigger_distance=gm.local_trigger_distance,
    )


@pytest.mark.parametrize(
    ("dist", "match", "escalate"),
    [(0.4, True, True), (1.5, False, True), (3.0, False, False)],
)
def test_escalation_is_wider_than_the_match_decision(
    cfg: Settings, dist: float, match: bool, escalate: bool
) -> None:
    """The ambiguous band between the two thresholds is what the local stage is for."""
    comparison = make_comparison(cfg, dist)

    assert comparison.is_match is match
    assert comparison.triggered_local is escalate


def test_the_evidence_records_the_thresholds_it_was_judged_against(
    cfg: Settings,
) -> None:
    """Bug 10 was four different values for one threshold; a report must say which."""
    evidence = make_comparison(cfg, 0.4).evidence

    assert evidence.distance_threshold == cfg.global_match.distance_threshold
    assert evidence.local_trigger_distance == cfg.global_match.local_trigger_distance


# ---------------------------------------------------------------------------
# batching and reuse (bug 11)
# ---------------------------------------------------------------------------
def test_batching_agrees_with_embedding_one_at_a_time(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """No cross-talk between batch members: the network is in eval mode.

    Would fail if BatchNorm were ever left training, which is the quiet way a batched
    inference path starts disagreeing with the single-image one it is supposed to match.
    """
    embedder = build(cfg)
    left, right = pair

    batched = embedder.embed_batch([left, right])
    singly = [embedder.embed(left), embedder.embed(right)]

    for from_batch, alone in zip(batched, singly, strict=True):
        assert np.allclose(from_batch.vector, alone.vector, atol=1e-6)


def test_an_empty_batch_embeds_to_nothing(cfg: Settings) -> None:
    assert build(cfg).embed_batch([]) == []


def test_attribution_can_be_declined_per_call(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """Corpus-scale scanning wants the distance without paying for two backward passes."""
    embedder = build(cfg)
    left, right = pair

    with_maps = embedder.compare(left, right)
    without = embedder.compare(left, right, attribution=False)

    assert with_maps.attribution_left is not None
    assert without.attribution_left is None
    assert without.attribution_right is None
    assert without.distance == pytest.approx(with_maps.distance)


def test_disabling_attribution_in_config_disables_it(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray]
) -> None:
    left, right = pair
    result = build(cfg, enabled=False).compare(left, right)

    assert result.attribution_left is None
    assert result.attribution_right is None


def test_supplying_a_model_skips_checkpoint_resolution(
    cfg: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``model=`` is the seam the weights-free tests hang on; it must not touch disk."""

    def explode(*args: object, **kwargs: object) -> Path:
        raise AssertionError("weight resolution should not have been attempted")

    monkeypatch.setattr("sciforensics.weights.ensure", explode)
    monkeypatch.setattr(torch, "load", explode)

    embedder = build(cfg)
    assert embedder.embed(textured_image()).vector.shape == (128,)


# ---------------------------------------------------------------------------
# what a random backbone can and cannot be asked
# ---------------------------------------------------------------------------
def test_a_random_backbone_barely_discriminates(
    cfg: Settings, pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """Documents the trap: an untrained trunk maps unrelated inputs to nearly one point.

    Every test above runs on a random backbone, which is safe because they assert
    structure -- shapes, symmetry, error paths, thresholds. None asserts that similar
    images score closer than dissimilar ones, and none can: measured here, two unrelated
    textures sit 0.0002 apart on an L1 norm of ~2.0. Discrimination claims need the real
    checkpoint and the ``weights`` marker.
    """
    embedder = build(cfg)
    left, right = pair

    left_embedded, right_embedded = embedder.embed(left), embedder.embed(right)
    norm = float(np.abs(left_embedded.vector).sum())

    assert norm > 1.0, "a collapsed-to-zero embedding would make the ratio meaningless"
    assert embedder.distance_to(left_embedded, right_embedded) / norm < 0.01


# ---------------------------------------------------------------------------
# the trained checkpoint
# ---------------------------------------------------------------------------
def read_asset(name: str) -> np.ndarray:
    """Load one of the repository's example panels as greyscale, or skip.

    ``inputs/`` is committed, so a miss means someone pruned the assets rather than that
    the environment is thin -- but skipping still beats failing, since these tests are
    about the checkpoint's behaviour and not about the fixtures being present.
    """
    path = Path(__file__).resolve().parents[1] / "inputs" / name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        pytest.skip(f"example asset {name} is not present")
    return image


@pytest.mark.weights
def test_the_trained_network_separates_a_manipulation_from_an_unrelated_image(
    cfg: Settings, weights_path: Path
) -> None:
    """The claim the random-backbone tests deliberately cannot make.

    Distances pinned so a change in preprocessing, normalisation or the checkpoint itself
    shows up here rather than as a quiet shift in every benchmark number.
    """
    base = read_asset("base_cell.png")
    manipulated = read_asset("base_cell_ex1_affine.png")
    unrelated = read_asset("base_cells_2.png")

    embedder = Embedder(cfg.global_match, image=cfg.image, weights=weights_path, device="cpu")
    near = embedder.compare(base, manipulated, attribution=False)
    far = embedder.compare(base, unrelated, attribution=False)

    assert near.distance == pytest.approx(0.436, abs=0.01)
    assert far.distance == pytest.approx(1.249, abs=0.01)
    assert near.similarity == pytest.approx(0.7556, abs=0.01)
    assert near.is_match and not far.is_match
    assert far.triggered_local, "an ambiguous negative still deserves the local stage"


@pytest.mark.weights
def test_the_legacy_pooling_annihilates_the_right_branch_on_the_real_checkpoint(
    cfg: Settings, weights_path: Path
) -> None:
    """Bug 5 exactly as it shipped: at ``block4`` the right map is uniformly zero.

    The weights-free version of this test asserts only that *some* branch is suppressed,
    because the side depends on initialisation. With the real checkpoint the side is
    fixed, and it is the right -- which is why every "Global Heatmap B" in the old
    reports is uniform blue.
    """
    left, right = textured_image(seed=SEED), textured_image(seed=SEED + 1)
    patched = cfg.global_match.model_copy(
        update={
            "attribution": cfg.global_match.attribution.model_copy(
                update={"stage": "block4", "gradient_pooling": "relu"}
            )
        }
    )
    embedder = Embedder(patched, image=cfg.image, weights=weights_path, device="cpu")
    result = embedder.compare(left, right)

    left_map, right_map = result.attribution_left, result.attribution_right
    assert left_map is not None and right_map is not None
    assert right_map.zero_fraction >= 0.98
    assert left_map.zero_fraction == pytest.approx(normalisation_floor(left_map.grid_size))


@pytest.mark.weights
def test_moving_the_stage_would_have_masked_the_bug_rather_than_fixed_it(
    cfg: Settings, weights_path: Path
) -> None:
    """The finding that inverts the obvious reading of bug 5.

    At ``block3`` the branches' Jacobians have diverged enough that the sign flip only
    partially survives: on the repository's asset pair the asymmetry drops from 64x to
    3.3x. Still broken, no longer visible in a report. Had the fix been "read a finer
    feature map", the maps would have looked plausible and stayed wrong -- so this pins
    that the *pooling* change is the load-bearing one.
    """
    base = read_asset("base_cell.png")
    manipulated = read_asset("base_cell_ex1_affine.png")

    def zero_fractions(stage: str) -> tuple[float, float]:
        patched = cfg.global_match.model_copy(
            update={
                "attribution": cfg.global_match.attribution.model_copy(
                    update={"stage": stage, "gradient_pooling": "relu"}
                )
            }
        )
        embedder = Embedder(patched, image=cfg.image, weights=weights_path, device="cpu")
        result = embedder.compare(base, manipulated)
        assert result.attribution_left is not None
        assert result.attribution_right is not None
        return result.attribution_left.zero_fraction, result.attribution_right.zero_fraction

    deep_left, deep_right = zero_fractions("block4")
    mid_left, mid_right = zero_fractions("block3")

    assert deep_right / deep_left > 20, "block4 annihilates the right branch outright"
    # Partial, and on the *other* side: both maps keep content, so a reader would see two
    # plausible heatmaps and no sign that one lost 70% of its cells.
    assert 2.0 < mid_left / mid_right < 10.0
    assert not any((mid_left == 1.0, mid_right == 1.0))


@pytest.mark.weights
def test_the_checkpoint_is_read_once_per_embedder(
    cfg: Settings, weights_path: Path, pair: tuple[np.ndarray, np.ndarray]
) -> None:
    """Bug 11: ``ForensicScanner`` was rebuilt inside the batch loop, so a 40-pair run
    performed 40 separate 35 MB ``torch.load`` calls."""
    calls = 0
    real_load = torch.load

    def counting_load(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_load(*args, **kwargs)  # type: ignore[arg-type]

    left, right = pair
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "load", counting_load)
        embedder = Embedder(cfg.global_match, image=cfg.image, weights=weights_path, device="cpu")
        for _ in range(3):
            embedder.compare(left, right, attribution=False)

    assert calls == 1
