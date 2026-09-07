"""Backbone: port fidelity, head detection, the similarity fix, safe loading.

Two kinds of test live here, and the distinction is the point.

Most are architecture tests that run on a randomly-initialised network and need no
checkpoint -- shapes, stage strides, head behaviour, error paths. They run
everywhere, in well under a second.

A handful are marked ``weights`` and assert against the real 35 MB checkpoint. The
most important of those is :func:`test_embedding_matches_the_recorded_golden_vector`,
which pins the exact output of the ported network. Its job is to outlive
``src/global_matching/``: right now port fidelity can be checked directly, by running
both implementations side by side, and that comparison gives max abs difference
``0.0``. Once the legacy package is deleted that comparison becomes impossible, so
the golden vector is the *transcript* of it -- the numbers the legacy model produced,
frozen, so any future refactor that changes what the network computes fails here
instead of quietly shifting every distance in the benchmark.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
import torch

from sciforensics.config import STAGE_STRIDES, Settings
from sciforensics.global_match import (
    EMBEDDING_DIM,
    BackboneLoadError,
    EmbeddingNet,
    FlattenHead,
    PoolHead,
    distance,
    load_backbone,
    parameter_count,
    similarity,
)

# Recorded from the legacy `Model(nin=True)` loaded with `models/weights.pth`, which
# the ported `EmbeddingNet` reproduces bit-exactly (max abs diff 0.0). See
# `golden_input` for why the input carries no RNG.
GOLDEN_FIRST_EIGHT = (
    0.352043,
    0.179415,
    -0.279414,
    0.091346,
    0.106646,
    -0.369761,
    -0.198156,
    -0.154272,
)
GOLDEN_NORM = 2.775151
GOLDEN_SUM = -0.003011


def golden_input(size: int = 128) -> torch.Tensor:
    """A deterministic, RNG-free test image.

    ``torch.manual_seed`` is not a stable contract across torch versions, so a
    golden vector generated from it could break on an upgrade for reasons that have
    nothing to do with this code. Two incommensurate spatial frequencies give
    varied activations from pure arithmetic instead.
    """
    rows = torch.arange(size, dtype=torch.float32).view(-1, 1)
    cols = torch.arange(size, dtype=torch.float32).view(1, -1)
    return (torch.sin(rows / 7.0) * torch.cos(cols / 11.0)).view(1, 1, size, size)


# ---------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------
def test_embedding_has_the_documented_width() -> None:
    model = EmbeddingNet()
    with torch.no_grad():
        out = model(golden_input())
    assert out.shape == (1, EMBEDDING_DIM)


def test_stage_strides_match_the_config_table() -> None:
    """``STAGE_STRIDES`` is what ``attribution.grid_size`` computes from.

    If the table and the network disagree, every heatmap is silently drawn at the
    wrong resolution and then stretched to fit -- which looks plausible and is
    wrong. Measuring the real feature maps is the only way to keep the constant
    honest.
    """
    model = EmbeddingNet()
    image = golden_input(128)
    for stage, stride in STAGE_STRIDES.items():
        captured = model.forward_to_stage(image, stage)
        assert captured.grid_size == 128 // stride, f"{stage} at stride {stride}"


def test_attribution_reads_a_finer_map_than_the_final_block() -> None:
    """Bug 5's other half: 8x8 is too coarse to point at anything.

    The prototype hooked the final block, giving a 16x16-pixel granularity on a
    128x128 input -- a heatmap that can indicate a quadrant, not a region. The
    default stage is ``block3``, four times finer in area.
    """
    model = EmbeddingNet()
    image = golden_input(128)
    assert model.forward_to_stage(image, "block3").grid_size == 16
    assert model.forward_to_stage(image, "block4").grid_size == 8


def test_stage_activations_carry_gradients(cfg: Settings) -> None:
    """The retained tensor must actually receive ``.grad``, or Grad-CAM is dead.

    A plain forward pass discards non-leaf gradients, so without ``retain_grad``
    the attribution code would read ``None`` and -- in the prototype's style --
    silently render a uniform map.
    """
    model = EmbeddingNet()
    captured = model.forward_to_stage(golden_input(), cfg.global_match.attribution.stage)
    assert captured.differentiable
    assert captured.activations.grad is None
    captured.embedding.abs().sum().backward()
    assert captured.activations.grad is not None
    assert torch.any(captured.activations.grad != 0.0)


def test_a_no_grad_forward_reports_itself_as_non_differentiable() -> None:
    """Inspecting activations under ``no_grad`` must work and must say so.

    Two failure modes are being avoided at once. Calling ``retain_grad``
    unconditionally raises ``RuntimeError`` under ``no_grad``, which makes the most
    natural way to ask for an embedding crash. Skipping it silently leaves ``.grad``
    as ``None`` forever, and an attribution routine that reads ``None`` as zero
    produces a uniform heatmap and presents it as an explanation -- bug 5's exact
    signature. The flag is what lets the consumer raise instead.
    """
    model = EmbeddingNet()
    with torch.no_grad():
        captured = model.forward_to_stage(golden_input(), "block3")
    assert captured.differentiable is False
    assert captured.activations.grad is None
    assert captured.grid_size == 16
    assert captured.embedding.shape == (1, EMBEDDING_DIM)


def test_the_attributed_activations_produced_the_embedding() -> None:
    """One forward pass, not two.

    ``forward_to_stage`` must return an embedding identical to ``forward``'s, or the
    heatmap would explain a different computation than the score reported beside it.
    """
    model = EmbeddingNet()
    image = golden_input()
    with torch.no_grad():
        direct = model(image)
        staged = model.forward_to_stage(image, "block3").embedding
    assert torch.equal(direct, staged)


def test_an_unknown_stage_is_rejected_by_name() -> None:
    model = EmbeddingNet()
    with pytest.raises(ValueError, match="unknown attribution stage"):
        model.forward_to_stage(golden_input(), "block9")


def test_every_config_stage_is_a_real_network_stage(cfg: Settings) -> None:
    """The config's allowed stages and the network's must be the same set."""
    assert set(EmbeddingNet().stage_names) == set(STAGE_STRIDES)
    assert cfg.global_match.attribution.stage in EmbeddingNet().stage_names


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------
def test_flatten_head_states_the_input_size_it_requires() -> None:
    assert FlattenHead(grid=8).required_input_size == 128
    assert FlattenHead(grid=14).required_input_size == 224


def test_pool_head_accepts_any_input_size() -> None:
    """The property that makes ``image.embed_size`` tunable in stage B2."""
    model = EmbeddingNet(head="gap")
    assert model.head.required_input_size is None
    with torch.no_grad():
        for size in (96, 128, 224):
            assert model(golden_input(size)).shape == (1, EMBEDDING_DIM)


def test_flatten_head_holds_almost_the_whole_network() -> None:
    """The 8.39M-parameter ``fc1`` is the reason stage B2 replaces this head."""
    counts = parameter_count(EmbeddingNet(head="flatten"))
    assert counts["total"] == pytest.approx(8_835_810, abs=0)
    assert counts["head"] / counts["total"] > 0.96


def test_pool_head_is_far_smaller_than_the_flatten_head() -> None:
    flatten = parameter_count(EmbeddingNet(head="flatten"))["total"]
    pooled = parameter_count(EmbeddingNet(head="gap"))["total"]
    assert pooled == pytest.approx(331_490, abs=0)
    assert flatten / pooled > 25.0


def test_parameter_count_excludes_buffers() -> None:
    """Running statistics are not parameters.

    The state dict holds 8,836,040 numbers but only 8,835,810 are trainable; the
    230-number difference is the BatchNorm buffers,
    ``2 * (1 + 16 + 32 + 64) + 4 num_batches_tracked``. Reporting the larger figure
    as a parameter count would be a small lie of exactly the kind the README's
    "~1.2M" was a large one.
    """
    model = EmbeddingNet()
    trainable = parameter_count(model)["total"]
    stated = sum(t.numel() for t in model.state_dict().values())
    assert stated - trainable == 230


def test_first_block_has_no_bottleneck() -> None:
    """It takes a 1-channel image, and the checkpoint has no 1x1 weights for it."""
    model = EmbeddingNet(bottleneck=True)
    assert model.block1.nin is None  # type: ignore[union-attr]
    assert model.block2.nin is not None  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# distance and similarity
# ---------------------------------------------------------------------------
def test_distance_is_l1_and_zero_on_identical_embeddings() -> None:
    left = torch.tensor([[1.0, -2.0, 3.0]])
    assert float(distance(left, left)) == 0.0
    assert float(distance(left, torch.zeros_like(left))) == 6.0


def test_distance_is_symmetric_and_batched() -> None:
    left = torch.randn(5, EMBEDDING_DIM)
    right = torch.randn(5, EMBEDDING_DIM)
    forward = distance(left, right)
    assert forward.shape == (5,)
    assert torch.allclose(forward, distance(right, left))


def test_similarity_reaches_above_the_legacy_ceiling(cfg: Settings) -> None:
    """Bug 9. ``sigmoid(1 - d)`` could not exceed 0.7311, ever.

    Identical images scored 0.72 in the legacy ``all_results.json`` and nothing in
    the output explained that this was the maximum. The replacement reaches 0.88 on
    the shipped configuration at ``d = 0`` and approaches 1 as the temperature
    sharpens -- the point being that the top of the range is now *reachable*.
    """
    threshold = cfg.global_match.distance_threshold
    temperature = cfg.global_match.similarity_temperature
    legacy_ceiling = float(torch.sigmoid(torch.tensor(1.0)))

    identical = similarity(0.0, threshold=threshold, temperature=temperature)
    assert identical > legacy_ceiling
    assert similarity(0.0, threshold=threshold, temperature=0.1) > 0.99


def test_similarity_is_one_half_exactly_at_the_threshold(cfg: Settings) -> None:
    """The score and the decision cannot disagree.

    Legacy code thresholded ``d`` but *reported* ``sigmoid(1 - d)``, so at the
    ``d = 1.0`` boundary the reported score was 0.5 only by coincidence of that
    particular threshold. Here it is structural: 0.5 means "on the line" for any
    configured threshold.
    """
    for threshold in (0.5, 1.0, 2.0):
        assert similarity(threshold, threshold=threshold, temperature=0.5) == pytest.approx(0.5)


def test_similarity_is_monotonically_decreasing_in_distance() -> None:
    scores = [similarity(d / 4.0, threshold=1.0, temperature=0.5) for d in range(40)]
    assert all(a > b for a, b in pairwise(scores))
    assert 0.0 < scores[-1] < scores[0] < 1.0


def test_a_nonpositive_temperature_is_rejected() -> None:
    """Zero would divide, and negative would silently invert the ranking."""
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="temperature must be positive"):
            similarity(1.0, threshold=1.0, temperature=bad)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def test_a_missing_checkpoint_is_reported_as_such(tmp_path: Path) -> None:
    with pytest.raises(BackboneLoadError, match="no checkpoint at"):
        load_backbone(tmp_path / "absent.pth")


def test_a_checkpoint_without_a_head_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "headless.pth"
    torch.save({"block1.conv1.weight": torch.zeros(16, 1, 3, 3)}, path)
    with pytest.raises(BackboneLoadError, match="no recognisable embedding head"):
        load_backbone(path)


def test_a_non_state_dict_payload_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tensor.pth"
    torch.save(torch.zeros(4), path)
    with pytest.raises(BackboneLoadError, match="does not contain a state dict"):
        load_backbone(path)


def test_the_head_is_detected_rather_than_configured(tmp_path: Path) -> None:
    """A round trip through disk must recover the head from the file alone.

    There is deliberately no ``head:`` config key. The head is a property of the
    checkpoint, so a configured value could only ever agree with the file or
    contradict it -- and a contradiction would surface as a shape error rather than
    as the configuration mistake it is.
    """
    for kind, grid in (("flatten", 8), ("gap", None)):
        path = tmp_path / f"{kind}.pth"
        saved = EmbeddingNet(head=kind, head_grid=grid)  # type: ignore[arg-type]
        torch.save(saved.state_dict(), path)
        loaded = load_backbone(path, embed_size=128)
        assert loaded.head_kind == kind
        assert isinstance(loaded.head, FlattenHead if kind == "flatten" else PoolHead)


def test_a_flatten_checkpoint_is_checked_against_the_configured_size(tmp_path: Path) -> None:
    """The size must come from ``fc1``, not from ``embed_size``.

    A check that derived the requirement from ``embed_size`` and then compared it
    against ``embed_size`` is a tautology: it passes for every value, and the real
    mismatch then surfaces from inside ``load_state_dict`` as the torch traceback
    the check exists to replace. This test fails against that implementation --
    ``embed_size=224`` yields ``grid=14``, ``14 * 16 == 224``, no error raised.
    """
    path = tmp_path / "flatten128.pth"
    torch.save(EmbeddingNet(head="flatten", head_grid=8).state_dict(), path)

    assert load_backbone(path, embed_size=128).head.required_input_size == 128
    with pytest.raises(BackboneLoadError, match=r"fixed to 128x128 input.*embed_size is 224"):
        load_backbone(path, embed_size=224)


def test_a_partial_checkpoint_is_refused_not_partially_applied(tmp_path: Path) -> None:
    """``strict=True``. Half-loaded weights would still produce confident numbers.

    A tolerant load is the worst available option here: the network would run, the
    distances would look ordinary, and some fraction of the model would be
    random. Refusing is the only safe answer.
    """
    path = tmp_path / "partial.pth"
    state = EmbeddingNet(head="flatten", head_grid=8).state_dict()
    del state["block3.conv1.weight"]
    torch.save(state, path)
    with pytest.raises(BackboneLoadError, match="not compatible with the backbone"):
        load_backbone(path)


def test_a_head_of_the_wrong_square_size_is_diagnosed(tmp_path: Path) -> None:
    """``fc1`` must describe a square feature map of the trunk's width."""
    path = tmp_path / "odd.pth"
    state = EmbeddingNet(head="flatten", head_grid=8).state_dict()
    state["head.fc1.weight"] = torch.zeros(1024, 8190)
    torch.save(state, path)
    with pytest.raises(BackboneLoadError, match="not a square feature map"):
        load_backbone(path)


# ---------------------------------------------------------------------------
# the real checkpoint
# ---------------------------------------------------------------------------
@pytest.mark.weights
def test_the_shipped_checkpoint_loads_with_weights_only(weights_path: Path) -> None:
    """Bug 12. ``weights_only=True`` is both safe and sufficient here.

    Worth asserting rather than assuming: had the artifact been a pickled ``Model``
    object instead of a plain ``state_dict``, the safe loader would refuse it and
    the fix would have required re-exporting the checkpoint, not just changing a
    keyword.
    """
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    assert any(key.startswith("features.") for key in payload), "expected the legacy key layout"

    model = load_backbone(weights_path, embed_size=128)
    assert model.head_kind == "flatten"
    assert not model.training, "an inference load must leave the model in eval mode"


@pytest.mark.weights
def test_legacy_keys_are_remapped_onto_named_stages(weights_path: Path) -> None:
    """The checkpoint's ``features.N`` layout survives the rename to ``blockN``.

    ``features.4`` is the trailing 1x1 projection, not a fifth block -- an
    off-by-one here would load the projection's weights into ``block4`` and every
    embedding would be wrong while everything still ran.
    """
    model = load_backbone(weights_path, embed_size=128)
    legacy = torch.load(weights_path, map_location="cpu", weights_only=True)
    loaded = model.state_dict()

    assert torch.equal(loaded["block1.conv1.weight"], legacy["features.0.conv1.weight"])
    assert torch.equal(loaded["block4.conv2.bias"], legacy["features.3.conv2.bias"])
    assert torch.equal(loaded["projection.0.weight"], legacy["features.4.weight"])
    assert torch.equal(loaded["head.fc1.weight"], legacy["fc1.weight"])


@pytest.mark.weights
def test_embedding_matches_the_recorded_golden_vector(weights_path: Path) -> None:
    """The transcript of the port-fidelity check, frozen for after the legacy delete.

    While ``src/global_matching/`` still exists this was verified directly: the same
    input through the legacy ``Model(nin=True)`` and through ``EmbeddingNet``, both
    loaded from this checkpoint, differ by a maximum absolute value of ``0.0`` --
    not approximately equal, identical. These constants are that output.

    A failure here means the network's computation changed. That is occasionally
    intentional (stage B2 retrains it, and these numbers are then expected to move),
    but it must never happen as a side effect of a refactor, because it would shift
    every distance in the benchmark without shifting anything that looks wrong.
    """
    model = load_backbone(weights_path, embed_size=128)
    with torch.no_grad():
        embedding = model(golden_input())[0]

    assert embedding.shape == (EMBEDDING_DIM,)
    assert float(embedding.norm()) == pytest.approx(GOLDEN_NORM, abs=1e-5)
    assert float(embedding.sum()) == pytest.approx(GOLDEN_SUM, abs=1e-5)
    for index, expected in enumerate(GOLDEN_FIRST_EIGHT):
        assert float(embedding[index]) == pytest.approx(expected, abs=1e-5), f"element {index}"


@pytest.mark.weights
def test_identical_inputs_embed_to_zero_distance(weights_path: Path) -> None:
    """The sanity check whose legacy answer was 0.72.

    An image against itself must give distance 0 and therefore the maximum
    similarity. That the prototype reported 0.7234 for this case, with no
    indication it was saturated, is bug 9 in one number.
    """
    model = load_backbone(weights_path, embed_size=128)
    image = golden_input()
    with torch.no_grad():
        embedding = model(image)
    assert float(distance(embedding, embedding)) == 0.0
    assert similarity(0.0, threshold=1.0, temperature=0.5) > 0.87


@pytest.mark.weights
def test_loading_is_deterministic(weights_path: Path) -> None:
    """Two loads of one file must embed identically.

    Eval mode is the substance of this: with BatchNorm left in training mode the
    running statistics would update on every forward pass, so the same image would
    embed differently depending on what had been embedded before it.
    """
    image = golden_input()
    with torch.no_grad():
        first = load_backbone(weights_path, embed_size=128)(image)
        second = load_backbone(weights_path, embed_size=128)(image)
    assert torch.equal(first, second)
