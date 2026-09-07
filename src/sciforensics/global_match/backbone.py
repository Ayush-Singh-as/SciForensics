"""The embedding backbone: a 4-block Siamese CNN over greyscale panels.

The architecture lineage is deliberately preserved. This is the network the
project trained, and "I found what was wrong with it and fixed it" is a more
useful claim than "I swapped in a pretrained ResNet" -- so the convolutional
trunk is ported faithfully, weight-for-weight, and the existing checkpoint loads
into it unchanged.

What changes here is everything *around* the trunk.

**Bug 9 -- the similarity score could never exceed 0.731.** The prototype reported
``sigmoid(1 - d)``, which attains ``sigmoid(1) = 0.7311`` at ``d = 0`` and falls
from there. Every "similarity" in the legacy ``all_results.json`` is at most
0.7234, *including an image compared against itself*, and a reader with no access
to the source had no way to know 0.72 meant "identical". :func:`similarity` divides
by a configured temperature and centres on the decision threshold, so 0.5 sits
exactly on the boundary and the full ``(0, 1)`` range is reachable.

**Bug 12 -- ``torch.load`` without ``weights_only``.** An unpickling checkpoint is
an arbitrary-code-execution surface, and since torch 2.6 the permissive default is
gone anyway, so the legacy call now fails outright on a current install.
:func:`load_backbone` passes ``weights_only=True``.

**Bug 8 (half) -- the flatten head hard-codes the input size.** ``Linear(8*8*128,
1024)`` holds 8.39M of the model's 8.84M trainable parameters -- 95% of the network
in one layer -- and silently requires exactly 128x128 input. Feeding it anything
else produced a bare shape-mismatch traceback from deep inside torch. The head is
now named and checked: :class:`FlattenHead` states its required input size, derived
from the *checkpoint's* ``fc1`` shape, and a mismatch against ``image.embed_size``
raises an explanatory error naming both numbers. :class:`PoolHead` is the stage-B2
replacement -- global average pooling, 0.33M parameters total, accepts any input
size -- and which one a checkpoint wants is *detected from the checkpoint* rather
than configured, so a config file can never contradict the file on disk.

The README's "~1.2M parameters" was wrong by 7.4x; :func:`parameter_count` exists so
that number is generated rather than remembered.

The port is verified bit-exact against the prototype: loading ``models/weights.pth``
into :class:`EmbeddingNet` and into the legacy ``Model(nin=True)`` and embedding the
same input gives a maximum absolute difference of ``0.0``. That equality is what
makes the surrounding fixes safe to claim as fixes -- nothing here silently changed
what the network computes. ``tests/test_backbone.py`` pins it with a golden vector so
it survives the legacy code's deletion.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from sciforensics.config import STAGE_STRIDES
from sciforensics.runtime import get_logger

__all__ = [
    "EMBEDDING_DIM",
    "BackboneLoadError",
    "ConvBlock",
    "EmbeddingNet",
    "FlattenHead",
    "HeadKind",
    "PoolHead",
    "StageActivations",
    "distance",
    "load_backbone",
    "parameter_count",
    "similarity",
]

_log = get_logger(__name__)

HeadKind = Literal["flatten", "gap"]

#: Channel widths of the four convolutional blocks. ``block4`` ends at 128, which
#: is also the embedding width, which is why the pooling head needs no projection
#: to hit the same dimensionality as the flatten head.
_WIDTHS: tuple[int, ...] = (16, 32, 64, 128)
EMBEDDING_DIM = 128


# ---------------------------------------------------------------------------
# trunk
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """One stage: normalise, optional 1x1 bottleneck, two 3x3 convs, pool.

    Ported from the prototype's ``ConvLayer`` with the parameter *order and
    naming* preserved exactly -- ``bn``, ``nin``, ``conv1``, ``conv2`` -- because
    the checkpoint's ``state_dict`` keys are those names. Renaming any of them for
    tidiness would silently break weight loading, and ``load_state_dict`` with
    ``strict=True`` is the only thing standing between that and a model of
    randomly-initialised weights reporting confident distances.

    ``BatchNorm`` leads rather than follows the convolutions, which is unusual but
    is what was trained; changing it would invalidate the checkpoint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bottleneck: bool = False,
        local_response_norm: bool = False,
        lrn_size: int = 5,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.lrn = nn.LocalResponseNorm(lrn_size) if local_response_norm else None
        self.nin = (
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) if bottleneck else None
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn(x)
        if self.lrn is not None:
            out = self.lrn(out)
        if self.nin is not None:
            out = self.relu(self.nin(out))
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        pooled: Tensor = self.pool(out)
        return pooled


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------
class FlattenHead(nn.Module):
    """The trained head: flatten the 8x8x128 map through two linear layers.

    Holds 8.52M of the network's 8.84M trainable parameters -- 96% -- with 8.39M of
    those in ``fc1`` alone. Its input dimension pins the image size to exactly
    :attr:`required_input_size`, which is the reason ``image.embed_size`` is not yet
    a free parameter, and it is stated here rather than discovered as a shape error.
    """

    def __init__(self, *, grid: int = 8, channels: int = 128, hidden: int = 1024) -> None:
        super().__init__()
        self._grid = grid
        self.fc1 = nn.Linear(grid * grid * channels, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, EMBEDDING_DIM)

    @property
    def grid(self) -> int:
        """Spatial edge length of the feature map this head expects."""
        return self._grid

    @property
    def required_input_size(self) -> int:
        """Input edge length this head demands, in pixels."""
        return self._grid * STAGE_STRIDES["block4"]

    def forward(self, features: Tensor) -> Tensor:
        flat = features.reshape(features.size(0), -1)
        embedding: Tensor = self.fc2(self.relu(self.fc1(flat)))
        return embedding


class PoolHead(nn.Module):
    """Stage B2's head: global average pooling, then one linear projection.

    Two properties the flatten head cannot offer. It accepts *any* input size,
    because pooling collapses the spatial dimensions before the linear layer sees
    them -- so ``image.embed_size`` becomes tunable and letterboxing to 224 stops
    being a rewrite. And it takes the network from 8.84M trainable parameters to
    0.33M, a 27x reduction, which matters when 670 real training images have to
    support it.

    Defined now, unused by the shipped checkpoint. It is here so that
    :func:`load_backbone` can recognise a B2 checkpoint the moment one exists,
    rather than the loader needing to change alongside the training script.
    """

    def __init__(self, *, channels: int = 128) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, EMBEDDING_DIM)

    @property
    def required_input_size(self) -> int | None:
        """``None``: any input size works."""
        return None

    def forward(self, features: Tensor) -> Tensor:
        pooled = self.pool(features).flatten(1)
        embedding: Tensor = self.fc(pooled)
        return embedding


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageActivations:
    """An embedding together with one intermediate feature map, for Grad-CAM.

    Returned alongside the embedding rather than stashed on the module: a module
    attribute would be shared mutable state between the two Siamese branches, and
    the whole point of bug 5 is that the two branches must be handled separately.

    :attr:`differentiable` records whether gradient tracking was active during the
    forward pass. It exists so that "no gradients available" and "the gradients are
    all zero" cannot be confused -- the first is a caller mistake that should raise,
    the second is a real, if uninformative, attribution result.
    """

    embedding: Tensor
    activations: Tensor
    stage: str

    @property
    def grid_size(self) -> int:
        """Spatial edge length of the retained feature map."""
        return int(self.activations.shape[-1])

    @property
    def differentiable(self) -> bool:
        """Whether a backward pass can populate ``activations.grad``."""
        return bool(self.activations.requires_grad)


class EmbeddingNet(nn.Module):
    """Greyscale panel -> 128-d embedding, with addressable intermediate stages.

    The prototype exposed only ``features`` as an opaque ``nn.Sequential``, so
    attribution code reached into it by index and the reports could not say which
    layer a heatmap came from. Blocks are named here (``block1`` .. ``block4``,
    matching ``STAGE_STRIDES`` and ``attribution.stage``) so a configuration can
    name a stage, a report can echo it, and the two cannot drift apart.
    """

    def __init__(
        self,
        *,
        head: HeadKind = "flatten",
        bottleneck: bool = True,
        local_response_norm: bool = False,
        embed_size: int = 128,
        head_grid: int | None = None,
    ) -> None:
        super().__init__()
        self.head_kind: HeadKind = head

        channels = 1
        blocks: list[tuple[str, nn.Module]] = []
        for index, width in enumerate(_WIDTHS, start=1):
            blocks.append(
                (
                    f"block{index}",
                    ConvBlock(
                        channels,
                        width,
                        # The first block takes a single-channel image; the
                        # prototype gave it no bottleneck and the checkpoint has
                        # no weights for one.
                        bottleneck=bottleneck and index > 1,
                        local_response_norm=local_response_norm and index > 1,
                    ),
                )
            )
            channels = width

        for name, block in blocks:
            self.add_module(name, block)
        self._stage_names = tuple(name for name, _ in blocks)

        # A trailing 1x1 + ReLU, present in the checkpoint only when bottlenecks
        # are enabled. Kept as `projection` rather than folded into block4 so its
        # state-dict keys stay distinguishable.
        self.projection: nn.Module | None = None
        if bottleneck:
            self.projection = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.ReLU()
            )

        self.head: FlattenHead | PoolHead
        if head == "flatten":
            # `head_grid` is supplied by `load_backbone`, which reads it out of the
            # checkpoint. Falling back to the `embed_size` implication is only for
            # constructing a fresh network, where there is no checkpoint to disagree
            # with.
            grid = head_grid if head_grid is not None else embed_size // STAGE_STRIDES["block4"]
            self.head = FlattenHead(grid=grid, channels=channels)
        else:
            self.head = PoolHead(channels=channels)

    # -- stage access -------------------------------------------------------
    @property
    def stage_names(self) -> tuple[str, ...]:
        return self._stage_names

    def blocks(self) -> Iterator[tuple[str, nn.Module]]:
        for name in self._stage_names:
            yield name, getattr(self, name)

    def trunk(self, x: Tensor) -> Tensor:
        """Run every block plus the projection: the full convolutional trunk."""
        for _, block in self.blocks():
            x = block(x)
        if self.projection is None:
            return x
        projected: Tensor = self.projection(x)
        return projected

    def forward(self, x: Tensor) -> Tensor:
        embedding: Tensor = self.head(self.trunk(x))
        return embedding

    def forward_to_stage(self, x: Tensor, stage: str) -> StageActivations:
        """Embed ``x`` while retaining the gradient of one named stage's output.

        A single forward pass produces both, so the activations that get
        attributed are provably the ones that produced the embedding -- running
        the trunk twice would be an invitation for BatchNorm or dropout state to
        make them disagree.

        Under ``torch.no_grad()`` this still returns the activations, but
        :attr:`StageActivations.differentiable` is ``False`` and ``.grad`` will stay
        ``None`` however the result is used. Callers that need gradients must check
        that flag rather than reading ``.grad`` and accepting whatever comes back --
        a ``None`` gradient treated as zero is precisely how the prototype came to
        render a uniform heatmap and present it as an explanation.
        """
        if stage not in self._stage_names:
            raise ValueError(
                f"unknown attribution stage {stage!r}; expected one of {self._stage_names}"
            )

        captured: Tensor | None = None
        out = x
        for name, block in self.blocks():
            out = block(out)
            if name == stage:
                if out.requires_grad:
                    out.retain_grad()
                captured = out
        assert captured is not None  # guaranteed by the membership check above

        if self.projection is not None:
            out = self.projection(out)
        return StageActivations(embedding=self.head(out), activations=captured, stage=stage)


# ---------------------------------------------------------------------------
# distance and similarity
# ---------------------------------------------------------------------------
def distance(left: Tensor, right: Tensor) -> Tensor:
    """L1 distance between embeddings, summed over the feature dimension.

    L1 rather than cosine or L2 because it is what the checkpoint was trained
    with, and ``global_match.distance_threshold`` is expressed in these units.
    Changing the metric would silently invalidate every configured threshold.
    """
    return torch.sum(torch.abs(left - right), dim=-1)


def similarity(dist: float, *, threshold: float, temperature: float) -> float:
    """Map an L1 distance to ``(0, 1)`` with 0.5 on the decision boundary.

    ``sigmoid((threshold - d) / temperature)``. **Bug 9's fix.** The legacy
    ``sigmoid(1 - d)`` was not a rescaling of this -- it was a *different function*
    whose maximum was ``sigmoid(1) = 0.7311``, so the top 27% of the range was
    unreachable and identical images scored 0.72. Here ``d == threshold`` gives
    exactly 0.5, ``d -> 0`` approaches 1, and ``d -> inf`` approaches 0.

    The result is still a monotone score, not a probability. Calling it calibrated
    requires stage B5, and :attr:`sciforensics.types.ScanResult.calibrated` is the
    flag that keeps the report honest about which it is.
    """
    if temperature <= 0.0:
        raise ValueError(f"similarity temperature must be positive, got {temperature}")
    logit = (threshold - dist) / temperature
    return float(torch.sigmoid(torch.tensor(logit, dtype=torch.float64)).item())


def parameter_count(model: nn.Module) -> dict[str, int]:
    """Trainable parameters, total and per top-level child.

    The README claimed "~1.2M parameters" while the real figure was 8.8M, 95% of
    it in one linear layer. Numbers that describe the code belong in the code.
    """
    counts = {
        name: sum(p.numel() for p in child.parameters() if p.requires_grad)
        for name, child in model.named_children()
    }
    counts["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
class BackboneLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded into a compatible network."""


def _detect_head(state: dict[str, Tensor]) -> HeadKind:
    """Infer which head a checkpoint carries from its parameter names.

    Detection rather than configuration: the head is a property of the file on
    disk, so a config knob for it could only ever agree or introduce a
    contradiction. Legacy checkpoints store the head at the top level
    (``fc1.weight``); ones written by this module nest it under ``head.``.
    """
    if any(key.endswith("fc1.weight") for key in state):
        return "flatten"
    if any(key.endswith("head.fc.weight") for key in state):
        return "gap"
    raise BackboneLoadError(
        "checkpoint contains no recognisable embedding head: expected 'fc1.weight' "
        f"(flatten) or 'head.fc.weight' (gap), found keys {sorted(state)[:8]}"
    )


def _remap_legacy_keys(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Translate the prototype's ``state_dict`` layout into this module's names.

    The prototype held its blocks in an ``nn.Sequential`` called ``features``, so
    the keys are ``features.0.bn.weight`` .. ``features.4.weight``, and its head
    sat at the top level as ``fc1`` / ``fc2``. Remapping here rather than
    reshaping the class is what lets the module have readable stage names *and*
    load the weights that already exist -- and it is reversible, so a checkpoint
    saved by this module needs no translation at all.
    """
    if not any(key.startswith("features.") for key in state):
        return state

    remapped: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.startswith("features."):
            _, index, *rest = key.split(".")
            suffix = ".".join(rest)
            position = int(index)
            if position < len(_WIDTHS):
                remapped[f"block{position + 1}.{suffix}"] = value
            else:
                # `features.4` is the trailing 1x1 conv; index 5 is its ReLU,
                # which has no parameters and therefore never appears here.
                remapped[f"projection.0.{suffix}"] = value
        elif key.startswith(("fc1.", "fc2.")):
            remapped[f"head.{key}"] = value
        else:
            remapped[key] = value
    return remapped


def _flatten_grid(state: dict[str, Tensor], *, channels: int = 128) -> int:
    """Recover the feature-map size a flatten-head checkpoint was built for.

    Read out of ``fc1.weight`` rather than inferred from the configured
    ``embed_size``, which is the whole point: comparing ``embed_size`` against a
    number *derived from* ``embed_size`` is a tautology that always passes, and the
    shape mismatch then surfaces from inside ``load_state_dict`` as the traceback
    this check exists to replace.
    """
    key = next((k for k in ("head.fc1.weight", "fc1.weight") if k in state), None)
    if key is None:  # pragma: no cover - guarded by _detect_head
        raise BackboneLoadError("flatten checkpoint has no fc1.weight")
    in_features = int(state[key].shape[1])
    grid_squared, remainder = divmod(in_features, channels)
    grid = math.isqrt(grid_squared)
    if remainder or grid * grid != grid_squared:
        raise BackboneLoadError(
            f"fc1 expects {in_features} inputs, which is not a square feature map of "
            f"{channels} channels; the checkpoint does not match this architecture"
        )
    return grid


def load_backbone(
    weights: str | Path,
    *,
    device: torch.device | str = "cpu",
    embed_size: int = 128,
) -> EmbeddingNet:
    """Build a network matching ``weights`` and load them into it.

    ``weights_only=True`` -- **bug 12's fix**. A checkpoint is data; unpickling it
    as code is an arbitrary-execution surface, and torch 2.6 made the safe
    behaviour the default, so the legacy call raises on any current install.

    ``strict=True`` on ``load_state_dict`` is equally deliberate. A tolerant load
    that skipped unmatched keys would leave part of the network randomly
    initialised and still return confident-looking distances -- a silent wrong
    answer of exactly the kind this refactor exists to eliminate.

    Everything the architecture needs is read *from the checkpoint*: which head it
    carries, whether its blocks have 1x1 bottlenecks, and what input size its head
    was sized for. ``embed_size`` is then checked against that, never used to derive
    it, so a configuration file cannot silently disagree with the file on disk.
    """
    path = Path(weights)
    if not path.is_file():
        raise BackboneLoadError(f"no checkpoint at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, dict):
        raise BackboneLoadError(f"{path} does not contain a state dict (got {type(raw).__name__})")
    state: dict[str, Tensor] = dict(raw)

    head = _detect_head(state)
    remapped = _remap_legacy_keys(state)
    bottleneck = any(".nin." in key for key in remapped)

    head_grid = _flatten_grid(remapped) if head == "flatten" else None
    if head_grid is not None:
        required = head_grid * STAGE_STRIDES["block4"]
        if required != embed_size:
            raise BackboneLoadError(
                f"{path} carries a flatten head fixed to {required}x{required} input "
                f"(fc1 expects a {head_grid}x{head_grid} feature map), but image.embed_size "
                f"is {embed_size}. Set image.embed_size={required}, or retrain with the "
                "pooling head (stage B2), which accepts any size."
            )

    model = EmbeddingNet(
        head=head, bottleneck=bottleneck, embed_size=embed_size, head_grid=head_grid
    )
    try:
        model.load_state_dict(remapped, strict=True)
    except RuntimeError as error:  # pragma: no cover - depends on the artifact
        raise BackboneLoadError(f"{path} is not compatible with the backbone: {error}") from error

    model.eval().to(device)
    counts = parameter_count(model)
    _log.debug(
        "loaded %s head from %s (%.2fM parameters) onto %s",
        head,
        path.name,
        counts["total"] / 1e6,
        device,
    )
    return model
