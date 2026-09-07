"""Global-stage inference: preprocessing, pairwise distance, and honest Grad-CAM.

Three audited defects live in this module's territory.

**Bug 5 — the right-hand heatmap was mathematically dead.** The distance is
``d = sum|e_L - e_R|``, so at the embedding the two branches' gradients are exact
negations: ``dd/de_R,k = -dd/de_L,k``. The prototype pooled the *signed* gradient into
Grad-CAM weights and then applied a ``ReLU``, so whichever branch landed negative was
annihilated. That is the ``relu`` mode below, and it is the only one of the three that
reproduces the defect -- ``signed`` pools the same signed gradient but takes ``abs`` of
the *map*, which restores the symmetry the ``ReLU`` destroyed.

Measured as the fraction of map cells at exactly zero, over the shipped checkpoint and
three input pairs -- ``base_cell.png`` against a manipulation of itself, against an
unrelated image, and a synthetic texture pair from ``tests/helpers.py``:

======  ======  ==========  ==========  =====
stage   mode    left zero%  right zero  ratio
======  ======  ==========  ==========  =====
block4  relu    1.6         **100.0**   64.0
block4  relu    1.6         **100.0**   64.0
block4  relu    1.6         **98.4**    63.0
block4  signed  1.6         1.6         1.0
block4  abs     1.6         1.6         1.0
block3  relu    **69.9**    21.1        3.3
block3  relu    **25.0**    6.6         3.8
block3  relu    **100.0**   0.4         256.0
block3  signed  0.4         0.4         1.0
block3  abs     0.4         0.4         1.0
======  ======  ==========  ==========  =====

The ``block4 relu`` rows are the bug in its shipped form: the right-hand map is zero
essentially everywhere, for every pair, which is why every "Global Heatmap B" in the old
reports is uniform blue.

Three things in that table are worth more than the headline.

*The asymmetry is the invariant; total annihilation is not.* On the synthetic pair the
right map is 98.4% zero rather than 100% -- one cell survived, so a check for a uniformly
zero map misses it while the heatmap is just as useless. What holds across every input
measured is the *ratio*: under ``relu`` one branch is zeroed 3-256x more than the other,
under ``abs`` and ``signed`` the ratio is exactly 1.0. The per-pair warning in
:meth:`Embedder.compare` and :attr:`Attribution.zero_fraction` both exist because of
this row.

*Moving the attribution stage would have hidden the bug, not fixed it.* At ``block3`` the
two branches' Jacobians have diverged enough that the sign flip only partially survives,
so the ratio drops from 64x to 3-4x on the asset pairs: still badly asymmetric, no longer
a blank image, and therefore no longer obvious to anyone looking at a report. Worse, it
becomes *intermittent* -- the synthetic pair still annihilates a branch completely at
``block3``. This module changes the stage for resolution reasons only; the pooling mode is
what fixes the bug.

*Under ``abs`` the single zeroed cell is an artifact of normalisation, not suppression.*
1.6% is exactly 1/64 at ``block4`` and 0.4% is exactly 1/256 at ``block3``: the argmin,
mapped to 0 by the rescale to ``[0, 1]``. Both branches are treated identically because
``alpha_c = mean(|dd/dA_c|)`` is non-negative and the activations already are too (every
block ends in ``maxpool(relu(.))``), so no ``ReLU`` is needed at all.

The signed modes stay selectable so the failure can be *demonstrated* rather than
asserted; ``tests/test_embed.py`` reproduces the table above.

The prototype had a second, quieter version of the same bug: ``_overlay_heatmap``
returned ``base_bgr.copy()`` when the gradients were ``None``, so an attribution that
could not be computed was presented as an image with no salient region. Here
:class:`AttributionUnavailableError` is raised, and a map that comes out genuinely flat
is reported as :attr:`Attribution.is_degenerate` rather than rendered as an explanation.

**Bug 8 — train/test preprocessing mismatch.** Handled in
:mod:`sciforensics.io.images`; this module's part is to hold the
:class:`~sciforensics.io.images.EmbedFit` that performed the forward transform and to
invert *through that object*, so the overlay geometry cannot drift from the
preprocessing. See ``image.embed_resize`` in ``configs/default.yaml`` for why the
shipped default is the legacy squash and not the letterbox.

**Bug 11 — the model was reloaded per image.** ``ForensicScanner`` was constructed
inside the batch loop, so a 40-pair run performed 40 separate 35 MB ``torch.load``
calls. :class:`Embedder` holds one loaded backbone and is the unit that gets reused;
constructing it is the expensive step and doing so is now explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch import Tensor

from sciforensics.config import GlobalMatchConfig, ImageConfig
from sciforensics.global_match.backbone import (
    EMBEDDING_DIM,
    EmbeddingNet,
    StageActivations,
    distance,
    load_backbone,
    similarity,
)
from sciforensics.io.images import EmbedFit, fit_for_embedding
from sciforensics.runtime import get_logger, resolve_device
from sciforensics.types import GlobalEvidence

__all__ = [
    "Attribution",
    "AttributionUnavailableError",
    "EmbeddedImage",
    "Embedder",
    "PairComparison",
    "blend_overlay",
    "colourise",
    "grad_cam",
    "preprocess",
    "resolve_colormap",
]

_log = get_logger(__name__)

#: Training normalisation, ``transforms.Normalize([0.5], [0.5])`` applied after
#: ``ToTensor``. Reproduced here as explicit constants because it is not a free
#: choice: the checkpoint's first BatchNorm was fitted to inputs in this range, and
#: silently switching to ImageNet statistics would shift every distance.
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5

#: Thresholds for the per-pair "one branch is suppressed" warning in
#: :meth:`Embedder.compare`, in units of :attr:`Attribution.zero_fraction`.
#:
#: Keyed on absolute suppression plus a gap rather than on
#: :attr:`Attribution.is_degenerate`, because degeneracy is a knife edge: measured at
#: ``block4`` under ``relu`` the right-hand map is 100% zero on the repository's asset
#: pairs but 98.4% zero on a synthetic pair -- one surviving cell, an equally useless
#: heatmap, and a degeneracy check that stays silent. These two numbers fire on all six
#: measured ``relu`` configurations and on none of the twelve ``abs``/``signed`` ones,
#: where the branches are zeroed identically (ratio exactly 1.0).
SUPPRESSED_FRACTION = 0.20
SUPPRESSED_GAP = 0.15


class AttributionUnavailableError(RuntimeError):
    """Raised when a heatmap was requested but no gradient reached the stage.

    Deliberately loud. The alternative -- the prototype's -- was to return the base
    image unmodified, which reads as "the model found nothing salient here" when the
    truth is "the model was never asked".
    """


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def preprocess(
    gray: np.ndarray,
    *,
    embed_size: int,
    mode: Literal["squash", "letterbox"] = "squash",
    pad_value: int = 0,
) -> tuple[Tensor, EmbedFit]:
    """Fit a greyscale panel to the network's input and normalise it.

    Returns the ``(1, 1, S, S)`` tensor *and* the fit object that produced it. The
    pairing is the point: any map computed from this tensor is inverted by calling
    :meth:`~sciforensics.io.images.Letterbox.project_to_source` on the returned fit,
    so the forward and reverse transforms cannot be written independently and drift.

    ``mode`` is ``image.embed_resize``. It is validated at runtime as well as by the
    annotation, because the value's origin is a YAML file and static typing does not
    reach that far.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a single-channel image, got shape {gray.shape}")
    if mode not in ("squash", "letterbox"):
        raise ValueError(f"unknown embed_resize mode {mode!r}; expected squash or letterbox")

    canvas, fit = fit_for_embedding(
        gray,
        embed_size,
        mode=mode,
        pad_value=pad_value,
        interpolation=cv2.INTER_AREA,
    )
    scaled = (canvas.astype(np.float32) / 255.0 - PIXEL_MEAN) / PIXEL_STD
    return torch.from_numpy(scaled).view(1, 1, embed_size, embed_size), fit


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def grad_cam(captured: StageActivations, *, pooling: str = "abs") -> np.ndarray:
    """Grad-CAM map for one Siamese branch, as a ``(grid, grid)`` float array.

    The returned map is normalised to ``[0, 1]``, or is all zeros when the raw map is
    constant (see :attr:`Attribution.is_degenerate`).

    Pooling modes
    -------------
    ``abs``
        ``alpha_c = mean_spatial(|dd/dA_c|)``. **Bug 5's fix, and the default.** The
        weights are non-negative and the activations are already non-negative (each
        block ends in ``maxpool(relu(.))``), so the map is non-negative without a
        ``ReLU`` and both branches are treated identically.

        This gives up the *sign* of each location's contribution, which is the right
        trade: once the branch identity is discarded the sign is not interpretable
        anyway, and the question a reader is actually asking -- "where do these two
        panels' representations differ" -- is a magnitude question.
    ``relu``
        The prototype's behaviour: pool the signed gradient, then ``ReLU`` the map. The
        only mode that reproduces bug 5, retained so the annihilation stays demonstrable.
        At ``block4`` -- the stage the prototype actually used -- the right-hand map comes
        out zero across 98-100% of its cells depending on the input; away from the final
        block the suppression is smaller but still 3-4x asymmetric. See the module
        docstring's table.
    ``signed``
        Pool the signed gradient and take the absolute value of the *map* rather than
        of the weights. Recovers the magnitude the ``ReLU`` would have thrown away, so
        it does not annihilate a branch, but the per-channel cancellation inside the
        sum still loses signal that ``abs`` keeps.

    Raises
    ------
    AttributionUnavailableError
        If the forward pass ran under ``torch.no_grad()``, or if ``backward()`` has
        not been called on a scalar derived from the embedding.
    """
    if not captured.differentiable:
        raise AttributionUnavailableError(
            f"stage {captured.stage!r} was captured under torch.no_grad(), so no "
            "gradient can reach it. Run the forward pass with gradients enabled."
        )
    gradient = captured.activations.grad
    if gradient is None:
        raise AttributionUnavailableError(
            f"stage {captured.stage!r} has no gradient yet; call backward() on the "
            "distance before requesting attribution"
        )

    activations = captured.activations.detach()
    if pooling == "abs":
        weights = gradient.abs().mean(dim=(0, 2, 3))
        raw = (weights.view(1, -1, 1, 1) * activations).sum(dim=1)
    elif pooling == "relu":
        weights = gradient.mean(dim=(0, 2, 3))
        raw = torch.relu((weights.view(1, -1, 1, 1) * activations).sum(dim=1))
    elif pooling == "signed":
        weights = gradient.mean(dim=(0, 2, 3))
        raw = (weights.view(1, -1, 1, 1) * activations).sum(dim=1).abs()
    else:
        raise ValueError(f"unknown gradient_pooling {pooling!r}; expected abs, relu or signed")

    cam = raw[0].detach().to(torch.float32).cpu().numpy()
    low, high = float(cam.min()), float(cam.max())
    if high - low <= 0.0:
        # Constant map. Returning zeros rather than dividing by zero, and
        # `Attribution.is_degenerate` is what tells the report not to draw it.
        return np.zeros_like(cam, dtype=np.float32)
    normalised: np.ndarray = (cam - low) / (high - low)
    return normalised.astype(np.float32)


@dataclass(frozen=True)
class Attribution:
    """A normalised Grad-CAM map plus the transform needed to place it on the source.

    The map is held at feature-map resolution (16x16 at the default
    ``block3``/``embed_size=128``) and upsampled only when projected, so nothing
    downstream can mistake an interpolated map for a measured one. :attr:`fit` is the
    exact object that preprocessed the image, which is what makes
    :meth:`to_source` correct for either ``embed_resize`` mode without branching.
    """

    grid: np.ndarray
    stage: str
    pooling: str
    fit: EmbedFit

    @property
    def grid_size(self) -> int:
        return int(self.grid.shape[-1])

    @property
    def is_degenerate(self) -> bool:
        """``True`` when the map is constant and therefore explains nothing.

        A uniform map is a legitimate outcome -- and under ``pooling='relu'`` at
        ``block4`` it is the near-universal outcome for one of the two branches. What is
        not legitimate is rendering it as though it localised something, which is what
        the prototype did.

        Necessary but not sufficient as a health check. It is a strict test on a
        continuous quantity, so it sits on a knife edge: the same ``relu`` configuration
        that zeroes 100% of a map on one input zeroes 98.4% on another, and this returns
        ``False`` for the second while the heatmap is equally unusable. Use
        :attr:`zero_fraction` to judge a map, and this only to decide whether it can be
        normalised at all.
        """
        return bool(np.ptp(self.grid) <= 0.0)

    @property
    def zero_fraction(self) -> float:
        """Fraction of cells at exactly zero: the measure of how suppressed a map is.

        The quantity bug 5 is actually visible in, and the reason this property exists.
        :attr:`is_degenerate` catches only total annihilation, which turns out to be
        input-dependent; *comparing this figure between the two branches* is what makes
        the asymmetry legible in every case measured -- 64x at ``block4`` under ``relu``,
        3-4x at ``block3``, exactly 1.0 under ``abs``.

        Note the floor. A healthy map still reports one zeroed cell, because normalising
        to ``[0, 1]`` sends the argmin to zero: ``1/64`` = 1.6% at ``block4``, ``1/256`` =
        0.4% at ``block3``. Anything at that level is the rescale, not suppression.
        """
        return float((self.grid == 0.0).mean())

    def to_canvas(self, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
        """Upsample to the network's square input resolution."""
        size = self.fit.out_size
        resized = cv2.resize(self.grid, (size, size), interpolation=interpolation)
        clipped: np.ndarray = np.clip(resized, 0.0, 1.0)
        return clipped

    def to_source(self, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
        """Project onto the original image's pixel grid, as ``(src_h, src_w)``.

        Two steps, both inverses of forward operations: upsample the feature map to
        canvas resolution, then undo the fit. For a letterbox that means cropping the
        padding away *before* the resize -- resizing the padded canvas directly would
        smear the pad into the borders and shift the whole map inward.
        """
        projected = self.fit.project_to_source(self.to_canvas(interpolation), interpolation)
        clipped: np.ndarray = np.clip(projected, 0.0, 1.0)
        return clipped


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def resolve_colormap(name: str) -> int:
    """Look up an OpenCV colormap constant by name, e.g. ``"JET"``.

    Validated here rather than trusted, so a typo in ``attribution.colormap``
    produces a message listing the alternatives instead of an ``AttributeError``
    from inside the renderer.
    """
    attribute = f"COLORMAP_{name.upper()}"
    value = getattr(cv2, attribute, None)
    if not isinstance(value, int):
        available = sorted(n[len("COLORMAP_") :] for n in dir(cv2) if n.startswith("COLORMAP_"))
        raise ValueError(f"unknown colormap {name!r}; available: {', '.join(available)}")
    return value


def colourise(heat: np.ndarray, *, colormap: str = "JET") -> np.ndarray:
    """Turn a ``[0, 1]`` float map into a BGR image."""
    scaled = np.clip(heat * 255.0, 0.0, 255.0).astype(np.uint8)
    coloured: np.ndarray = cv2.applyColorMap(scaled, resolve_colormap(colormap))
    return coloured


def blend_overlay(
    base_bgr: np.ndarray,
    heat: np.ndarray,
    *,
    colormap: str = "JET",
    alpha: float = 0.45,
) -> np.ndarray:
    """Composite a heatmap over an image.

    ``heat`` must already be in ``base_bgr``'s pixel frame -- normally the output of
    :meth:`Attribution.to_source`. The shape is checked rather than resized: a
    silent resize here would hide exactly the geometry mismatch that
    :class:`Attribution` exists to prevent.
    """
    if heat.shape[:2] != base_bgr.shape[:2]:
        raise ValueError(
            f"heatmap is {heat.shape[1]}x{heat.shape[0]} but the image is "
            f"{base_bgr.shape[1]}x{base_bgr.shape[0]}; project it with "
            "Attribution.to_source() first rather than resizing it here"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    coloured = colourise(heat, colormap=colormap)
    blended: np.ndarray = cv2.addWeighted(base_bgr, 1.0 - alpha, coloured, alpha, 0.0)
    return blended


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddedImage:
    """One image's embedding and the preprocessing that produced it."""

    embedding: Tensor
    fit: EmbedFit
    captured: StageActivations | None = None

    @property
    def vector(self) -> np.ndarray:
        """The embedding as a 1-D float32 array, for indexing and serialisation."""
        return self.embedding.detach()[0].to(torch.float32).cpu().numpy()


@dataclass(frozen=True)
class PairComparison:
    """The global stage's complete output for one pair.

    Carries the thresholds it was judged against, not just the verdict. A bare
    ``is_match`` boolean is unreproducible six months later when the configured
    threshold has moved; the pair of numbers is what makes the decision auditable.
    """

    left: EmbeddedImage
    right: EmbeddedImage
    distance: float
    similarity: float
    distance_threshold: float
    local_trigger_distance: float
    attribution_left: Attribution | None = None
    attribution_right: Attribution | None = None

    @property
    def is_match(self) -> bool:
        return self.distance < self.distance_threshold

    @property
    def triggered_local(self) -> bool:
        """Whether the pair warrants the expensive keypoint stage.

        True for a match *and* for the ambiguous band above it. The global stage
        cannot produce evidence, only a reason to look, so escalation is deliberately
        wider than the match decision.
        """
        return self.distance < self.local_trigger_distance

    @property
    def evidence(self) -> GlobalEvidence:
        """Serialisable evidence record.

        ``attribution_left`` / ``attribution_right`` are left ``None``: they are
        *file paths*, filled in by whichever consumer writes the overlays to disk.
        Use ``model_copy(update=...)`` to populate them.
        """
        return GlobalEvidence(
            distance=self.distance,
            similarity=self.similarity,
            distance_threshold=self.distance_threshold,
            local_trigger_distance=self.local_trigger_distance,
            is_match=self.is_match,
            triggered_local=self.triggered_local,
            embedding_dim=int(self.left.embedding.shape[-1]),
        )


# ---------------------------------------------------------------------------
# the embedder
# ---------------------------------------------------------------------------
class Embedder:
    """A loaded backbone, reused across every image in a run. **Bug 11's fix.**

    Construction is the expensive part -- a 35 MB read, a state-dict load and a
    device transfer -- so it happens once and is visible in the call graph. The
    prototype rebuilt its scanner inside the batch loop, which cost a fresh
    ``torch.load`` per image and made a 40-pair sweep spend most of its time in
    deserialisation.

    Not thread-safe by design. :meth:`compare` runs a backward pass to obtain
    attribution, which mutates ``.grad`` on the captured activations, so concurrent
    calls on one instance would interleave gradients between pairs. The HTTP API
    serialises work through a job queue for this reason; a thread pool would need one
    instance per worker.
    """

    def __init__(
        self,
        cfg: GlobalMatchConfig,
        *,
        image: ImageConfig,
        weights: Path | None = None,
        device: torch.device | str | None = None,
        model: EmbeddingNet | None = None,
    ) -> None:
        """
        Parameters
        ----------
        cfg
            ``global_match`` settings: thresholds, temperature, attribution.
        image
            ``image`` settings. Only ``embed_size``, ``embed_resize`` and
            ``letterbox_pad_value`` are read.
        weights
            Checkpoint path. Defaults to ``cfg.weights``, resolved through
            :func:`sciforensics.weights.ensure` when that is ``None``.
        device
            Defaults to ``runtime.device`` resolution. Passing ``"cpu"`` explicitly
            is worth doing in tests: CUDA reductions are not bit-reproducible across
            architectures, so a golden vector recorded on one GPU may not match on
            another.
        model
            Pre-built network, bypassing checkpoint loading entirely. Exists so
            architecture tests can use a randomly-initialised net without a 35 MB
            artifact, and so a training run can attach the net it just fitted.
        """
        self.cfg = cfg
        self.image = image
        self.device = torch.device(device) if device is not None else resolve_device()

        if model is not None:
            self.model = model.to(self.device).eval()
            self.weights_path: Path | None = None
        else:
            from sciforensics import weights as weights_module

            resolved = weights_module.ensure(weights if weights is not None else cfg.weights)
            self.model = load_backbone(resolved, device=self.device, embed_size=image.embed_size)
            self.weights_path = resolved
            _log.debug("loaded backbone from %s onto %s", resolved, self.device)

        required = self.model.head.required_input_size
        if required is not None and required != image.embed_size:
            raise ValueError(
                f"the backbone requires {required}x{required} input but "
                f"image.embed_size is {image.embed_size}"
            )

        if cfg.attribution.enabled and cfg.attribution.gradient_pooling != "abs":
            # Warned once here rather than per pair, because Embedder construction is
            # rare by design (bug 11) and per-pair logging would be noise. Not an
            # error: reproducing bug 5 on demand is a legitimate use, and
            # tests/test_embed.py depends on it.
            #
            # The two modes get different text because they are different problems, and
            # a warning that conflates them would send a `signed` user looking for a
            # dead branch they do not have. Measured, `signed` zeroes both branches
            # identically (ratio 1.0 on every pair tried); only `relu` annihilates.
            if cfg.attribution.gradient_pooling == "relu":
                _log.warning(
                    "attribution.gradient_pooling='relu' reproduces bug 5. The Siamese "
                    "branches' embedding gradients are exact negations, so pooling the "
                    "signed gradient and then clipping at zero suppresses whichever "
                    "branch lands negative -- 98-100%% of the map at block4, 3-4x "
                    "asymmetric elsewhere. Use 'abs' for a map a reader can trust."
                )
            else:
                _log.warning(
                    "attribution.gradient_pooling='signed' does not annihilate a branch "
                    "-- taking abs of the map restores the symmetry bug 5 destroyed -- "
                    "but positive and negative channel contributions still cancel "
                    "spatially inside the sum, which loses localisation 'abs' keeps."
                )

    # -- single images ------------------------------------------------------
    def _fit(self, gray: np.ndarray) -> tuple[Tensor, EmbedFit]:
        tensor, fit = preprocess(
            gray,
            embed_size=self.image.embed_size,
            mode=self.image.embed_resize,
            pad_value=self.image.letterbox_pad_value,
        )
        return tensor.to(self.device), fit

    def embed(self, gray: np.ndarray) -> EmbeddedImage:
        """Embed one panel, without gradients.

        The cheap path, used for corpus indexing and for pairs where attribution is
        switched off. No activations are captured, so :attr:`EmbeddedImage.captured`
        is ``None``.
        """
        tensor, fit = self._fit(gray)
        with torch.no_grad():
            embedding = self.model(tensor)
        return EmbeddedImage(embedding=embedding, fit=fit)

    def embed_batch(self, grays: list[np.ndarray]) -> list[EmbeddedImage]:
        """Embed several panels in one forward pass.

        Batching is safe here only because the network is in ``eval`` mode: with
        BatchNorm still updating running statistics, an image's embedding would
        depend on what it was batched with. :func:`load_backbone` guarantees eval
        mode, and ``tests/test_backbone.py::test_loading_is_deterministic`` pins it.
        """
        if not grays:
            return []
        fitted = [self._fit(gray) for gray in grays]
        stacked = torch.cat([tensor for tensor, _ in fitted], dim=0)
        with torch.no_grad():
            embeddings = self.model(stacked)
        return [
            EmbeddedImage(embedding=embeddings[index : index + 1], fit=fit)
            for index, (_, fit) in enumerate(fitted)
        ]

    # -- pairs --------------------------------------------------------------
    def compare(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        *,
        attribution: bool | None = None,
    ) -> PairComparison:
        """Embed a pair, score it, and attribute the distance to both branches.

        One backward pass on the scalar distance supplies both heatmaps, which is
        what makes them comparable: they explain the same number. Attributing each
        branch from its own separate objective -- the obvious alternative -- would
        produce two maps that answer two different questions and invite a reader to
        compare them anyway.

        Parameters
        ----------
        attribution
            Overrides ``cfg.attribution.enabled``. ``False`` skips gradient tracking
            entirely, which is the right setting for a benchmark sweep that only
            needs distances.
        """
        want = self.cfg.attribution.enabled if attribution is None else attribution
        if not want:
            left, right = self.embed(left_gray), self.embed(right_gray)
            dist = float(distance(left.embedding, right.embedding).item())
            return self._assemble(left, right, dist, None, None)

        stage = self.cfg.attribution.stage
        pooling = self.cfg.attribution.gradient_pooling
        left_tensor, left_fit = self._fit(left_gray)
        right_tensor, right_fit = self._fit(right_gray)

        # Three things are being arranged here.
        #
        # `enable_grad` rather than relying on ambient state: the pipeline may well be
        # running inside an outer `torch.no_grad()`, and a silently non-differentiable
        # capture is the precise failure `StageActivations.differentiable` exists to
        # surface. Better to make it impossible than to diagnose it.
        #
        # `requires_grad_` on the *inputs* rather than depending on the parameters to
        # carry the graph. Both work today, but differentiating with respect to the
        # image is what Grad-CAM actually means, and it keeps attribution working if
        # the backbone is ever handed over with its parameters frozen -- at which
        # point relying on the parameters would leave `differentiable` False and this
        # method raising for no reason a caller could see.
        #
        # `zero_grad` on both sides of the backward. Before, so nothing accumulates
        # across pairs; after, to release the parameter gradients the backward pass
        # unavoidably fills in -- 35 MB that would otherwise stay resident in an
        # inference worker for the lifetime of the process.
        with torch.enable_grad():
            left_capture = self.model.forward_to_stage(left_tensor.requires_grad_(True), stage)
            right_capture = self.model.forward_to_stage(right_tensor.requires_grad_(True), stage)
            scalar = distance(left_capture.embedding, right_capture.embedding).sum()
            self.model.zero_grad(set_to_none=True)
            scalar.backward()
            self.model.zero_grad(set_to_none=True)

        dist = float(scalar.detach().item())
        left_map = Attribution(
            grid=grad_cam(left_capture, pooling=pooling),
            stage=stage,
            pooling=pooling,
            fit=left_fit,
        )
        right_map = Attribution(
            grid=grad_cam(right_capture, pooling=pooling),
            stage=stage,
            pooling=pooling,
            fit=right_fit,
        )
        left_zero, right_zero = left_map.zero_fraction, right_map.zero_fraction
        dead, dead_zero, live_zero = (
            ("right", right_zero, left_zero)
            if right_zero > left_zero
            else ("left", left_zero, right_zero)
        )
        if dead_zero >= SUPPRESSED_FRACTION and dead_zero - live_zero >= SUPPRESSED_GAP:
            # One branch's map is mostly dead while the other's is not: a strictly worse
            # event than the mode-level caveat the constructor already warned about,
            # because it means one of the two images in *this* report has little or no
            # explanation. Named per pair for that reason.
            _log.warning(
                "%s-hand attribution is %.0f%% zero at %s under gradient_pooling=%r "
                "while the other side is %.0f%% zero; this pair's %s heatmap explains "
                "little and should not be rendered as though it localised something",
                dead,
                100.0 * dead_zero,
                stage,
                pooling,
                100.0 * live_zero,
                dead,
            )

        left_embedded = EmbeddedImage(
            embedding=left_capture.embedding.detach(), fit=left_fit, captured=left_capture
        )
        right_embedded = EmbeddedImage(
            embedding=right_capture.embedding.detach(), fit=right_fit, captured=right_capture
        )
        return self._assemble(left_embedded, right_embedded, dist, left_map, right_map)

    def distance_to(self, left: EmbeddedImage, right: EmbeddedImage) -> float:
        """Score an already-embedded pair, e.g. two hits from the corpus index."""
        return float(distance(left.embedding, right.embedding).item())

    def score(self, dist: float) -> float:
        """Map a distance to the configured similarity. **Bug 9's fix**, applied."""
        return similarity(
            dist,
            threshold=self.cfg.distance_threshold,
            temperature=self.cfg.similarity_temperature,
        )

    def _assemble(
        self,
        left: EmbeddedImage,
        right: EmbeddedImage,
        dist: float,
        left_map: Attribution | None,
        right_map: Attribution | None,
    ) -> PairComparison:
        return PairComparison(
            left=left,
            right=right,
            distance=dist,
            similarity=self.score(dist),
            distance_threshold=self.cfg.distance_threshold,
            local_trigger_distance=self.cfg.local_trigger_distance,
            attribution_left=left_map,
            attribution_right=right_map,
        )

    def __repr__(self) -> str:
        source = self.weights_path.name if self.weights_path else "in-memory"
        return (
            f"Embedder(weights={source!r}, device={self.device}, "
            f"embed_size={self.image.embed_size}, dim={EMBEDDING_DIM})"
        )
