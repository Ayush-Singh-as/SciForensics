"""Global stage: whole-image embedding, distance, and gradient attribution.

The first pass of the pipeline. It answers "are these two panels plausibly the
same content?" cheaply enough to run over a corpus, and hands anything suspicious
to :mod:`sciforensics.local_match` for keypoint-level proof.

The split matters for how results should be read. This stage produces a *score*
and a heatmap; it never produces evidence. A high similarity here is a reason to
look, not a finding -- the finding is a geometrically verified correspondence set,
and only the local stage can supply one.
"""

from __future__ import annotations

from sciforensics.global_match.backbone import (
    EMBEDDING_DIM,
    BackboneLoadError,
    ConvBlock,
    EmbeddingNet,
    FlattenHead,
    HeadKind,
    PoolHead,
    StageActivations,
    distance,
    load_backbone,
    parameter_count,
    similarity,
)
from sciforensics.global_match.embed import (
    Attribution,
    AttributionUnavailableError,
    EmbeddedImage,
    Embedder,
    PairComparison,
    blend_overlay,
    colourise,
    grad_cam,
    preprocess,
)

__all__ = [
    "EMBEDDING_DIM",
    "Attribution",
    "AttributionUnavailableError",
    "BackboneLoadError",
    "ConvBlock",
    "EmbeddedImage",
    "Embedder",
    "EmbeddingNet",
    "FlattenHead",
    "HeadKind",
    "PairComparison",
    "PoolHead",
    "StageActivations",
    "blend_overlay",
    "colourise",
    "distance",
    "grad_cam",
    "load_backbone",
    "parameter_count",
    "preprocess",
    "similarity",
]
