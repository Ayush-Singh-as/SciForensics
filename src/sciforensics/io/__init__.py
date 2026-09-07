"""Input handling: image loading, PDF figure extraction, panel splitting."""

from __future__ import annotations

from sciforensics.io.images import (
    EmbedFit,
    ImageLoadError,
    Letterbox,
    LoadedImage,
    Squash,
    fit_for_embedding,
    letterbox,
    load_image,
    squash,
)

__all__ = [
    "EmbedFit",
    "ImageLoadError",
    "Letterbox",
    "LoadedImage",
    "Squash",
    "fit_for_embedding",
    "letterbox",
    "load_image",
    "squash",
]
