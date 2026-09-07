"""Image loading, resolution capping and aspect-preserving letterboxing.

Two of the audited defects live here.

**Bug 8 — a train/test preprocessing mismatch.** The audit originally recorded this
as "the overlay points at the wrong pixels", and that part was wrong: the legacy code
squashed the frame to 128x128 and then resized the 8x8 map back to the original
width and height, which *is* the exact inverse of a squash, so the overlay geometry
was consistent. The real defect is upstream of the overlay. Training preprocessed
with ``transforms.Resize(256)`` followed by ``CenterCrop(128)`` — short side to 256
with aspect *preserved*, then a centre crop — while inference squashed the whole
frame. The network was therefore evaluated on a different aspect handling *and* a
much wider field of view than it was ever trained on.

:class:`Letterbox` is the general machinery for fixing that: an aspect-preserving
resize paired with an exact inverse, so a map computed in network space projects back
onto source pixels whatever the forward transform was. It is not, however, a drop-in
improvement for the *current* checkpoint, which has never seen padding either —
measured over ``inputs/``, letterboxing costs 0.075 AUC against the legacy squash.
``image.embed_resize`` therefore ships as ``squash``, with the measurement recorded in
``configs/default.yaml``, and flips to ``letterbox`` when stage B2 retrains on
letterboxed frames. The honest summary is that no inference-time preprocessing is
correct for this checkpoint; retraining is the fix, and this module is what makes the
switch a one-line configuration change.

That switch is what :class:`Squash` exists for. The legacy squash was *not* wrong to
invert — it inverted correctly — but the inverse was a second hand-written
``cv2.resize`` 120 lines from the forward one. Giving the squash the same interface as
:class:`Letterbox` means :func:`fit_for_embedding` can return either, and the caller
inverts by asking the object rather than by re-deriving the geometry itself.

**Bug 14 — no resolution cap.** Nothing bounded the input size before
:func:`~sciforensics.local_match.keypoints.enhance` applied its 4x upscale. A
3840x2400 photograph became a 15360x9600 (147 megapixel) working image, and a
single pair took over two minutes. :func:`load_image` caps the longest edge and
records the fact in :class:`~sciforensics.types.ImageMeta`, while every
coordinate reported downstream is mapped back to the original pixel frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from sciforensics.audit import file_sha256
from sciforensics.runtime import get_logger
from sciforensics.types import ImageMeta

__all__ = [
    "EmbedFit",
    "ImageLoadError",
    "Letterbox",
    "LoadedImage",
    "Squash",
    "fit_for_embedding",
    "letterbox",
    "load_image",
    "probe_dimensions",
    "squash",
]

_log = get_logger(__name__)

INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

#: Refuse to decode anything above this many pixels unless the caller raises the
#: limit. A 20000x20000 PNG compresses to a few hundred kilobytes but wants
#: 1.2 GB of RAM as BGR; that is a denial-of-service vector once the HTTP API is
#: exposed, not merely a performance problem.
DEFAULT_MAX_DECODED_PIXELS = 50_000_000


class ImageLoadError(ValueError):
    """Raised when an input cannot be read, or is implausibly large."""


# ---------------------------------------------------------------------------
# letterboxing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Letterbox:
    """Invertible aspect-preserving resize into a square canvas.

    The source is scaled by a single factor (so aspect is preserved), centred,
    and padded to ``out_size`` x ``out_size``. Holding the parameters lets any
    map computed in canvas space be projected back onto the source exactly,
    which is what makes the attribution overlay trustworthy.

    Attributes
    ----------
    scale
        Applied to source pixels to reach canvas pixels.
    pad_x, pad_y
        Left and top padding in canvas pixels.
    content_w, content_h
        Size of the scaled source inside the canvas.
    """

    out_size: int
    src_w: int
    src_h: int
    scale: float
    pad_x: int
    pad_y: int
    content_w: int
    content_h: int

    @classmethod
    def compute(cls, src_w: int, src_h: int, out_size: int) -> Letterbox:
        if src_w <= 0 or src_h <= 0:
            raise ImageLoadError(f"degenerate image size {src_w}x{src_h}")
        scale = min(out_size / src_w, out_size / src_h)
        # Clamp to >=1 px: a 4000x3 input would otherwise round to zero height
        # and produce an empty canvas rather than a visible error.
        content_w = max(1, min(out_size, round(src_w * scale)))
        content_h = max(1, min(out_size, round(src_h * scale)))
        return cls(
            out_size=out_size,
            src_w=src_w,
            src_h=src_h,
            scale=scale,
            pad_x=(out_size - content_w) // 2,
            pad_y=(out_size - content_h) // 2,
            content_w=content_w,
            content_h=content_h,
        )

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` of the real image content within the canvas."""
        return (self.pad_x, self.pad_y, self.content_w, self.content_h)

    def to_canvas(self, points: np.ndarray) -> np.ndarray:
        """Map source ``(N, 2)`` coordinates into canvas space."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = pts * self.scale
        out[:, 0] += self.pad_x
        out[:, 1] += self.pad_y
        return out

    def to_source(self, points: np.ndarray) -> np.ndarray:
        """Map canvas ``(N, 2)`` coordinates back to source space."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        pts[:, 0] -= self.pad_x
        pts[:, 1] -= self.pad_y
        return pts / self.scale

    def unpad(self, canvas_map: np.ndarray) -> np.ndarray:
        """Crop a canvas-sized map down to just the image content.

        Use before resizing an attribution map back onto the source: resizing
        the padded canvas directly would smear the padding into the borders and
        shift the whole map inward.
        """
        if canvas_map.shape[:2] != (self.out_size, self.out_size):
            raise ValueError(
                f"expected a {self.out_size}x{self.out_size} canvas map, got "
                f"{canvas_map.shape[0]}x{canvas_map.shape[1]}"
            )
        x, y, w, h = self.content_box
        return canvas_map[y : y + h, x : x + w]

    def project_to_source(
        self, canvas_map: np.ndarray, interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        """Un-letterbox a canvas-space map onto the source's pixel grid."""
        content = self.unpad(canvas_map)
        return cv2.resize(content, (self.src_w, self.src_h), interpolation=interpolation)


def letterbox(
    image: np.ndarray,
    out_size: int,
    *,
    pad_value: int = 0,
    interpolation: int = cv2.INTER_AREA,
) -> tuple[np.ndarray, Letterbox]:
    """Resize ``image`` into a square canvas, preserving aspect ratio.

    ``INTER_AREA`` is the right default because letterboxing for the embedding
    net is almost always a downscale, and it is the only OpenCV filter that
    properly averages the discarded detail instead of point-sampling it.
    """
    height, width = image.shape[:2]
    box = Letterbox.compute(width, height, out_size)

    resized = cv2.resize(image, (box.content_w, box.content_h), interpolation=interpolation)

    canvas_shape: tuple[int, ...] = (out_size, out_size)
    if image.ndim == 3:
        canvas_shape = (out_size, out_size, image.shape[2])
    canvas = np.full(canvas_shape, pad_value, dtype=image.dtype)
    canvas[box.pad_y : box.pad_y + box.content_h, box.pad_x : box.pad_x + box.content_w] = resized
    return canvas, box


@dataclass(frozen=True)
class Squash:
    """Anisotropic resize into a square canvas: aspect ratio is not preserved.

    The legacy inference transform, kept because it is what the shipped checkpoint
    discriminates best under (see ``image.embed_resize`` in ``configs/default.yaml``
    for the measurement) and retained as a first-class object rather than an inline
    ``cv2.resize`` so that it carries its own inverse.

    That is the structural point of this class. A squash *is* invertible — a plain
    resize back to the source dimensions — and the legacy code did in fact invert it
    correctly. But it did so with a separate, hand-written ``cv2.resize`` call sited
    120 lines away from the forward one, which is how a forward transform and its
    inverse come to disagree. Pairing them here means switching
    :class:`Letterbox` in cannot leave a stale inverse behind: both types expose
    :meth:`project_to_source`, and the caller never writes the inversion itself.

    Deliberately mirrors :class:`Letterbox`'s interface, with ``scale`` degenerating
    into a separate factor per axis. Padding is always zero, so :meth:`unpad` is the
    identity and exists only so the two are substitutable.
    """

    out_size: int
    src_w: int
    src_h: int
    scale_x: float
    scale_y: float

    @classmethod
    def compute(cls, src_w: int, src_h: int, out_size: int) -> Squash:
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"degenerate source size {src_w}x{src_h}")
        return cls(
            out_size=out_size,
            src_w=src_w,
            src_h=src_h,
            scale_x=out_size / src_w,
            scale_y=out_size / src_h,
        )

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        """The whole canvas: a squash discards nothing and pads nothing."""
        return (0, 0, self.out_size, self.out_size)

    def to_canvas(self, points: np.ndarray) -> np.ndarray:
        """Map source ``(N, 2)`` coordinates into canvas space."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        scaled: np.ndarray = pts * np.array([self.scale_x, self.scale_y])
        return scaled

    def to_source(self, points: np.ndarray) -> np.ndarray:
        """Map canvas ``(N, 2)`` coordinates back to source space."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        unscaled: np.ndarray = pts / np.array([self.scale_x, self.scale_y])
        return unscaled

    def unpad(self, canvas_map: np.ndarray) -> np.ndarray:
        """Identity, modulo a shape check: there is no padding to remove."""
        if canvas_map.shape[:2] != (self.out_size, self.out_size):
            raise ValueError(
                f"expected a {self.out_size}x{self.out_size} canvas map, got "
                f"{canvas_map.shape[0]}x{canvas_map.shape[1]}"
            )
        return canvas_map

    def project_to_source(
        self, canvas_map: np.ndarray, interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        """Stretch a canvas-space map back onto the source's pixel grid."""
        content = self.unpad(canvas_map)
        return cv2.resize(content, (self.src_w, self.src_h), interpolation=interpolation)


def squash(
    image: np.ndarray, out_size: int, *, interpolation: int = cv2.INTER_AREA
) -> tuple[np.ndarray, Squash]:
    """Resize ``image`` to a square canvas without preserving aspect ratio."""
    height, width = image.shape[:2]
    box = Squash.compute(width, height, out_size)
    return cv2.resize(image, (out_size, out_size), interpolation=interpolation), box


#: Either fitting strategy. Both expose ``project_to_source``, ``to_canvas``,
#: ``to_source``, ``unpad`` and ``content_box``, so attribution code can invert
#: whichever was used without branching on the mode.
EmbedFit = Letterbox | Squash


def fit_for_embedding(
    image: np.ndarray,
    out_size: int,
    *,
    mode: Literal["squash", "letterbox"],
    pad_value: int = 0,
    interpolation: int = cv2.INTER_AREA,
) -> tuple[np.ndarray, EmbedFit]:
    """Fit ``image`` to ``out_size`` square by the configured strategy.

    The single place the ``image.embed_resize`` choice is acted on. Returning the
    fit object alongside the canvas is what keeps the attribution overlay honest:
    whatever transform was applied on the way in is the object that inverts it on
    the way out.
    """
    if mode == "letterbox":
        return letterbox(image, out_size, pad_value=pad_value, interpolation=interpolation)
    return squash(image, out_size, interpolation=interpolation)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedImage:
    """An input image plus everything needed to report coordinates honestly.

    ``bgr`` and ``gray`` are the *analysis* arrays: already capped to
    ``max_dimension``. ``analysis_scale`` converts analysis coordinates back to
    original-image coordinates, and :meth:`to_original` applies it. Downstream
    stages work exclusively in analysis space and convert once, at the boundary,
    which is why the reported boxes line up with the file the user handed in
    rather than with some intermediate the report never mentions.
    """

    meta: ImageMeta
    bgr: np.ndarray
    gray: np.ndarray
    analysis_scale: float

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the analysis arrays."""
        return (self.gray.shape[0], self.gray.shape[1])

    @property
    def diagonal(self) -> float:
        h, w = self.shape
        return float(np.hypot(w, h))

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """Map analysis-space ``(N, 2)`` coordinates to original-image pixels."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return pts / self.analysis_scale

    def box_to_original(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Map an analysis-space ``(x, y, w, h)`` box to original pixels."""
        inv = 1.0 / self.analysis_scale
        x, y, w, h = box
        return (
            round(x * inv),
            round(y * inv),
            max(1, round(w * inv)),
            max(1, round(h * inv)),
        )


def probe_dimensions(path: str | Path) -> tuple[int, int]:
    """Read ``(width, height)`` from headers without decoding pixel data.

    Lets the API reject a decompression bomb from its declared dimensions before
    allocating anything, which is the only point at which rejection is cheap.
    """
    from PIL import Image

    with Image.open(path) as handle:
        return int(handle.width), int(handle.height)


def load_image(
    path: str | Path,
    *,
    max_dimension: int = 2048,
    max_decoded_pixels: int = DEFAULT_MAX_DECODED_PIXELS,
    compute_hash: bool = True,
) -> LoadedImage:
    """Load an image for analysis, capped and hashed.

    Parameters
    ----------
    max_dimension
        Longest edge of the analysis arrays. Images at or below it are untouched;
        larger ones are downscaled with ``INTER_AREA`` and the original size is
        preserved in the returned metadata.
    max_decoded_pixels
        Reject before decoding if the declared dimensions exceed this.
    compute_hash
        SHA-256 the file for the audit trail. Disable in tight benchmark loops
        where the same file is loaded repeatedly.

    Raises
    ------
    ImageLoadError
        If the path is missing, the payload is not a decodable image, or the
        declared dimensions exceed ``max_decoded_pixels``.
    """
    src = Path(path)
    if not src.is_file():
        raise ImageLoadError(f"input image not found: {src}")

    file_format: str | None = None
    try:
        declared_w, declared_h = probe_dimensions(src)
        file_format = src.suffix.lstrip(".").lower() or None
        if declared_w * declared_h > max_decoded_pixels:
            raise ImageLoadError(
                f"{src.name} declares {declared_w}x{declared_h} = "
                f"{declared_w * declared_h:,} pixels, above the "
                f"{max_decoded_pixels:,} limit. Raise api.max_decoded_pixels if this "
                "image is genuinely expected."
            )
    except ImageLoadError:
        raise
    except Exception:
        # A format Pillow cannot probe may still be readable by OpenCV; fall
        # through and let imread decide rather than rejecting outright.
        _log.debug("could not probe dimensions of %s from headers", src, exc_info=True)

    # IMREAD_COLOR normalises the channel count, so a greyscale PNG, a
    # palettised GIF and an RGBA TIFF all arrive as 3-channel BGR and the rest
    # of the pipeline needs no per-format branching.
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageLoadError(
            f"could not decode {src} as an image (unsupported format or corrupt file)"
        )

    original_h, original_w = bgr.shape[:2]
    scale = 1.0
    longest = max(original_w, original_h)
    if longest > max_dimension:
        scale = max_dimension / longest
        new_size = (max(1, round(original_w * scale)), max(1, round(original_h * scale)))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
        _log.info(
            "downscaled %s from %dx%d to %dx%d for analysis (image.max_dimension=%d)",
            src.name,
            original_w,
            original_h,
            new_size[0],
            new_size[1],
            max_dimension,
        )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    analysed_h, analysed_w = bgr.shape[:2]

    meta = ImageMeta(
        path=src,
        sha256=file_sha256(src) if compute_hash else "",
        width=original_w,
        height=original_h,
        channels=3,
        file_format=file_format,
        size_bytes=src.stat().st_size,
        analysed_at=(analysed_w, analysed_h),
    )
    return LoadedImage(meta=meta, bgr=bgr, gray=gray, analysis_scale=scale)
