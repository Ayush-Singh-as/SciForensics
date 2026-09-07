"""Deterministic builders shared across test modules.

Kept separate from ``conftest.py`` on purpose: ``conftest`` is a pytest plugin
module and importing it directly is fragile, whereas this is an ordinary module
that both pytest and mypy resolve as ``tests.helpers``. Fixtures live in
``conftest``; the functions they wrap live here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

# A fixed seed everywhere: a flaky forensics test suite is worse than none, since
# the whole point of the tool is that the same input yields the same evidence.
SEED = 20260904


def textured_image(size: tuple[int, int] = (320, 400), *, seed: int = SEED) -> np.ndarray:
    """A deterministic greyscale image with enough corner structure for ORB.

    Random noise alone gives ORB nothing stable to latch onto across a resample,
    and a smooth gradient gives it nothing at all. Overlapping bright blobs plus a
    few hard-edged rectangles produce repeatable, well-localised corners -- which
    is what makes the matching assertions meaningful rather than lucky.
    """
    height, width = size
    generator = np.random.default_rng(seed)
    image = np.full((height, width), 40, dtype=np.uint8)

    for _ in range(60):
        cx, cy = generator.integers(0, width), generator.integers(0, height)
        radius = int(generator.integers(6, 22))
        value = int(generator.integers(90, 255))
        cv2.circle(image, (int(cx), int(cy)), radius, value, -1)

    for _ in range(18):
        x, y = generator.integers(0, width - 40), generator.integers(0, height - 40)
        w, h = generator.integers(12, 38), generator.integers(12, 38)
        value = int(generator.integers(120, 255))
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), value, -1)

    blurred: np.ndarray = cv2.GaussianBlur(image, (0, 0), 0.7)
    return blurred


def unrepeated_texture(size: tuple[int, int] = (320, 400), *, seed: int = SEED) -> np.ndarray:
    """Rich corner structure in which no primitive is ever drawn twice.

    The negative control for copy-move detection, and the reason it cannot be
    :func:`textured_image`. That builder draws 60 filled circles with radii from
    ``[6, 22)`` and 18 rectangles from a similarly narrow range, which on a
    320x400 frame means near-duplicate primitives by the pigeonhole principle --
    genuinely repeated content, which a copy-move detector is arguably *right* to
    flag. Measured across 12 seeds it reports a region on 7 of them; the default
    ``SEED`` is simply one of the clean ones, so a negative control built on it
    asserts seed luck rather than precision.

    Two distinct failure modes were measured on that fixture, and only one of
    them was a detector defect:

    * A filled disc is rotationally symmetric, so it maps onto *itself* under a
      180-degree rotation about its own centre. That is now rejected in
      :func:`~sciforensics.local_match.copymove._build_region` -- see the
      median-displacement gate there.
    * Two circles of equal radius and near-equal fill are real duplicate content.
      No gate should suppress that, because on a real figure it is exactly the
      finding the tool exists to report.

    Band-limited Gaussian noise has neither property: every neighbourhood is
    unique, while the blur leaves gradients smooth enough for ORB to localise
    corners repeatably. Measured 0/15 false positives across seeds, and 15/15
    exact recall once a patch is cloned into it -- so a test using it as a
    negative control is falsifiable rather than vacuous.
    """
    height, width = size
    generator = np.random.default_rng(seed)
    field = cv2.GaussianBlur(generator.normal(128.0, 70.0, (height, width)).astype(np.float32), (0, 0), 1.5)
    low, high = float(field.min()), float(field.max())
    # Rescale to a fixed dynamic range so the DoG and Otsu thresholds downstream
    # see the same contrast regardless of what the draw happened to produce.
    normalised: np.ndarray = (field - low) / (high - low) * 235.0 + 10.0
    return normalised.astype(np.uint8)


def localised_texture(
    centres: Sequence[tuple[int, int]] = ((80, 75), (290, 235)),
    size: tuple[int, int] = (320, 400),
    *,
    radius: int = 55,
    seed: int = SEED,
) -> np.ndarray:
    """Structure confined to a few clusters on an otherwise flat background.

    :func:`textured_image` is textured edge to edge, which is right for matching
    tests -- it gives ORB plenty to work with -- but it makes region *proposal*
    untestable, because a single connected component spans the whole frame. This
    is the layout region proposal actually exists for: a western blot is a couple
    of bands on empty film, and a correct ROI should bound them and exclude the
    rest. Callers assert against the ``centres`` they passed in.
    """
    height, width = size
    generator = np.random.default_rng(seed)
    image = np.full((height, width), 40, dtype=np.uint8)

    for cx, cy in centres:
        for _ in range(14):
            x = int(cx + generator.integers(-radius, radius))
            y = int(cy + generator.integers(-radius, radius))
            value = int(generator.integers(140, 255))
            cv2.circle(image, (x, y), int(generator.integers(5, 13)), value, -1)

    blurred: np.ndarray = cv2.GaussianBlur(image, (0, 0), 0.7)
    return blurred


def clone_patch(
    image: np.ndarray,
    source: tuple[int, int],
    target: tuple[int, int],
    *,
    size: int = 90,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Copy a square patch within one image, optionally rotated or resized.

    The copy-move fixture builder: ``source`` and ``target`` are the top-left
    corners of a ``size``-square region, both in ``(x, y)``. Returns a new array;
    the input is not modified.

    Rotation and scaling are applied about the patch centre with
    ``BORDER_REFLECT``, so the warp cannot introduce a black margin -- a hard
    black edge is a corner feature in its own right, and ORB would happily match
    the *border* between two clones and manufacture agreement the content does not
    support.

    Note the sign convention. ``cv2.getRotationMatrix2D`` takes a
    counter-clockwise angle in a y-down frame, which is a clockwise rotation of
    the content, whereas :func:`sciforensics.local_match.decompose_affine`
    reports the transform that maps source pixels to target pixels. A patch built
    with ``rotation_deg=30`` is therefore reported at roughly ``-30``, and callers
    assert against the negated value.
    """
    out = image.copy()
    sx, sy = source
    tx, ty = target
    patch = image[sy : sy + size, sx : sx + size]
    if rotation_deg or scale != 1.0:
        centre = (size / 2.0 - 0.5, size / 2.0 - 0.5)
        matrix = cv2.getRotationMatrix2D(centre, rotation_deg, scale)
        patch = cv2.warpAffine(
            patch,
            matrix,
            (size, size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
    out[ty : ty + size, tx : tx + size] = patch
    return out


def affine_matrix(
    *,
    rotation_deg: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float | None = None,
    shear_deg: float = 0.0,
    translation: tuple[float, float] = (0.0, 0.0),
    flip: bool = False,
) -> np.ndarray:
    """Build a ``2x3`` affine matrix from interpretable components.

    Mirrors the convention in :func:`sciforensics.local_match.compose_affine`
    (``A = R(theta) @ [[sx, m], [0, sy]]``, reflection carried in ``sign(sy)``) so
    a test can state a transform in human terms and assert the decomposition
    recovers exactly it.
    """
    sy = scale_x if scale_y is None else scale_y
    signed_sy = -sy if flip else sy
    theta = math.radians(rotation_deg)
    shear = math.tan(math.radians(shear_deg)) * abs(signed_sy)
    upper = np.array([[scale_x, shear], [0.0, signed_sy]], dtype=np.float64)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    linear = rotation @ upper
    return np.hstack([linear, np.array([[translation[0]], [translation[1]]])])


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a ``2x3`` affine to an ``(N, 2)`` point set."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    moved: np.ndarray = (pts @ matrix[:, :2].T) + matrix[:, 2]
    return moved


def warp_image(
    image: np.ndarray, matrix: np.ndarray, size: tuple[int, int] | None = None
) -> np.ndarray:
    """Warp ``image`` by a ``2x3`` affine, keeping its size unless told otherwise."""
    height, width = image.shape[:2] if size is None else size
    warped: np.ndarray = cv2.warpAffine(
        image, matrix.astype(np.float32), (width, height), flags=cv2.INTER_CUBIC
    )
    return warped
