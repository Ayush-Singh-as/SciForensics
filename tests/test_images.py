"""Image loading: letterboxing, the resolution cap, and coordinate honesty.

Bug 8 is the regression here. The legacy code resized every input to a flat
``(128, 128)`` regardless of aspect ratio, then stretched the resulting 8x8
attribution map back over the *original* aspect -- so for any non-square image,
and most are, the overlay pointed at the wrong pixels. Nothing about the output
looked broken; the heatmap was smooth, plausible, and displaced.

Bug 14 also lives here: nothing bounded input resolution before the 4x
enhancement upscale, so a 3840x2400 photograph became a 147-megapixel working
image and a single pair took over two minutes.

The tests below are mostly *round-trip* tests, because the property that matters
is not "letterboxing produces a square" but "a coordinate survives the journey
into network space and back". That is the invariant whose violation is invisible.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from sciforensics.io.images import (
    DEFAULT_MAX_DECODED_PIXELS,
    ImageLoadError,
    Letterbox,
    letterbox,
    load_image,
    probe_dimensions,
)
from tests.helpers import textured_image

# Deliberately awkward shapes. A square input cannot expose an aspect bug -- it is
# the one case the legacy code got right -- and 2:1 / 1:3 are the ratios a
# multi-panel figure and a gel lane actually have.
SHAPES = [(320, 400), (100, 300), (300, 100), (64, 64), (37, 211), (1, 500)]


def _save(tmp_path: Path, image: np.ndarray, name: str = "input.png") -> Path:
    path = tmp_path / name
    assert cv2.imwrite(str(path), image)
    return path


# ---------------------------------------------------------------------------
# bug 8: aspect ratio survives the trip into network space and back
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("height", "width"), SHAPES)
def test_letterbox_preserves_aspect_ratio(height: int, width: int) -> None:
    """Content keeps its shape; the canvas is squared by padding, not by stretching."""
    canvas, box = letterbox(np.full((height, width), 200, np.uint8), 224)

    assert canvas.shape == (224, 224)
    source_ratio = width / height
    content_ratio = box.content_w / box.content_h
    # Rounding to whole pixels is the only permitted deviation, and it is bounded
    # by one pixel on the smaller edge.
    tolerance = source_ratio * (1.0 / min(box.content_w, box.content_h) + 1e-9)
    assert content_ratio == pytest.approx(source_ratio, abs=tolerance)


@pytest.mark.parametrize(("height", "width"), SHAPES)
def test_coordinates_round_trip_through_canvas_space(height: int, width: int) -> None:
    """The invariant the attribution overlay rests on.

    A point identified in canvas space must map back to the source pixel it came
    from. Without this, a heatmap can be computed correctly and still be drawn in
    the wrong place -- which is exactly what shipped.
    """
    box = Letterbox.compute(width, height, 224)
    rng = np.random.default_rng(4)
    points = np.column_stack([rng.uniform(0, width, size=64), rng.uniform(0, height, size=64)])

    recovered = box.to_source(box.to_canvas(points))

    assert np.allclose(recovered, points, atol=1e-9)


@pytest.mark.parametrize(("height", "width"), SHAPES)
def test_content_occupies_the_canvas_without_overflowing_it(height: int, width: int) -> None:
    """One edge must touch the canvas exactly; neither may exceed it."""
    box = Letterbox.compute(width, height, 224)

    assert box.content_w <= 224 and box.content_h <= 224
    assert max(box.content_w, box.content_h) == 224 or min(width, height) == 1
    assert box.pad_x >= 0 and box.pad_y >= 0
    assert box.pad_x + box.content_w <= 224
    assert box.pad_y + box.content_h <= 224


def test_the_corners_of_the_source_land_on_the_corners_of_the_content() -> None:
    """States the mapping concretely, in the terms a reader can check by hand."""
    box = Letterbox.compute(400, 200, 224)  # 2:1 -> 224x112, padded 56 top and bottom

    assert box.content_box == (0, 56, 224, 112)
    corners = box.to_canvas(np.array([[0.0, 0.0], [400.0, 200.0]]))
    assert corners[0] == pytest.approx([0.0, 56.0])
    assert corners[1] == pytest.approx([224.0, 168.0])


def test_padding_is_centred_so_an_overlay_is_not_offset() -> None:
    box = Letterbox.compute(400, 200, 224)
    assert box.pad_y == (224 - box.content_h) // 2
    assert 224 - box.content_h - box.pad_y == pytest.approx(box.pad_y, abs=1)


@pytest.mark.parametrize(("height", "width"), SHAPES)
def test_projecting_a_canvas_map_back_yields_the_source_grid(height: int, width: int) -> None:
    """An attribution map must come back at the source's exact resolution.

    The legacy overlay resized an 8x8 map straight onto the original frame, which
    both smeared the padding into the borders and applied the wrong aspect. Here
    the padding is cropped first, so what gets stretched is only real content.
    """
    box = Letterbox.compute(width, height, 224)
    canvas_map = np.zeros((224, 224), np.float32)
    x, y, w, h = box.content_box
    canvas_map[y : y + h, x : x + w] = 1.0

    projected = box.project_to_source(canvas_map)

    assert projected.shape == (height, width)
    # The content region was uniformly 1.0, so nothing but padding could have
    # reduced it -- an off-by-one in `unpad` shows up as a dark border here.
    assert projected.min() > 0.99


def test_unpad_rejects_a_map_that_is_not_canvas_sized() -> None:
    """Silently accepting a mis-sized map would misplace every overlay."""
    box = Letterbox.compute(400, 320, 224)
    with pytest.raises(ValueError, match="224x224"):
        box.unpad(np.zeros((112, 112), np.float32))


def test_letterbox_padding_value_is_respected() -> None:
    canvas, box = letterbox(np.full((100, 300), 200, np.uint8), 224, pad_value=17)
    assert canvas[0, 0] == 17
    assert canvas[box.pad_y + box.content_h // 2, box.pad_x + box.content_w // 2] == 200


def test_letterbox_keeps_colour_channels() -> None:
    canvas, _ = letterbox(np.full((100, 300, 3), 200, np.uint8), 224)
    assert canvas.shape == (224, 224, 3)


def test_letterbox_refuses_a_degenerate_size() -> None:
    with pytest.raises(ImageLoadError, match="degenerate"):
        Letterbox.compute(0, 100, 224)


def test_an_extreme_aspect_ratio_keeps_at_least_one_pixel() -> None:
    """A 4000x3 strip must not round to an empty canvas.

    Clamping to one pixel is a visible, checkable degradation; a zero-height
    content box is an empty array that looks like a black image.
    """
    box = Letterbox.compute(4000, 3, 224)
    assert box.content_h >= 1
    assert box.content_w == 224


# ---------------------------------------------------------------------------
# bug 14: the resolution cap, and reporting in the user's coordinate frame
# ---------------------------------------------------------------------------
def test_a_large_image_is_capped_and_says_so(tmp_path: Path) -> None:
    """Both the cap and the record of it. The record is what makes it auditable."""
    path = _save(tmp_path, textured_image(size=(1200, 1600)))
    loaded = load_image(path, max_dimension=512)

    assert max(loaded.shape) == 512
    assert loaded.meta.width == 1600, "the original size must survive in the metadata"
    assert loaded.meta.height == 1200
    assert loaded.meta.analysed_at == (512, 384)
    assert loaded.analysis_scale == pytest.approx(512 / 1600)


def test_a_small_image_is_left_alone(tmp_path: Path) -> None:
    """No resampling below the cap: an untouched image needs no scale correction."""
    original = textured_image(size=(320, 400))
    loaded = load_image(_save(tmp_path, original), max_dimension=2048)

    assert loaded.analysis_scale == 1.0
    assert loaded.shape == (320, 400)
    assert np.array_equal(loaded.gray, original)


def test_capping_preserves_aspect_ratio(tmp_path: Path) -> None:
    path = _save(tmp_path, textured_image(size=(600, 1500)))
    loaded = load_image(path, max_dimension=500)

    height, width = loaded.shape
    assert width / height == pytest.approx(1500 / 600, rel=0.01)


def test_analysis_coordinates_map_back_to_the_users_pixels(tmp_path: Path) -> None:
    """A box in the report must refer to the file the user handed in.

    Reporting analysis-space coordinates as if they were original-space was one of
    the defects I introduced *during* this refactor, so it is worth pinning: at a
    0.32 scale factor every coordinate would be understated by a factor of three
    and still look entirely reasonable.
    """
    path = _save(tmp_path, textured_image(size=(1200, 1600)))
    loaded = load_image(path, max_dimension=512)
    height, width = loaded.shape

    corners = np.array([[0.0, 0.0], [width, height]])
    mapped = loaded.to_original(corners)

    assert mapped[0] == pytest.approx([0.0, 0.0])
    assert mapped[1] == pytest.approx([1600.0, 1200.0], rel=0.01)
    # scale is exactly 512/1600 = 0.32, so the inverse is 3.125 and these four
    # products are whole or clearly-rounding numbers -- chosen to avoid landing on
    # a .5 tie, since which way a half-pixel breaks is not what this test is about.
    assert loaded.box_to_original((10, 24, 32, 40)) == (31, 75, 100, 125)


def test_box_conversion_never_collapses_a_region_to_nothing(tmp_path: Path) -> None:
    """A thin box must stay visible after conversion, not round away to zero."""
    path = _save(tmp_path, textured_image(size=(1200, 1600)))
    loaded = load_image(path, max_dimension=512)

    _, _, w, h = loaded.box_to_original((5, 5, 0, 0))
    assert w >= 1 and h >= 1


def test_diagonal_is_measured_in_analysis_space(tmp_path: Path) -> None:
    """Geometry gates compare spreads against this, so its frame must be fixed."""
    loaded = load_image(_save(tmp_path, textured_image(size=(320, 400))), max_dimension=2048)
    assert loaded.diagonal == pytest.approx(math.hypot(400, 320))


# ---------------------------------------------------------------------------
# refusing bad input
# ---------------------------------------------------------------------------
def test_a_decompression_bomb_is_rejected_from_its_header(tmp_path: Path) -> None:
    """Rejected on declared dimensions, before any allocation.

    This is the only point at which refusal is cheap: a 20000x20000 PNG is a few
    hundred kilobytes on disk and 1.2 GB decoded, so a limit enforced after
    decoding is not a limit at all once the HTTP API is exposed.
    """
    path = _save(tmp_path, np.full((1000, 1000), 128, np.uint8), "big.png")
    declared = probe_dimensions(path)
    assert declared == (1000, 1000)

    with pytest.raises(ImageLoadError, match="above the"):
        load_image(path, max_decoded_pixels=500_000)


def test_the_default_pixel_limit_admits_ordinary_scientific_figures() -> None:
    """A limit that rejects real inputs would just be turned off.

    50 megapixels is comfortably above a 600-dpi full-page journal figure, which
    is the largest thing this tool should expect to be handed.
    """
    assert DEFAULT_MAX_DECODED_PIXELS >= 8000 * 6000


def test_a_missing_file_is_named_in_the_error(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError, match="not found"):
        load_image(tmp_path / "absent.png")


def test_a_file_that_is_not_an_image_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.png"
    path.write_text("this is not a PNG\n", encoding="utf-8")
    with pytest.raises(ImageLoadError, match="could not decode"):
        load_image(path)


def test_an_unprobeable_but_decodable_file_still_loads(tmp_path: Path) -> None:
    """Pillow failing to read a header is not grounds for refusing the image.

    OpenCV reads formats Pillow cannot probe. Falling through to ``imread`` keeps
    the loader permissive about *formats* while staying strict about *sizes*.
    """
    path = _save(tmp_path, textured_image(size=(64, 64)), "odd.pnm")
    assert load_image(path).shape == (64, 64)


# ---------------------------------------------------------------------------
# normalisation and the audit trail
# ---------------------------------------------------------------------------
def test_every_input_arrives_as_three_channel_bgr_plus_grey(tmp_path: Path) -> None:
    """Uniform channel count is what keeps per-format branching out of the pipeline."""
    grey_path = _save(tmp_path, textured_image(size=(64, 64)), "grey.png")
    colour = cv2.cvtColor(textured_image(size=(64, 64)), cv2.COLOR_GRAY2BGR)
    colour_path = _save(tmp_path, colour, "colour.png")

    for path in (grey_path, colour_path):
        loaded = load_image(path)
        assert loaded.bgr.ndim == 3 and loaded.bgr.shape[2] == 3
        assert loaded.gray.ndim == 2
        assert loaded.meta.channels == 3


def test_the_hash_identifies_the_file_and_nothing_else(tmp_path: Path) -> None:
    """The audit block's job: same bytes, same hash; one pixel different, different hash.

    A report that cannot be tied to the exact bytes it describes is not evidence.
    """
    image = textured_image(size=(64, 64))
    first = _save(tmp_path, image, "a.png")
    same = _save(tmp_path, image, "b.png")
    altered = image.copy()
    altered[0, 0] = 255 - altered[0, 0]
    different = _save(tmp_path, altered, "c.png")

    digest = load_image(first).meta.sha256
    assert len(digest) == 64
    assert digest == load_image(same).meta.sha256
    assert digest != load_image(different).meta.sha256


def test_hashing_can_be_skipped_for_benchmark_loops(tmp_path: Path) -> None:
    path = _save(tmp_path, textured_image(size=(64, 64)))
    assert load_image(path, compute_hash=False).meta.sha256 == ""


def test_metadata_records_the_format_and_size_on_disk(tmp_path: Path) -> None:
    path = _save(tmp_path, textured_image(size=(64, 64)), "panel.png")
    meta = load_image(path).meta

    assert meta.file_format == "png"
    assert meta.size_bytes == path.stat().st_size
    assert meta.path == path
