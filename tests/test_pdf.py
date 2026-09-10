"""PDF ingestion and panel splitting. Stage C1.

Panel splitting is where this module's real bugs were, and both were found by
building a fixture with a *known* layout and checking the panel count against
it. Neither would have been visible from reading the code:

1. A projection-only gutter test cannot separate "gap between panels" from
   "dark region inside one panel". A mountain photo's shadowed foreground read
   as a gutter and split a two-tile figure into three, **slicing a planted
   duplicate into a 78px fragment** that was then compared against a 239px
   panel. Fixed by requiring a gutter to match the page background colour.
2. Sparse figures defeat projection outright. A fluorescence panel at 4.4%
   occupancy has large genuinely-empty regions inside one panel, and no
   projection statistic distinguishes them from separators. Fixed by refusing
   to split below an occupancy floor and reporting ``confident=False``.

So these tests assert panel *counts and boxes* against fixtures whose layout is
constructed, not merely that the code runs.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fitz", reason="PDF ingestion needs PyMuPDF ([bench] extra)")

from sciforensics.io.pdf import (
    PdfError,
    Route,
    extract_figures,
    foreground_mask,
    ingest,
    page_background,
    split_panels,
    write,
)

GUTTER = 26
TILE_W, TILE_H = 200, 160


def _tile(seed: int) -> np.ndarray:
    """A dense, textured tile. Dense matters: sparse tiles hit the occupancy floor."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 220, (TILE_H, TILE_W, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(base, (0, 0), 1.1)
    cv2.circle(blurred, (TILE_W // 2, TILE_H // 2), 40, (230, 230, 230), -1)
    return blurred


def _figure(columns: int, rows: int = 1, seed: int = 0) -> np.ndarray:
    """A grid of tiles on a white page with uniform gutters."""
    height = rows * TILE_H + (rows + 1) * GUTTER
    width = columns * TILE_W + (columns + 1) * GUTTER
    canvas = np.full((height, width, 3), 255, np.uint8)
    for row in range(rows):
        for column in range(columns):
            y = GUTTER + row * (TILE_H + GUTTER)
            x = GUTTER + column * (TILE_W + GUTTER)
            canvas[y : y + TILE_H, x : x + TILE_W] = _tile(seed + row * columns + column)
    return canvas


def _pdf(tmp_path: Path, figures: list[np.ndarray]) -> Path:
    import fitz

    document = fitz.open()
    for index, image in enumerate(figures):
        png = tmp_path / f"_fig{index}.png"
        cv2.imwrite(str(png), image)
        page = document.new_page(width=595, height=842)
        height, width = image.shape[:2]
        scale = 460 / width
        page.insert_image(
            fitz.Rect(60, 100, 60 + width * scale, 100 + height * scale), filename=str(png)
        )
    path = tmp_path / "fixture.pdf"
    document.save(path)
    document.close()
    return path


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def test_embedded_route_is_preferred_and_recorded(tmp_path: Path) -> None:
    """Provenance is recorded, never inferred later.

    A reader must be able to tell a submitted pixel from a re-rendered one; a
    finding on resampled pixels is still a finding but carries less weight.
    """
    manuscript = extract_figures(_pdf(tmp_path, [_figure(2)]))
    assert len(manuscript.figures) == 1
    figure = manuscript.figures[0]
    assert figure.route is Route.EMBEDDED
    assert figure.is_evidential
    assert manuscript.evidential_fraction == 1.0


def test_render_route_is_marked_non_evidential(tmp_path: Path) -> None:
    manuscript = extract_figures(_pdf(tmp_path, [_figure(2)]), prefer=Route.RENDER, dpi=72)
    assert all(f.route is Route.RENDER for f in manuscript.figures)
    assert not any(f.is_evidential for f in manuscript.figures)
    assert manuscript.evidential_fraction == 0.0


def test_repeated_placement_is_not_a_second_figure(tmp_path: Path) -> None:
    """The same image placed twice on a page is one figure, not two.

    Emitting both would hand the scanner a guaranteed self-match: a pair of
    byte-identical panels reported as reuse, which is a layout fact rather than
    a finding.
    """
    import fitz

    image = _figure(1)
    png = tmp_path / "one.png"
    cv2.imwrite(str(png), image)

    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(60, 80, 260, 240), filename=str(png))
    page.insert_image(fitz.Rect(60, 300, 260, 460), filename=str(png))
    path = tmp_path / "twice.pdf"
    document.save(path)
    document.close()

    assert len(extract_figures(path).figures) == 1


def test_tiny_images_are_discarded(tmp_path: Path) -> None:
    """Logos and rules are embedded images too, and match on branding alone.

    The floor applies to *embedded* images only. When it rejects everything on a
    page, the render fallback must still fire -- applying the same floor there
    made an aggressive `min_side` raise "no figures found" rather than falling
    back, defeating the point of having a fallback.
    """
    manuscript = extract_figures(_pdf(tmp_path, [_figure(2)]), min_side=10_000)
    assert manuscript.figures, "the render fallback must still produce a figure"
    assert all(f.route is Route.RENDER for f in manuscript.figures)


def test_unopenable_file_is_an_error(tmp_path: Path) -> None:
    broken = tmp_path / "not.pdf"
    broken.write_bytes(b"this is not a pdf")
    with pytest.raises(PdfError):
        extract_figures(broken)


# ---------------------------------------------------------------------------
# panel splitting -- where the bugs were
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("columns", "rows"), [(2, 1), (3, 1), (3, 2), (2, 2)])
def test_grid_splits_into_exactly_its_tiles(tmp_path: Path, columns: int, rows: int) -> None:
    """The count is constructed, so an off-by-one split fails loudly.

    This is the assertion that caught the spurious-gutter bug: a 2-tile figure
    was producing 3 panels.
    """
    manuscript = ingest(_pdf(tmp_path, [_figure(columns, rows)]))
    panels = manuscript.panels
    assert len(panels) == columns * rows, [p.box for p in panels]
    assert all(p.confident for p in panels)


def test_split_panels_recover_plausible_boxes(tmp_path: Path) -> None:
    """Boxes must tile the figure without overlapping."""
    manuscript = extract_figures(_pdf(tmp_path, [_figure(3)]))
    panels = split_panels(manuscript.figures[0])
    assert len(panels) == 3

    boxes = sorted((p.box for p in panels), key=lambda b: b[0])
    for (x, _, w, _), (next_x, *_) in pairwise(boxes):
        assert x + w <= next_x + 1, f"panel boxes overlap: {boxes}"


def test_dark_content_is_not_mistaken_for_a_gutter(tmp_path: Path) -> None:
    """Bug 1 of this module, directly.

    A tile with a large dark band produced a spurious gutter under a
    projection-only test. A gutter must match the *page background*; dark image
    content does not, however dark it is.
    """
    figure = _figure(2)
    # Paint a wide black band down the middle of the left tile.
    figure[GUTTER : GUTTER + TILE_H, GUTTER + 60 : GUTTER + 140] = 0

    import fitz

    png = tmp_path / "dark.png"
    cv2.imwrite(str(png), figure)
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    height, width = figure.shape[:2]
    page.insert_image(fitz.Rect(60, 100, 60 + width * 0.7, 100 + height * 0.7), filename=str(png))
    path = tmp_path / "dark.pdf"
    document.save(path)
    document.close()

    panels = ingest(path).panels
    assert len(panels) == 2, f"dark band split a tile: {[p.box for p in panels]}"


def test_sparse_figure_refuses_to_split(tmp_path: Path) -> None:
    """Bug 2 of this module.

    Below the occupancy floor, projection cannot tell a panel gap from a gap
    between cells. Returning one flagged panel beats inventing a boundary: a
    wrongly split panel gets compared against its own other half downstream,
    which manufactures a guaranteed false match.
    """
    sparse = np.full((240, 320, 3), 255, np.uint8)
    for centre in ((40, 40), (150, 60), (90, 180), (260, 200)):
        cv2.circle(sparse, centre, 14, (30, 30, 30), -1)

    manuscript = extract_figures(_pdf(tmp_path, [sparse]))
    panels = split_panels(manuscript.figures[0])

    assert len(panels) == 1
    assert not panels[0].confident, "a refused split must say so"
    assert panels[0].box == (0, 0, 320, 240)


def test_single_panel_figure_is_returned_confidently(tmp_path: Path) -> None:
    """One dense tile is one panel, and that is a *confident* answer."""
    manuscript = extract_figures(_pdf(tmp_path, [_figure(1)]))
    panels = split_panels(manuscript.figures[0])
    assert len(panels) == 1
    assert panels[0].confident


def test_page_background_reads_the_border() -> None:
    canvas = np.full((80, 80), 255, np.uint8)
    canvas[20:60, 20:60] = 0
    assert page_background(canvas) == pytest.approx(255.0)


def test_foreground_mask_handles_both_polarities() -> None:
    """Fluorescence is light-on-dark; blots are dark-on-light. Both must work."""
    dark_on_light = np.full((60, 60), 240, np.uint8)
    dark_on_light[20:40, 20:40] = 20
    light_on_dark = np.full((60, 60), 20, np.uint8)
    light_on_dark[20:40, 20:40] = 240

    for image in (dark_on_light, light_on_dark):
        mask = foreground_mask(image)
        # The 20x20 square is the content: ~11% of a 60x60 canvas.
        assert 0.05 < mask.mean() < 0.25, mask.mean()
        assert mask[30, 30] == 1, "the square must be foreground in both polarities"


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def test_write_encodes_the_route_in_the_filename(tmp_path: Path) -> None:
    """Provenance must survive a file being picked up off disk."""
    manuscript = ingest(_pdf(tmp_path, [_figure(2)]))
    written = write(manuscript, tmp_path / "out")

    figures = [p for p in written if p.parent.name == "out"]
    assert figures and all("_embedded" in p.name for p in figures)
    assert (tmp_path / "out" / "panels").is_dir()


def test_doubtful_panels_are_marked_on_disk(tmp_path: Path) -> None:
    """A downstream consumer reading filenames must see the flag too."""
    sparse = np.full((240, 320, 3), 255, np.uint8)
    cv2.circle(sparse, (80, 80), 15, (20, 20, 20), -1)
    cv2.circle(sparse, (240, 170), 15, (20, 20, 20), -1)

    manuscript = ingest(_pdf(tmp_path, [sparse]))
    write(manuscript, tmp_path / "out")
    names = [p.name for p in (tmp_path / "out" / "panels").glob("*.png")]
    assert any("_unsplit" in name for name in names), names
