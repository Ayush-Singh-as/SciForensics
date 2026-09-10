"""PDF → figures → panels. Stage C1.

**Why this changes what the tool is.** Everything before this compares two
images a user has already isolated. Research-integrity screening does not start
there: it starts with a manuscript. Proofig and ImageTwin ingest a PDF, pull out
the figures, split each into panels, and compare every panel against every other
one. This module is the front of that pipeline, and it is what turns "compare
two PNGs" into "audit this submission".

**Two extraction routes, because neither alone is sufficient.**

``embedded``
    Pull the actual embedded image XObjects. Lossless — these are the bytes the
    author submitted, at full resolution, which is what a forensic comparison
    should see. Fails when a figure is drawn as vector paths, or assembled from
    dozens of tiny tiles.
``render``
    Rasterise the page region at a chosen DPI. Always works, but it is a
    *rendering*: resampled, possibly recompressed, and carrying whatever the
    viewer did to it. Fine for triage, weaker as evidence.

``embedded`` is preferred and the route actually used is recorded on every
:class:`Figure`, because a reader must be able to tell a submitted pixel from a
re-rendered one.

**Panel splitting is gutter projection, not a learned segmenter.** Multi-panel
figures are separated by near-uniform background bands. Summing the foreground
mask along each axis gives a 1-D profile whose low regions are gutters; cutting
there recovers the panels. It handles the regular grids that dominate biomedical
figures and does not pretend to handle overlapping or diagonal layouts —
:attr:`Panel.confident` records which case a panel came from, so a caller can
decline to draw conclusions from a doubtful split rather than being told
nothing.

BioFors is panel-organised, so it doubles as the evaluation set here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sciforensics.runtime import get_logger
from sciforensics.types import BBox

_log = get_logger(__name__)


class PdfError(ValueError):
    """Raised when a PDF cannot be opened or contains nothing usable."""


class Route(str, Enum):
    """How a figure's pixels were obtained. Recorded, never inferred later."""

    #: The embedded image stream, byte-for-byte as submitted.
    EMBEDDED = "embedded"
    #: A rasterisation of the page region at a chosen DPI.
    RENDER = "render"


@dataclass(frozen=True)
class Figure:
    """One extracted figure and its provenance."""

    image: np.ndarray
    page: int
    #: 0-based index of the figure within its page.
    index: int
    route: Route
    #: Where on the page it came from, in PDF points. ``None`` for embedded
    #: images whose placement could not be resolved.
    rect: tuple[float, float, float, float] | None = None
    dpi: int | None = None

    @property
    def name(self) -> str:
        return f"p{self.page:03d}_f{self.index:02d}"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.image.shape[0], self.image.shape[1])

    @property
    def is_evidential(self) -> bool:
        """True when these are the submitted pixels rather than a re-rendering.

        A finding on a rendered figure is still a finding, but the reader should
        know the comparison ran on resampled pixels.
        """
        return self.route is Route.EMBEDDED


@dataclass(frozen=True)
class Panel:
    """One sub-panel of a figure."""

    image: np.ndarray
    #: Box within the parent figure, in figure pixels.
    box: BBox
    figure: str
    index: int
    #: False when the split fell back to "the whole figure is one panel", or
    #: when the gutter structure was ambiguous. A caller comparing panels
    #: across a manuscript should surface this rather than silently treating a
    #: doubtful split as ground truth.
    confident: bool = True

    @property
    def name(self) -> str:
        return f"{self.figure}_panel{self.index:02d}"


@dataclass
class Manuscript:
    """Everything extracted from one PDF."""

    path: Path
    pages: int
    figures: list[Figure] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def evidential_fraction(self) -> float:
        """Share of figures obtained losslessly. 1.0 is the good case."""
        if not self.figures:
            return 0.0
        return sum(1 for f in self.figures if f.is_evidential) / len(self.figures)


def _to_bgr(pixmap: Any) -> np.ndarray:
    """PyMuPDF pixmap → BGR ndarray.

    Typed ``Any`` rather than ``object``: PyMuPDF ships no stubs, and `object`
    makes every attribute access an error while asserting nothing. `fitz.*` is
    already in mypy's ignore-missing-imports list.
    """
    import fitz

    # Drop alpha and any exotic colourspace by normalising to RGB first;
    # comparing against an unconverted CMYK buffer would silently misread
    # every channel.
    if pixmap.alpha or pixmap.colorspace is None or pixmap.n not in (1, 3):
        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 1:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def extract_figures(
    path: str | Path,
    *,
    prefer: Route = Route.EMBEDDED,
    dpi: int = 200,
    min_side: int = 96,
    max_pixels: int = 50_000_000,
) -> Manuscript:
    """Pull every figure out of a PDF.

    Parameters
    ----------
    prefer
        ``EMBEDDED`` tries the lossless route first and falls back to rendering
        per page when a page yields nothing. ``RENDER`` rasterises throughout.
    min_side
        Discard images whose shorter side is below this. Publisher logos, rules
        and icons are embedded images too, and comparing them across a corpus
        produces nothing but matches on the journal's own branding.
    max_pixels
        Skip an image that decodes larger than this. The same
        decompression-bomb guard the API applies to uploads — a PDF is untrusted
        input in exactly the same way.

    Raises
    ------
    PdfError
        If the file cannot be opened.
    """
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - optional extra
        raise PdfError(
            f"PyMuPDF is not installed ({exc}); install with: pip install 'sciforensics[bench]'"
        ) from exc

    path = Path(path)
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise PdfError(f"could not open {path}: {exc}") from exc

    manuscript = Manuscript(path=path, pages=document.page_count)

    with document:
        for page_number in range(document.page_count):
            page = document[page_number]
            found = 0

            if prefer is Route.EMBEDDED:
                found = _extract_embedded(
                    document,
                    page,
                    page_number,
                    manuscript,
                    min_side=min_side,
                    max_pixels=max_pixels,
                )

            if found == 0:
                # Either the page draws its figures as vectors, or the caller
                # asked to rasterise. Recorded as RENDER so the provenance is
                # never ambiguous.
                _render_page(page, page_number, manuscript, dpi=dpi)

    if not manuscript.figures:
        raise PdfError(f"no figures found in {path} ({manuscript.pages} page(s))")

    _log.info(
        "extracted %d figures from %d pages (%.0f%% lossless)",
        len(manuscript.figures),
        manuscript.pages,
        manuscript.evidential_fraction * 100,
    )
    return manuscript


def _extract_embedded(
    document: Any,
    page: Any,
    page_number: int,
    manuscript: Manuscript,
    *,
    min_side: int,
    max_pixels: int,
) -> int:
    """Pull embedded image XObjects from one page. Returns the count kept."""
    import fitz

    kept = 0
    seen: set[int] = set()

    for info in page.get_images(full=True):
        xref = int(info[0])
        # The same image placed twice on a page appears twice here. Deduplicate
        # by xref: a repeated *placement* is a layout fact, not a second figure,
        # and emitting both would produce a guaranteed self-match downstream.
        if xref in seen:
            continue
        seen.add(xref)

        width, height = int(info[2]), int(info[3])
        if min(width, height) < min_side:
            continue
        if width * height > max_pixels:
            manuscript.warnings.append(
                f"page {page_number}: skipped a {width}x{height} image over the "
                f"{max_pixels}-pixel guard"
            )
            continue

        try:
            pixmap = fitz.Pixmap(document, xref)
            image = _to_bgr(pixmap)
        except Exception as exc:
            manuscript.warnings.append(f"page {page_number}: could not decode xref {xref}: {exc}")
            continue

        rect: tuple[float, float, float, float] | None = None
        boxes = page.get_image_rects(xref)
        if boxes:
            box = boxes[0]
            rect = (float(box.x0), float(box.y0), float(box.x1), float(box.y1))

        manuscript.figures.append(
            Figure(
                image=image,
                page=page_number,
                index=kept,
                route=Route.EMBEDDED,
                rect=rect,
            )
        )
        kept += 1

    return kept


def _render_page(
    page: Any,
    page_number: int,
    manuscript: Manuscript,
    *,
    dpi: int,
) -> None:
    """Rasterise a whole page as one figure."""
    try:
        pixmap = page.get_pixmap(dpi=dpi)
        image = _to_bgr(pixmap)
    except Exception as exc:
        manuscript.warnings.append(f"page {page_number}: render failed: {exc}")
        return

    # `min_side` deliberately does *not* apply here. It exists to reject logos,
    # rules and icons from among the *embedded* images, where a small XObject
    # really is chrome. A rendered page is never a logo, and applying the same
    # floor made an aggressive `min_side` reject the fallback too -- so a PDF
    # whose only images were below the floor raised "no figures found" instead
    # of falling back to rendering, which is exactly what the fallback is for.
    # Only a degenerate page (zero-area) is skipped.
    if min(image.shape[:2]) < 2:
        return

    rect = page.rect
    manuscript.figures.append(
        Figure(
            image=image,
            page=page_number,
            index=len([f for f in manuscript.figures if f.page == page_number]),
            route=Route.RENDER,
            rect=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
            dpi=dpi,
        )
    )


# ---------------------------------------------------------------------------
# panel splitting
# ---------------------------------------------------------------------------
def foreground_mask(image: np.ndarray) -> np.ndarray:
    """Non-background pixels, as ``uint8`` in ``{0, 1}``.

    Otsu against the *border* population rather than a fixed threshold:
    biomedical figures are as often light-on-dark (fluorescence) as
    dark-on-light (blots, charts), and a fixed polarity would treat one of those
    as entirely background.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sample the border to decide polarity: whichever side of the Otsu split
    # the margins fall on is the background.
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if border.mean() > 0.5:
        binary = 1 - binary

    mask: np.ndarray = binary.astype(np.uint8)
    return mask


def _gutters(profile: np.ndarray, *, min_run: int, tolerance: float) -> list[tuple[int, int]]:
    """Runs of near-empty rows/columns, as ``(start, end)`` half-open pairs.

    ``tolerance`` is a fraction of the profile's own maximum rather than an
    absolute count, so the same value works for a 200px thumbnail and a 4000px
    figure.

    **Measured against the median, not the maximum.** Scaling by the max makes
    the test depend on the single densest row in the figure, so one dark band
    inside a sparse panel drags the bar down and every genuinely-dark region
    reads as a separator. A 320x240 fluorescence panel with 4.4% foreground
    produced four spurious column gutters that way and split one image into
    two panels. The median is the typical occupancy of a *content* row, which is
    the quantity a gutter is actually supposed to fall below.
    """
    if profile.size == 0:
        return []
    occupied = profile[profile > 0]
    if occupied.size == 0:
        return []
    reference = float(np.median(occupied))
    if reference <= 0:
        return []

    empty = profile <= reference * tolerance
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_empty in enumerate(empty):
        if is_empty and start is None:
            start = index
        elif not is_empty and start is not None:
            if index - start >= min_run:
                runs.append((start, index))
            start = None
    if start is not None and len(empty) - start >= min_run:
        runs.append((start, len(empty)))
    return runs


def page_background(gray: np.ndarray) -> float:
    """Modal intensity of the figure's border, i.e. the page colour."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return float(np.median(border))


def _is_background_band(
    gray: np.ndarray,
    run: tuple[int, int],
    *,
    axis: int,
    background: float,
    intensity_tolerance: float = 12.0,
    min_share: float = 0.9,
) -> bool:
    """Is this band actually the page showing through?

    **This replaced a projection-only test, and the difference is decisive.**
    Occupancy alone cannot separate "gap between panels" from "dark region
    inside one panel": a mountain photo's shadowed foreground and a fluorescence
    panel's empty background both read as gutters. On the measured fixture that
    split a two-tile figure into three panels and sliced a planted duplicate in
    half, leaving a 78px fragment to be compared against a 239px panel.

    Colour separates them cleanly. Measured on that fixture:

        true gutter      100.0% of pixels within 12 of background, std   0.0
        spurious gutter   24.5% of pixels within 12 of background, std 108.8

    A real gutter *is* the page: uniform, and equal to the border colour. Image
    content is neither, however dark it happens to be.

    ``axis=0`` checks a horizontal band (rows), ``axis=1`` a vertical one.
    """
    band = gray[run[0] : run[1], :] if axis == 0 else gray[:, run[0] : run[1]]
    if band.size == 0:
        return False
    share = float((np.abs(band.astype(np.float32) - background) < intensity_tolerance).mean())
    return share >= min_share


def _cut_points(length: int, gutters: list[tuple[int, int]]) -> list[int]:
    """Split coordinates: the middle of each interior gutter.

    Gutters touching an edge are margins, not separators, so they set the
    content bounds instead of producing a zero-width panel.
    """
    interior = [(a, b) for a, b in gutters if a > 0 and b < length]
    return [(a + b) // 2 for a, b in interior]


def split_panels(
    figure: Figure,
    *,
    min_panel_px: int = 64,
    min_gutter_frac: float = 0.02,
    tolerance: float = 0.02,
    max_panels: int = 64,
    min_occupancy: float = 0.10,
) -> list[Panel]:
    """Split one figure into panels by gutter projection.

    Returns a single whole-figure panel with ``confident=False`` when no
    plausible gutter structure is found. That is deliberately not an empty list:
    a figure that could not be split is still a figure worth comparing, and
    returning nothing would silently drop it from a manuscript-wide scan.

    Parameters
    ----------
    min_occupancy
        Foreground fraction below which projection is not attempted at all. See
        the comment on the check -- sparse figures defeat the method outright.
    """
    image = figure.image
    height, width = image.shape[:2]
    mask = foreground_mask(image)

    # Sparse figures cannot be split by projection, and this is a limit of the
    # method rather than a threshold to tune. A fluorescence panel of scattered
    # nuclei at 4.4% occupancy has large genuinely-empty regions *inside* one
    # panel, and no projection statistic distinguishes "gap between panels" from
    # "gap between cells" -- the measured case that produced two panels from a
    # single 320x240 image. Refusing to split, and saying so via `confident`,
    # beats inventing a boundary: a wrongly split panel is compared against its
    # own other half downstream, which manufactures a guaranteed false match.
    occupancy = float(mask.mean())
    if occupancy < min_occupancy:
        _log.info(
            "%s is %.1f%% foreground, below the %.1f%% floor for gutter projection; "
            "treating as one panel",
            figure.name,
            occupancy * 100,
            min_occupancy * 100,
        )
        return [
            Panel(
                image=image,
                box=(0, 0, width, height),
                figure=figure.name,
                index=0,
                confident=False,
            )
        ]

    min_gutter = max(4, round(min(height, width) * min_gutter_frac))
    rows = _gutters(mask.sum(axis=1), min_run=min_gutter, tolerance=tolerance)
    columns = _gutters(mask.sum(axis=0), min_run=min_gutter, tolerance=tolerance)

    # The projection proposes candidate bands; this confirms which ones are
    # genuinely the page showing between panels rather than dark image content.
    # See `_is_background_band` for the measurement that motivated it.
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = page_background(gray)
    rows = [g for g in rows if _is_background_band(gray, g, axis=0, background=background)]
    columns = [g for g in columns if _is_background_band(gray, g, axis=1, background=background)]

    y_cuts = [0, *_cut_points(height, rows), height]
    x_cuts = [0, *_cut_points(width, columns), width]

    panels: list[Panel] = []
    for top, bottom in pairwise(y_cuts):
        for left, right in pairwise(x_cuts):
            if bottom - top < min_panel_px or right - left < min_panel_px:
                continue
            tile = image[top:bottom, left:right]
            # A tile that is entirely background is a gap in an irregular grid,
            # not a panel.
            if foreground_mask(tile).mean() < 0.005:
                continue
            panels.append(
                Panel(
                    image=tile,
                    box=(left, top, right - left, bottom - top),
                    figure=figure.name,
                    index=len(panels),
                )
            )

    if len(panels) > max_panels:
        _log.info(
            "%s split into %d panels, over the %d cap; treating as one panel",
            figure.name,
            len(panels),
            max_panels,
        )
        panels = []

    if len(panels) <= 1:
        # Either genuinely one panel, or the projection found nothing usable.
        # Both are reported the same way and flagged, because this function
        # cannot tell them apart and should not guess.
        return [
            Panel(
                image=image,
                box=(0, 0, width, height),
                figure=figure.name,
                index=0,
                confident=len(panels) == 1,
            )
        ]

    return panels


def ingest(
    path: str | Path,
    *,
    prefer: Route = Route.EMBEDDED,
    dpi: int = 200,
    split: bool = True,
) -> Manuscript:
    """Extract figures and, optionally, split them into panels."""
    manuscript = extract_figures(path, prefer=prefer, dpi=dpi)
    if split:
        for figure in manuscript.figures:
            manuscript.panels.extend(split_panels(figure))
        _log.info(
            "split %d figures into %d panels", len(manuscript.figures), len(manuscript.panels)
        )
    return manuscript


def write(manuscript: Manuscript, directory: str | Path, *, panels: bool = True) -> list[Path]:
    """Write figures (and panels) as PNGs. Returns what was written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for figure in manuscript.figures:
        # The route is in the filename, not only in the metadata: a reader
        # picking a file off disk should be able to tell a submitted pixel from
        # a re-rendered one without consulting a manifest.
        path = directory / f"{figure.name}_{figure.route.value}.png"
        cv2.imwrite(str(path), figure.image)
        written.append(path)

    if panels:
        panel_dir = directory / "panels"
        panel_dir.mkdir(exist_ok=True)
        for panel in manuscript.panels:
            suffix = "" if panel.confident else "_unsplit"
            path = panel_dir / f"{panel.name}{suffix}.png"
            cv2.imwrite(str(path), panel.image)
            written.append(path)

    return written
