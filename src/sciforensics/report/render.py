"""Render a scan into HTML, PDF and JSON.

**Bug 7's fix.** The prototype drew its verdict with ``cv2.putText`` onto a
fixed 420 px canvas; a full result put its last wrapped line at baseline y=415,
so the closing words were sliced off in every report it ever produced. Text here
is flowed HTML with no fixed height in the text path, so it cannot clip at any
length. The regression test asserts the property that matters -- the whole
verdict string is present in the output -- rather than measuring a pixel.

**PDF is optional by design.** WeasyPrint pulls in native libraries (Pango,
cairo) that are awkward on Windows and in slim containers. When it is absent the
HTML is still written and the caller is told, because a report the reader can
open in a browser and print is far better than a hard failure. ``formats``
requesting ``pdf`` without WeasyPrint installed is a warning, not an error.

Images are written beside the HTML and referenced relatively, so the output
directory is self-contained and can be zipped or served as-is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from sciforensics.config import Settings
from sciforensics.pipeline import CopyMoveAnalysis, PairAnalysis
from sciforensics.report import overlays as ov
from sciforensics.runtime import get_logger
from sciforensics.types import CopyMoveResult, EvidenceContribution, ScanResult, Verdict

_log = get_logger(__name__)

_TEMPLATES = Path(__file__).parent / "templates"

# Verdict -> the CSS class that colours the finding block. Kept here rather than
# in the template so a new Verdict member is a KeyError in one obvious place
# instead of an unstyled block nobody notices.
_VERDICT_CLASS = {
    Verdict.LIKELY_MANIPULATED: "is-alert",
    Verdict.SUSPICIOUS: "is-suspicious",
    Verdict.INCONCLUSIVE: "",
    Verdict.CLEAN: "is-clean",
}


class ReportError(RuntimeError):
    """Raised when a report cannot be written at all."""


@dataclass(frozen=True)
class Written:
    """What actually landed on disk, plus anything the caller should hear about."""

    directory: Path
    html: Path | None = None
    pdf: Path | None = None
    json_path: Path | None = None
    images: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def primary(self) -> Path | None:
        """The file to point a human at."""
        return self.pdf or self.html or self.json_path


def _environment() -> Environment:
    """Jinja with ``StrictUndefined``.

    A typo'd field in a forensic report must fail loudly. Jinja's default would
    render a missing value as an empty string, which in this document means a
    silently absent measurement -- indistinguishable from a measurement of zero.
    """
    return Environment(
        loader=FileSystemLoader(_TEMPLATES),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _contribution_rows(
    contributions: tuple[EvidenceContribution, ...],
) -> list[dict[str, Any]]:
    """Sort by magnitude and scale bar widths to the largest contribution.

    Normalising to the largest rather than to a fixed maximum keeps the chart
    readable whether the strongest signal is 0.4 or 4.0 log-odds; the numeric
    column carries the absolute value, so nothing is lost.
    """
    if not contributions:
        return []
    ordered = sorted(contributions, key=lambda c: abs(c.contribution), reverse=True)
    largest = max(abs(c.contribution) for c in ordered) or 1.0
    rows: list[dict[str, Any]] = []
    for c in ordered:
        rows.append(
            {
                "name": c.name,
                "description": c.description,
                "value": c.value,
                "contribution": c.contribution,
                # Half-width track per side, so 50% is a full-length bar.
                "width": min(50.0, abs(c.contribution) / largest * 50.0),
            }
        )
    return rows


def _audit_fields(result: ScanResult | CopyMoveResult) -> dict[str, Any]:
    """Audit block minus the per-image entries the template renders itself."""
    skip = {"left_sha256", "right_sha256", "inputs", "image_sha256"}
    dump = result.audit.model_dump(mode="json")
    return {k: v for k, v in dump.items() if v is not None and v != [] and k not in skip}


def _write_overlays(
    items: list[ov.Overlay | None], directory: Path, *, max_px: int
) -> tuple[dict[str, dict[str, Any]], tuple[Path, ...]]:
    """Write each overlay and return a name -> descriptor map for the template."""
    assets = directory / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, dict[str, Any]] = {}
    written: list[Path] = []
    for item in items:
        if item is None:
            continue
        path = item.write(assets, max_px=max_px)
        written.append(path)
        mapping[item.name] = {
            # Relative so the directory stays portable.
            "href": f"assets/{path.name}",
            "caption": item.caption,
            "trust": item.trust,
        }
    return mapping, tuple(written)


def _render_pdf(html: str, destination: Path, base_url: Path) -> str | None:
    """Write a PDF, or return why it could not be written.

    ``base_url`` must be the output directory so WeasyPrint resolves the
    relative image hrefs.
    """
    try:
        from weasyprint import HTML  # type: ignore[import-untyped]
    except ImportError:
        return (
            "PDF skipped: WeasyPrint is not installed (pip install 'sciforensics[report]'). "
            "The HTML report was written instead and prints to PDF from any browser."
        )
    try:
        HTML(string=html, base_url=str(base_url)).write_pdf(str(destination))
    except Exception as exc:  # pragma: no cover - native library failures
        return f"PDF rendering failed: {exc}. The HTML report was written instead."
    return None


def render_pair(
    analysis: PairAnalysis,
    cfg: Settings,
    directory: Path,
    *,
    formats: tuple[str, ...] | None = None,
) -> Written:
    """Render a comparison. ``directory`` is created if needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_cfg = cfg.report
    wanted = tuple(formats) if formats else report_cfg.formats
    result = analysis.result

    items: list[ov.Overlay | None] = [
        ov.thumbnail(analysis.left, side="left"),
        ov.thumbnail(analysis.right, side="right"),
    ]

    for side, image, attribution in (
        ("left", analysis.left, analysis.comparison.attribution_left),
        ("right", analysis.right, analysis.comparison.attribution_right),
    ):
        if attribution is not None:
            items.append(
                ov.attribution_overlay(
                    image,
                    attribution,
                    side=side,
                    colormap=cfg.global_match.attribution.colormap,
                    alpha=cfg.global_match.attribution.alpha,
                )
            )

    if analysis.correspondences is not None:
        items.append(
            ov.match_overlay(
                analysis.left,
                analysis.right,
                analysis.correspondences,
                analysis.verification,
                report_cfg,
            )
        )

    overlay_map, images = _write_overlays(items, directory, max_px=report_cfg.thumbnail_max_px)

    warnings: list[str] = []
    html_path: Path | None = None
    pdf_path: Path | None = None
    json_path: Path | None = None

    if "json" in wanted:
        json_path = directory / "report.json"
        json_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
        )

    if "html" in wanted or "pdf" in wanted:
        env = _environment()
        html = env.get_template("report.html.j2").render(
            title="Image Comparison",
            css=(_TEMPLATES / "base.css").read_text(encoding="utf-8"),
            result=result,
            g=result.global_evidence,
            k=result.keypoints,
            m=result.matches,
            geo=result.geometry,
            t=result.geometry.transform if result.geometry else None,
            overlays=overlay_map,
            contributions=_contribution_rows(result.contributions),
            audit=_audit_fields(result) if report_cfg.audit_trail else None,
            verdict_class=_VERDICT_CLASS[result.verdict],
        )
        # Always write the HTML when a PDF is requested: it is the fallback if
        # WeasyPrint is missing, and it costs nothing.
        html_path = directory / "report.html"
        html_path.write_text(html, encoding="utf-8")

        if "pdf" in wanted:
            pdf_path = directory / "report.pdf"
            problem = _render_pdf(html, pdf_path, directory)
            if problem:
                warnings.append(problem)
                pdf_path = None

        if "html" not in wanted and pdf_path is not None:
            html_path.unlink()
            html_path = None

    if "png" in wanted:
        # The contact sheet is a Stage A3 nice-to-have for README embedding; the
        # individual overlays already cover the same ground, so rather than ship
        # a half-built composite the request is reported as unhandled.
        warnings.append("PNG contact-sheet output is not implemented; overlays were written.")

    return Written(
        directory=directory,
        html=html_path,
        pdf=pdf_path,
        json_path=json_path,
        images=images,
        warnings=tuple(warnings),
    )


def render_copy_move(
    analysis: CopyMoveAnalysis,
    cfg: Settings,
    directory: Path,
    *,
    formats: tuple[str, ...] | None = None,
) -> Written:
    """Render a copy-move search.

    Shares the template with :func:`render_pair` only where the two results
    genuinely coincide; CMFD has no second image, so it gets its own document
    rather than a pair template with half its fields blanked.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_cfg = cfg.report
    wanted = tuple(formats) if formats else report_cfg.formats
    result = analysis.result

    items: list[ov.Overlay | None] = [
        ov.thumbnail(analysis.image, side="input"),
        ov.copy_move_overlay(analysis.image, analysis.detection),
    ]
    overlay_map, images = _write_overlays(items, directory, max_px=report_cfg.thumbnail_max_px)

    # Region masks are rasters the evidence model points at by path, so write
    # them and record where they went.
    mask_paths: list[Path] = []
    if analysis.detection.masks:
        mask_dir = directory / "assets"
        for index, mask in enumerate(analysis.detection.masks, start=1):
            import cv2

            path = mask_dir / f"region_{index}_mask.png"
            cv2.imwrite(str(path), mask)
            mask_paths.append(path)

    warnings: list[str] = []
    html_path: Path | None = None
    pdf_path: Path | None = None
    json_path: Path | None = None

    if "json" in wanted:
        json_path = directory / "report.json"
        json_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
        )

    if "html" in wanted or "pdf" in wanted:
        env = _environment()
        html = env.get_template("copymove.html.j2").render(
            title="Copy-Move Analysis",
            css=(_TEMPLATES / "base.css").read_text(encoding="utf-8"),
            result=result,
            ev=result.evidence,
            overlays=overlay_map,
            contributions=_contribution_rows(result.contributions),
            audit=_audit_fields(result) if report_cfg.audit_trail else None,
            verdict_class=_VERDICT_CLASS[result.verdict],
        )
        html_path = directory / "report.html"
        html_path.write_text(html, encoding="utf-8")

        if "pdf" in wanted:
            pdf_path = directory / "report.pdf"
            problem = _render_pdf(html, pdf_path, directory)
            if problem:
                warnings.append(problem)
                pdf_path = None

        if "html" not in wanted and pdf_path is not None:
            html_path.unlink()
            html_path = None

    return Written(
        directory=directory,
        html=html_path,
        pdf=pdf_path,
        json_path=json_path,
        images=tuple(images) + tuple(mask_paths),
        warnings=tuple(warnings),
    )
