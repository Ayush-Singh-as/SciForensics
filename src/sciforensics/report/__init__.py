"""Report rendering: HTML, PDF and JSON, plus the raster overlays they embed.

Stage A3. Replaces the prototype's ``cv2.putText`` dashboard, whose fixed 420 px
summary panel clipped the verdict text in every report it produced (bug 7).

``render`` imports Jinja2 and OpenCV, so it is not pulled in at package scope --
:mod:`sciforensics.cli` imports it only when a report is actually requested.
"""

from __future__ import annotations

__all__ = ["ReportError", "Written", "render_copy_move", "render_pair"]


def __getattr__(name: str) -> object:
    """Re-export lazily, so importing this package stays cheap."""
    if name in __all__:
        from sciforensics.report import render

        return getattr(render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
