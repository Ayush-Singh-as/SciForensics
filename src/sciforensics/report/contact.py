"""Compact PNG reports with dynamically measured, fully visible prose."""

from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from sciforensics.report.overlays import Overlay
from sciforensics.types import CopyMoveResult, ScanResult


def contact_sheet(
    items: list[Overlay | None], result: ScanResult | CopyMoveResult, destination: Path
) -> Path:
    font = ImageFont.load_default(size=16)
    rows = [
        result.verdict.label,
        f"Confidence: {result.confidence:.3f} "
        f"({'calibrated' if result.calibrated else 'uncalibrated score'})",
    ]
    if isinstance(result, ScanResult) and result.geometry is not None:
        g = result.geometry
        rows.append(
            f"Geometry: {'verified' if g.verified else g.rejection_reason.value}; "
            f"{g.inlier_count} inliers"
        )
        if g.transform:
            t = g.transform
            rows.append(
                f"Rotation {t.rotation_deg:.2f} degrees; reflection {t.flip}; "
                f"scale {abs(t.scale_x * t.scale_y) ** 0.5:.4f}"
            )
    rows.extend(result.notes)
    rows.append(
        "Automated screening. A finding requires expert review; "
        "absence of a finding is not proof of integrity."
    )
    lines = [line for row in rows for line in textwrap.wrap(row, 100)]
    overlays = [
        item for item in items if item is not None and not item.name.startswith("keypoints_")
    ]
    cell_w, cell_h = 800, 360
    header_h = 36 + len(lines) * 24
    canvas = Image.new(
        "RGB", (cell_w * 2, header_h + ((len(overlays) + 1) // 2) * cell_h), "#10161b"
    )
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((24, 18 + index * 24), line, fill="#e3e8eb", font=font)
    for index, item in enumerate(overlays):
        image = Image.fromarray(cv2.cvtColor(item.image, cv2.COLOR_BGR2RGB))
        image.thumbnail((cell_w - 24, cell_h - 44))
        x = (index % 2) * cell_w + 12
        y = header_h + (index // 2) * cell_h
        draw.text((x, y), item.name.replace("_", " "), fill="#e3e8eb", font=font)
        canvas.paste(image, (x, y + 28))
    canvas.save(destination)
    return destination
