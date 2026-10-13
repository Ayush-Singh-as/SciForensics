"""Curated demo assets, included in wheels and containers."""

from pathlib import Path
from typing import Any

ASSET_ROOT = Path(__file__).parent / "data" / "examples"
EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "mountains-rotation",
        "label": "Rotated reuse",
        "note": "A 180-degree rotation. Both axes reverse, so reflection is correctly absent.",
        "kind": "positive",
        "mode": "compare",
        "left": "mountains.jpg",
        "right": "mountains_manipulated.jpg",
    },
    {
        "id": "cells-unrelated",
        "label": "Unrelated cell panels",
        "note": "Negative control: different source images should not verify geometrically.",
        "kind": "negative",
        "mode": "compare",
        "left": "base_cell.png",
        "right": "base_cells_2.png",
    },
    {
        "id": "cells-clone",
        "label": "Cloned cell region",
        "note": "A planted 128-pixel clone. The demo uses the documented CMFD tuning.",
        "kind": "positive",
        "mode": "cmfd",
        "left": "cells_clone.png",
        "right": None,
    },
]
