"""
Comprehensive Forensic Scanner Demo
====================================
Runs the full forensic pipeline on all input images and generates:
  1. Individual forensic reports for each (base, manipulation) pair
  2. One consolidated per-base summary image showing all 5 manipulations
  3. CMFD (copy-move forgery detection) reports for the copy-move examples
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(SRC_DIR / "global_matching"))
    sys.path.insert(0, str(SRC_DIR / "local_matching"))

from forensic_scanner import ForensicScanner, ScannerConfig

# ── Configuration ──────────────────────────────────────────────────────────
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUT_DIR = ROOT_DIR / "outputs" / "demo"

BASE_IMAGES = {
    "base_cell": {
        "base": "base_cell.png",
        "manipulations": [
            ("ex1_affine",    "base_cell_ex1_affine.png",    "Affine (Rotate+Flip)"),
            ("ex2_degraded",  "base_cell_ex2_degraded.jpg",  "Signal Degradation"),
            ("ex3_copymove",  "base_cell_ex3_copymove.png",  "Copy-Move Forgery"),
            ("ex4_exposure",  "base_cell_ex4_exposure.png",  "Exposure Shift"),
            ("ex5_blackout",  "base_cell_ex5_blackout.png",  "Blackout Masking"),
        ],
    },
    "base_cells_1": {
        "base": "base_cells_1.png",
        "manipulations": [
            ("ex1_affine",    "base_cells_1_ex1_affine.png",    "Affine (Rotate+Flip)"),
            ("ex2_degraded",  "base_cells_1_ex2_degraded.jpg",  "Signal Degradation"),
            ("ex3_copymove",  "base_cells_1_ex3_copymove.png",  "Copy-Move Forgery"),
            ("ex4_exposure",  "base_cells_1_ex4_exposure.png",  "Exposure Shift"),
            ("ex5_blackout",  "base_cells_1_ex5_blackout.png",  "Blackout Masking"),
        ],
    },
    "base_cells_2": {
        "base": "base_cells_2.png",
        "manipulations": [
            ("ex1_affine",    "base_cells_2_ex1_affine.png",    "Affine (Rotate+Flip)"),
            ("ex2_degraded",  "base_cells_2_ex2_degraded.jpg",  "Signal Degradation"),
            ("ex3_copymove",  "base_cells_2_ex3_copymove.png",  "Copy-Move Forgery"),
            ("ex4_exposure",  "base_cells_2_ex4_exposure.png",  "Exposure Shift"),
            ("ex5_blackout",  "base_cells_2_ex5_blackout.png",  "Blackout Masking"),
        ],
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────

def fit_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image to fit within (width, height) while preserving aspect ratio,
    then center on a light-gray canvas."""
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((height, width, 3), 240, dtype=np.uint8)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def make_title_bar(text: str, width: int, height: int = 40, font_scale: float = 0.7,
                   bg_color=(30, 30, 30), fg_color=(240, 240, 240)) -> np.ndarray:
    bar = np.full((height, width, 3), bg_color[0], dtype=np.uint8)
    bar[:, :, 0] = bg_color[0]
    bar[:, :, 1] = bg_color[1]
    bar[:, :, 2] = bg_color[2]
    cv2.putText(bar, text, (14, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg_color, 2, cv2.LINE_AA)
    return bar


def make_verdict_bar(verdict: str, width: int, height: int = 50) -> np.ndarray:
    """Color-coded verdict bar."""
    is_positive = any(w in verdict.lower() for w in ["strong", "verified", "detected"])
    bg = (0, 0, 140) if is_positive else (0, 120, 0)  # BGR: red vs green
    bar = np.full((height, width, 3), 30, dtype=np.uint8)
    bar[:, :, 0] = bg[0]
    bar[:, :, 1] = bg[1]
    bar[:, :, 2] = bg[2]

    # Wrap the verdict text if needed
    words = verdict.split()
    lines = []
    current = ""
    for w in words:
        if len(current) + len(w) > 55:
            lines.append(current.strip())
            current = w + " "
        else:
            current += w + " "
    if current.strip():
        lines.append(current.strip())

    y = 20
    for line in lines:
        cv2.putText(bar, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return bar


def make_stats_panel(summary: dict, width: int, height: int) -> np.ndarray:
    """Compact metrics panel for the consolidated view."""
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    lines = [
        f"Dist: {summary['global_distance']:.3f}   Sim: {summary['similarity_score']:.3f}",
        f"ORB matches: {summary['good_matches']}   Inliers: {summary['inlier_count']}",
        f"Verified: {'YES' if summary['homography_verified'] else 'no'}   "
        f"Ratio: {summary['inlier_ratio']:.2f}",
    ]
    y = 22
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (30, 30, 30), 1, cv2.LINE_AA)
        y += 20
    return canvas


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = ScannerConfig(
        weights_path=ROOT_DIR / "models" / "weights.pth",
        output_path=OUTPUT_DIR / "placeholder.png",
        same_threshold=1.1,
        local_trigger_threshold=3.0,
    )
    scanner = ForensicScanner(config)

    all_summaries = {}

    for base_key, spec in BASE_IMAGES.items():
        base_path = INPUTS_DIR / spec["base"]
        if not base_path.exists():
            print(f"⚠ Base image not found: {base_path}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing: {base_key}")
        print(f"{'='*60}")

        base_bgr = cv2.imread(str(base_path), cv2.IMREAD_COLOR)

        # ── Build the base image column (leftmost) ──
        base_title = make_title_bar(f"Original: {spec['base']}", CELL_W, 36, 0.50)
        base_fitted = fit_image(base_bgr, CELL_W, CELL_H)
        base_label_bar = np.full((68, CELL_W, 3), 248, dtype=np.uint8)
        cv2.putText(base_label_bar, "Base image (anchor)", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(base_label_bar, f"Size: {base_bgr.shape[1]}x{base_bgr.shape[0]}",
                    (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)
        base_verdict = make_verdict_bar("Reference Image", CELL_W, 44)
        base_column = np.vstack([base_title, base_fitted, base_label_bar, base_verdict])

        # ── Build a column for each manipulation ──
        columns = [base_column]
        summaries_for_base = []

        for manip_id, manip_file, manip_label in spec["manipulations"]:
            manip_path = INPUTS_DIR / manip_file
            if not manip_path.exists():
                print(f"  ⚠ Not found: {manip_file}, skipping")
                continue

            print(f"  → {manip_label} ({manip_file})")

            # Generate individual detailed report
            config.output_path = OUTPUT_DIR / f"report_{base_key}_{manip_id}.png"
            summary = scanner.scan(str(base_path), str(manip_path))

            # Build column for consolidated view
            manip_bgr = cv2.imread(str(manip_path), cv2.IMREAD_COLOR)
            manip_fitted = fit_image(manip_bgr, CELL_W, CELL_H)

            title = make_title_bar(manip_label, CELL_W, 36, 0.55)
            stats = make_stats_panel(summary, CELL_W, 68)
            verdict = make_verdict_bar(summary["decision"], CELL_W, 44)
            col = np.vstack([title, manip_fitted, stats, verdict])
            columns.append(col)

            summaries_for_base.append({"manipulation": manip_label, **summary})
            print(f"    Distance={summary['global_distance']:.3f}  "
                  f"Verified={summary['homography_verified']}  "
                  f"→ {summary['decision']}")

        all_summaries[base_key] = summaries_for_base

        # ── CMFD on the copy-move example ──
        for manip_id, manip_file, manip_label in spec["manipulations"]:
            if "copymove" in manip_id:
                manip_path = INPUTS_DIR / manip_file
                if manip_path.exists():
                    print(f"  → CMFD scan on {manip_file}")
                    config.output_path = OUTPUT_DIR / f"cmfd_{base_key}.png"
                    cmfd_summary = scanner.scan_single(str(manip_path))
                    print(f"    Clone detected: {cmfd_summary['clone_detected']}")

        # ── Assemble consolidated image ──
        max_h = max(c.shape[0] for c in columns)
        unified = []
        for c in columns:
            if c.shape[0] < max_h:
                pad = np.full((max_h - c.shape[0], c.shape[1], 3), 240, dtype=np.uint8)
                c = np.vstack([c, pad])
            unified.append(c)

        # Add 2px separator between columns
        sep = np.full((max_h, 2, 3), 180, dtype=np.uint8)
        parts = []
        for i, col in enumerate(unified):
            if i > 0:
                parts.append(sep)
            parts.append(col)
        row = np.hstack(parts)

        # Add a big title at the top
        big_title = make_title_bar(
            f"Forensic Analysis: {spec['base']}  |  5 Manipulation Types",
            row.shape[1], 48, 0.75, bg_color=(25, 25, 25)
        )
        consolidated = np.vstack([big_title, row])

        consolidated_path = OUTPUT_DIR / f"consolidated_{base_key}.png"
        cv2.imwrite(str(consolidated_path), consolidated)
        print(f"\n  >>> Consolidated report: {consolidated_path}")

    # ── Save JSON summary ──
    json_path = OUTPUT_DIR / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  All done! Results saved to {OUTPUT_DIR}")
    print(f"  JSON summary: {json_path}")
    print(f"{'='*60}")


CELL_W = 320
CELL_H = 240


if __name__ == "__main__":
    main()
