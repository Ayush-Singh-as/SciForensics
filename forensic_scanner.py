from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(SRC_DIR / "global_matching"))
    sys.path.insert(0, str(SRC_DIR / "local_matching"))

from global_matching.pairwise_inference import PairwiseEmbeddingModel
import local_matching.local_detector as cmfd


@dataclass
class GeometricVerificationResult:
    verified: bool
    homography: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]
    inlier_count: int
    inlier_ratio: float
    left_bbox: Optional[tuple[int, int, int, int]]
    right_bbox: Optional[tuple[int, int, int, int]]
    rotation_deg: Optional[float]
    scale: Optional[float]
    translation: Optional[tuple[float, float]]
    flip_detected: Optional[bool]


@dataclass
class ScannerConfig:
    weights_path: Path
    output_path: Path
    same_threshold: float = 1.1
    local_trigger_threshold: float = 2.0
    nn_ratio: float = 0.8
    max_features: int = 2000
    min_matches: int = 8
    min_inliers: int = 8
    min_inlier_ratio: float = 0.18
    ransac_reproj_threshold: float = 4.0


class ForensicScanner:
    def __init__(self, config: ScannerConfig) -> None:
        self.config = config
        self.global_model = PairwiseEmbeddingModel(
            weights_path=config.weights_path,
            same_threshold=config.same_threshold,
            local_trigger_threshold=config.local_trigger_threshold,
        )

    def scan(self, left_path: str | Path, right_path: str | Path) -> dict:
        left_path = Path(left_path)
        right_path = Path(right_path)
        left_bgr = self._read_color(left_path)
        right_bgr = self._read_color(right_path)

        global_result = self.global_model.compare(left_path, right_path, generate_heatmaps=True)
        local_result = None
        geometry_result = None

        if global_result.trigger_local:
            local_result = cmfd.match_pair(
                left_bgr,
                right_bgr,
                max_features=self.config.max_features,
                nn_ratio=self.config.nn_ratio,
                min_matches=self.config.min_matches,
            )
            geometry_result = self._verify_geometry(local_result)

        report_image = self._build_report(left_bgr, right_bgr, global_result, local_result, geometry_result)
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.config.output_path), report_image)

        summary = self._summarize(global_result, local_result, geometry_result)
        summary["report_path"] = str(self.config.output_path.resolve())
        return summary

    def _verify_geometry(self, local_result: cmfd.PairwiseMatchResult) -> GeometricVerificationResult:
        if local_result.good_match_count < 4 or local_result.left_points is None or local_result.right_points is None:
            return GeometricVerificationResult(
                verified=False,
                homography=None,
                inlier_mask=None,
                inlier_count=0,
                inlier_ratio=0.0,
                left_bbox=None,
                right_bbox=None,
                rotation_deg=None,
                scale=None,
                translation=None,
                flip_detected=None,
            )

        homography, mask = cv2.findHomography(
            local_result.left_points,
            local_result.right_points,
            cv2.RANSAC,
            self.config.ransac_reproj_threshold,
        )

        if homography is None or mask is None:
            return GeometricVerificationResult(
                verified=False,
                homography=None,
                inlier_mask=None,
                inlier_count=0,
                inlier_ratio=0.0,
                left_bbox=None,
                right_bbox=None,
                rotation_deg=None,
                scale=None,
                translation=None,
                flip_detected=None,
            )

        inlier_mask = mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = float(inlier_count / max(local_result.good_match_count, 1))
        verified = inlier_count >= self.config.min_inliers and inlier_ratio >= self.config.min_inlier_ratio

        left_bbox = None
        right_bbox = None
        rotation_deg = None
        scale = None
        translation = None
        flip_detected = None

        if inlier_count >= 4:
            inlier_left = local_result.left_points[inlier_mask].reshape(-1, 2)
            inlier_right = local_result.right_points[inlier_mask].reshape(-1, 2)
            left_bbox = self._points_to_bbox(inlier_left)
            right_bbox = self._points_to_bbox(inlier_right)
            affine, _ = cv2.estimateAffinePartial2D(inlier_left, inlier_right, method=cv2.LMEDS)
            if affine is not None:
                rotation_deg, scale, translation, flip_detected = self._decode_affine(affine)

        return GeometricVerificationResult(
            verified=verified,
            homography=homography,
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            left_bbox=left_bbox,
            right_bbox=right_bbox,
            rotation_deg=rotation_deg,
            scale=scale,
            translation=translation,
            flip_detected=flip_detected,
        )

    def _build_report(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        global_result,
        local_result: Optional[cmfd.PairwiseMatchResult],
        geometry_result: Optional[GeometricVerificationResult],
    ) -> np.ndarray:
        annotated_left = left_bgr.copy()
        annotated_right = right_bgr.copy()

        if geometry_result and geometry_result.left_bbox:
            self._draw_bbox(annotated_left, geometry_result.left_bbox, "Verified region")
        if geometry_result and geometry_result.right_bbox:
            self._draw_bbox(annotated_right, geometry_result.right_bbox, "Matched region")

        left_heatmap = global_result.left_heatmap if global_result.left_heatmap is not None else self._placeholder_panel("Global heatmap skipped")
        right_heatmap = global_result.right_heatmap if global_result.right_heatmap is not None else self._placeholder_panel("Global heatmap skipped")

        if local_result is not None:
            match_panel = cmfd.draw_match_visualization(
                local_result,
                inlier_mask=geometry_result.inlier_mask if geometry_result else None,
                only_inliers=bool(geometry_result and geometry_result.verified),
            )
        else:
            match_panel = self._placeholder_panel("Local ORB stage not triggered")

        summary_panel = self._build_summary_panel(global_result, local_result, geometry_result)

        top_row = self._hstack_panels(
            [
                self._panel_with_title(annotated_left, "Input A"),
                self._panel_with_title(annotated_right, "Input B"),
                self._panel_with_title(summary_panel, "Forensic Summary"),
            ]
        )
        bottom_row = self._hstack_panels(
            [
                self._panel_with_title(left_heatmap, "Global Heatmap A"),
                self._panel_with_title(right_heatmap, "Global Heatmap B"),
                self._panel_with_title(match_panel, "Local Matches"),
            ]
        )
        return np.vstack([top_row, bottom_row])

    def _build_summary_panel(self, global_result, local_result, geometry_result) -> np.ndarray:
        canvas = np.full((420, 540, 3), 248, dtype=np.uint8)
        lines = [
            f"Global distance: {global_result.distance:.3f}",
            f"Similarity score: {global_result.similarity_score:.3f}",
            f"Local stage triggered: {'yes' if global_result.trigger_local else 'no'}",
        ]

        if local_result is not None:
            lines.extend(
                [
                    f"Keypoints: {local_result.left_keypoint_count} vs {local_result.right_keypoint_count}",
                    f"Good ORB matches: {local_result.good_match_count}",
                ]
            )
        else:
            lines.append("Good ORB matches: n/a")

        if geometry_result is not None:
            lines.extend(
                [
                    f"Homography verified: {'yes' if geometry_result.verified else 'no'}",
                    f"Inliers: {geometry_result.inlier_count}",
                    f"Inlier ratio: {geometry_result.inlier_ratio:.3f}",
                ]
            )
            if geometry_result.rotation_deg is not None:
                lines.append(f"Estimated rotation: {geometry_result.rotation_deg:.1f} deg")
            if geometry_result.scale is not None:
                lines.append(f"Estimated scale: {geometry_result.scale:.3f}")
            if geometry_result.translation is not None:
                lines.append(
                    f"Estimated translation: ({geometry_result.translation[0]:.1f}, {geometry_result.translation[1]:.1f})"
                )
            if geometry_result.flip_detected is not None:
                lines.append(f"Flip detected: {'yes' if geometry_result.flip_detected else 'no'}")
        else:
            lines.append("Homography verified: n/a")

        decision = self._decision_text(global_result, geometry_result)
        lines.append("")
        lines.append(f"Decision: {decision}")

        y = 40
        for line in lines:
            if line.startswith("Decision:"):
                # wrap Decision line
                words = line.split()
                current_line = ""
                cv2.putText(canvas, "Decision:", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
                y += 25
                words = words[1:]
                for word in words:
                    if len(current_line) + len(word) > 40:
                        cv2.putText(canvas, current_line.strip(), (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
                        y += 25
                        current_line = word + " "
                    else:
                        current_line += word + " "
                if current_line:
                    cv2.putText(canvas, current_line.strip(), (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
                y += 25
        return canvas

    def _summarize(self, global_result, local_result, geometry_result) -> dict:
        return {
            "global_distance": round(global_result.distance, 6),
            "similarity_score": round(global_result.similarity_score, 6),
            "triggered_local_stage": bool(global_result.trigger_local),
            "good_matches": int(local_result.good_match_count) if local_result else 0,
            "homography_verified": bool(geometry_result.verified) if geometry_result else False,
            "inlier_count": int(geometry_result.inlier_count) if geometry_result else 0,
            "inlier_ratio": round(geometry_result.inlier_ratio, 6) if geometry_result else 0.0,
            "decision": self._decision_text(global_result, geometry_result),
        }

    def _decision_text(self, global_result, geometry_result: Optional[GeometricVerificationResult]) -> str:
        if geometry_result and geometry_result.verified and global_result.suspicious:
            return "Strong evidence of reused or manipulated scientific imagery"
        if geometry_result and geometry_result.verified:
            return "Potential partial reuse verified geometrically"
        if global_result.suspicious:
            return "Globally similar pair; local verification inconclusive"
        return "No strong evidence of manipulated reuse"

    def _decode_affine(
        self, affine: np.ndarray
    ) -> tuple[float, float, tuple[float, float], bool]:
        linear = affine[:, :2]
        tx, ty = float(affine[0, 2]), float(affine[1, 2])
        det = float(np.linalg.det(linear))
        flip_detected = det < 0
        col0 = linear[:, 0]
        scale = float(np.linalg.norm(col0))
        rotation_deg = float(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
        return rotation_deg, scale, (tx, ty), flip_detected

    def _points_to_bbox(self, points: np.ndarray) -> tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(points.astype(np.float32).reshape(-1, 1, 2))
        return int(x), int(y), int(w), int(h)

    def _draw_bbox(self, image: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> None:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 220, 255), 2)
        cv2.putText(image, label, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

    def _panel_with_title(self, image: np.ndarray, title: str) -> np.ndarray:
        image = self._ensure_bgr(image)
        fitted = self._fit_to_panel(image, width=540, height=420)
        title_bar = np.full((48, fitted.shape[1], 3), 35, dtype=np.uint8)
        cv2.putText(title_bar, title, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
        return np.vstack([title_bar, fitted])

    def _fit_to_panel(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        image = self._ensure_bgr(image)
        h, w = image.shape[:2]
        scale = min(width / max(w, 1), height / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        canvas = np.full((height, width, 3), 250, dtype=np.uint8)
        offset_x = (width - new_w) // 2
        offset_y = (height - new_h) // 2
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
        return canvas

    def _hstack_panels(self, panels: list[np.ndarray]) -> np.ndarray:
        height = max(panel.shape[0] for panel in panels)
        normalized = []
        for panel in panels:
            if panel.shape[0] == height:
                normalized.append(panel)
                continue
            pad = np.full((height - panel.shape[0], panel.shape[1], 3), 250, dtype=np.uint8)
            normalized.append(np.vstack([panel, pad]))
        return np.hstack(normalized)

    def _placeholder_panel(self, text: str) -> np.ndarray:
        canvas = np.full((420, 540, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, text, (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70, 70, 70), 2, cv2.LINE_AA)
        return canvas

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _read_color(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image

    # ------------------------------------------------------------------
    #  Intra-image Copy-Move Forgery Detection (CMFD)
    # ------------------------------------------------------------------

    def scan_single(self, image_path: str | Path) -> dict:
        """Run intra-image copy-move forgery detection on a single image."""
        image_path = Path(image_path)
        image_bgr = self._read_color(image_path)

        cmfd_result = cmfd.detect_copy_move(
            image_bgr,
            max_features=self.config.max_features,
            nn_ratio=self.config.nn_ratio,
            min_inliers=self.config.min_inliers,
            min_inlier_ratio=self.config.min_inlier_ratio,
            ransac_reproj_threshold=self.config.ransac_reproj_threshold,
        )

        report_image = self._build_cmfd_report(image_bgr, cmfd_result)
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.config.output_path), report_image)

        summary = self._summarize_cmfd(cmfd_result)
        summary["image_path"] = str(image_path.resolve())
        summary["report_path"] = str(self.config.output_path.resolve())
        return summary

    def _build_cmfd_report(
        self,
        image_bgr: np.ndarray,
        cmfd_result: cmfd.CopyMoveResult,
    ) -> np.ndarray:
        """Build a forensic report for intra-image CMFD."""
        annotated = cmfd.draw_copy_move_visualization(cmfd_result)
        summary_panel = self._build_cmfd_summary_panel(cmfd_result)

        top_row = self._hstack_panels(
            [
                self._panel_with_title(image_bgr, "Original Image"),
                self._panel_with_title(annotated, "CMFD Annotated"),
                self._panel_with_title(summary_panel, "CMFD Summary"),
            ]
        )
        return top_row

    def _build_cmfd_summary_panel(self, cmfd_result: cmfd.CopyMoveResult) -> np.ndarray:
        canvas = np.full((420, 540, 3), 248, dtype=np.uint8)
        lines = [
            "Mode: Intra-Image CMFD",
            "",
            f"Total keypoints: {cmfd_result.keypoint_count}",
            f"Spatial self-matches: {cmfd_result.self_match_count}",
            f"Clone regions found: {len(cmfd_result.regions)}",
        ]

        for idx, region in enumerate(cmfd_result.regions):
            lines.append("")
            lines.append(f"--- Region {idx + 1} ---")
            lines.append(f"  Inliers: {region.inlier_count}")
            lines.append(f"  Inlier ratio: {region.inlier_ratio:.3f}")
            sx, sy, sw, sh = region.source_bbox
            lines.append(f"  Source: ({sx},{sy}) {sw}x{sh}")
            cx, cy, cw, ch = region.clone_bbox
            lines.append(f"  Clone:  ({cx},{cy}) {cw}x{ch}")

        lines.append("")
        if cmfd_result.clone_detected:
            lines.append("Decision: Copy-move forgery DETECTED")
        else:
            lines.append("Decision: No copy-move forgery found")

        y = 40
        for line in lines:
            color = (0, 0, 200) if "DETECTED" in line else (30, 30, 30)
            cv2.putText(
                canvas, line, (24, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
            )
            y += 25
        return canvas

    def _summarize_cmfd(self, cmfd_result: cmfd.CopyMoveResult) -> dict:
        regions_data = []
        for region in cmfd_result.regions:
            regions_data.append({
                "source_bbox": list(region.source_bbox),
                "clone_bbox": list(region.clone_bbox),
                "inlier_count": region.inlier_count,
                "inlier_ratio": round(region.inlier_ratio, 6),
            })
        return {
            "mode": "cmfd",
            "keypoint_count": cmfd_result.keypoint_count,
            "self_match_count": cmfd_result.self_match_count,
            "clone_region_count": len(cmfd_result.regions),
            "clone_detected": cmfd_result.clone_detected,
            "regions": regions_data,
            "decision": "Copy-move forgery DETECTED" if cmfd_result.clone_detected else "No copy-move forgery found",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scientific image plagiarism and manipulation scanner")
    parser.add_argument("left_image", help="path to the base scientific image")
    parser.add_argument("right_image", nargs="?", help="path to the second scientific image, or a directory of images to test against (omit if using --cmfd)")
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "outputs"),
        help="path for the combined forensic report image, or output directory for dataset scanning",
    )
    parser.add_argument(
        "--weights",
        default=str(ROOT_DIR / "models" / "weights.pth"),
        help="path to the PyTorch weights file",
    )
    parser.add_argument("--cmfd", action="store_true", help="run intra-image copy-move forgery detection on left_image instead of pairwise comparison")
    parser.add_argument("--same-threshold", type=float, default=1.1, help="distance threshold for strong global similarity")
    parser.add_argument("--local-threshold", type=float, default=3.0, help="distance threshold for triggering local ORB matching")
    parser.add_argument("--ratio", type=float, default=0.8, help="nearest-neighbor ratio test threshold for ORB matches")
    parser.add_argument("--max-features", type=int, default=2000, help="maximum number of ORB features")
    parser.add_argument("--min-matches", type=int, default=8, help="minimum good matches expected from the local stage")
    parser.add_argument("--min-inliers", type=int, default=8, help="minimum RANSAC inliers required for geometric verification")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.18, help="minimum inlier ratio for geometric verification")
    parser.add_argument("--ransac-threshold", type=float, default=4.0, help="RANSAC reprojection error threshold in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cmfd and not args.right_image:
        print("Error: pairwise scanning requires both a left_image and a right_image. Use --cmfd if you want to scan a single image.")
        sys.exit(1)
    
    out_path = Path(args.output)

    # CMFD mode: intra-image copy-move forgery detection
    if args.cmfd:
        report_path = out_path if out_path.suffix else out_path / "cmfd_report.png"
        config = ScannerConfig(
            weights_path=Path(args.weights),
            output_path=report_path,
            same_threshold=args.same_threshold,
            local_trigger_threshold=args.local_threshold,
            nn_ratio=args.ratio,
            max_features=args.max_features,
            min_matches=args.min_matches,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            ransac_reproj_threshold=args.ransac_threshold,
        )
        scanner = ForensicScanner(config)
        summary = scanner.scan_single(args.left_image)
        print(json.dumps(summary, indent=2))
        return

    # Evaluate Dataset mode if right_image is a directory
    if args.right_image and Path(args.right_image).is_dir():
        right_path = Path(args.right_image)
        out_path = Path(args.output)
        if out_path.suffix:
            out_path = out_path.parent
        out_path.mkdir(parents=True, exist_ok=True)
            
        test_images = list(right_path.glob("*.jpg")) + list(right_path.glob("*.png"))
        print(f"Starting evaluation: Computing results for {len(test_images)} images under '{right_path.name}'...")
        
        overall_summary = []
        for img_path in test_images:
            print(f"Scanning against {img_path.name}...")
            report_path = out_path / f"report_{img_path.stem}.png"
            
            config = ScannerConfig(
                weights_path=Path(args.weights),
                output_path=report_path,
                same_threshold=args.same_threshold,
                local_trigger_threshold=args.local_threshold,
                nn_ratio=args.ratio,
                max_features=args.max_features,
                min_matches=args.min_matches,
                min_inliers=args.min_inliers,
                min_inlier_ratio=args.min_inlier_ratio,
                ransac_reproj_threshold=args.ransac_threshold,
            )
            scanner = ForensicScanner(config)
            summary = scanner.scan(args.left_image, img_path)
            
            # Save individual JSON summary
            json_path = out_path / f"summary_{img_path.stem}.json"
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)
                
            summary['test_image'] = img_path.name
            overall_summary.append(summary)

        # Save overall summary
        with open(out_path / "overall_summary.json", "w") as f:
            json.dump(overall_summary, f, indent=2)
            
        print(f"Evaluation complete! All visual reports and JSON logs are saved in '{out_path}'.")
        
    else:
        # Standard Single File Scan mode
        config = ScannerConfig(
            weights_path=Path(args.weights),
            output_path=out_path if out_path.suffix else out_path / "forensic_report.png",
            same_threshold=args.same_threshold,
            local_trigger_threshold=args.local_threshold,
            nn_ratio=args.ratio,
            max_features=args.max_features,
            min_matches=args.min_matches,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            ransac_reproj_threshold=args.ransac_threshold,
        )
        scanner = ForensicScanner(config)
        summary = scanner.scan(args.left_image, args.right_image)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()