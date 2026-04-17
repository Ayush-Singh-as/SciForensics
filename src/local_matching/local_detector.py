#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from skimage.feature import blob_dog
except ImportError:
    blob_dog = None


@dataclass
class PairwiseMatchResult:
    left_work_bgr: np.ndarray
    right_work_bgr: np.ndarray
    left_keypoints: list[cv2.KeyPoint]
    right_keypoints: list[cv2.KeyPoint]
    good_matches: list[cv2.DMatch]
    left_points: Optional[np.ndarray]
    right_points: Optional[np.ndarray]
    dog_boxes_left: list[tuple[int, int, int, int]]
    dog_boxes_right: list[tuple[int, int, int, int]]
    left_keypoint_count: int
    right_keypoint_count: int
    good_match_count: int


@dataclass
class CopyMoveRegion:
    """A single pair of source and clone regions within one image."""
    source_bbox: tuple[int, int, int, int]  # (x, y, w, h) bounding box of source
    clone_bbox: tuple[int, int, int, int]   # (x, y, w, h) bounding box of clone
    source_points: np.ndarray               # Nx2 inlier keypoints in source
    clone_points: np.ndarray                # Nx2 inlier keypoints in clone
    homography: np.ndarray                  # 3x3 homography mapping source → clone
    inlier_count: int
    inlier_ratio: float


@dataclass
class CopyMoveResult:
    """Result of intra-image copy-move forgery detection."""
    image_bgr: np.ndarray                   # Original image for reference
    keypoint_count: int                     # Total ORB keypoints detected
    self_match_count: int                   # Spatially-filtered self-matches
    regions: list[CopyMoveRegion]           # Verified clone region pairs
    clone_detected: bool                    # True if at least one region verified


def sobel_f(image: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(dx, dy)
    max_value = float(np.max(magnitude))
    if max_value > 0:
        magnitude *= 255.0 / max_value
    return np.uint8(magnitude)


def dog_regions(gray: np.ndarray, max_regions: int = 24, min_area: int = 40) -> list[tuple[int, int, int, int]]:
    if blob_dog is not None:
        blobs = blob_dog(gray, min_sigma=2, max_sigma=30, threshold=0.08)
        boxes = []
        for y, x, sigma in blobs[:max_regions]:
            radius = max(4, int(round(float(sigma) * np.sqrt(2))))
            boxes.append(_circle_to_box(int(round(x)), int(round(y)), radius, gray.shape))
        if boxes:
            return boxes

    gray_float = gray.astype(np.float32) / 255.0
    blur_small = cv2.GaussianBlur(gray_float, (0, 0), 1.0)
    blur_large = cv2.GaussianBlur(gray_float, (0, 0), 2.4)
    dog = cv2.absdiff(blur_small, blur_large)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((int(x), int(y), int(w), int(h)))

    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes[:max_regions]


def enhance_for_orb(gray: np.ndarray, scale: float = 4.0) -> np.ndarray:
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (0, 0), 0.8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blurred)


def match_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    max_features: int = 2000,
    nn_ratio: float = 0.8,
    min_matches: int = 8,
) -> PairwiseMatchResult:
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    left_work = enhance_for_orb(left_gray)
    right_work = enhance_for_orb(right_gray)
    left_boxes = dog_regions(sobel_f(left_work))
    right_boxes = dog_regions(sobel_f(right_work))

    orb = cv2.ORB_create(
        nfeatures=max_features,
        fastThreshold=5,
        edgeThreshold=15,
        patchSize=31,
        scaleFactor=1.2,
        nlevels=8,
    )
    left_keypoints, left_descriptors = orb.detectAndCompute(left_work, None)
    right_keypoints, right_descriptors = orb.detectAndCompute(right_work, None)

    if left_boxes:
        left_keypoints, left_descriptors = _filter_to_regions(left_keypoints, left_descriptors, left_boxes)
    if right_boxes:
        right_keypoints, right_descriptors = _filter_to_regions(right_keypoints, right_descriptors, right_boxes)

    good_matches: list[cv2.DMatch] = []
    left_points = None
    right_points = None

    if (
        left_descriptors is not None
        and right_descriptors is not None
        and len(left_keypoints) > 0
        and len(right_keypoints) > 0
    ):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            match_a, match_b = pair
            if match_a.distance < nn_ratio * match_b.distance:
                good_matches.append(match_a)

        if len(good_matches) >= min_matches:
            left_points = np.float32([left_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            right_points = np.float32([right_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            left_points /= 4.0
            right_points /= 4.0

    left_keypoints_1x = [cv2.KeyPoint(x=kp.pt[0]/4.0, y=kp.pt[1]/4.0, size=kp.size/4.0, angle=kp.angle, response=kp.response, octave=kp.octave, class_id=kp.class_id) for kp in left_keypoints]
    right_keypoints_1x = [cv2.KeyPoint(x=kp.pt[0]/4.0, y=kp.pt[1]/4.0, size=kp.size/4.0, angle=kp.angle, response=kp.response, octave=kp.octave, class_id=kp.class_id) for kp in right_keypoints]

    left_draw = left_bgr.copy()
    right_draw = right_bgr.copy()
    dog_boxes_left = [_downscale_box(box, 4.0) for box in left_boxes]
    dog_boxes_right = [_downscale_box(box, 4.0) for box in right_boxes]

    _draw_boxes(left_draw, dog_boxes_left)
    _draw_boxes(right_draw, dog_boxes_right)

    return PairwiseMatchResult(
        left_work_bgr=left_draw,
        right_work_bgr=right_draw,
        left_keypoints=left_keypoints_1x,
        right_keypoints=right_keypoints_1x,
        good_matches=good_matches,
        left_points=left_points,
        right_points=right_points,
        dog_boxes_left=dog_boxes_left,
        dog_boxes_right=dog_boxes_right,
        left_keypoint_count=len(left_keypoints),
        right_keypoint_count=len(right_keypoints),
        good_match_count=len(good_matches),
    )


def draw_match_visualization(
    result: PairwiseMatchResult,
    inlier_mask: Optional[np.ndarray] = None,
    only_inliers: bool = True,
) -> np.ndarray:
    matches = result.good_matches
    if inlier_mask is not None:
        selected_matches = [match for match, keep in zip(matches, inlier_mask) if bool(keep)]
        if selected_matches or only_inliers:
            matches = selected_matches

    if not matches:
        canvas = np.full((420, 540, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No matches to visualize", (55, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70, 70, 70), 2, cv2.LINE_AA)
        return canvas

    return cv2.drawMatches(
        result.left_work_bgr,
        result.left_keypoints,
        result.right_work_bgr,
        result.right_keypoints,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 180, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def _filter_to_regions(
    keypoints: list[cv2.KeyPoint],
    descriptors: Optional[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray]]:
    if descriptors is None or not keypoints:
        return [], None

    keep_indices = [index for index, keypoint in enumerate(keypoints) if _point_in_any_box(keypoint.pt, boxes)]
    if len(keep_indices) < 8:
        return keypoints, descriptors

    filtered_keypoints = [keypoints[index] for index in keep_indices]
    filtered_descriptors = descriptors[keep_indices]
    return filtered_keypoints, filtered_descriptors


def _point_in_any_box(point: tuple[float, float], boxes: list[tuple[int, int, int, int]]) -> bool:
    x, y = point
    for box_x, box_y, box_w, box_h in boxes:
        if box_x <= x <= box_x + box_w and box_y <= y <= box_y + box_h:
            return True
    return False


def _draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 120, 0), 1)


def _circle_to_box(x: int, y: int, radius: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(width - 1, x + radius)
    bottom = min(height - 1, y + radius)
    return left, top, max(1, right - left), max(1, bottom - top)


def _downscale_box(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return int(round(x / scale)), int(round(y / scale)), int(round(w / scale)), int(round(h / scale))


def _read_color(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


# ---------------------------------------------------------------------------
#  Intra-image Copy-Move Forgery Detection (CMFD)
# ---------------------------------------------------------------------------

def detect_copy_move(
    image_bgr: np.ndarray,
    max_features: int = 4000,
    nn_ratio: float = 0.75,
    min_spatial_distance_ratio: float = 0.10,
    min_cluster_matches: int = 8,
    ransac_reproj_threshold: float = 4.0,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.20,
) -> CopyMoveResult:
    """Detect duplicated (copy-pasted) regions within a single image.

    Strategy:
      1. Enhance and detect ORB keypoints across the full image.
      2. Self-match: match the descriptor set against itself.
      3. Spatial filter: discard matches where the two keypoints are
         physically close together (< min_spatial_distance_ratio of the
         image diagonal), because these are just same-region self-hits.
      4. RANSAC: estimate a homography on the surviving matches to verify
         geometric consistency and group inlier clusters.
      5. Extract bounding boxes around source / clone inlier groups.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    work = enhance_for_orb(gray)
    h_img, w_img = image_bgr.shape[:2]
    diag = float(np.sqrt(h_img ** 2 + w_img ** 2))
    min_dist = min_spatial_distance_ratio * diag

    # Detect ORB keypoints
    orb = cv2.ORB_create(
        nfeatures=max_features,
        fastThreshold=5,
        edgeThreshold=15,
        patchSize=31,
        scaleFactor=1.2,
        nlevels=8,
    )
    keypoints, descriptors = orb.detectAndCompute(work, None)

    if descriptors is None or len(keypoints) < 2 * min_cluster_matches:
        return CopyMoveResult(
            image_bgr=image_bgr,
            keypoint_count=len(keypoints) if keypoints else 0,
            self_match_count=0,
            regions=[],
            clone_detected=False,
        )

    # Self-match with k=2 for ratio test (but k=3 so we can skip the identity match)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(descriptors, descriptors, k=3)

    # Filter: remove identity matches and apply ratio test + spatial distance
    scale = 4.0  # enhance_for_orb uses 4x upscale
    good_src_pts: list[tuple[float, float]] = []
    good_dst_pts: list[tuple[float, float]] = []

    for group in knn_matches:
        # Skip the first match (identity: descriptor matched to itself)
        candidates = [m for m in group if m.queryIdx != m.trainIdx]
        if len(candidates) < 2:
            continue
        m1, m2 = candidates[0], candidates[1]
        if m1.distance >= nn_ratio * m2.distance:
            continue
        pt_q = keypoints[m1.queryIdx].pt
        pt_t = keypoints[m1.trainIdx].pt
        # Map back to original image coordinates
        qx, qy = pt_q[0] / scale, pt_q[1] / scale
        tx, ty = pt_t[0] / scale, pt_t[1] / scale
        
        # VERY IMPORTANT: Enforce directional ordering so we don't get contradicting
        # translation vectors (A->B and B->A) which destroys RANSAC homography.
        if (qx < tx) or (qx == tx and qy < ty):
            spatial_dist = np.sqrt((qx - tx) ** 2 + (qy - ty) ** 2)
            if spatial_dist >= min_dist:
                good_src_pts.append((qx, qy))
                good_dst_pts.append((tx, ty))

    self_match_count = len(good_src_pts)

    if self_match_count < min_cluster_matches:
        return CopyMoveResult(
            image_bgr=image_bgr,
            keypoint_count=len(keypoints),
            self_match_count=self_match_count,
            regions=[],
            clone_detected=False,
        )

    src_pts = np.float32(good_src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(good_dst_pts).reshape(-1, 1, 2)

    transform, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_threshold
    )

    regions: list[CopyMoveRegion] = []

    if transform is not None and mask is not None:
        inlier_mask = mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = float(inlier_count / max(self_match_count, 1))

        if inlier_count >= min_inliers and inlier_ratio >= min_inlier_ratio:
            # Pad 2x3 affine matrix to 3x3 homography matrix for standard representation
            homography = np.vstack([transform, [0, 0, 1]])
            
            inlier_src = src_pts[inlier_mask].reshape(-1, 2)
            inlier_dst = dst_pts[inlier_mask].reshape(-1, 2)
            src_bbox = _pts_to_bbox(inlier_src)
            dst_bbox = _pts_to_bbox(inlier_dst)
            regions.append(CopyMoveRegion(
                source_bbox=src_bbox,
                clone_bbox=dst_bbox,
                source_points=inlier_src,
                clone_points=inlier_dst,
                homography=homography,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
            ))

    return CopyMoveResult(
        image_bgr=image_bgr,
        keypoint_count=len(keypoints),
        self_match_count=self_match_count,
        regions=regions,
        clone_detected=len(regions) > 0,
    )


def _pts_to_bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    """Convert Nx2 points to (x, y, w, h) bounding box."""
    x, y, w, h = cv2.boundingRect(points.astype(np.float32).reshape(-1, 1, 2))
    return int(x), int(y), int(w), int(h)


def draw_copy_move_visualization(
    result: CopyMoveResult,
) -> np.ndarray:
    """Draw annotated visualization for intra-image CMFD results."""
    canvas = result.image_bgr.copy()

    if not result.regions:
        cv2.putText(
            canvas, "No copy-move forgery detected",
            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
        )
        return canvas

    for idx, region in enumerate(result.regions):
        # Draw source region in blue
        sx, sy, sw, sh = region.source_bbox
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (255, 100, 0), 2)
        cv2.putText(
            canvas, f"Source {idx + 1}",
            (sx, max(24, sy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2, cv2.LINE_AA,
        )

        # Draw clone region in red
        cx, cy, cw, ch = region.clone_bbox
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
        cv2.putText(
            canvas, f"Clone {idx + 1}",
            (cx, max(24, cy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

        # Draw match lines between inlier pairs
        for src_pt, dst_pt in zip(region.source_points, region.clone_points):
            pt1 = (int(round(src_pt[0])), int(round(src_pt[1])))
            pt2 = (int(round(dst_pt[0])), int(round(dst_pt[1])))
            cv2.line(canvas, pt1, pt2, (0, 220, 255), 1, cv2.LINE_AA)

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise DoG + ORB scientific image matcher")
    parser.add_argument("left_image", help="path to the first image")
    parser.add_argument("right_image", nargs="?", default=None, help="path to the second image (omit for CMFD mode)")
    parser.add_argument("--output", default="orb_matches.png", help="path to the match visualization")
    parser.add_argument("--cmfd", action="store_true", help="run intra-image copy-move forgery detection on left_image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_bgr = _read_color(args.left_image)

    if args.cmfd or args.right_image is None:
        # Intra-image CMFD mode
        result = detect_copy_move(left_bgr)
        visualization = draw_copy_move_visualization(result)
        cv2.imwrite(str(args.output), visualization)
        print(f"Saved CMFD visualization to {Path(args.output).resolve()}")
        print(f"Keypoints: {result.keypoint_count}")
        print(f"Spatially-filtered self-matches: {result.self_match_count}")
        print(f"Clone regions found: {len(result.regions)}")
        print(f"Clone detected: {result.clone_detected}")
    else:
        right_bgr = _read_color(args.right_image)
        result = match_pair(left_bgr, right_bgr)
        visualization = draw_match_visualization(result, only_inliers=False)
        cv2.imwrite(str(args.output), visualization)
        print(f"Saved visualization to {Path(args.output).resolve()}")
        print(f"Good matches: {result.good_match_count}")


if __name__ == "__main__":
    main()
