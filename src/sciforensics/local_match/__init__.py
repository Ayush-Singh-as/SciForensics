"""Local (keypoint-level) matching: detection, correspondence, geometry, copy-move."""

from __future__ import annotations

from sciforensics.local_match.copymove import (
    CopyMoveDetection,
    canonical_orientation,
    cluster_offsets,
    detect_copy_move,
    offset_features,
    orient_correspondences,
)
from sciforensics.local_match.geometry import (
    Verification,
    compose_affine,
    decompose_affine,
    distinct_count,
    estimate_affine,
    rescale_affine,
    rescale_box,
    rescale_homography,
    verify,
    verify_with_mask,
)
from sciforensics.local_match.keypoints import (
    Detection,
    KeypointDetector,
    OrbDetector,
    build_detector,
    dog_regions,
    enhance,
    sobel_magnitude,
)
from sciforensics.local_match.matcher import (
    BruteForceMatcher,
    Correspondences,
    Matcher,
    build_matcher,
    keypoint_evidence,
    self_match,
)

__all__ = [
    "BruteForceMatcher",
    "CopyMoveDetection",
    "Correspondences",
    "Detection",
    "KeypointDetector",
    "Matcher",
    "OrbDetector",
    "Verification",
    "build_detector",
    "build_matcher",
    "canonical_orientation",
    "cluster_offsets",
    "compose_affine",
    "decompose_affine",
    "detect_copy_move",
    "distinct_count",
    "dog_regions",
    "enhance",
    "estimate_affine",
    "keypoint_evidence",
    "offset_features",
    "orient_correspondences",
    "rescale_affine",
    "rescale_box",
    "rescale_homography",
    "self_match",
    "sobel_magnitude",
    "verify",
    "verify_with_mask",
]
