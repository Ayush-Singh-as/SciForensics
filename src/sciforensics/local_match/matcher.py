"""Descriptor matching with a mutual-nearest-neighbour constraint.

This module fixes the matching half of **bug 4**, which is the single most
consequential defect in the prototype.

The legacy code built ``cv2.BFMatcher(cv2.NORM_HAMMING)`` with no ``crossCheck``
and applied only Lowe's ratio test. The ratio test is a *unilateral* filter: it
asks whether descriptor ``i`` in the left image has one clearly-best partner in
the right image. It says nothing about whether that partner also prefers ``i``.
So when the two sides are unevenly sampled -- 2000 keypoints against 12, exactly
what the ROI bug produced -- hundreds of left keypoints can all pass the ratio
test against the *same* handful of right keypoints. The prototype reported "231
good matches" and RANSAC then found "107 inliers" among them, but those inliers
landed on only 4 geometrically distinct locations. Four points can be fitted
perfectly by an infinite family of homographies. The verdict was "strong
evidence"; the evidence was a counting artefact.

Requiring mutual agreement makes the surviving correspondence set *injective* by
construction: if ``(i, j)`` and ``(i', j)`` both survive then ``j``'s unique best
partner is both ``i`` and ``i'``, so ``i == i'``. Match count can therefore never
again exceed the smaller keypoint set, and the pathology is unrepresentable
rather than merely unlikely. :attr:`MatchEvidence.is_injective` asserts this
property on every result.

Both filters are kept, because they reject different things: the ratio test
removes ambiguous descriptors (repeated texture), mutual NN removes asymmetric
ones (many-to-one collapse).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from sciforensics.config import CopyMoveConfig, LocalMatchConfig
from sciforensics.local_match.geometry import distinct_count
from sciforensics.local_match.keypoints import Detection
from sciforensics.runtime import get_logger
from sciforensics.types import KeypointEvidence, MatchEvidence

__all__ = [
    "BruteForceMatcher",
    "Correspondences",
    "Matcher",
    "build_matcher",
    "keypoint_evidence",
    "self_match",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class Correspondences:
    """Surviving matches as parallel point arrays, in analysis space.

    ``src`` and ``dst`` are what geometry estimation consumes; the index arrays
    are retained so the report can draw lines back to specific keypoints.
    """

    src: np.ndarray  # (M, 2) float32 -- left/query points
    dst: np.ndarray  # (M, 2) float32 -- right/train points
    src_index: np.ndarray  # (M,) int32 into the left Detection
    dst_index: np.ndarray  # (M,) int32 into the right Detection
    distances: np.ndarray  # (M,) float32 descriptor distances
    evidence: MatchEvidence

    def __len__(self) -> int:
        return len(self.src)

    @property
    def count(self) -> int:
        return len(self.src)


@runtime_checkable
class Matcher(Protocol):
    """Interface every matcher backend satisfies.

    LightGlue (stage B3) is a *joint* matcher -- it consumes both keypoint sets
    and their descriptors together rather than doing independent nearest-neighbour
    lookups -- but it still produces an injective correspondence set, so it fits
    this signature without the pipeline changing.
    """

    name: str

    def match(self, left: Detection, right: Detection) -> Correspondences: ...


class BruteForceMatcher:
    """Exhaustive descriptor matching, ratio test, then mutual-NN agreement.

    Brute force rather than FLANN: with ``max_features`` capped at a few thousand
    the exhaustive Hamming comparison costs milliseconds, and it is *exact*.
    FLANN's approximation would introduce a second, silent source of asymmetry
    into the very set of matches this module exists to make symmetric.
    """

    name = "mutual_nn"

    def __init__(self, cfg: LocalMatchConfig, *, norm: int = cv2.NORM_HAMMING) -> None:
        self._cfg = cfg
        self._norm = norm
        # crossCheck is deliberately left False here: it is incompatible with
        # knnMatch(k=2), which the ratio test needs. Mutual agreement is enforced
        # explicitly below so both filters can be applied and both counts reported.
        self._matcher = cv2.BFMatcher(norm, crossCheck=False)

    def match(self, left: Detection, right: Detection) -> Correspondences:
        cfg = self._cfg
        if left.descriptors is None or right.descriptors is None:
            return _empty(self.name, cfg)
        if len(left.descriptors) == 0 or len(right.descriptors) == 0:
            return _empty(self.name, cfg)

        forward = self._matcher.knnMatch(left.descriptors, right.descriptors, k=2)
        raw = sum(1 for candidates in forward if candidates)

        pairs: list[tuple[int, int, float]] = []
        for candidates in forward:
            if not candidates:
                continue
            best = candidates[0]
            if len(candidates) >= 2:
                second = candidates[1]
                # Guard the degenerate case of two identical descriptors: a zero
                # denominator would otherwise raise, and a tie carries no
                # discriminative information anyway.
                if second.distance <= 0.0:
                    continue
                if best.distance >= cfg.nn_ratio * second.distance:
                    continue
            pairs.append((best.queryIdx, best.trainIdx, float(best.distance)))

        ratio_passed = len(pairs)

        if cfg.mutual_nn and pairs:
            backward = self._matcher.knnMatch(right.descriptors, left.descriptors, k=1)
            # best_left[j] = index of the left descriptor that right descriptor j
            # prefers. -1 marks "no candidate", which OpenCV can return for
            # masked or empty rows.
            best_left = np.full(len(right.descriptors), -1, dtype=np.int64)
            for candidates in backward:
                if candidates:
                    best_left[candidates[0].queryIdx] = candidates[0].trainIdx
            pairs = [(i, j, d) for i, j, d in pairs if best_left[j] == i]

        if not pairs:
            return _empty(self.name, cfg, raw=raw, ratio_passed=ratio_passed)

        src_index = np.array([i for i, _, _ in pairs], dtype=np.int32)
        dst_index = np.array([j for _, j, _ in pairs], dtype=np.int32)
        distances = np.array([d for _, _, d in pairs], dtype=np.float32)
        src = np.ascontiguousarray(left.points[src_index], dtype=np.float32)
        dst = np.ascontiguousarray(right.points[dst_index], dtype=np.float32)

        evidence = MatchEvidence(
            matcher=self.name if cfg.mutual_nn else "ratio_only",
            raw=raw,
            ratio_passed=ratio_passed,
            good=len(pairs),
            # Counted on the same half-pixel grid geometry.verify uses, so
            # "distinct" means one thing across the whole codebase. Note this
            # counts distinct *locations*, not distinct indices: two ORB
            # keypoints at the same point on different octaves are one
            # geometric constraint, not two.
            distinct_left=distinct_count(src),
            distinct_right=distinct_count(dst),
            mutual_nn=cfg.mutual_nn,
            ratio=cfg.nn_ratio,
        )
        if not evidence.is_injective and cfg.mutual_nn:
            # Reachable only through coincident keypoint locations, never through
            # index collisions; worth logging because it means `good` overstates
            # the independent constraints available to the geometry stage.
            _log.debug(
                "coincident keypoints: %d matches span %d/%d distinct locations",
                evidence.good,
                evidence.distinct_left,
                evidence.distinct_right,
            )

        return Correspondences(
            src=src,
            dst=dst,
            src_index=src_index,
            dst_index=dst_index,
            distances=distances,
            evidence=evidence,
        )


def _g2nn_accept(distances: Sequence[float], ratio: float) -> int:
    """How many of the sorted neighbours in ``distances`` are genuine partners.

    Lowe's ratio test asks a single question -- is ``d1`` much smaller than
    ``d2``? -- and that question has no good answer when a region has been cloned
    more than once. Three copies of a patch give every one of its keypoints *two*
    partners at essentially equal distance, so ``d1 / d2`` is close to 1 and the
    ratio test throws away both. A matcher restricted to 2-NN can therefore never
    report more than one clone per keypoint, no matter how many copies exist or
    how large ``copy_move.knn`` is set.

    The generalisation (g2NN, Amerini et al. 2011, the standard formulation for
    keypoint copy-move detection) looks for the *gap* instead of the top-two
    ratio: accept the first ``t`` neighbours, where ``t`` is the first position
    followed by a distance jump of at least ``1 / ratio``. At ``t == 1`` this
    reduces exactly to Lowe's test, so the single-clone behaviour is unchanged.

    Returns 0 when no such gap exists -- every candidate is equally plausible,
    which is the signature of flat or repetitive texture rather than of a clone.
    """
    n = len(distances)
    for t in range(1, n):
        following = distances[t]
        if following <= 0.0:
            # Exact duplicates: 0 / 0 carries no information about a gap, so keep
            # looking. This is the common case for a losslessly cloned region,
            # where the first several distances are all zero.
            continue
        if distances[t - 1] < ratio * following:
            return t
    return 0


def self_match(
    detection: Detection, cfg: CopyMoveConfig, *, norm: int = cv2.NORM_HAMMING
) -> Correspondences:
    """Match an image against itself for copy-move detection.

    The trivial ``i -> i`` self-correspondence and everything within
    ``copy_move.min_spatial_separation`` pixels must be suppressed, otherwise
    every keypoint matches itself and the "clone" is the whole image.

    Uses ``copy_move.knn`` neighbours and the g2NN acceptance rule (see
    :func:`_g2nn_accept`) rather than a plain 2-NN ratio test, which is what makes
    a region pasted three or more times detectable. ``knn`` therefore bounds the
    clone multiplicity this stage can discover: a patch appearing ``m`` times
    needs ``knn >= m`` to be fully recovered.
    """
    if detection.descriptors is None or len(detection.descriptors) < 2:
        return _empty_self(cfg)

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = min(cfg.knn, len(detection.descriptors))
    neighbours = matcher.knnMatch(detection.descriptors, detection.descriptors, k=knn)

    raw = 0
    pairs: list[tuple[int, int, float]] = []
    separation_sq = float(cfg.min_spatial_separation) ** 2
    for candidates in neighbours:
        # knnMatch returns the descriptor itself at distance 0; that is not a
        # clone, it is the same keypoint.
        rest = [c for c in candidates if c.queryIdx != c.trainIdx]
        raw += len(rest)
        if len(rest) < 2:
            continue

        accepted = _g2nn_accept([float(c.distance) for c in rest], cfg.nn_ratio)
        for candidate in rest[:accepted]:
            pi = detection.points[candidate.queryIdx]
            pj = detection.points[candidate.trainIdx]
            if float((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2) < separation_sq:
                continue
            pairs.append((candidate.queryIdx, candidate.trainIdx, float(candidate.distance)))

    ratio_passed = len(pairs)
    if not pairs:
        return _empty_self(cfg, raw=raw, ratio_passed=ratio_passed)

    # Self-matching is symmetric: (i, j) and (j, i) are the same clone
    # relationship seen from both ends. Canonicalise on i < j and deduplicate,
    # so a two-region clone contributes each correspondence once. The legacy
    # code instead ordered by `qx < tx`, which silently discarded every clone
    # offset purely vertically -- part of bug 6.
    seen: set[tuple[int, int]] = set()
    canonical: list[tuple[int, int, float]] = []
    for i, j, d in pairs:
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        canonical.append((key[0], key[1], d))

    src_index = np.array([i for i, _, _ in canonical], dtype=np.int32)
    dst_index = np.array([j for _, j, _ in canonical], dtype=np.int32)
    distances = np.array([d for _, _, d in canonical], dtype=np.float32)
    src = np.ascontiguousarray(detection.points[src_index], dtype=np.float32)
    dst = np.ascontiguousarray(detection.points[dst_index], dtype=np.float32)

    return Correspondences(
        src=src,
        dst=dst,
        src_index=src_index,
        dst_index=dst_index,
        distances=distances,
        evidence=MatchEvidence(
            matcher="self_knn",
            raw=raw,
            ratio_passed=ratio_passed,
            good=len(canonical),
            distinct_left=distinct_count(src),
            distinct_right=distinct_count(dst),
            mutual_nn=False,
            ratio=cfg.nn_ratio,
        ),
    )


def keypoint_evidence(left: Detection, right: Detection) -> KeypointEvidence:
    """Summarise both detections into the reportable evidence record."""
    return KeypointEvidence(
        detector=left.detector,
        detected_left=left.detected,
        detected_right=right.detected,
        kept_left=left.count,
        kept_right=right.count,
        roi_abandoned_left=left.roi_abandoned,
        roi_abandoned_right=right.roi_abandoned,
        enhancement_scale=left.enhancement_scale,
    )


def _empty(
    name: str, cfg: LocalMatchConfig, *, raw: int = 0, ratio_passed: int = 0
) -> Correspondences:
    return _blank(
        MatchEvidence(
            matcher=name,
            raw=raw,
            ratio_passed=ratio_passed,
            good=0,
            distinct_left=0,
            distinct_right=0,
            mutual_nn=cfg.mutual_nn,
            ratio=cfg.nn_ratio,
        )
    )


def _empty_self(cfg: CopyMoveConfig, *, raw: int = 0, ratio_passed: int = 0) -> Correspondences:
    return _blank(
        MatchEvidence(
            matcher="self_knn",
            raw=raw,
            ratio_passed=ratio_passed,
            good=0,
            distinct_left=0,
            distinct_right=0,
            mutual_nn=False,
            ratio=cfg.nn_ratio,
        )
    )


def _blank(evidence: MatchEvidence) -> Correspondences:
    return Correspondences(
        src=np.zeros((0, 2), np.float32),
        dst=np.zeros((0, 2), np.float32),
        src_index=np.zeros((0,), np.int32),
        dst_index=np.zeros((0,), np.int32),
        distances=np.zeros((0,), np.float32),
        evidence=evidence,
    )


def build_matcher(cfg: LocalMatchConfig, *, norm: int = cv2.NORM_HAMMING) -> Matcher:
    """Instantiate the configured matcher backend."""
    if cfg.matcher in {"mutual_nn", "bf"}:
        return BruteForceMatcher(cfg, norm=norm)
    raise NotImplementedError(
        f"matcher {cfg.matcher!r} is not available yet (LightGlue lands in stage B3). "
        "Use local_match.matcher=mutual_nn."
    )
