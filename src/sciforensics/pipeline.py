"""Stage orchestration: file paths in, a scored :class:`ScanResult` out.

This is the module the CLI, the HTTP API and the benchmark harness all call, and
the only place that knows the *order* of the stages. Everything below it is a pure
function of its inputs; everything above it renders what this returns.

Three things live here and nowhere else.

**Expensive objects are built once.** The backbone, the detector, the matcher and
the calibrator are all constructed with the :class:`Pipeline` and reused across
every pair it is asked about. Bug 11 was the prototype rebuilding its scanner
*inside* the batch loop, which spent most of a 40-pair sweep re-reading a 35 MB
checkpoint from disk. Holding them on an object rather than in a closure is what
makes that structural rather than a comment asking the next person not to.

**Escalation is a decision, and skipping a stage is recorded.** The local stage is
expensive and only runs when the global stage says a pair is worth the keypoints
(:attr:`~sciforensics.global_match.PairComparison.triggered_local`). A result whose
``geometry`` is ``None`` because nothing escalated is a genuinely different claim
from one whose geometry ran and found nothing -- which is the distinction
:mod:`sciforensics.fusion.features` is built around -- so both the
:attr:`~sciforensics.types.ScanResult.stages_run` tuple and an explicit warning
say which happened.

**The keypoint floor is enforced here.** ``local_match.min_keypoints_per_side``
asks whether the *surviving* counts are enough to compare at all, which is an
orchestration question because it is a fact about the pair rather than about
either image. (The detector has its own, separate threshold --
``local_match.roi.min_keypoints`` -- for deciding whether to abandon the ROI
restriction; the two were one knob until that conflation made every small panel
report a keypoint shortfall for what was really a matching one.) This is the half
of bug 4 that lives above the matcher: on the ``mountains`` pair the legacy
pipeline matched 2000 keypoints against 12 and reported 231 good matches and 121
"inliers". Refusing to match at all is a cheaper and clearer answer than gating
the resulting garbage downstream, and it is the one case that produces
:attr:`~sciforensics.types.RejectionReason.TOO_FEW_KEYPOINTS`.

Coordinate frames follow the convention the stages already established: analysis
space throughout, converted to original-image pixels at the point of report. See
:func:`sciforensics.local_match.rescale_homography` for why that is not cosmetic.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from sciforensics import fusion, weights
from sciforensics.audit import build_audit_block
from sciforensics.config import Settings
from sciforensics.global_match.embed import Embedder, PairComparison
from sciforensics.io.images import LoadedImage, load_image
from sciforensics.local_match.copymove import CopyMoveDetection, detect_copy_move
from sciforensics.local_match.geometry import Verification, verify_with_mask
from sciforensics.local_match.keypoints import Detection, KeypointDetector, build_detector
from sciforensics.local_match.matcher import (
    Correspondences,
    Matcher,
    build_matcher,
    keypoint_evidence,
)
from sciforensics.runtime import get_logger, resolve_device
from sciforensics.types import (
    AuditBlock,
    CopyMoveResult,
    GeometryEvidence,
    RejectionReason,
    ScanResult,
    Stage,
    Verdict,
)

__all__ = [
    "CopyMoveAnalysis",
    "PairAnalysis",
    "Pipeline",
    "PipelineError",
]

_log = get_logger(__name__)


class PipelineError(RuntimeError):
    """Raised when a run cannot proceed, as distinct from finding nothing.

    The two must never be conflated. "I was configured not to look" and "I looked
    and found nothing" produce identical reports unless one of them raises.
    """


@contextmanager
def _timed(timings: dict[str, float], stage: Stage) -> Iterator[None]:
    """Record a stage's wall-clock cost, including when it raises.

    ``finally`` rather than a plain trailing assignment: a stage that failed still
    consumed time, and a timings dict that silently omits the slow thing that blew
    up is worse than useless when someone is diagnosing a timeout.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage.value] = time.perf_counter() - start


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairAnalysis:
    """A scored comparison plus the working data needed to *draw* it.

    The same split as :class:`~sciforensics.local_match.Verification` and
    :class:`~sciforensics.local_match.CopyMoveDetection`, applied at the top level:
    :attr:`result` is the serialisable record -- it is what the JSON output and the
    HTTP response are made of -- and everything else is numpy and torch working
    data that the report renderer consumes and then discards.

    Keeping them apart is what lets a result be serialised without deciding where
    its overlay images live. The renderer writes the heatmaps and the masks, then
    fills in the paths with ``model_copy(update=...)``; nothing here touches the
    filesystem.
    """

    result: ScanResult
    left: LoadedImage
    right: LoadedImage
    #: Carries both :class:`~sciforensics.global_match.Attribution` maps, when
    #: attribution ran.
    comparison: PairComparison
    #: ``None`` when the local stage did not run. Both are ``None`` or neither is.
    left_keypoints: Detection | None = None
    right_keypoints: Detection | None = None
    #: ``None`` when the local stage did not run, or when the keypoint floor
    #: refused the pair before matching.
    correspondences: Correspondences | None = None
    #: Carries the per-correspondence consensus mask, so the renderer can draw
    #: which lines held. ``None`` whenever :attr:`correspondences` is.
    verification: Verification | None = None

    @property
    def verdict(self) -> Verdict:
        return self.result.verdict


@dataclass(frozen=True)
class CopyMoveAnalysis:
    """A scored copy-move search plus its masks and keypoints."""

    result: CopyMoveResult
    image: LoadedImage
    keypoints: Detection
    #: Region masks, in analysis space; see
    #: :class:`~sciforensics.local_match.CopyMoveDetection`.
    detection: CopyMoveDetection

    @property
    def verdict(self) -> Verdict:
        return self.result.verdict


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
@dataclass
class Pipeline:
    """The stages, wired up, with their expensive parts held open.

    Construct one per configuration and call it many times::

        pipeline = Pipeline(load_config())
        for left, right in pairs:
            analysis = pipeline.compare(left, right)

    Not thread-safe, for the same reason :class:`~sciforensics.global_match.Embedder`
    is not: attribution runs a backward pass that mutates ``.grad`` on captured
    activations, so two concurrent :meth:`compare` calls would interleave gradients
    between pairs. Use one instance per worker.
    """

    cfg: Settings
    #: Overrides ``runtime.device``. Mostly for tests, which want CPU regardless
    #: of what the machine has.
    device: str | None = None
    #: Pre-built backbone, bypassing checkpoint resolution. Lets a test drive the
    #: whole pipeline against a randomly-initialised net with no 35 MB artifact.
    embedder: Embedder | None = None
    _weights_sha256: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.torch_device = (
            resolve_device(self.device) if self.device else resolve_device(self.cfg.runtime.device)
        )
        # An injected embedder carries its *own* `global_match` and `image` config,
        # and the escalation threshold is read off the comparison it returns rather
        # than off `self.cfg`. Two live sources for one threshold is bug 10's exact
        # shape -- there, the local-stage trigger existed as a dataclass default, an
        # argparse default and a documented value at once, and which one applied
        # depended on the entry point. Nothing reads the pipeline's copy today, so
        # this is latent rather than broken; refusing the mismatched pair now is
        # what keeps it that way when someone adds the first read.
        #
        # Only these two sections are constrained, so the genuinely useful case
        # still works: reusing one loaded backbone across pipelines that sweep
        # `geometry` or `local_match` thresholds, which is what the ablation grid
        # needs and what a 35 MB reload per cell would make prohibitive.
        if self.embedder is not None:
            for section, mine, theirs in (
                ("global_match", self.cfg.global_match, self.embedder.cfg),
                ("image", self.cfg.image, self.embedder.image),
            ):
                if mine != theirs:
                    raise PipelineError(
                        f"the injected embedder was built with a different `{section}` "
                        f"configuration than this pipeline's. The thresholds it reports and "
                        f"the ones in cfg.{section} would then disagree, and the report would "
                        f"cite whichever the code happened to read. Build the embedder from "
                        f"the same Settings, reusing `model=` to avoid reloading the backbone."
                    )
        # Read once, at construction. A calibrator re-read per comparison is bug
        # 11 in miniature -- see the note in ``sciforensics.fusion``.
        self.calibrator = fusion.load_calibrator(self.cfg.fusion)

    # -- lazily built, then held -------------------------------------------
    @cached_property
    def _embedder(self) -> Embedder:
        """The backbone, built on first use and reused thereafter.

        Lazy rather than eager so ``sciforensics cmfd`` -- which is a single-image
        question with no embedding in it -- never resolves or downloads a
        checkpoint it will not use. Every pair after the first pays nothing, which
        is the property bug 11 was missing.
        """
        if self.embedder is not None:
            return self.embedder
        return Embedder(
            self.cfg.global_match,
            image=self.cfg.image,
            device=self.torch_device,
        )

    @cached_property
    def detector(self) -> KeypointDetector:
        return build_detector(self.cfg.local_match)

    @cached_property
    def matcher(self) -> Matcher:
        return build_matcher(self.cfg.local_match, norm=self.detector.norm)

    # -- inputs ------------------------------------------------------------
    def load(self, path: str | Path) -> LoadedImage:
        """Load one input under the active ``image`` configuration."""
        return load_image(
            path,
            max_dimension=self.cfg.image.max_dimension,
            max_decoded_pixels=self.cfg.api.max_decoded_pixels,
        )

    # -- audit -------------------------------------------------------------
    def audit(self, *, weighted: bool) -> AuditBlock:
        """Assemble the provenance block for one result.

        ``weighted`` says whether the backbone actually participated. A copy-move
        search never loads it, and stamping a report with the digest of a
        checkpoint that took no part in producing it would be a false claim of
        provenance -- the audit block exists precisely so that a reader can trust
        the opposite.
        """
        path: Path | None = None
        if weighted:
            path = self._embedder.weights_path
            if self._weights_sha256 is None and path is not None:
                self._weights_sha256 = weights.verify(path, expected=None)
        return build_audit_block(
            config_fingerprint=self.cfg.fingerprint(),
            device=str(self.torch_device),
            seed=self.cfg.runtime.seed,
            weights_path=path,
            weights_sha256=self._weights_sha256 if weighted else None,
        )

    # -- pair comparison ---------------------------------------------------
    def compare(
        self,
        left_path: str | Path,
        right_path: str | Path,
        *,
        force_local: bool = False,
        attribution: bool | None = None,
    ) -> PairAnalysis:
        """Compare two images and score the evidence.

        Parameters
        ----------
        force_local
            Run the local stage even when the global stage did not escalate. The
            benchmark needs this: measuring what the keypoint stage *would* have
            found on pairs the screen rejected is the only way to attribute a miss
            to the right stage. It is not a default, because paying for keypoints
            on every pair of a corpus sweep defeats the point of having a screen.
        attribution
            Overrides ``global_match.attribution.enabled``. ``False`` skips the
            backward pass, which is worth doing in a sweep that only needs scores.
        """
        timings: dict[str, float] = {}
        warnings: list[str] = []
        stages: list[Stage] = []

        left = self.load(left_path)
        right = self.load(right_path)
        for image, side in ((left, "left"), (right, "right")):
            if image.meta.was_downscaled:
                warnings.append(
                    f"{side} image was analysed at {image.shape[1]}x{image.shape[0]} "
                    f"(downscaled from {image.meta.width}x{image.meta.height} by "
                    f"image.max_dimension); reported coordinates are in original pixels"
                )

        # --- stage 1: global screen ---------------------------------------
        stages.append(Stage.GLOBAL)
        with _timed(timings, Stage.GLOBAL):
            comparison = self._embedder.compare(left.gray, right.gray, attribution=attribution)

        keypoints: tuple[Detection, Detection] | None = None
        correspondences: Correspondences | None = None
        verification: Verification | None = None

        escalate = comparison.triggered_local or force_local
        if not escalate:
            warnings.append(
                f"local stage skipped: embedding distance {comparison.distance:.3f} is above the "
                f"escalation threshold {comparison.local_trigger_distance:.3f}, so no keypoint "
                f"evidence was gathered. This is an unexamined pair, not a cleared one."
            )
        else:
            # --- stage 2: keypoints and correspondence --------------------
            stages.append(Stage.LOCAL)
            with _timed(timings, Stage.LOCAL):
                keypoints = (self.detector.detect(left.gray), self.detector.detect(right.gray))
                warnings.extend(keypoints[0].warnings)
                warnings.extend(keypoints[1].warnings)
                floor = self._below_keypoint_floor(keypoints)
                if floor is None:
                    correspondences = self.matcher.match(*keypoints)
                else:
                    warnings.append(floor)

            # --- stage 3: geometry ----------------------------------------
            stages.append(Stage.GEOMETRY)
            with _timed(timings, Stage.GEOMETRY):
                verification = (
                    self._verify(left, right, correspondences)
                    if correspondences is not None
                    else _refuse(RejectionReason.TOO_FEW_KEYPOINTS, self.cfg.geometry.method)
                )

        # --- stage 4: fusion ----------------------------------------------
        stages.append(Stage.FUSION)
        with _timed(timings, Stage.FUSION):
            bundle = fusion.EvidenceBundle(
                global_evidence=comparison.evidence,
                keypoints=keypoint_evidence(*keypoints) if keypoints else None,
                matches=correspondences.evidence if correspondences is not None else None,
                geometry=verification.evidence if verification is not None else None,
                reproj_rms_limit=self.cfg.geometry.max_reproj_rms,
            )
            fused = fusion.score(bundle, self.cfg.fusion, calibrator=self.calibrator)

        result = ScanResult(
            left=left.meta,
            right=right.meta,
            global_evidence=bundle.global_evidence,
            keypoints=bundle.keypoints,
            matches=bundle.matches,
            geometry=bundle.geometry,
            verdict=fused.verdict,
            confidence=fused.confidence,
            calibrated=fused.calibrated,
            contributions=fused.contributions,
            notes=fused.notes,
            stages_run=tuple(stages),
            timings=timings,
            warnings=tuple(warnings),
            audit=self.audit(weighted=True),
        )
        _log.info(
            "%s vs %s: %s (%.3f%s) in %.2fs",
            left.meta.path.name,
            right.meta.path.name,
            result.verdict.value,
            result.confidence,
            "" if result.calibrated else ", uncalibrated",
            sum(timings.values()),
        )
        return PairAnalysis(
            result=result,
            left=left,
            right=right,
            comparison=comparison,
            left_keypoints=keypoints[0] if keypoints else None,
            right_keypoints=keypoints[1] if keypoints else None,
            correspondences=correspondences,
            verification=verification,
        )

    def _below_keypoint_floor(self, keypoints: tuple[Detection, Detection]) -> str | None:
        """Refuse to match two sides that were not comparably sampled.

        Returns the diagnostic, or ``None`` when the pair may proceed. Bug 4's
        orchestration-layer gate -- see the module docstring.
        """
        floor = self.cfg.local_match.min_keypoints_per_side
        left, right = keypoints
        if left.count >= floor and right.count >= floor:
            return None
        return (
            f"keypoint floor not met: {left.count} left and {right.count} right survived "
            f"detection, against local_match.min_keypoints_per_side={floor}. Matching was not "
            f"attempted, because correspondences drawn from a starved side collapse onto a "
            f"handful of points and inflate the inlier count without adding constraints."
        )

    def _verify(
        self, left: LoadedImage, right: LoadedImage, correspondences: Correspondences
    ) -> Verification:
        """Fit and gate a transform between the surviving correspondences.

        Areas and diagonals go in as *analysis*-space quantities, which is where
        the gates are calibrated; the scales are what convert the reported
        homography, decomposition and hulls back to original pixels.
        """
        height, width = left.shape
        return verify_with_mask(
            correspondences.src,
            correspondences.dst,
            self.cfg.geometry,
            src_diagonal=left.diagonal,
            dst_diagonal=right.diagonal,
            src_area=float(height * width),
            src_scale=left.analysis_scale,
            dst_scale=right.analysis_scale,
        )

    # -- copy-move ---------------------------------------------------------
    def copy_move(self, path: str | Path) -> CopyMoveAnalysis:
        """Search one image for cloned regions.

        Raises
        ------
        PipelineError
            If ``copy_move.enabled`` is false. A report reading "no cloned regions
            found" from a detector that was switched off is indistinguishable from
            one that looked, and that ambiguity is the class of bug this refactor
            exists to remove.
        """
        if not self.cfg.copy_move.enabled:
            raise PipelineError(
                "copy_move.enabled is false, so no search was performed. Enable it rather than "
                "reading this run as 'nothing found' -- the two are not the same claim."
            )

        timings: dict[str, float] = {}
        warnings: list[str] = []
        image = self.load(path)
        if image.meta.was_downscaled:
            warnings.append(
                f"image was analysed at {image.shape[1]}x{image.shape[0]} (downscaled from "
                f"{image.meta.width}x{image.meta.height} by image.max_dimension); reported "
                f"regions are in original pixels"
            )

        with _timed(timings, Stage.LOCAL):
            detected = self.detector.detect(image.gray)
            warnings.extend(detected.warnings)

        with _timed(timings, Stage.COPY_MOVE):
            detection = detect_copy_move(
                detected,
                self.cfg.copy_move,
                affine=self.cfg.geometry.affine,
                shape=image.shape,
                scale=image.analysis_scale,
                norm=self.detector.norm,
            )

        with _timed(timings, Stage.FUSION):
            bundle = fusion.CopyMoveBundle(
                evidence=detection.evidence,
                image_area=float(image.meta.width * image.meta.height),
                reproj_rms_limit=self.cfg.copy_move.reproj_threshold,
            )
            fused = fusion.score_copy_move(bundle, self.cfg.fusion)

        surviving = len(detection.evidence.regions)
        clusters = detection.evidence.cluster_count
        if clusters > surviving:
            warnings.append(
                f"{clusters} candidate offset cluster{'s' if clusters != 1 else ''} were found but "
                f"{surviving} survived per-cluster geometric verification"
            )

        result = CopyMoveResult(
            image=image.meta,
            evidence=detection.evidence,
            verdict=fused.verdict,
            confidence=fused.confidence,
            calibrated=fused.calibrated,
            contributions=fused.contributions,
            notes=fused.notes,
            timings=timings,
            warnings=tuple(warnings),
            audit=self.audit(weighted=False),
        )
        _log.info(
            "%s: %s (%.3f) in %.2fs",
            image.meta.path.name,
            result.summary_line,
            result.confidence,
            sum(timings.values()),
        )
        return CopyMoveAnalysis(result=result, image=image, keypoints=detected, detection=detection)


def _refuse(reason: RejectionReason, method: str) -> Verification:
    """A geometry refusal for a pair that never reached the estimator.

    Distinct from :func:`sciforensics.local_match.geometry.verify_with_mask`'s own
    rejections in one respect that matters: those describe a fit that was attempted
    and failed a gate, whereas this records that the stage declined to try. Both are
    abstentions to :mod:`sciforensics.fusion.features`, which is correct -- neither
    is evidence *against* reuse.
    """
    return Verification(
        evidence=GeometryEvidence(verified=False, rejection_reason=reason, method=method),
        inlier_mask=np.zeros(0, dtype=bool),
    )
