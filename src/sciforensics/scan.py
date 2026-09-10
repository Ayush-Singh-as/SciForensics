"""Manuscript-wide scanning. Stages C1 + C2.

Compares every panel in a manuscript against every other one, which is the
actual research-integrity workflow: an editor has a submission, not a curated
pair of PNGs.

**The quadratic problem, and why a prefilter is not optional.** A 40-page
manuscript yields on the order of 200 panels, so 19,900 pairs. At the measured
0.30 s per pair (ORB, one core) that is **100 minutes**. Perceptual hashing
reduces it to a screen: pHash is a 64-bit descriptor comparable in nanoseconds,
so all 19,900 pairs are triaged instantly and only the survivors reach the
pipeline. On a manuscript with no reuse that is typically a handful.

**A prefilter is only sound if it cannot discard a true positive, and the first
version of this one did.** pHash is invariant to compression, mild blur and
brightness; it is *not* invariant to rotation or reflection, which are precisely
the manipulations this tool hunts. The scan fixture plants a panel reused at 180
degrees, and plain pHash scored that pair at Hamming **34** -- above even a gate
set at 32, half the code length. The true positive was screened out before the
pipeline saw it and the manuscript came back clean.

Two things follow, both load-bearing:

* Hashes are computed over **all eight dihedral orientations** and compared by
  minimum (:func:`dihedral_phashes`). The same pair then scores **0**. The
  dihedral group is finite, so this is exact rather than a loosened threshold.
* Coverage is **always reported**. Every scan states how many pairs were
  screened out, because a prefilter buys latency at a recall cost and that cost
  must be visible. ``hamming_max=64`` disables it and compares exhaustively.

Crop and scale remain outside pHash's invariance, so the gate stays loose rather
than tuned tight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sciforensics.config import Settings
from sciforensics.runtime import get_logger
from sciforensics.types import Verdict

if TYPE_CHECKING:  # pragma: no cover - annotation only
    pass

_log = get_logger(__name__)

#: Hamming distance on a 64-bit pHash below which a pair is *not* screened out.
#: 32 is half the code length -- essentially chance -- because pHash is blind to
#: rotation and reflection, and those are precisely the manipulations we hunt.
#: A tight threshold would be fast and would discard true positives.
DEFAULT_HAMMING_MAX = 32


@dataclass(frozen=True)
class Candidate:
    """One panel pair that survived the prefilter."""

    left: Path
    right: Path
    hamming: int


@dataclass
class Finding:
    """A scored panel pair."""

    left: str
    right: str
    verdict: str
    confidence: float
    hamming: int
    matches: int
    inliers: int
    rejection: str
    #: Recorded so a finding can be discounted when either panel came from a
    #: doubtful split, or from a re-rendered rather than embedded figure.
    trustworthy: bool = True

    @property
    def flagged(self) -> bool:
        return self.verdict in {Verdict.LIKELY_MANIPULATED.value, Verdict.SUSPICIOUS.value}


@dataclass
class ScanReport:
    """The outcome of scanning one manuscript."""

    source: Path
    figures: int
    panels: int
    #: Pairs before the prefilter -- the exhaustive count.
    pairs_total: int
    #: Pairs that survived it and were actually analysed.
    pairs_analysed: int
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def flagged(self) -> list[Finding]:
        return sorted(
            (f for f in self.findings if f.flagged),
            key=lambda f: f.confidence,
            reverse=True,
        )

    @property
    def screened_out(self) -> int:
        return self.pairs_total - self.pairs_analysed

    @property
    def coverage(self) -> float:
        """Fraction of possible pairs actually analysed.

        Reported rather than assumed. A scan that screened out 98% of pairs has
        bought its speed with recall, and the reader is entitled to know.
        """
        if self.pairs_total == 0:
            return 1.0
        return self.pairs_analysed / self.pairs_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "figures": self.figures,
            "panels": self.panels,
            "pairs_total": self.pairs_total,
            "pairs_analysed": self.pairs_analysed,
            "pairs_screened_out": self.screened_out,
            "coverage": self.coverage,
            "seconds": self.seconds,
            "flagged": len(self.flagged),
            "findings": [
                {
                    "left": f.left,
                    "right": f.right,
                    "verdict": f.verdict,
                    "confidence": f.confidence,
                    "hamming": f.hamming,
                    "matches": f.matches,
                    "inliers": f.inliers,
                    "rejection": f.rejection,
                    "trustworthy": f.trustworthy,
                }
                for f in self.findings
            ],
            "warnings": self.warnings,
        }


def _phash_of(image: Any, *, hash_size: int) -> int:
    import imagehash

    return int(str(imagehash.phash(image, hash_size=hash_size)), 16)


def phash(path: str | Path, *, hash_size: int = 8) -> int:
    """64-bit perceptual hash as an int.

    DCT-based (``imagehash.phash``): robust to recompression and mild blur,
    which is what a prefilter needs. **Not** robust to rotation or reflection --
    see :func:`dihedral_phashes`.
    """
    from PIL import Image

    try:
        import imagehash  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            f"ImageHash is not installed ({exc}); pip install 'sciforensics[report]'"
        ) from exc

    with Image.open(path) as handle:
        return _phash_of(handle.convert("L"), hash_size=hash_size)


def dihedral_phashes(path: str | Path, *, hash_size: int = 8) -> tuple[int, ...]:
    """pHash under all eight dihedral orientations of the square.

    **This is not a refinement, it is a correctness fix, and it was caught by
    measurement.** The scan fixture plants a panel reused at 180 degrees -- the
    single most common figure manipulation. Plain pHash put that pair at Hamming
    **34**, above the (already loose) gate of 32, so the prefilter *discarded the
    true positive before the pipeline ever saw it* and the scan reported the
    manuscript clean. Undoing the rotation first put the same pair at **0**.

    A prefilter that silently drops the manipulation class the tool exists to
    detect is worse than no prefilter. pHash cannot be made rotation-invariant,
    but the dihedral group is finite and tiny: 8 hashes per panel, computed
    once, and :func:`dihedral_distance` takes the minimum over them. Cost is 8x
    a nanosecond-scale operation; the alternative was a false negative.

    Reflections are included because mirroring is as common as rotation, and
    ``det < 0`` is exactly what bug 2 existed to make detectable.
    """
    from PIL import Image

    try:
        import imagehash  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            f"ImageHash is not installed ({exc}); pip install 'sciforensics[report]'"
        ) from exc

    with Image.open(path) as handle:
        gray = handle.convert("L")
        variants = [
            gray,
            gray.transpose(Image.Transpose.ROTATE_90),
            gray.transpose(Image.Transpose.ROTATE_180),
            gray.transpose(Image.Transpose.ROTATE_270),
            gray.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            gray.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            gray.transpose(Image.Transpose.TRANSPOSE),
            gray.transpose(Image.Transpose.TRANSVERSE),
        ]
        return tuple(_phash_of(variant, hash_size=hash_size) for variant in variants)


def hamming(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


def dihedral_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Smallest Hamming distance over the orientations of either panel.

    Comparing every orientation of one against the *identity* of the other is
    sufficient -- the dihedral group is closed, so any relative orientation is
    already covered -- but taking the full minimum costs nothing and removes the
    need to reason about which side was transformed.
    """
    return min(hamming(a, b) for a in left for b in right)


def prefilter(
    paths: list[Path], *, hamming_max: int = DEFAULT_HAMMING_MAX
) -> tuple[list[Candidate], int]:
    """Triage all pairs by pHash. Returns ``(candidates, total_pairs)``.

    ``hamming_max >= 64`` disables the filter, so an exhaustive scan is a
    configuration rather than a different code path.
    """
    hashes: dict[Path, tuple[int, ...]] = {}
    for path in paths:
        try:
            # All eight orientations, so a rotated or mirrored reuse is not
            # screened out. See `dihedral_phashes` -- plain pHash put the
            # fixture's 180-degree duplicate at Hamming 34 and discarded it.
            hashes[path] = dihedral_phashes(path)
        except Exception as exc:
            _log.warning("could not hash %s: %s", path, exc)

    usable = sorted(hashes)
    total = len(usable) * (len(usable) - 1) // 2

    candidates = [
        Candidate(left=a, right=b, hamming=distance)
        for a, b in combinations(usable, 2)
        if (distance := dihedral_distance(hashes[a], hashes[b])) <= hamming_max
    ]
    _log.info("prefilter kept %d of %d pairs (hamming <= %d)", len(candidates), total, hamming_max)
    return candidates, total


def scan_manuscript(
    pdf: str | Path,
    cfg: Settings,
    *,
    workdir: str | Path,
    device: str = "cpu",
    hamming_max: int = DEFAULT_HAMMING_MAX,
    max_pairs: int | None = 500,
) -> ScanReport:
    """Extract, split, prefilter and compare every surviving panel pair.

    ``max_pairs`` bounds the work after prefiltering. When it truncates, the
    fact lands in :attr:`ScanReport.warnings` and in ``coverage`` -- a
    silently-truncated scan reporting "no findings" would be indistinguishable
    from a clean manuscript.
    """
    import time

    from sciforensics.io.pdf import ingest, write
    from sciforensics.pipeline import Pipeline

    started = time.perf_counter()
    workdir = Path(workdir)

    manuscript = ingest(pdf, split=True)
    write(manuscript, workdir, panels=True)

    panel_dir = workdir / "panels"
    panels = sorted(panel_dir.glob("*.png"))

    # `_unsplit` in the filename marks a panel whose split was not confident;
    # `_render` marks a figure that was rasterised rather than embedded.
    doubtful = {p.name for p in panels if "_unsplit" in p.name}

    report = ScanReport(
        source=Path(pdf),
        figures=len(manuscript.figures),
        panels=len(panels),
        pairs_total=0,
        pairs_analysed=0,
        warnings=list(manuscript.warnings),
    )

    if manuscript.evidential_fraction < 1.0:
        report.warnings.append(
            f"{(1 - manuscript.evidential_fraction) * 100:.0f}% of figures were rasterised rather "
            "than extracted losslessly; comparisons on those ran on resampled pixels"
        )

    candidates, total = prefilter(panels, hamming_max=hamming_max)
    report.pairs_total = total

    if max_pairs is not None and len(candidates) > max_pairs:
        report.warnings.append(
            f"{len(candidates)} pairs survived the prefilter but only {max_pairs} were analysed "
            f"(--max-pairs); this scan is not exhaustive"
        )
        candidates = sorted(candidates, key=lambda c: c.hamming)[:max_pairs]

    pipeline = Pipeline(cfg, device=device)
    for candidate in candidates:
        try:
            analysis = pipeline.compare(candidate.left, candidate.right)
        except Exception as exc:
            _log.warning("pair %s/%s failed: %s", candidate.left.name, candidate.right.name, exc)
            report.warnings.append(
                f"{candidate.left.name} vs {candidate.right.name}: analysis failed ({exc})"
            )
            continue

        result = analysis.result
        geometry = result.geometry
        report.findings.append(
            Finding(
                left=candidate.left.name,
                right=candidate.right.name,
                verdict=result.verdict.value,
                confidence=result.confidence,
                hamming=candidate.hamming,
                matches=result.matches.good if result.matches else 0,
                inliers=geometry.inlier_count if geometry else 0,
                rejection=(
                    geometry.rejection_reason.value if geometry and not geometry.verified else ""
                ),
                trustworthy=(
                    candidate.left.name not in doubtful and candidate.right.name not in doubtful
                ),
            )
        )
        report.pairs_analysed += 1

    report.seconds = time.perf_counter() - started
    _log.info(
        "scanned %s: %d panels, %d/%d pairs analysed, %d flagged in %.1fs",
        Path(pdf).name,
        report.panels,
        report.pairs_analysed,
        report.pairs_total,
        len(report.flagged),
        report.seconds,
    )
    return report


def build_index(paths: list[Path]) -> dict[str, str]:
    """pHash index over a corpus, as ``{relative path: hex hash}``.

    Deliberately a dict, not FAISS. FAISS indexes *dense embedding vectors* and
    earns its keep past ~100k items; a pHash corpus is 64-bit integers, and
    below that scale a linear scan over ints is faster than the index build.
    Reaching for it here would be complexity bought with nothing.

    ``ponytail: linear pHash scan -- add a FAISS embedding index when a corpus
    exceeds ~100k panels and needs semantic rather than perceptual recall.``
    """
    index: dict[str, str] = {}
    for path in paths:
        try:
            # Store the *canonical* orientation hash -- the numerically smallest
            # of the eight -- so two panels differing only by rotation or
            # reflection land on the same key. Storing the as-is hash instead
            # would make retrieval blind to exactly the reuse the scan path was
            # fixed to catch.
            index[path.as_posix()] = f"{min(dihedral_phashes(path)):016x}"
        except Exception as exc:
            _log.warning("could not hash %s: %s", path, exc)
    return index


def query_index(
    index: dict[str, str],
    target: str | Path,
    *,
    hamming_max: int = DEFAULT_HAMMING_MAX,
    k: int = 10,
) -> list[tuple[str, int]]:
    """Nearest entries to ``target`` by pHash Hamming distance.

    The needle is canonicalised to the same smallest-of-eight orientation the
    index stores, so a query with a rotated or mirrored copy still retrieves its
    original. Hashing the needle as-is would make retrieval miss precisely the
    reuse `prefilter` was fixed to catch.
    """
    needle = min(dihedral_phashes(target))
    scored = [
        (name, distance)
        for name, value in index.items()
        if (distance := hamming(needle, int(value, 16))) <= hamming_max
    ]
    return sorted(scored, key=lambda pair: pair[1])[:k]


def latency_note(report: ScanReport) -> str:
    """One line quantifying what the prefilter bought, for the CLI and reports."""
    if report.pairs_total == 0:
        return "no pairs to compare"
    saved = report.screened_out
    return (
        f"{report.pairs_analysed} of {report.pairs_total} pairs analysed "
        f"({report.coverage:.1%} coverage); {saved} screened out by pHash"
    )


def summarise(report: ScanReport) -> np.ndarray:
    """Confidences of every analysed pair, for a histogram or a threshold sweep."""
    return np.asarray([f.confidence for f in report.findings], dtype=np.float64)
