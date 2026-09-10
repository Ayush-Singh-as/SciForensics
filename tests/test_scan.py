"""Manuscript scanning and pHash retrieval. Stages C1 + C2.

**The prefilter bug this file exists to prevent from recurring.** The first
version hashed each panel once and gated on Hamming distance. The scan fixture
plants a panel reused at 180 degrees — the single most common figure
manipulation — and plain pHash scored that pair at Hamming **34**, above even a
gate set at 32 (half the code length). The true positive was screened out
*before the pipeline ever saw it* and the manuscript came back clean, with no
warning, because from the report's point of view nothing had gone wrong.

Undoing the rotation put the same pair at **0**. So hashes are now taken over
all eight dihedral orientations and compared by minimum.

A prefilter that silently drops the manipulation class the tool exists to detect
is worse than no prefilter, and it fails *quietly* — which is why the guard is a
test rather than a comment.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("imagehash", reason="retrieval needs ImageHash ([report] extra)")

from sciforensics.scan import (
    DEFAULT_HAMMING_MAX,
    Finding,
    ScanReport,
    build_index,
    dihedral_distance,
    dihedral_phashes,
    hamming,
    latency_note,
    phash,
    prefilter,
    query_index,
)


@pytest.fixture
def panel(tmp_path: Path) -> Path:
    """A textured panel. Texture matters: pHash on flat noise is unstable."""
    rng = np.random.default_rng(99)
    noise = rng.integers(30, 220, (160, 200), dtype=np.uint8)
    image: np.ndarray = cv2.GaussianBlur(noise, (0, 0), 1.2)
    cv2.circle(image, (60, 50), 26, 240, -1)
    cv2.rectangle(image, (120, 90), (180, 140), 40, -1)
    path = tmp_path / "panel.png"
    cv2.imwrite(str(path), image)
    return path


def _read(path: Path) -> np.ndarray:
    """`cv2.imread` returns None on failure rather than raising."""
    image = cv2.imread(str(path))
    assert image is not None, f"could not read {path}"
    return image


def _variant(source: Path, destination: Path, transform: int | None) -> Path:
    image = _read(source)
    if transform is not None:
        image = cv2.rotate(image, transform)
    cv2.imwrite(str(destination), image)
    return destination


# ---------------------------------------------------------------------------
# the prefilter's soundness
# ---------------------------------------------------------------------------
def test_plain_phash_cannot_see_a_rotated_duplicate(panel: Path, tmp_path: Path) -> None:
    """The defect itself, reproduced.

    Asserting the failure is deliberate: it documents why the dihedral variant
    is required, and fails loudly if someone concludes plain pHash was good
    enough after all.

    The bound is stated as "no better than chance" rather than a specific
    number. On the real scan fixture plain pHash scored 34, above the gate of
    32; on this synthetic panel it scores exactly 32, at the boundary. Either
    way it carries **no usable signal** about a rotated duplicate -- a pair that
    is byte-identical up to rotation should be near 0, and half the code length
    is what two unrelated images score. Pinning the exact value would make the
    test a fixture detail rather than a statement about the method.
    """
    rotated = _variant(panel, tmp_path / "rot180.png", cv2.ROTATE_180)
    plain = hamming(phash(panel), phash(rotated))
    dihedral = dihedral_distance(dihedral_phashes(panel), dihedral_phashes(rotated))

    assert plain >= DEFAULT_HAMMING_MAX, (
        f"plain pHash scored the rotated duplicate at {plain}, comfortably inside the "
        f"gate of {DEFAULT_HAMMING_MAX} -- if this now passes, the fixture no longer "
        "reproduces the bug"
    )
    # The comparison that matters: the same pair, seen correctly.
    assert dihedral == 0
    assert plain - dihedral >= DEFAULT_HAMMING_MAX


@pytest.mark.parametrize(
    "transform",
    [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE],
)
def test_dihedral_phash_finds_a_rotated_duplicate(
    panel: Path, tmp_path: Path, transform: int
) -> None:
    """The fix: every 90-degree rotation must land at distance 0."""
    rotated = _variant(panel, tmp_path / f"rot{transform}.png", transform)
    distance = dihedral_distance(dihedral_phashes(panel), dihedral_phashes(rotated))
    assert distance == 0, distance


def test_dihedral_phash_finds_a_mirrored_duplicate(panel: Path, tmp_path: Path) -> None:
    """Mirroring is as common as rotation, and `det < 0` is bug 2's whole point."""
    mirrored = tmp_path / "mirror.png"
    cv2.imwrite(str(mirrored), cv2.flip(_read(panel), 1))
    distance = dihedral_distance(dihedral_phashes(panel), dihedral_phashes(mirrored))
    assert distance == 0, distance


def test_dihedral_phashes_returns_the_whole_group(panel: Path) -> None:
    hashes = dihedral_phashes(panel)
    assert len(hashes) == 8, "the dihedral group of the square has eight elements"


def test_prefilter_keeps_a_rotated_pair(panel: Path, tmp_path: Path) -> None:
    """End to end through `prefilter`, which is where the loss happened."""
    rotated = _variant(panel, tmp_path / "b.png", cv2.ROTATE_180)
    candidates, total = prefilter([panel, rotated])

    assert total == 1
    assert len(candidates) == 1, "the rotated duplicate was screened out again"
    assert candidates[0].hamming == 0


def test_prefilter_screens_out_unrelated_panels(tmp_path: Path) -> None:
    """The filter must still filter, or it buys nothing."""
    rng = np.random.default_rng(7)
    paths = []
    for index in range(4):
        # Structured but mutually unrelated: distinct shapes, not just noise,
        # since pHash on pure noise clusters near the middle of the space.
        base = np.full((160, 200), 20 + index * 50, np.uint8)
        cv2.circle(base, (40 + index * 30, 50 + index * 20), 20 + index * 8, 250, -1)
        cv2.rectangle(base, (10, 110), (60 + index * 30, 150), 120, -1)
        image: np.ndarray = cv2.add(base, rng.integers(0, 20, base.shape, dtype=np.uint8))
        path = tmp_path / f"u{index}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)

    candidates, total = prefilter(paths, hamming_max=8)
    assert total == 6
    assert len(candidates) < total, "a gate of 8 should reject some unrelated pairs"


def test_prefilter_can_be_disabled(panel: Path, tmp_path: Path) -> None:
    """`hamming_max=64` is an exhaustive scan, not a separate code path."""
    others = [_variant(panel, tmp_path / f"v{i}.png", None) for i in range(3)]
    candidates, total = prefilter([panel, *others], hamming_max=64)
    assert len(candidates) == total == 6


def test_unreadable_file_is_skipped_not_fatal(panel: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    candidates, total = prefilter([panel, broken])
    # Only the readable panel survives, so there is no pair at all.
    assert total == 0
    assert candidates == []


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
def test_index_retrieval_is_orientation_invariant(panel: Path, tmp_path: Path) -> None:
    """A query with a rotated copy must retrieve the original.

    The index stores the canonical (smallest) dihedral hash and the query
    canonicalises the needle the same way. Hashing either as-is would make
    retrieval blind to exactly the reuse `prefilter` was fixed to catch.
    """
    rotated = _variant(panel, tmp_path / "rotated.png", cv2.ROTATE_90_CLOCKWISE)
    index = build_index([panel])

    hits = query_index(index, rotated, hamming_max=4)
    assert hits, "the rotated query found nothing"
    assert Path(hits[0][0]).name == panel.name
    assert hits[0][1] == 0


def test_index_skips_unreadable_entries(panel: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"nope")
    index = build_index([panel, broken])
    assert len(index) == 1


def test_query_respects_the_distance_gate(panel: Path, tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.png"
    image = np.zeros((160, 200), np.uint8)
    cv2.rectangle(image, (0, 0), (199, 79), 255, -1)
    cv2.imwrite(str(unrelated), image)

    index = build_index([unrelated])
    assert query_index(index, panel, hamming_max=0) == []


# ---------------------------------------------------------------------------
# reporting -- coverage must never be silently omitted
# ---------------------------------------------------------------------------
def _finding(confidence: float, verdict: str) -> Finding:
    return Finding(
        left="a.png",
        right="b.png",
        verdict=verdict,
        confidence=confidence,
        hamming=0,
        matches=100,
        inliers=90,
        rejection="",
    )


def test_coverage_reports_what_the_prefilter_cost() -> None:
    """A scan reporting "nothing found" over 12% coverage means something very
    different from one over 100%, and the difference must be visible."""
    report = ScanReport(
        source=Path("m.pdf"), figures=2, panels=10, pairs_total=45, pairs_analysed=6
    )
    assert report.screened_out == 39
    assert report.coverage == pytest.approx(6 / 45)
    assert "6 of 45" in latency_note(report)
    assert "39 screened out" in latency_note(report)


def test_flagged_excludes_clean_and_sorts_by_confidence() -> None:
    report = ScanReport(source=Path("m.pdf"), figures=1, panels=4, pairs_total=6, pairs_analysed=6)
    report.findings = [
        _finding(0.10, "clean"),
        _finding(0.95, "likely_manipulated"),
        _finding(0.62, "suspicious"),
        _finding(0.40, "inconclusive"),
    ]
    flagged = report.flagged
    assert [f.verdict for f in flagged] == ["likely_manipulated", "suspicious"]
    assert flagged[0].confidence > flagged[1].confidence


def test_empty_scan_reports_full_coverage() -> None:
    """Zero pairs is 100% covered, not 0% -- there was nothing to miss."""
    report = ScanReport(source=Path("m.pdf"), figures=1, panels=1, pairs_total=0, pairs_analysed=0)
    assert report.coverage == 1.0
    assert "no pairs" in latency_note(report)


def test_report_serialises_coverage_and_trust() -> None:
    report = ScanReport(source=Path("m.pdf"), figures=2, panels=5, pairs_total=10, pairs_analysed=4)
    report.findings = [_finding(0.9, "likely_manipulated")]
    payload = report.to_dict()
    assert payload["pairs_screened_out"] == 6
    assert payload["coverage"] == pytest.approx(0.4)
    assert payload["findings"][0]["trustworthy"] is True
