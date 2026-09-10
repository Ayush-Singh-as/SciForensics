"""Benchmark harness. Stage B6.

The metric definitions are what get tested here, not the pipeline. That is
deliberate: a harness bug does not produce an obviously broken number, it
produces a *plausible wrong* one, and this module found exactly that in its own
first run — ``mountains``/``mountains_manipulated`` share content but do not use
the ``<base>_<suffix>`` naming, so they were paired as unrelated "controls" and a
**correct detection was scored as a false positive**. FPR read 10.0% and
precision 90.9% when the truth was 0.0% and 100.0%. The sign of the headline
number was wrong.

So the tests below pin the classification rules and the arithmetic, with no
model in the loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sciforensics.benchmark import (
    IDENTITY_SUFFIXES,
    KNOWN_POSITIVES,
    SUFFIXES,
    Case,
    Outcome,
    Report,
    correct_correspondences,
    discover_cases,
    latency_percentiles,
    write_results,
)


def _case(positive: bool, manipulation: str = "", label: str = "case") -> Case:
    return Case(
        left=Path("a.png"),
        right=Path("b.png"),
        positive=positive,
        manipulation=manipulation,
        label=label,
    )


def _outcome(positive: bool, detected: bool, **kwargs: object) -> Outcome:
    defaults: dict[str, object] = {
        "verdict": "likely_manipulated" if detected else "clean",
        "confidence": 0.9 if detected else 0.1,
        "matches": 100,
        "correct": None,
        "rejection": "",
        "seconds": 0.5,
    }
    defaults.update(kwargs)
    return Outcome(
        case=_case(positive, str(defaults.pop("manipulation", ""))),
        detected=detected,
        **defaults,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# case discovery -- where the sign error lived
# ---------------------------------------------------------------------------
def _corpus(root: Path, bases: list[str], suffixes: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for base in bases:
        (root / f"{base}.png").write_bytes(f"base-{base}".encode())
        for suffix in suffixes:
            (root / f"{base}_{suffix}.png").write_bytes(f"{base}-{suffix}".encode())


def test_positives_come_from_the_suffix_naming(tmp_path: Path) -> None:
    _corpus(tmp_path, ["alpha", "beta"], ["ex2_degraded", "ex4_exposure"])
    cases = discover_cases(tmp_path)

    positives = [c for c in cases if c.positive]
    assert len(positives) == 4
    assert {c.manipulation for c in positives} == {"ex2_degraded", "ex4_exposure"}


def test_controls_are_every_cross_base_pair(tmp_path: Path) -> None:
    """The measurement `run_demo.py` structurally could not make.

    It paired each base only with its own manipulations, so there was not one
    unrelated pair and therefore no false-positive rate.
    """
    _corpus(tmp_path, ["alpha", "beta", "gamma"], ["ex2_degraded"])
    controls = [c for c in discover_cases(tmp_path) if not c.positive]

    # 3 choose 2.
    assert len(controls) == 3
    assert all("control" in c.label for c in controls)


def test_known_related_pairs_are_not_treated_as_controls(tmp_path: Path) -> None:
    """The bug this harness found in itself.

    `mountains`/`mountains_manipulated` share content but use neither the
    `<base>_<suffix>` naming nor a common stem prefix that discovery would
    notice. Paired as a control, a *correct* detection was counted as a false
    positive: FPR 10.0% and precision 90.9% instead of 0.0% and 100.0%.
    """
    stem, partner = next(iter(KNOWN_POSITIVES.items()))
    (tmp_path / f"{stem}.png").write_bytes(b"left")
    (tmp_path / f"{partner}.png").write_bytes(b"right")
    (tmp_path / "unrelated.png").write_bytes(b"other")

    cases = discover_cases(tmp_path)
    labels = {c.label: c for c in cases}

    assert labels[f"{stem} -> {partner}"].positive is True

    # Neither member may appear in a control pair -- with `unrelated` or with
    # each other.
    for case in cases:
        if not case.positive:
            assert stem not in case.label
            assert partner not in case.label


def test_empty_corpus_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no base images"):
        discover_cases(tmp_path)


def test_identity_suffixes_are_a_subset_of_known_suffixes() -> None:
    """Correspondence precision is only checkable where the transform is known."""
    assert set(SUFFIXES) >= IDENTITY_SUFFIXES
    # A geometric manipulation must never be treated as identity.
    assert "ex1_affine" not in IDENTITY_SUFFIXES
    assert "ex3_copymove" not in IDENTITY_SUFFIXES


# ---------------------------------------------------------------------------
# correspondence correctness -- the measurement that changed a conclusion
# ---------------------------------------------------------------------------
def test_correct_correspondences_counts_only_near_identity() -> None:
    """Why match count is not the metric.

    DISK produced 25 matches on `base_cell -> ex2_degraded` against ORB's 4, and
    **zero** of the 25 were geometrically correct. A harness counting matches
    would have reported a 6x improvement while the pipeline got worse.
    """
    src = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    dst = np.array([[10.5, 10.2], [21.0, 19.8], [90.0, 12.0], [40.1, 41.9]])

    # Max-axis displacements: 0.5, 1.0, 60.0, 1.9. The bound is strict (`<`),
    # so a tolerance of exactly 1.0 excludes the 1.0 pair.
    assert correct_correspondences(src, dst, tolerance_px=3.0) == 3
    assert correct_correspondences(src, dst, tolerance_px=1.5) == 2
    assert correct_correspondences(src, dst, tolerance_px=1.0) == 1
    assert correct_correspondences(src, dst, tolerance_px=0.1) == 0


def test_correct_correspondences_handles_no_matches() -> None:
    empty = np.zeros((0, 2))
    assert correct_correspondences(empty, empty, tolerance_px=3.0) == 0


# ---------------------------------------------------------------------------
# metric arithmetic
# ---------------------------------------------------------------------------
def test_recall_counts_detected_positives() -> None:
    report = Report(backend="test")
    report.outcomes = [
        _outcome(True, True),
        _outcome(True, True),
        _outcome(True, False),
        _outcome(False, False),
    ]
    assert report.recall() == pytest.approx(2 / 3)


def test_false_positive_rate_is_over_controls_only() -> None:
    report = Report(backend="test")
    report.outcomes = [
        _outcome(True, False),  # a missed positive must not affect FPR
        _outcome(False, True),
        _outcome(False, False),
        _outcome(False, False),
        _outcome(False, False),
    ]
    assert report.false_positive_rate() == pytest.approx(0.25)


def test_false_positive_rate_is_nan_without_controls() -> None:
    """A recall figure with no controls is not a result, and must not read as 0%."""
    report = Report(backend="test")
    report.outcomes = [_outcome(True, True)]
    assert np.isnan(report.false_positive_rate())


def test_precision_is_over_detections() -> None:
    report = Report(backend="test")
    report.outcomes = [
        _outcome(True, True),
        _outcome(True, True),
        _outcome(False, True),  # one false positive
        _outcome(True, False),
    ]
    assert report.precision() == pytest.approx(2 / 3)


def test_precision_is_nan_when_nothing_was_detected() -> None:
    report = Report(backend="test")
    report.outcomes = [_outcome(True, False), _outcome(False, False)]
    assert np.isnan(report.precision())


def test_recall_by_manipulation_partitions_positives() -> None:
    report = Report(backend="test")
    report.outcomes = [
        _outcome(True, True, manipulation="ex2_degraded"),
        _outcome(True, False, manipulation="ex2_degraded"),
        _outcome(True, True, manipulation="ex4_exposure"),
        _outcome(False, False),
    ]
    breakdown = report.recall_by_manipulation()
    assert breakdown["ex2_degraded"] == (1, 2)
    assert breakdown["ex4_exposure"] == (1, 1)
    assert "" not in breakdown, "controls must not appear in a recall breakdown"


def test_outcome_precision_is_none_without_ground_truth() -> None:
    assert _outcome(True, True, correct=None).precision is None
    assert _outcome(True, True, matches=0, correct=0).precision is None
    assert _outcome(True, True, matches=10, correct=4).precision == pytest.approx(0.4)


def test_latency_percentiles_are_ordered() -> None:
    report = Report(backend="test")
    report.outcomes = [_outcome(True, True, seconds=float(s)) for s in range(1, 101)]
    pct = latency_percentiles(report)
    assert pct["p50"] <= pct["p90"] <= pct["p99"]


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def test_write_results_emits_json_and_markdown(tmp_path: Path) -> None:
    """Both, from one object, so the prose cannot drift from the numbers.

    Documentation drift is what the original audit found throughout the
    prototype (a documented `grad_loc.py` that did not exist, a documented eval
    path that failed, "~1.2M parameters" against a real 8.8M).
    """
    report = Report(backend="orb+mutual_nn")
    report.outcomes = [
        _outcome(True, True, manipulation="ex2_degraded", matches=110, correct=88),
        _outcome(False, False),
    ]
    json_path, md_path = write_results([report], tmp_path / "out")

    assert json_path.is_file() and md_path.is_file()
    markdown = md_path.read_text(encoding="utf-8")
    assert "orb+mutual_nn" in markdown
    # The controls must be visible in the rendered table, not just the JSON.
    assert "Controls" in markdown
    assert "do not edit by hand" in markdown
