"""Benchmark harness: recall, and the precision the project could not state.

**The gap this closes.** ``run_demo.py`` paired each base image only with *its
own* manipulations — not one unrelated pair. A tool evaluated that way cannot
report a false-positive rate at all, so "detects manipulation" was an
unfalsifiable claim. Every run here builds the negative controls first, from
**all cross-base pairs**, and reports precision alongside recall. A recall
number without them is not a result.

**Correctness is measured against the transform, not the match count.** This is
not pedantry — it is the finding that changed a conclusion. On the degradation
sweep, DISK+LightGlue produces more matches than ORB on two of three pairs, and
**zero** of those extra matches are geometrically correct:

    pair              backend  matches  correct  precision
    base_cell         orb            4        2      50.0%
    base_cell         disk          25        0       0.0%
    base_cells_2      orb           34        1       2.9%
    base_cells_2      disk           5        0       0.0%

A harness that counted matches would have reported DISK as a 6x improvement on
``base_cell`` while the pipeline got strictly worse. The generated pairs carry a
known transform, so ``correct`` is checkable: a correspondence is correct when it
lands within ``tolerance_px`` of where the true transform sends it.

**Detection is the end-to-end outcome**, taken from the pipeline's own verdict
rather than recomputed here, so the benchmark cannot disagree with the tool.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from sciforensics.audit import build_audit_block
from sciforensics.config import Settings
from sciforensics.runtime import get_logger
from sciforensics.types import Verdict

_log = get_logger(__name__)

#: How the generated examples are named, and what transform each carries.
#: `_ex2_degraded` is signal-only, so its true transform is the identity --
#: which is what makes "correct correspondence" checkable at all.
SUFFIXES: dict[str, str] = {
    "ex1_affine": "45 deg rotation, 1.2x scale, horizontal mirror",
    "ex2_degraded": "JPEG q25 + Gaussian noise (no geometric change)",
    "ex3_copymove": "intra-image clone of a 25%-side patch",
    "ex4_exposure": "contrast 1.8x, brightness 0.5x (no geometric change)",
    "ex5_blackout": "region blacked out",
}

#: Suffixes whose true image-to-image transform is the identity. Only these can
#: score correspondence precision without solving for the transform first.
IDENTITY_SUFFIXES = frozenset({"ex2_degraded", "ex4_exposure", "ex5_blackout"})


@dataclass(frozen=True)
class Case:
    """One pair to evaluate, and whether it *should* be detected."""

    left: Path
    right: Path
    #: True when the two images share content by construction.
    positive: bool
    #: ``""`` for negative controls.
    manipulation: str
    label: str

    @property
    def identity(self) -> bool:
        return self.manipulation in IDENTITY_SUFFIXES


@dataclass
class Outcome:
    """What the pipeline said about one case."""

    case: Case
    detected: bool
    verdict: str
    confidence: float
    matches: int
    #: Correspondences landing within tolerance of the true transform. ``None``
    #: when the transform is not known in closed form.
    correct: int | None
    rejection: str
    seconds: float

    @property
    def precision(self) -> float | None:
        if self.correct is None or self.matches == 0:
            return None
        return self.correct / self.matches


@dataclass
class Report:
    """Aggregated metrics plus every individual outcome."""

    backend: str
    outcomes: list[Outcome] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def positives(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.case.positive]

    @property
    def negatives(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.case.positive]

    def recall(self) -> float:
        hits = [o for o in self.positives if o.detected]
        return len(hits) / len(self.positives) if self.positives else 0.0

    def false_positive_rate(self) -> float:
        """The number ``run_demo.py`` structurally could not produce."""
        if not self.negatives:
            return float("nan")
        return sum(1 for o in self.negatives if o.detected) / len(self.negatives)

    def precision(self) -> float:
        """Over *detections*, not over correspondences."""
        detected = [o for o in self.outcomes if o.detected]
        if not detected:
            return float("nan")
        return sum(1 for o in detected if o.case.positive) / len(detected)

    def recall_by_manipulation(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for outcome in self.positives:
            hits, total = out.get(outcome.case.manipulation, (0, 0))
            out[outcome.case.manipulation] = (hits + int(outcome.detected), total + 1)
        return dict(sorted(out.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "audit": self.audit,
            "summary": {
                "positives": len(self.positives),
                "negatives": len(self.negatives),
                "recall": self.recall(),
                "false_positive_rate": self.false_positive_rate(),
                "precision": self.precision(),
                "recall_by_manipulation": {
                    key: {"detected": hits, "total": total}
                    for key, (hits, total) in self.recall_by_manipulation().items()
                },
            },
            "cases": [
                {
                    "label": o.case.label,
                    "positive": o.case.positive,
                    "manipulation": o.case.manipulation,
                    "detected": o.detected,
                    "verdict": o.verdict,
                    "confidence": o.confidence,
                    "matches": o.matches,
                    "correct_correspondences": o.correct,
                    "correspondence_precision": o.precision,
                    "rejection": o.rejection,
                    "seconds": o.seconds,
                }
                for o in self.outcomes
            ],
        }


#: Pairs that share content but do not follow the ``<base>_<suffix>`` naming.
#: Without this, `discover_cases` treated `mountains`/`mountains_manipulated` as
#: two unrelated bases and scored a correct detection on it as a *false
#: positive* -- inverting the sign of the headline metric. Ground truth for this
#: pair was established by correlation in Stage A: it is a 180-degree rotation.
KNOWN_POSITIVES: dict[str, str] = {
    "mountains": "mountains_manipulated",
}


def discover_cases(inputs: Path) -> list[Case]:
    """Build positives from ``<base>_<suffix>`` and negatives from cross-base pairs.

    The negatives are the point. Every distinct base image is unrelated content
    by construction, so each cross-base pair is a case where a detection is a
    *false positive* — which is what makes a precision figure possible.

    :data:`KNOWN_POSITIVES` is subtracted from the control set first. A pair that
    genuinely shares content is not a control, and counting it as one does not
    merely add noise: it reports a *correct* detection as a false positive.
    """
    related = set(KNOWN_POSITIVES) | set(KNOWN_POSITIVES.values())
    bases = sorted(
        path
        for path in inputs.glob("*")
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and not any(f"_{suffix}" in path.stem for suffix in SUFFIXES)
    )
    if not bases:
        raise FileNotFoundError(f"no base images under {inputs}")

    by_stem = {path.stem: path for path in bases}

    cases: list[Case] = []
    for base in bases:
        for suffix in SUFFIXES:
            for extension in (".png", ".jpg", ".jpeg"):
                candidate = inputs / f"{base.stem}_{suffix}{extension}"
                if candidate.is_file():
                    cases.append(
                        Case(
                            left=base,
                            right=candidate,
                            positive=True,
                            manipulation=suffix,
                            label=f"{base.stem} -> {suffix}",
                        )
                    )
                    break

    for stem, partner_stem in KNOWN_POSITIVES.items():
        left, right = by_stem.get(stem), by_stem.get(partner_stem)
        if left is not None and right is not None:
            cases.append(
                Case(
                    left=left,
                    right=right,
                    positive=True,
                    manipulation="known_pair",
                    label=f"{stem} -> {partner_stem}",
                )
            )

    # Controls: distinct source images only. Anything in `related` shares
    # content with something else here and cannot serve as a control.
    controls = [path for path in bases if path.stem not in related]
    for left, right in combinations(controls, 2):
        cases.append(
            Case(
                left=left,
                right=right,
                positive=False,
                manipulation="",
                label=f"{left.stem} vs {right.stem} (control)",
            )
        )

    _log.info(
        "discovered %d cases: %d positive, %d negative controls",
        len(cases),
        sum(1 for c in cases if c.positive),
        sum(1 for c in cases if not c.positive),
    )
    return cases


def correct_correspondences(src: np.ndarray, dst: np.ndarray, *, tolerance_px: float) -> int:
    """Correspondences consistent with an identity transform.

    Only meaningful for the signal-degradation and exposure cases, where the two
    images are pixel-aligned by construction. This is the measurement that
    exposed DISK's extra matches on ``base_cell`` as 0% correct.
    """
    if len(src) == 0:
        return 0
    displacement = np.abs(np.asarray(dst) - np.asarray(src)).max(axis=1)
    return int((displacement < tolerance_px).sum())


def run(
    cfg: Settings,
    cases: Sequence[Case],
    *,
    device: str = "cpu",
    tolerance_px: float = 3.0,
    backend: str | None = None,
) -> Report:
    """Evaluate every case with one pipeline instance.

    One ``Pipeline`` for the whole sweep, not one per pair -- bug 11 was a 35 MB
    ``torch.load`` per image in exactly this loop.
    """
    import time

    from sciforensics.pipeline import Pipeline

    pipeline = Pipeline(cfg, device=device)
    label = backend or f"{cfg.local_match.detector}+{cfg.local_match.matcher}"
    report = Report(
        backend=label,
        audit=build_audit_block(
            config_fingerprint=cfg.fingerprint(),
            device=device,
            seed=cfg.runtime.seed,
        ).model_dump(mode="json"),
    )

    for case in cases:
        started = time.perf_counter()
        try:
            analysis = pipeline.compare(case.left, case.right)
        except Exception as exc:  # a crash is a result, not a reason to stop
            _log.warning("case %s failed: %s", case.label, exc)
            report.outcomes.append(
                Outcome(
                    case=case,
                    detected=False,
                    verdict="error",
                    confidence=0.0,
                    matches=0,
                    correct=None,
                    rejection=str(exc)[:120],
                    seconds=time.perf_counter() - started,
                )
            )
            continue

        elapsed = time.perf_counter() - started
        result = analysis.result
        geometry = result.geometry

        correct: int | None = None
        if case.identity and analysis.correspondences is not None:
            correct = correct_correspondences(
                analysis.correspondences.src,
                analysis.correspondences.dst,
                tolerance_px=tolerance_px,
            )

        report.outcomes.append(
            Outcome(
                case=case,
                # Taken from the pipeline's own verdict, so the benchmark cannot
                # disagree with the tool it is measuring.
                detected=result.verdict in {Verdict.LIKELY_MANIPULATED, Verdict.SUSPICIOUS},
                verdict=result.verdict.value,
                confidence=result.confidence,
                matches=result.matches.good if result.matches else 0,
                correct=correct,
                rejection=(
                    geometry.rejection_reason.value if geometry and not geometry.verified else ""
                ),
                seconds=elapsed,
            )
        )

    return report


def latency_percentiles(report: Report) -> dict[str, float]:
    times = sorted(o.seconds for o in report.outcomes)
    if not times:
        return {}
    array = np.asarray(times)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
    }


def write_results(reports: Sequence[Report], directory: Path) -> tuple[Path, Path]:
    """Write ``results.json`` plus a Markdown table.

    Both, deliberately: the JSON is what CI regression-guards, the Markdown is
    what a reader sees. Generating the prose from the same object is what stops
    ``RESULTS.md`` drifting from the numbers behind it -- the documentation drift
    the audit found throughout the prototype.
    """
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "reports": [report.to_dict() for report in reports],
        "latency": {report.backend: latency_percentiles(report) for report in reports},
    }
    json_path = directory / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = directory / "RESULTS.md"
    md_path.write_text(_render_markdown(reports), encoding="utf-8")
    return json_path, md_path


def _render_markdown(reports: Sequence[Report]) -> str:
    lines: list[str] = [
        "# Benchmark results",
        "",
        "Generated by `sciforensics bench`. Every number below comes from",
        "`results.json` in this directory; do not edit by hand.",
        "",
        "## Why the negative controls matter",
        "",
        "`run_demo.py` paired each base image only with its own manipulations, so the",
        "project could not state a false-positive rate at all. The controls here are",
        "every cross-base pair: distinct source images, so any detection is a false",
        "positive by construction.",
        "",
        "## Summary",
        "",
        "| Backend | Positives | Controls | Recall | FPR | Precision |",
        "|---|---|---|---|---|---|",
    ]
    for report in reports:
        lines.append(
            f"| `{report.backend}` | {len(report.positives)} | {len(report.negatives)} | "
            f"{report.recall():.1%} | {report.false_positive_rate():.1%} | "
            f"{report.precision():.1%} |"
        )

    # A rate over a handful of controls is not a rate. Saying so in the
    # generated prose is the difference between a measurement and a claim --
    # and the whole reason this harness exists is that the prototype made the
    # claim without the measurement.
    smallest = min((len(r.negatives) for r in reports), default=0)
    if smallest < 30:
        lines += [
            "",
            f"> **Caveat: only {smallest} negative control(s).** A false-positive rate over so",
            "> few pairs has a confidence interval far wider than the point estimate, and a",
            "> reported 0.0% is consistent with a true rate of several percent. It bounds the",
            "> obvious failure modes and nothing more. A publishable figure needs a corpus",
            "> (BioFors, held out per `docs/DATA_CARD.md`), not the repository's example images.",
        ]

    lines += [
        "",
        "## Recall by manipulation",
        "",
        "| Manipulation | " + " | ".join(f"`{r.backend}`" for r in reports) + " |",
        "|---" * (len(reports) + 1) + "|",
    ]
    manipulations = sorted({m for r in reports for m in r.recall_by_manipulation()})
    for manipulation in manipulations:
        cells = []
        for report in reports:
            hits, total = report.recall_by_manipulation().get(manipulation, (0, 0))
            cells.append(f"{hits}/{total}" if total else "—")
        lines.append(f"| `{manipulation}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Correspondence precision (identity-transform cases)",
        "",
        "`ex2_degraded`, `ex4_exposure` and `ex5_blackout` apply **no geometric**",
        "change, so a correspondence is correct only if it lands within tolerance of",
        "the same pixel. This is the column that matters: a matcher can raise its",
        "match count while every extra match is wrong.",
        "",
        "| Case | Backend | Matches | Correct | Precision |",
        "|---|---|---|---|---|",
    ]
    for report in reports:
        for outcome in report.outcomes:
            if outcome.correct is None:
                continue
            share = outcome.precision
            lines.append(
                f"| {outcome.case.label} | `{report.backend}` | {outcome.matches} | "
                f"{outcome.correct} | " + (f"{share:.1%} |" if share is not None else "— |")
            )

    lines += [
        "",
        "## Latency (seconds per pair)",
        "",
        "| Backend | p50 | p90 | p99 |",
        "|---|---|---|---|",
    ]
    for report in reports:
        pct = latency_percentiles(report)
        if pct:
            lines.append(
                f"| `{report.backend}` | {pct['p50']:.2f} | {pct['p90']:.2f} | {pct['p99']:.2f} |"
            )

    return "\n".join(lines) + "\n"


def iter_backends(cfg: Settings, names: Sequence[str]) -> Iterator[tuple[str, Settings]]:
    """Yield ``(label, settings)`` for each requested backend."""
    pairs = {
        "orb": ("orb", "mutual_nn"),
        "disk": ("disk", "lightglue"),
    }
    for name in names:
        if name not in pairs:
            raise ValueError(f"unknown backend {name!r}; known: {sorted(pairs)}")
        detector, matcher = pairs[name]
        yield (
            f"{detector}+{matcher}",
            cfg.model_copy(
                update={
                    "local_match": cfg.local_match.model_copy(
                        update={"detector": detector, "matcher": matcher}
                    )
                }
            ),
        )
