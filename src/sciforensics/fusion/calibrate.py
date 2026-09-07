"""Applying a fitted calibrator, and the file format one is stored in.

This module *applies* a calibrator; it does not fit one. Fitting is stage B5, which
needs a held-out split and labelled negatives that do not exist yet -- so until
``fusion.calibrator`` names a file, :mod:`sciforensics.fusion.rules` scores every
result and ``calibrated=False`` travels with it. That flag is the whole point of the
separation: a reader can always tell whether the number in front of them came from
weights someone reasoned about or from weights fitted to data.

The format is defined here rather than in the trainer, and the writer
(:meth:`LogisticCalibrator.save`) lives next to the reader (:func:`load`), because a
serialisation format with its two halves in different modules drifts.

**The feature-schema check is the reason this module is not three lines long.** A
calibrator is a vector of coefficients, and a vector is meaningless without the
column order it was fitted against. If someone appends a feature to
:data:`~sciforensics.fusion.features.FEATURES` and re-runs against a calibrator
fitted before that change, every coefficient after the insertion point applies to
the wrong quantity -- and the result is not an error, it is a plausible number from a
model nobody reviewed. :func:`load` therefore refuses any file whose recorded
feature names are not exactly the current schema, in order, and says which names
moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from sciforensics.config import BandsConfig
from sciforensics.fusion.features import FEATURE_NAMES, FEATURES, EvidenceBundle, extract
from sciforensics.fusion.rules import Fused, band, logistic
from sciforensics.runtime import get_logger
from sciforensics.types import EvidenceContribution

__all__ = [
    "SUPPORTED_KINDS",
    "CalibratorError",
    "Isotonic",
    "LogisticCalibrator",
    "apply",
    "load",
]

_log = get_logger(__name__)

#: Values accepted in a calibrator file's ``kind`` field. Dispatched on by
#: :func:`load`, which is the extension point for a future non-linear calibrator.
SUPPORTED_KINDS = ("logistic",)

#: Version of the on-disk format, so a future breaking change can be diagnosed
#: rather than mis-parsed.
FORMAT_VERSION = 1


class CalibratorError(RuntimeError):
    """Raised when a calibrator cannot be read, or does not match this build."""


@dataclass(frozen=True)
class Isotonic:
    """A piecewise-linear monotone map, as fitted by isotonic regression.

    Stored as knots rather than as a scikit-learn object so that applying a
    calibrator needs no sklearn at inference time and the artifact stays readable:
    someone auditing a report can open the JSON and see the mapping.
    """

    #: Knot inputs, strictly increasing. Uncalibrated probabilities.
    x: tuple[float, ...]
    #: Knot outputs, non-decreasing. Calibrated probabilities.
    y: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise CalibratorError(f"isotonic knots disagree: {len(self.x)} x, {len(self.y)} y")
        if len(self.x) < 2:
            raise CalibratorError("an isotonic map needs at least two knots")
        if any(b <= a for a, b in pairwise(self.x)):
            raise CalibratorError("isotonic x knots must be strictly increasing")
        if any(b < a for a, b in pairwise(self.y)):
            # A decreasing segment would make the calibrated score non-monotone in
            # the raw one, which no isotonic fit produces -- so this means a
            # hand-edited or corrupted file, not an unusual model.
            raise CalibratorError("isotonic y knots must be non-decreasing")
        if any(not 0.0 <= value <= 1.0 for value in self.y):
            raise CalibratorError("isotonic y knots must be probabilities in [0, 1]")

    def __call__(self, value: float) -> float:
        """Interpolate, clamping outside the fitted range.

        Clamping rather than extrapolating: beyond the knots there was no data, and a
        linear extension of the last segment would invent confidence exactly where
        the fit has none.
        """
        if value <= self.x[0]:
            return self.y[0]
        if value >= self.x[-1]:
            return self.y[-1]
        for index in range(1, len(self.x)):
            if value <= self.x[index]:
                x0, x1 = self.x[index - 1], self.x[index]
                y0, y1 = self.y[index - 1], self.y[index]
                return y0 + (y1 - y0) * (value - x0) / (x1 - x0)
        return self.y[-1]  # pragma: no cover -- unreachable given the clamp above


@dataclass(frozen=True)
class LogisticCalibrator:
    """Fitted logistic regression over the feature table, optionally isotonic-wrapped.

    Deliberately the same functional form as :mod:`sciforensics.fusion.rules`, so
    swapping the rule table for a fitted model changes the numbers and nothing about
    how a result is read or rendered.
    """

    #: Feature names in the order the coefficients were fitted. Checked against
    #: :data:`~sciforensics.fusion.features.FEATURE_NAMES` on load.
    features: tuple[str, ...]
    coef: tuple[float, ...]
    intercept: float
    isotonic: Isotonic | None = None
    #: Provenance for the audit block: which frozen split this was fitted on, and the
    #: metrics it achieved there. Free-form because B6 decides what it reports, but
    #: carried through to the report so a calibrated number can be traced to a run.
    fitted_on: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.coef) != len(self.features):
            raise CalibratorError(
                f"calibrator has {len(self.coef)} coefficients for "
                f"{len(self.features)} feature names"
            )

    def logit(self, values: dict[str, float]) -> float:
        """Weighted sum over the fitted features, plus the intercept."""
        missing = [name for name in self.features if name not in values]
        if missing:
            raise CalibratorError(f"feature vector is missing {', '.join(missing)}")
        total = self.intercept
        for name, weight in zip(self.features, self.coef, strict=True):
            total += values[name] * weight
        return total

    def predict(self, values: dict[str, float]) -> float:
        """Calibrated probability: logistic, then the isotonic map if there is one."""
        raw = logistic(self.logit(values))
        return raw if self.isotonic is None else self.isotonic(raw)

    @property
    def weights(self) -> dict[str, float]:
        """Coefficients by feature name, for the contribution bars."""
        return dict(zip(self.features, self.coef, strict=True))

    def to_dict(self) -> dict[str, object]:
        """Serialisable form. The inverse of :func:`load`'s parsing."""
        payload: dict[str, object] = {
            "format_version": FORMAT_VERSION,
            "kind": "logistic",
            "features": list(self.features),
            "coef": list(self.coef),
            "intercept": self.intercept,
            "fitted_on": self.fitted_on,
            "metrics": dict(self.metrics),
        }
        if self.isotonic is not None:
            payload["isotonic"] = {"x": list(self.isotonic.x), "y": list(self.isotonic.y)}
        return payload

    def save(self, path: Path) -> Path:
        """Write to ``path`` as indented JSON, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def _require_schema(names: list[str]) -> tuple[str, ...]:
    """Confirm a file's feature names are exactly the current schema, in order."""
    if tuple(names) == FEATURE_NAMES:
        return FEATURE_NAMES

    recorded, current = set(names), set(FEATURE_NAMES)
    detail: list[str] = []
    if added := sorted(current - recorded):
        detail.append(f"features added since it was fitted: {', '.join(added)}")
    if removed := sorted(recorded - current):
        detail.append(f"features it expects that no longer exist: {', '.join(removed)}")
    if not detail:
        detail.append(f"same features in a different order (expected {', '.join(FEATURE_NAMES)})")
    raise CalibratorError(
        "calibrator was fitted against a different feature schema; "
        + "; ".join(detail)
        + ". Refit it against this build rather than reordering the file: its "
        "coefficients are positional and would otherwise be applied to the wrong "
        "quantities without any error."
    )


def load(path: Path) -> LogisticCalibrator:
    """Read and validate a calibrator file.

    Raises
    ------
    CalibratorError
        If the file is missing, is not JSON, names an unsupported ``kind``, or was
        fitted against a different feature schema. Every one of these is fatal on
        purpose: ``fusion.calibrator`` being set means someone asked for calibrated
        output, and silently falling back to the rule table would produce a report
        labelled with the wrong provenance.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise CalibratorError(f"fusion.calibrator points at {path}, which does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibratorError(f"could not read calibrator at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CalibratorError(f"calibrator at {path} is not a JSON object")

    kind = payload.get("kind")
    if kind not in SUPPORTED_KINDS:
        raise CalibratorError(
            f"calibrator at {path} has kind {kind!r}; this build supports "
            f"{', '.join(SUPPORTED_KINDS)}"
        )
    version = payload.get("format_version", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise CalibratorError(
            f"calibrator at {path} is format version {version}; this build reads {FORMAT_VERSION}"
        )

    try:
        features = _require_schema([str(name) for name in payload["features"]])
        coef = tuple(float(value) for value in payload["coef"])
        intercept = float(payload["intercept"])
    except KeyError as exc:
        raise CalibratorError(f"calibrator at {path} is missing {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise CalibratorError(f"calibrator at {path} has malformed coefficients: {exc}") from exc

    isotonic: Isotonic | None = None
    if (knots := payload.get("isotonic")) is not None:
        try:
            isotonic = Isotonic(
                x=tuple(float(v) for v in knots["x"]),
                y=tuple(float(v) for v in knots["y"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibratorError(
                f"calibrator at {path} has malformed isotonic knots: {exc}"
            ) from exc

    calibrator = LogisticCalibrator(
        features=features,
        coef=coef,
        intercept=intercept,
        isotonic=isotonic,
        fitted_on=str(payload.get("fitted_on", "")),
        metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()},
    )
    _log.info(
        "loaded %s calibrator from %s (fitted on %s%s)",
        kind,
        path,
        calibrator.fitted_on or "an unrecorded split",
        ", isotonic-wrapped" if isotonic is not None else "",
    )
    return calibrator


def apply(calibrator: LogisticCalibrator, bundle: EvidenceBundle, bands: BandsConfig) -> Fused:
    """Score a pair comparison with a fitted calibrator instead of the rule table.

    The returned :class:`~sciforensics.fusion.rules.Fused` carries
    ``calibrated=True``, which is the only path by which that flag becomes true.
    """
    values = extract(bundle)
    descriptions = {f.name: f.description for f in FEATURES}
    weights = calibrator.weights

    logit = calibrator.logit(values)
    contributions = [
        EvidenceContribution(
            name="prior",
            value=1.0,
            weight=calibrator.intercept,
            contribution=calibrator.intercept,
            description=f"Fitted base rate ({calibrator.fitted_on or 'unrecorded split'})",
        )
    ]
    contributions.extend(
        sorted(
            (
                EvidenceContribution(
                    name=name,
                    value=value,
                    weight=weights[name],
                    contribution=value * weights[name],
                    description=descriptions[name],
                )
                for name, value in values.items()
                if value != 0.0
            ),
            key=lambda c: abs(c.contribution),
            reverse=True,
        )
    )

    probability = calibrator.predict(values)
    notes: tuple[str, ...] = ()
    if calibrator.isotonic is not None:
        # Worth saying explicitly: the contributions above sum to the logistic
        # stage's logit, and the isotonic map is applied to its output. So the bars
        # explain the ranking faithfully but no longer add up to the final number.
        notes = (
            "This score passed through an isotonic calibration step after the weighted sum, so "
            "the contributions below explain its ranking rather than summing to the final value.",
        )
    return Fused(
        verdict=band(probability, bands),
        confidence=probability,
        contributions=tuple(contributions),
        features=values,
        logit=logit,
        notes=notes,
        calibrated=True,
    )
