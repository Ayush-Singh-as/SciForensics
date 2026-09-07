"""Evidence fusion: many measurements to one verdict, with the arithmetic shown.

Three modules, one seam:

* :mod:`~sciforensics.fusion.features` turns whatever evidence the stages produced
  into a bounded, named feature vector -- the schema both scorers agree on.
* :mod:`~sciforensics.fusion.rules` scores that vector with hand-set weights and
  labels the result ``calibrated=False``. This is what ships today.
* :mod:`~sciforensics.fusion.calibrate` scores it with a fitted logistic regression
  and labels the result ``calibrated=True``. Stage B5 fits the artifact; everything
  needed to load, validate and apply one is already here.

Which of the two runs is decided by ``fusion.calibrator`` in the configuration and by
nothing else, so a report's provenance is a property of the config it was produced
with. :func:`load_calibrator` is separate from :func:`score` on purpose: the
calibrator is read from disk once, when the pipeline is built, rather than once per
comparison. That is the same mistake as bug 11 (a 35 MB checkpoint re-read per image)
in a smaller and easier-to-miss form.
"""

from __future__ import annotations

from sciforensics.config import FusionConfig
from sciforensics.fusion import calibrate, features, rules
from sciforensics.fusion.calibrate import (
    CalibratorError,
    Isotonic,
    LogisticCalibrator,
)
from sciforensics.fusion.features import (
    COPY_MOVE_FEATURE_NAMES,
    FEATURE_NAMES,
    CopyMoveBundle,
    EvidenceBundle,
    Feature,
    extract,
    extract_copy_move,
)
from sciforensics.fusion.rules import (
    COPY_MOVE_WEIGHTS,
    PRIOR_LOGIT,
    WEIGHTS,
    Fused,
    band,
    fuse,
    fuse_copy_move,
    logistic,
)

__all__ = [
    "COPY_MOVE_FEATURE_NAMES",
    "COPY_MOVE_WEIGHTS",
    "FEATURE_NAMES",
    "PRIOR_LOGIT",
    "WEIGHTS",
    "CalibratorError",
    "CopyMoveBundle",
    "EvidenceBundle",
    "Feature",
    "Fused",
    "Isotonic",
    "LogisticCalibrator",
    "band",
    "calibrate",
    "extract",
    "extract_copy_move",
    "features",
    "fuse",
    "fuse_copy_move",
    "load_calibrator",
    "logistic",
    "rules",
    "score",
    "score_copy_move",
]


def load_calibrator(cfg: FusionConfig) -> LogisticCalibrator | None:
    """Read the configured calibrator, or ``None`` if the rule table is in use.

    Call once and hold the result. A missing or mismatched file raises
    :class:`~sciforensics.fusion.calibrate.CalibratorError` rather than falling back
    to the rule table, because ``fusion.calibrator`` being set means someone asked
    for calibrated output and a silent downgrade would mislabel the report.
    """
    return None if cfg.calibrator is None else calibrate.load(cfg.calibrator)


def score(
    bundle: EvidenceBundle,
    cfg: FusionConfig,
    *,
    calibrator: LogisticCalibrator | None = None,
) -> Fused:
    """Score a pair comparison, with the calibrator if one was supplied.

    ``calibrator`` is passed in rather than resolved here so the disk read stays at
    pipeline-construction time. Passing ``None`` when ``cfg.calibrator`` is set is
    therefore not an error -- it is how a caller deliberately scores against the rule
    table for comparison, and ``Fused.calibrated`` records which happened.
    """
    if calibrator is None:
        return rules.fuse(bundle, cfg.bands)
    return calibrate.apply(calibrator, bundle, cfg.bands)


def score_copy_move(bundle: CopyMoveBundle, cfg: FusionConfig) -> Fused:
    """Score a copy-move search. Always the rule table -- see :mod:`.features`."""
    return rules.fuse_copy_move(bundle, cfg.bands)
