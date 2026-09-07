"""Typed, validated configuration for the whole pipeline.

Design note
-----------
Every threshold lives in ``configs/default.yaml``; the models here only declare
*types and invariants*, never competing defaults. That is deliberate. The
pre-refactor code carried the local-stage trigger threshold as a dataclass
default of ``2.0``, an argparse default of ``3.0`` and a documented value of
``2.0`` simultaneously, so the answer you got depended on which entry point you
used. Fields below are therefore required (no ``= value``) unless a default is
genuinely semantic, and :func:`load_config` is the only way to build a
:class:`Settings`.

Override precedence, lowest to highest::

    configs/default.yaml  <  --config user.yaml  <  SCIFORENSICS_* env  <  --set k=v

That order is implemented explicitly in :func:`load_config` rather than through
``BaseSettings`` source magic, because ``BaseSettings`` ranks constructor
arguments above environment variables and a fully-populated YAML file would
therefore shadow every env var. ``pydantic_settings`` is still used for the
service-only settings in :mod:`sciforensics.api.settings`, where its
environment/secret handling is the right tool.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AffineConfig",
    "ApiConfig",
    "AttributionConfig",
    "ClusterConfig",
    "ConfigError",
    "CopyMoveConfig",
    "EnhanceConfig",
    "FusionConfig",
    "GeometryConfig",
    "GlobalMatchConfig",
    "ImageConfig",
    "LocalMatchConfig",
    "OrbConfig",
    "ReportConfig",
    "RoiConfig",
    "RuntimeConfig",
    "Settings",
    "default_config_path",
    "load_config",
]

ENV_PREFIX = "SCIFORENSICS_"
ENV_NESTED_DELIMITER = "__"

# Spatial downsampling factor of each named backbone stage relative to the
# input. Used to validate `attribution.stage` and to report the grid size.
STAGE_STRIDES: dict[str, int] = {"block1": 2, "block2": 4, "block3": 8, "block4": 16}

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


class _Base(BaseModel):
    """Strict base: unknown keys are errors, not silently ignored.

    A typo in a YAML key used to mean "the documented threshold is quietly not
    applied". ``extra="forbid"`` turns that into a startup failure naming the
    offending key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
class RuntimeConfig(_Base):
    seed: int | None
    device: str
    deterministic: bool
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_format: Literal["console", "json"]
    num_workers: Annotated[int, Field(ge=0)]


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------
class ImageConfig(_Base):
    max_dimension: Annotated[int, Field(ge=64, le=16384)]
    embed_size: Annotated[int, Field(ge=32, le=1024)]
    embed_resize: Literal["squash", "letterbox"]
    letterbox_pad_value: Annotated[int, Field(ge=0, le=255)]

    @model_validator(mode="after")
    def _embed_size_divisible(self) -> ImageConfig:
        # Four stride-2 pooling stages: a size not divisible by 16 silently
        # truncates, so the attribution grid stops aligning with the image.
        if self.embed_size % 16 != 0:
            raise ValueError(
                "image.embed_size must be divisible by 16 (four pooling stages); "
                f"got {self.embed_size}"
            )
        return self


# ---------------------------------------------------------------------------
# global_match
# ---------------------------------------------------------------------------
class AttributionConfig(_Base):
    enabled: bool
    stage: Literal["block1", "block2", "block3", "block4"]
    gradient_pooling: Literal["abs", "relu", "signed"]
    colormap: str
    alpha: UnitFloat

    def grid_size(self, embed_size: int) -> int:
        """Spatial resolution of the attribution map for a given input size."""
        return embed_size // STAGE_STRIDES[self.stage]


class GlobalMatchConfig(_Base):
    weights: Path | None
    distance_threshold: PositiveFloat
    local_trigger_distance: PositiveFloat
    similarity_temperature: PositiveFloat
    attribution: AttributionConfig

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> GlobalMatchConfig:
        if self.local_trigger_distance <= self.distance_threshold:
            raise ValueError(
                "global_match.local_trigger_distance must exceed distance_threshold "
                f"(got {self.local_trigger_distance} <= {self.distance_threshold}); "
                "otherwise the escalation band is empty and the local stage can never fire "
                "on an ambiguous pair."
            )
        return self


# ---------------------------------------------------------------------------
# local_match
# ---------------------------------------------------------------------------
class OrbConfig(_Base):
    scale_factor: Annotated[float, Field(gt=1.0)]
    n_levels: PositiveInt
    edge_threshold: Annotated[int, Field(ge=0)]
    patch_size: Annotated[int, Field(ge=2)]
    fast_threshold: Annotated[int, Field(ge=0)]


class EnhanceConfig(_Base):
    enabled: bool
    scale: Annotated[float, Field(ge=1.0, le=8.0)]
    interpolation: Literal["nearest", "linear", "cubic", "lanczos"]
    clahe_clip_limit: PositiveFloat
    clahe_tile_grid: PositiveInt


class RoiConfig(_Base):
    enabled: bool
    dog_sigma_low: PositiveFloat
    dog_sigma_high: PositiveFloat
    dog_threshold: PositiveFloat
    min_area: PositiveInt
    max_regions: PositiveInt
    dilate_px: Annotated[int, Field(ge=0)]
    #: Below this many keypoints inside the ROI mask, the restriction is dropped
    #: for that side and ``Detection.roi_abandoned`` records it. This is a
    #: *quality* judgement about the mask -- "the band-pass filter threw away too
    #: much to trust it" -- and is deliberately separate from
    #: :attr:`LocalMatchConfig.min_keypoints_per_side`, which is an arithmetic
    #: floor on whether matching is attempted at all. The two were one knob
    #: until measurement showed they want different values: raising the ROI
    #: threshold to a sensible 100 silently made the matching floor 100 too,
    #: and every 256x256 panel in ``inputs/`` was then refused with
    #: ``too_few_keypoints`` when the real cause lay one gate further on.
    min_keypoints: PositiveInt

    @model_validator(mode="after")
    def _sigmas_ordered(self) -> RoiConfig:
        if self.dog_sigma_high <= self.dog_sigma_low:
            raise ValueError(
                "local_match.roi.dog_sigma_high must exceed dog_sigma_low for the "
                f"difference-of-Gaussians to be a band-pass filter (got "
                f"{self.dog_sigma_high} <= {self.dog_sigma_low})"
            )
        return self


class LocalMatchConfig(_Base):
    detector: Literal["orb", "disk", "superpoint"]
    matcher: Literal["mutual_nn", "lightglue"]
    max_features: PositiveInt
    #: Hard floor on surviving keypoints per side, below which matching is not
    #: attempted at all. Under injective mutual-NN matching the number of
    #: correspondences -- and therefore of inliers -- cannot exceed
    #: ``min(kp_left, kp_right)``, so below this floor geometric verification is
    #: arithmetically impossible rather than merely unlikely. See the
    #: ``Settings`` cross-section invariant tying it to ``geometry.min_inliers``.
    #: For the ROI-mask quality threshold, see :attr:`RoiConfig.min_keypoints`.
    min_keypoints_per_side: PositiveInt
    nn_ratio: Annotated[float, Field(gt=0.0, le=1.0)]
    mutual_nn: bool
    orb: OrbConfig
    enhance: EnhanceConfig
    roi: RoiConfig

    @model_validator(mode="after")
    def _floors_below_budget(self) -> LocalMatchConfig:
        # Both floors are compared against counts that ORB can never exceed, so
        # either one set above the feature budget is unsatisfiable by
        # construction -- a config that rejects every input rather than a strict
        # one.
        for name, floor in (
            ("min_keypoints_per_side", self.min_keypoints_per_side),
            ("roi.min_keypoints", self.roi.min_keypoints),
        ):
            if floor > self.max_features:
                raise ValueError(
                    f"local_match.{name} cannot exceed max_features "
                    f"({floor} > {self.max_features}); the floor would be unsatisfiable "
                    "and every pair would be rejected."
                )
        return self


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
class AffineConfig(_Base):
    model: Literal["full", "similarity"]
    method: Literal["magsac", "ransac", "lmeds"]
    reproj_threshold: PositiveFloat
    anisotropy_tolerance: PositiveFloat
    shear_tolerance_deg: PositiveFloat

    @property
    def can_detect_flip(self) -> bool:
        """Whether the chosen model can represent a reflection at all.

        ``similarity`` (``cv2.estimateAffinePartial2D``) produces
        ``[[a, -b, tx], [b, a, ty]]`` whose determinant is ``a**2 + b**2``, so
        it is strictly positive for any non-degenerate fit and a ``det < 0``
        flip test can never fire. Only ``full`` has the degrees of freedom.
        """
        return self.model == "full"


class GeometryConfig(_Base):
    method: Literal["magsac", "ransac", "lmeds"]
    reproj_threshold: PositiveFloat
    max_iters: PositiveInt
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    # A planar homography has 8 degrees of freedom and therefore needs at least
    # 4 point correspondences. Anything below that is not a loose threshold, it
    # is an unsolvable system.
    min_matches: Annotated[int, Field(ge=4)]
    min_inliers: Annotated[int, Field(ge=4)]
    min_inlier_ratio: UnitFloat
    min_distinct_inliers: Annotated[int, Field(ge=4)]
    min_inlier_spread: UnitFloat
    max_reproj_rms: PositiveFloat
    max_condition_number: PositiveFloat
    affine: AffineConfig

    @model_validator(mode="after")
    def _gates_consistent(self) -> GeometryConfig:
        # Note there is deliberately NO constraint tying min_inliers to
        # min_matches. They gate different things -- min_matches decides whether
        # a fit is attempted at all, min_inliers whether the result is accepted
        # -- and requiring, say, min_inliers <= min_matches would forbid the
        # entirely reasonable "try at 8 correspondences, only trust it at 40".
        if self.min_distinct_inliers > self.min_inliers:
            raise ValueError(
                "geometry.min_distinct_inliers cannot exceed min_inliers "
                f"({self.min_distinct_inliers} > {self.min_inliers}); distinct correspondences "
                "are a subset of inlier rows, so the stricter gate would make the looser one "
                "unreachable."
            )
        return self


# ---------------------------------------------------------------------------
# copy_move
# ---------------------------------------------------------------------------
class ClusterConfig(_Base):
    eps: PositiveFloat
    min_samples: PositiveInt
    feature_weights: tuple[float, float, float, float]

    @model_validator(mode="after")
    def _weights_nonnegative(self) -> ClusterConfig:
        if any(w < 0 for w in self.feature_weights):
            raise ValueError("copy_move.cluster.feature_weights must all be >= 0")
        if not any(w > 0 for w in self.feature_weights):
            raise ValueError(
                "copy_move.cluster.feature_weights must contain at least one positive weight, "
                "otherwise every offset collapses to the origin and DBSCAN returns one cluster."
            )
        return self


class CopyMoveConfig(_Base):
    enabled: bool
    max_features: PositiveInt
    nn_ratio: Annotated[float, Field(gt=0.0, le=1.0)]
    min_spatial_separation: Annotated[float, Field(ge=0.0)]
    knn: Annotated[int, Field(ge=2)]
    cluster: ClusterConfig
    min_cluster_inliers: PositiveInt
    reproj_threshold: PositiveFloat
    mask_close_px: Annotated[int, Field(ge=0)]
    max_regions: PositiveInt


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------
class BandsConfig(_Base):
    likely_manipulated: UnitFloat
    suspicious: UnitFloat
    inconclusive: UnitFloat

    @model_validator(mode="after")
    def _monotone(self) -> BandsConfig:
        if not (self.likely_manipulated > self.suspicious > self.inconclusive):
            raise ValueError(
                "fusion.bands must be strictly decreasing: likely_manipulated > suspicious > "
                f"inconclusive (got {self.likely_manipulated} > {self.suspicious} > "
                f"{self.inconclusive})"
            )
        return self


class FusionConfig(_Base):
    calibrator: Path | None
    bands: BandsConfig


# ---------------------------------------------------------------------------
# report / api
# ---------------------------------------------------------------------------
class ReportConfig(_Base):
    formats: tuple[Literal["pdf", "html", "png", "json"], ...]
    audit_trail: bool
    thumbnail_max_px: PositiveInt
    draw_inlier_hull: bool
    draw_match_lines: bool
    max_match_lines: PositiveInt

    @model_validator(mode="after")
    def _at_least_one_format(self) -> ReportConfig:
        if not self.formats:
            raise ValueError("report.formats must list at least one output format")
        if len(set(self.formats)) != len(self.formats):
            raise ValueError(f"report.formats contains duplicates: {self.formats}")
        return self


class ApiConfig(_Base):
    host: str
    port: Annotated[int, Field(ge=1, le=65535)]
    cors_origins: tuple[str, ...]
    max_upload_bytes: PositiveInt
    max_decoded_pixels: PositiveInt
    job_ttl_seconds: PositiveInt
    max_concurrent_jobs: PositiveInt


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------
class Settings(_Base):
    runtime: RuntimeConfig
    image: ImageConfig
    global_match: GlobalMatchConfig
    local_match: LocalMatchConfig
    geometry: GeometryConfig
    copy_move: CopyMoveConfig
    fusion: FusionConfig
    report: ReportConfig
    api: ApiConfig

    @model_validator(mode="after")
    def _cross_section_invariants(self) -> Settings:
        # Flip detection is a headline capability; refusing to run with a
        # configuration that makes it unreachable is better than silently
        # reporting "flip: no" for every mirrored image, which is exactly what
        # the pre-refactor pipeline did.
        if not self.geometry.affine.can_detect_flip:
            raise ValueError(
                "geometry.affine.model='similarity' cannot represent a reflection: its "
                "determinant is a**2 + b**2 > 0, so flip detection is mathematically "
                "unreachable. Use model='full' (cv2.estimateAffine2D). If you deliberately "
                "want a 4-DOF fit, set it per-call rather than globally."
            )
        # Mutual-NN matching is injective, so correspondences -- and therefore
        # inlier rows, which are a subset -- are bounded by the smaller keypoint
        # set. A keypoint floor beneath min_inliers admits pairs whose
        # verification cannot possibly pass, and they then fail one gate later
        # with `too_few_matches`, blaming the matcher for a starved detector.
        # Requiring the floor to dominate keeps each rejection reason truthful
        # about its own cause.
        if self.local_match.min_keypoints_per_side < self.geometry.min_inliers:
            raise ValueError(
                "local_match.min_keypoints_per_side must be >= geometry.min_inliers "
                f"({self.local_match.min_keypoints_per_side} < {self.geometry.min_inliers}): "
                "mutual-nearest-neighbour matching is injective, so inlier rows can never "
                "exceed the smaller keypoint set and verification would be arithmetically "
                "impossible for a pair that nonetheless passed the floor."
            )
        return self

    # -- audit -------------------------------------------------------------
    def fingerprint(self) -> str:
        """Stable SHA-256 over the resolved configuration.

        Goes into every report's audit block so a result can be tied to the
        exact settings that produced it. Key order is canonicalised, so
        semantically identical configs written differently hash the same.
        """
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json"), sort_keys=True, default_flow_style=False
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def default_config_path() -> Path:
    """Path to the packaged ``default.yaml``.

    Resolves the repo checkout first (editable installs, which is how anyone
    developing this will run it) and falls back to the data shipped alongside
    the package.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "configs" / "default.yaml",  # <repo>/configs
        here.parent / "data" / "default.yaml",  # packaged copy
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConfigError(
        "could not locate configs/default.yaml; looked in " + ", ".join(str(c) for c in candidates)
    )


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new dict.

    Mappings merge key-by-key; every other type (including lists) replaces
    wholesale. Replacing lists is intentional: element-wise merging of, say,
    ``report.formats`` would make it impossible to *remove* a format.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI/env string into the narrowest sensible Python type.

    Uses the YAML scalar rules so ``true``/``null``/``2.5``/``[1, 2]`` all behave
    the way they do in the config file itself, rather than inventing a second,
    subtly different syntax for overrides. That fidelity includes the quirks: YAML
    1.1 needs a signed exponent, so ``1.6e+3`` is 1600 while ``1e7`` is the string
    ``"1e7"``. Inheriting the quirk is still better than diverging from the file,
    and a typed field rejects the string loudly rather than coercing it.
    """
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _assign_path(target: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ConfigError(f"empty configuration key in override: {dotted!r}")
    cursor: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _env_overlay(environ: Mapping[str, str]) -> dict[str, Any]:
    """Build an override tree from ``SCIFORENSICS_A__B=value`` variables.

    ``SCIFORENSICS_GEOMETRY__MIN_INLIERS=20`` maps to
    ``{"geometry": {"min_inliers": 20}}``. Case is folded to lower because
    environment variables are conventionally upper-case while config keys are
    not.
    """
    overlay: dict[str, Any] = {}
    for raw_key, raw_value in environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        suffix = raw_key[len(ENV_PREFIX) :]
        if not suffix:
            continue
        dotted = suffix.lower().replace(ENV_NESTED_DELIMITER, ".")
        _assign_path(overlay, dotted, _coerce_scalar(raw_value))
    return overlay


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a YAML mapping at the top level, got {type(data).__name__}"
        )
    return data


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Iterable[str] | None = None,
    env: Mapping[str, str] | None = None,
    use_env: bool = True,
) -> Settings:
    """Build a validated :class:`Settings`.

    Parameters
    ----------
    path
        Optional user config layered on top of ``configs/default.yaml``. It may
        be partial -- only the keys it sets are overridden.
    overrides
        ``"dotted.key=value"`` strings, highest precedence. Values are parsed
        with YAML scalar rules, so ``geometry.method=ransac``,
        ``report.formats=[pdf]`` and ``global_match.weights=null`` all work.
    env
        Environment mapping to read ``SCIFORENSICS_*`` from; defaults to
        :data:`os.environ`.
    use_env
        Set ``False`` to ignore the environment entirely. Tests use this so a
        developer's exported variables cannot change assertion outcomes.

    Raises
    ------
    ConfigError
        If a file is unreadable/malformed, an override is not ``key=value``, an
        unknown key is present, or a documented invariant is violated.
    """
    data = _read_yaml(default_config_path())

    if path is not None:
        user_path = Path(path)
        if not user_path.is_file():
            raise ConfigError(f"config file not found: {user_path}")
        data = _deep_merge(data, _read_yaml(user_path))

    if use_env:
        data = _deep_merge(data, _env_overlay(os.environ if env is None else env))

    if overrides:
        overlay: dict[str, Any] = {}
        for item in overrides:
            key, sep, raw = item.partition("=")
            if not sep:
                raise ConfigError(
                    f"override {item!r} is not in key=value form, e.g. geometry.min_inliers=20"
                )
            _assign_path(overlay, key.strip(), _coerce_scalar(raw))
        data = _deep_merge(data, overlay)

    try:
        return Settings.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError, plus our ValueErrors
        raise ConfigError(f"invalid configuration:\n{exc}") from exc
