"""Configuration: one source of truth, and a precedence order you can predict.

Bug 10 is the regression here. The prototype carried the local-stage trigger
threshold as a dataclass default of ``2.0``, an argparse default of ``3.0`` and a
documented value of ``2.0`` at the same time, so the threshold you got depended
on which entry point you happened to use -- and no single file was wrong. The fix
is structural rather than a corrected number: every threshold is a *required*
field with no model-level default, so ``configs/default.yaml`` is the only place a
value can come from, and a missing key is a startup failure instead of a silent
fallback to whichever default the code path reached first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sciforensics.config import (
    ConfigError,
    Settings,
    default_config_path,
    load_config,
)

# Two independent knobs used throughout as override targets. `max_iters` is an
# int under `geometry`, `nn_ratio` a float under `local_match`, so a test that
# moves one can assert the other did not follow.
#
# Both are deliberately chosen from the fields that participate in *no*
# cross-field validator. `min_inliers` was the original int probe and had to be
# swapped out: once a cross-section invariant tied it to
# `local_match.min_keypoints_per_side`, six loader tests began failing on a
# geometry constraint while asserting nothing about geometry. A knob used to
# prove that layering and deep-merge work must be inert, or the loader's tests
# break every time an unrelated invariant is added.
ITERS = ("geometry", "max_iters")
RATIO = ("local_match", "nn_ratio")


def _write(tmp_path: Path, payload: dict[str, object], name: str = "user.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _get(cfg: Settings, dotted: tuple[str, ...]) -> object:
    value: object = cfg
    for part in dotted:
        value = getattr(value, part)
    return value


# ---------------------------------------------------------------------------
# bug 10: nowhere for a second default to hide
# ---------------------------------------------------------------------------
def test_no_field_carries_its_own_default(cfg: Settings) -> None:
    """Every threshold must be required, so ``default.yaml`` is the only source.

    This is the structural form of the fix. A field with a model-level default is
    exactly the affordance that let three different trigger thresholds coexist:
    delete the key from the YAML and the code keeps running on a value nobody
    documented. Walking the model tree is worth the indirection because it holds
    for fields added later, which a hand-written list of keys would not.
    """
    semantic_defaults = {
        # Optional by meaning, not by oversight: absent weights mean "resolve
        # from the weights cache", and both ROI/enhance toggles read better as
        # explicit booleans in YAML than as required fields nobody varies.
        ("global_match", "weights"),
    }

    def walk(model: object, prefix: tuple[str, ...] = ()) -> list[str]:
        fields = getattr(type(model), "model_fields", None)
        if fields is None:
            return []
        offenders: list[str] = []
        for name, field in fields.items():
            path = (*prefix, name)
            if field.is_required():
                offenders.extend(walk(getattr(model, name), path))
            elif path not in semantic_defaults:
                offenders.append(".".join(path))
        return offenders

    assert walk(cfg) == []


def test_every_required_key_is_present_in_the_shipped_yaml() -> None:
    """The corollary: if fields are required, the file must be complete.

    Together with the test above this closes the loop -- no defaults in code, and
    no gaps in the file -- which is what makes "the value is in default.yaml" a
    true statement rather than an aspiration.
    """
    assert load_config(use_env=False) is not None
    assert default_config_path().is_file()


def test_a_missing_key_is_a_startup_error_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting a threshold must fail loudly instead of falling back.

    Aimed at the *base* file rather than a user config, because a user config is
    merged over the defaults and so cannot create a gap -- which is the point of
    the layering. Redirecting ``default_config_path`` is the only way to reach the
    state this test is about, and it is a state worth pinning: it is what a
    packaging mistake looks like from the inside.
    """
    data = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    del data["geometry"]["min_inliers"]
    incomplete = _write(tmp_path, data, "incomplete.yaml")
    monkeypatch.setattr("sciforensics.config.default_config_path", lambda: incomplete)

    with pytest.raises(ConfigError, match="min_inliers"):
        load_config(use_env=False)


def test_a_typo_names_the_offending_key_rather_than_being_ignored(tmp_path: Path) -> None:
    """``extra="forbid"`` turns a misspelt override into an error.

    Silently ignoring an unknown key means a documented threshold quietly not
    applying -- indistinguishable, from the outside, from the threshold not
    working.
    """
    path = _write(tmp_path, {"geometry": {"min_inlier": 25}})
    with pytest.raises(ConfigError, match="min_inlier"):
        load_config(path, use_env=False)


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------
def test_precedence_runs_default_then_file_then_env_then_set(tmp_path: Path) -> None:
    """All four layers at once, each overriding every layer beneath it.

    Asserted as a stack rather than as four separate tests because the ordering
    *between* layers is the property that matters, and it is the one an
    implementation built on ``BaseSettings`` source ranking would get wrong: that
    machinery ranks constructor arguments above environment variables, so a fully
    populated YAML file would shadow every env var.
    """
    base = load_config(use_env=False)
    path = _write(tmp_path, {"geometry": {"max_iters": 41}})
    env = {"SCIFORENSICS_GEOMETRY__MAX_ITERS": "52"}

    from_file = load_config(path, use_env=False)
    from_env = load_config(path, env=env)
    from_set = load_config(path, env=env, overrides=["geometry.max_iters=63"])

    assert _get(base, ITERS) not in {41, 52, 63}, "premise: the default differs from the overrides"
    assert _get(from_file, ITERS) == 41
    assert _get(from_env, ITERS) == 52, "env must beat the file"
    assert _get(from_set, ITERS) == 63, "--set must beat env"


def test_use_env_false_ignores_the_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property every other test in this suite depends on.

    A developer with ``SCIFORENSICS_*`` exported must not be able to change
    assertion outcomes, which is why the ``cfg`` fixture passes ``use_env=False``.
    """
    monkeypatch.setenv("SCIFORENSICS_GEOMETRY__MAX_ITERS", "99")

    assert _get(load_config(use_env=False), ITERS) != 99
    assert _get(load_config(), ITERS) == 99, "premise: the variable would otherwise apply"


def test_an_explicit_env_mapping_beats_the_ambient_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCIFORENSICS_GEOMETRY__MAX_ITERS", "99")
    cfg = load_config(env={"SCIFORENSICS_GEOMETRY__MAX_ITERS": "42"})
    assert _get(cfg, ITERS) == 42


def test_unprefixed_variables_are_not_consulted() -> None:
    """Only ``SCIFORENSICS_*`` participates; a bare ``MAX_ITERS`` is not ours."""
    cfg = load_config(env={"MAX_ITERS": "7", "GEOMETRY__MAX_ITERS": "7"})
    assert _get(cfg, ITERS) != 7


def test_a_partial_user_config_leaves_its_siblings_alone(tmp_path: Path) -> None:
    """Merging, not replacing: setting one nested key must not blank the rest."""
    base = load_config(use_env=False)
    path = _write(tmp_path, {"local_match": {"roi": {"dilate_px": 3}}})
    merged = load_config(path, use_env=False)

    assert merged.local_match.roi.dilate_px == 3
    assert merged.local_match.roi.dog_threshold == base.local_match.roi.dog_threshold
    assert merged.local_match.nn_ratio == base.local_match.nn_ratio
    assert merged.geometry == base.geometry


def test_lists_are_replaced_wholesale_not_merged(tmp_path: Path) -> None:
    """Element-wise list merging would make a format impossible to *remove*."""
    base = load_config(use_env=False)
    assert len(base.report.formats) > 1, "premise: the default ships several formats"

    path = _write(tmp_path, {"report": {"formats": ["json"]}})
    assert load_config(path, use_env=False).report.formats == ("json",)


# ---------------------------------------------------------------------------
# override syntax
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("override", "dotted", "expected"),
    [
        ("geometry.max_iters=25", ITERS, 25),
        ("local_match.nn_ratio=0.65", RATIO, 0.65),
        ("local_match.mutual_nn=false", ("local_match", "mutual_nn"), False),
        ("global_match.weights=null", ("global_match", "weights"), None),
        ("report.formats=[json]", ("report", "formats"), ("json",)),
        # YAML 1.1 only reads an exponent as a number when it is signed, so this
        # is 1600 while a bare `1.6e3` is the string "1.6e3". Pinned because it is
        # the kind of thing that would otherwise be discovered by a user.
        ("image.max_dimension=1.6e+3", ("image", "max_dimension"), 1600),
    ],
)
def test_overrides_use_yaml_scalar_rules(
    override: str, dotted: tuple[str, ...], expected: object
) -> None:
    """One syntax, not two.

    ``--set`` values parse exactly as they would in the config file, so ``false``
    and ``null`` and ``[json]`` mean what a reader of ``default.yaml`` expects
    rather than arriving as the strings ``"false"``, ``"null"``, ``"[json]"``.
    """
    assert _get(load_config(use_env=False, overrides=[override]), dotted) == expected


def test_an_unparseable_number_fails_loudly_rather_than_arriving_as_a_string() -> None:
    """The other half of "one syntax": YAML's rules apply even when they surprise.

    ``1e7`` is a *string* to YAML 1.1 -- the exponent needs a sign. Inheriting that
    rule is right, because the alternative is a second override syntax that
    disagrees with the config file, but it means the failure has to be loud. A
    field typed ``int`` refusing the string is what makes it loud.
    """
    with pytest.raises(ConfigError, match="max_dimension"):
        load_config(use_env=False, overrides=["image.max_dimension=1e7"])


def test_overrides_are_independent(tmp_path: Path) -> None:
    """Setting one knob must not disturb another -- the deep-merge sanity check."""
    base = load_config(use_env=False)
    cfg = load_config(use_env=False, overrides=["geometry.max_iters=25"])

    assert _get(cfg, ITERS) == 25
    assert _get(cfg, RATIO) == _get(base, RATIO)


def test_an_override_without_an_equals_sign_is_rejected() -> None:
    with pytest.raises(ConfigError, match="key=value"):
        load_config(use_env=False, overrides=["geometry.min_inliers"])


def test_an_empty_override_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="empty configuration key"):
        load_config(use_env=False, overrides=["=20"])


def test_an_override_naming_an_unknown_section_is_rejected() -> None:
    """A dotted path is not a licence to invent structure."""
    with pytest.raises(ConfigError, match=r"geomtery|extra"):
        load_config(use_env=False, overrides=["geomtery.min_inliers=20"])


# ---------------------------------------------------------------------------
# file handling
# ---------------------------------------------------------------------------
def test_a_missing_config_file_is_an_error_not_a_silent_default(tmp_path: Path) -> None:
    """Asking for a file that isn't there must not quietly run the defaults."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_malformed_yaml_is_reported_as_such(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("geometry: {min_inliers: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_a_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- geometry\n- local_match\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_config(path)


def test_an_empty_config_file_is_a_no_op(tmp_path: Path) -> None:
    """A file with nothing in it means "change nothing", not "reset everything"."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path, use_env=False) == load_config(use_env=False)


# ---------------------------------------------------------------------------
# invariants that catch a self-inconsistent configuration at startup
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("override", "match"),
    [
        # An escalation band that cannot contain anything: the local stage would
        # never fire on an ambiguous pair, which is the case it exists for.
        (
            ["global_match.local_trigger_distance=0.5", "global_match.distance_threshold=1.0"],
            "escalation band",
        ),
        # Not a band-pass filter any more, so the ROI response is meaningless.
        (["local_match.roi.dog_sigma_high=0.5", "local_match.roi.dog_sigma_low=2.0"], "band-pass"),
        # A floor above the detection budget rejects every pair by construction.
        (
            ["local_match.min_keypoints_per_side=5000", "local_match.max_features=2000"],
            "unsatisfiable",
        ),
        # Four stride-2 stages: a size off the 16-grid misaligns attribution.
        (["image.embed_size=100"], "divisible by 16"),
    ],
)
def test_self_contradictory_settings_are_refused_at_load(override: list[str], match: str) -> None:
    """Each of these is a configuration whose parts are individually valid.

    Range checks cannot catch them -- every value is positive and in bounds -- so
    the invariant has to be stated across fields. Failing at load is what keeps a
    run from producing plausible output under settings that cannot mean anything.
    """
    with pytest.raises(ConfigError, match=match):
        load_config(use_env=False, overrides=override)


@pytest.mark.parametrize(
    "override",
    [
        "local_match.nn_ratio=1.5",  # a ratio above 1 accepts everything
        "local_match.nn_ratio=0.0",
        "image.max_dimension=32",  # below the floor
        "local_match.roi.min_area=0",
        "local_match.enhance.scale=0.5",  # would be a downscale
        "global_match.attribution.stage=block5",  # no such stage
        "geometry.affine.method=nonsense",
    ],
)
def test_out_of_range_values_are_refused(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config(use_env=False, overrides=[override])


# ---------------------------------------------------------------------------
# audit trail
# ---------------------------------------------------------------------------
def test_fingerprint_is_stable_across_runs(cfg: Settings) -> None:
    assert cfg.fingerprint() == load_config(use_env=False).fingerprint()


def test_fingerprint_ignores_key_order(tmp_path: Path) -> None:
    """Two files that say the same thing differently must hash the same.

    The fingerprint goes in every report's audit block to tie a result to the
    settings that produced it. If cosmetic reordering changed the hash, two
    identical runs would look like different experiments.
    """
    first = _write(
        tmp_path, {"geometry": {"max_iters": 15}, "local_match": {"nn_ratio": 0.7}}, "a.yaml"
    )
    second = _write(
        tmp_path, {"local_match": {"nn_ratio": 0.7}, "geometry": {"max_iters": 15}}, "b.yaml"
    )

    assert load_config(first, use_env=False).fingerprint() == (
        load_config(second, use_env=False).fingerprint()
    )


def test_fingerprint_changes_when_any_value_changes(cfg: Settings) -> None:
    """The property that makes it evidence: a different config, a different hash."""
    nudged = load_config(use_env=False, overrides=["geometry.max_iters=17"])
    deeper = load_config(use_env=False, overrides=["local_match.roi.dilate_px=9"])

    fingerprints = {cfg.fingerprint(), nudged.fingerprint(), deeper.fingerprint()}
    assert len(fingerprints) == 3


def test_to_yaml_round_trips_through_the_loader(cfg: Settings, tmp_path: Path) -> None:
    """A dumped config must reload to the same settings, hash included.

    This is what lets a report ship the exact configuration that produced it and
    a reader reproduce the run from that file alone.
    """
    path = tmp_path / "dumped.yaml"
    path.write_text(cfg.to_yaml(), encoding="utf-8")
    reloaded = load_config(path, use_env=False)

    assert reloaded == cfg
    assert reloaded.fingerprint() == cfg.fingerprint()


def test_settings_are_frozen(cfg: Settings) -> None:
    """No stage may retune a threshold mid-run and leave the audit block lying."""
    with pytest.raises(ValidationError, match=r"frozen|immutable"):
        cfg.geometry.min_inliers = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# cross-section invariants
# ---------------------------------------------------------------------------
def test_a_similarity_model_is_refused_because_it_cannot_flip() -> None:
    """Bug 2, made unreachable by construction rather than by a comment.

    ``estimateAffinePartial2D`` returns ``[[a, -b, tx], [b, a, ty]]``, whose
    determinant is ``a**2 + b**2 > 0``, so a ``det < 0`` flip test can never fire.
    The prototype shipped in exactly that state and reported "Flip detected: no"
    for every input, which reads as a measurement and is not one. Refusing to start
    is better than running a configuration whose headline capability is
    mathematically absent.
    """
    with pytest.raises(ConfigError, match="unreachable"):
        load_config(use_env=False, overrides=["geometry.affine.model=similarity"])


def test_a_keypoint_floor_beneath_min_inliers_is_refused() -> None:
    """Mutual-NN matching is injective, so the two gates are arithmetically linked.

    Correspondences -- and therefore inlier rows, which are a subset of them --
    cannot exceed the smaller keypoint set. A floor beneath ``min_inliers`` admits
    pairs whose verification is impossible before it is attempted, and they then
    fail one gate later as ``too_few_matches``, blaming the matcher for a starved
    detector. The invariant is what keeps each rejection reason true about its own
    cause.
    """
    with pytest.raises(ConfigError, match="arithmetically impossible"):
        load_config(
            use_env=False,
            overrides=["local_match.min_keypoints_per_side=10", "geometry.min_inliers=15"],
        )


def test_the_shipped_floor_is_exactly_the_loosest_admissible_one(cfg: Settings) -> None:
    """And it should stay that way, because over-refusing is the failure mode.

    The floor's job is to skip work that cannot succeed, not to express a quality
    opinion -- that is ``roi.min_keypoints``, which is a separate knob for exactly
    this reason. When the two were one, the value chosen for ROI quality (100)
    silently became the matching floor, and every 256x256 panel in ``inputs/`` was
    refused with ``too_few_keypoints`` for a shortfall that was really the
    matcher's. Pinning equality makes a future raise a deliberate act with a test
    to change rather than a default someone drifts.
    """
    assert cfg.local_match.min_keypoints_per_side == cfg.geometry.min_inliers
    assert cfg.local_match.roi.min_keypoints > cfg.local_match.min_keypoints_per_side, (
        "premise: the two knobs are genuinely set to different values"
    )


def test_either_floor_above_the_feature_budget_is_refused() -> None:
    """A floor ORB cannot reach rejects every input, which is not a strict config."""
    for key in ("local_match.min_keypoints_per_side", "local_match.roi.min_keypoints"):
        with pytest.raises(ConfigError, match="unsatisfiable"):
            load_config(use_env=False, overrides=[f"{key}=3000", "local_match.max_features=2000"])
