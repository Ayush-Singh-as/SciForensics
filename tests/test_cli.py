"""The console script: every command must at least *build*.

The regression here is structural rather than numeric. ``pyproject.toml``
declared ``sciforensics = "sciforensics.cli:app"`` while ``cli.py`` did not
exist, so the entry point could not resolve: the wheel-install CI job failed and
every command in the plan's verification section was unrunnable. A 14,500-line
library with no way to invoke it is not a tool.

The subtler failure this guards is that Typer builds its click parameters by
*introspecting annotations at import time*, so a type it cannot interpret is not
a lint nit -- it is an ``AssertionError`` during command construction that takes
down every subcommand at once, including ``--help``. That is exactly what
``Optional[list[str]]`` written as ``list[str] | None`` does here. A test that
only imported the module would not catch it; the failure needs the command
actually built, which is what invoking ``--help`` forces.

Deliberately torch-free. These tests drive the argument, config and error paths,
none of which need the model, so they run in a bare environment and stay fast.
The rendering of a real result is covered by ``test_pipeline.py``, which has the
pipeline available to produce one.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sciforensics.cli import app

runner = CliRunner()

# Every command registered on the app. Parametrising over this rather than
# listing names twice means a new subcommand is covered the moment it is added.
COMMANDS = ["compare", "cmfd", "config", "splits", "bench", "serve", "version"]


def test_app_exposes_the_documented_commands() -> None:
    registered = {
        info.name or info.callback.__name__  # type: ignore[union-attr]
        for info in app.registered_commands
    }
    assert registered == set(COMMANDS)


def test_root_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_builds(command: str) -> None:
    """``--help`` forces Typer to construct the click parameters.

    This is the assertion that fails on an annotation Typer cannot introspect,
    and it fails for *all* commands at once rather than just the offending one.
    """
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


def test_version_reports_provenance() -> None:
    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    # git_commit/git_dirty may legitimately be None outside a checkout, so only
    # the keys that are always derivable are asserted on.
    assert payload["version"]
    assert payload["python"]
    assert payload["default_config"].endswith("default.yaml")


def test_config_dumps_resolved_settings() -> None:
    result = runner.invoke(app, ["config", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "geometry" in payload
    assert "local_match" in payload


def test_config_applies_an_override() -> None:
    """``--set`` is the repeatable list option that broke command construction.

    Asserting the value actually lands proves the annotation is not merely
    *accepted* by Typer but still parsed as a list of strings.
    """
    base = json.loads(runner.invoke(app, ["config", "--json"]).stdout)
    probe = base["geometry"]["max_iters"] + 1000

    result = runner.invoke(app, ["config", "--json", "--set", f"geometry.max_iters={probe}"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["geometry"]["max_iters"] == probe
    # An override must not leak into a neighbouring key.
    assert payload["local_match"]["nn_ratio"] == base["local_match"]["nn_ratio"]


def test_repeated_overrides_all_apply() -> None:
    base = json.loads(runner.invoke(app, ["config", "--json"]).stdout)
    iters = base["geometry"]["max_iters"] + 500

    result = runner.invoke(
        app,
        [
            "config",
            "--json",
            "--set",
            f"geometry.max_iters={iters}",
            "--set",
            "local_match.nn_ratio=0.65",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["geometry"]["max_iters"] == iters
    assert payload["local_match"]["nn_ratio"] == pytest.approx(0.65)


def test_malformed_override_exits_one_with_a_readable_message() -> None:
    result = runner.invoke(app, ["config", "--set", "nonsense"])
    assert result.exit_code == 1
    assert "key=value" in result.output


def test_unknown_config_key_is_rejected() -> None:
    """Bug 10's structural fix: an unknown key is a startup failure, not a no-op."""
    result = runner.invoke(app, ["config", "--set", "geometry.definitely_not_a_key=1"])
    assert result.exit_code == 1


def test_bad_log_format_exits_one() -> None:
    result = runner.invoke(app, ["config", "--log-format", "yaml"])
    assert result.exit_code == 1
    assert "console" in result.output


def test_missing_image_is_an_argument_error() -> None:
    """Exit 2, from Typer's own `exists=True` check, before any model loads."""
    result = runner.invoke(app, ["compare", "does_not_exist_a.png", "does_not_exist_b.png"])
    assert result.exit_code == 2


def test_compare_exposes_report_options() -> None:
    """A3's `--report` must be reachable from the command the plan verifies."""
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--report" in result.output
    assert "--format" in result.output
