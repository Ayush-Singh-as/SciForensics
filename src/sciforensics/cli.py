"""Command-line entry point — the ``sciforensics`` console script.

This module is what ``[project.scripts]`` in ``pyproject.toml`` points at, so
its absence was not a missing convenience: the declared entry point could not
resolve, which broke the wheel-install CI job and every command in the plan's
verification section.

**Scope.** This is a thin adapter, deliberately. It parses arguments, loads
config, builds one :class:`~sciforensics.pipeline.Pipeline`, and renders what
comes back. It holds no thresholds (bug 10: every threshold lives in
``configs/default.yaml``), and it makes no forensic decisions of its own — a
verdict rendered here is exactly the verdict :mod:`sciforensics.fusion`
produced. When a number looks wrong, the bug is upstream of this file.

**Commands.** ``compare`` and ``cmfd`` are the two the plan verifies against.
``config`` dumps the resolved settings (the fastest way to answer "which
threshold is actually in effect?"), ``version`` prints provenance, and ``serve``
runs the API behind the web demo. Reports are written by ``--report`` on the two
analysis commands rather than by a separate verb, since a report is always *of*
an analysis. The ``index``/``eval`` verbs named in the plan belong to stages
B6/C2 and are not stubbed here: a subcommand that exists but does nothing is the
same lie as a detector that was switched off but reported "no regions found".

**Exit codes.** ``0`` success, ``1`` a handled error (bad config, unreadable
image, disabled stage), ``2`` argument-parsing failure (Typer's own). The
verdict deliberately does *not* affect the exit code — "clean" is a successful
run, and a CI job that treats a finding as a build failure would encourage
suppressing findings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from sciforensics import __version__
from sciforensics.config import ConfigError, Settings, default_config_path, load_config
from sciforensics.runtime import get_logger, seed_everything, setup_logging
from sciforensics.types import BBox, CopyMoveResult, RejectionReason, ScanResult

app = typer.Typer(
    name="sciforensics",
    help="Forensic detection of image reuse and manipulation in scientific figures.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# stdout carries results, stderr carries diagnostics. Keeping them apart is what
# lets `sciforensics compare ... --json | jq` work while logs stay visible.
_out = Console()
_err = Console(stderr=True)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# shared options
# ---------------------------------------------------------------------------
# Typer resolves annotations at import time, so these are module-level singletons
# rather than being rebuilt per command.
_ConfigOpt = typer.Option(
    None,
    "--config",
    "-c",
    help="YAML config layered over configs/default.yaml. May be partial.",
    exists=True,
    dir_okay=False,
    readable=True,
)
_SetOpt = typer.Option(
    None,
    "--set",
    "-s",
    metavar="KEY=VALUE",
    help="Override one config key, e.g. -s geometry.min_inliers=20. Repeatable.",
)
_DeviceOpt = typer.Option(
    None, "--device", help="Torch device: auto, cpu, cuda, cuda:0. Overrides runtime.device."
)
_JsonOpt = typer.Option(False, "--json", help="Emit the full result as JSON on stdout.")
_OutOpt = typer.Option(
    None,
    "--out",
    "-o",
    help="Write the result JSON to this path (in addition to the rendered summary).",
    dir_okay=False,
)
_ReportOpt = typer.Option(
    None,
    "--report",
    "-r",
    help="Write an HTML/PDF/JSON report into this directory (created if needed).",
    file_okay=False,
)
_FormatOpt = typer.Option(
    None,
    "--format",
    "-f",
    help="Report format(s), overriding report.formats: pdf, html, png, json. Repeatable.",
)
_LogLevelOpt = typer.Option("WARNING", "--log-level", help="DEBUG, INFO, WARNING, ERROR.")
_LogFormatOpt = typer.Option("console", "--log-format", help="console or json.")


def _bootstrap(
    config: Path | None,
    overrides: list[str] | None,
    log_level: str,
    log_format: str,
) -> Settings:
    """Load config and configure logging, or exit 1 with a readable message.

    Logging is configured *before* config is loaded so that a config failure is
    itself reported through the same handler.
    """
    if log_format not in {"console", "json"}:
        _err.print(f"[red]error:[/] --log-format must be console or json, got {log_format!r}")
        raise typer.Exit(1)

    setup_logging(level=log_level, fmt=log_format)  # type: ignore[arg-type]

    try:
        return load_config(config, overrides=overrides or None)
    except ConfigError as exc:
        # ConfigError already carries the offending key and the invariant it
        # violated; re-wrapping would only bury it.
        _err.print(f"[red]config error:[/] {exc}")
        raise typer.Exit(1) from exc


def _build_pipeline(cfg: Settings, device: str | None) -> Any:
    """Import and construct the pipeline.

    ``torch`` is imported here rather than at module scope so ``--help`` and
    ``version`` stay fast and keep working in an environment where the heavy
    optional stack is broken or absent.
    """
    try:
        from sciforensics.pipeline import Pipeline
    except ImportError as exc:  # pragma: no cover - environment-dependent
        _err.print(
            f"[red]error:[/] could not import the pipeline ({exc}).\n"
            'Install the full stack with: pip install -e ".[dev]"'
        )
        raise typer.Exit(1) from exc

    seed_everything(cfg.runtime.seed)
    return Pipeline(cfg, device=device)


def _emit(result: ScanResult | CopyMoveResult, *, as_json: bool, out: Path | None) -> None:
    """Render to stdout and/or write JSON.

    ``mode="json"`` so ``Path`` and ``Enum`` values serialise as strings; the
    result models are pydantic, so this is their own schema rather than a
    hand-maintained dict that could drift from it.
    """
    payload = result.model_dump(mode="json")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _err.print(f"[dim]wrote {out}[/]")

    if as_json:
        _out.print_json(json.dumps(payload))


def _verdict_style(result: ScanResult | CopyMoveResult) -> str:
    if result.confidence >= 0.75:
        return "bold red"
    if result.confidence >= 0.5:
        return "bold yellow"
    return "green"


def _render_header(result: ScanResult | CopyMoveResult) -> None:
    _out.print(f"\n[{_verdict_style(result)}]{result.summary_line}[/]")
    if not result.calibrated:
        # Bug-adjacent honesty: rules.py weights are hand-set, not fitted (B5).
        # Printing a bare 0.87 without this line is the same overclaiming the
        # legacy verdict strings committed.
        _out.print(
            "[dim]Confidence is a monotone score, not a calibrated probability "
            "(no calibrator fitted).[/]"
        )
    for note in result.notes:
        _out.print(f"  [yellow]note:[/] {note}")
    for warning in result.warnings:
        _out.print(f"  [magenta]warning:[/] {warning}")


def _render_contributions(result: ScanResult | CopyMoveResult) -> None:
    """Per-evidence contribution bars — which signal actually drove the score."""
    if not result.contributions:
        return
    table = Table(title="Evidence contributions", title_justify="left", header_style="bold")
    table.add_column("feature")
    table.add_column("value", justify="right")
    table.add_column("weight", justify="right")
    table.add_column("log-odds", justify="right")
    table.add_column("")
    # Rank by magnitude: the reader wants the drivers, not the schema order.
    for item in sorted(result.contributions, key=lambda c: abs(c.contribution), reverse=True):
        bar_len = min(20, int(abs(item.contribution) * 4))
        bar = ("[red]" if item.contribution > 0 else "[green]") + "#" * bar_len + "[/]"
        table.add_row(
            item.name,
            f"{item.value:.3f}",
            f"{item.weight:+.2f}",
            f"{item.contribution:+.2f}",
            bar if bar_len else "",
        )
    _out.print(table)


def _render_scan(result: ScanResult, *, verbose: bool) -> None:
    _render_header(result)

    g = result.global_evidence
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Global distance", f"{g.distance:.4f}  (threshold {g.distance_threshold:.2f})")
    table.add_row("Global similarity", f"{g.similarity:.4f}")
    table.add_row("Local stage", "ran" if g.triggered_local else "not triggered")

    if result.keypoints is not None:
        k = result.keypoints
        table.add_row("Keypoints", f"{k.left} vs {k.right}")

    if result.matches is not None:
        m = result.matches
        table.add_row("Matches", f"{m.raw} raw -> {m.ratio_passed} ratio -> {m.good} good")
        # The 12-vs-2000 failure (bug 4) is visible exactly here.
        table.add_row(
            "Distinct",
            f"{m.distinct_left} left / {m.distinct_right} right"
            + ("" if m.is_injective else "  [red](not injective)[/]"),
        )

    if result.geometry is not None:
        geo = result.geometry
        if geo.verified:
            table.add_row("Geometry", f"[green]verified[/] via {geo.method}")
        else:
            # Naming the gate is the whole point of RejectionReason: "not
            # verified" and "degenerate correspondences" are different findings.
            reason = (
                geo.rejection_reason.value
                if geo.rejection_reason is not RejectionReason.NONE
                else "not verified"
            )
            table.add_row("Geometry", f"[yellow]{reason}[/] via {geo.method}")
        table.add_row("Inliers", f"{geo.inlier_count} ({geo.inlier_ratio:.2f} of matches)")
        table.add_row("Distinct inliers", str(geo.distinct_inliers))
        if geo.reproj_rms is not None:
            table.add_row("Reprojection RMS", f"{geo.reproj_rms:.2f} px")
        table.add_row("Inlier spread", f"{geo.inlier_spread:.3f} of diagonal")
        table.add_row("Matched area", f"{geo.matched_area_fraction:.3f}")

        if geo.transform is not None:
            t = geo.transform
            # Bugs 2 and 3 are read off these three lines.
            table.add_row("Rotation", f"{t.rotation_deg:+.2f} deg")
            table.add_row(
                "Scale",
                f"{t.scale:.4f}"
                + (f"  (x {t.scale_x:.3f}, y {t.scale_y:.3f})" if t.anisotropic else ""),
            )
            table.add_row("Flip", "[bold]yes[/]" if t.flip else "no")
            if t.sheared:
                table.add_row("Shear", f"{t.shear_deg:+.2f} deg")
            table.add_row("Affine model", t.model)

    _out.print(table)

    if verbose:
        _render_contributions(result)
        _render_audit(result)


def _render_copy_move(result: CopyMoveResult, *, verbose: bool) -> None:
    _render_header(result)

    ev = result.evidence
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Keypoints", str(ev.keypoints))
    table.add_row("Self-matches", str(ev.self_matches))
    # cluster_count vs len(regions) is the funnel: candidates that failed
    # per-cluster geometric verification. Bug 6 made this distinction impossible.
    table.add_row("Offset clusters", str(ev.cluster_count))
    table.add_row("Verified regions", str(len(ev.regions)))
    _out.print(table)

    if ev.regions:
        regions = Table(title="Cloned regions", title_justify="left", header_style="bold")
        regions.add_column("#", justify="right")
        regions.add_column("source bbox")
        regions.add_column("clone bbox")
        regions.add_column("inliers", justify="right")
        regions.add_column("rot", justify="right")
        regions.add_column("scale", justify="right")
        regions.add_column("flip")
        for i, region in enumerate(ev.regions, start=1):
            regions.add_row(
                str(i),
                _fmt_box(region.source_box),
                _fmt_box(region.target_box),
                str(region.inlier_count),
                f"{region.rotation_deg:+.1f}",
                f"{region.scale:.3f}",
                "[bold]yes[/]" if region.flip else "no",
            )
        _out.print(regions)

    if verbose:
        _render_contributions(result)
        _render_audit(result)


def _fmt_box(box: BBox) -> str:
    """``(x, y, w, h)`` in original image pixels — the frame the user handed in."""
    x, y, w, h = box
    return f"({x},{y}) {w}x{h}"


def _render_audit(result: ScanResult | CopyMoveResult) -> None:
    audit = result.audit.model_dump(mode="json")
    table = Table(title="Audit", title_justify="left", show_header=False, box=None, padding=(0, 2))
    for key, value in audit.items():
        if value is None or value == [] or value == {}:
            continue
        table.add_row(key, str(value))
    if result.timings:
        table.add_row("timings", ", ".join(f"{k} {v:.3f}s" for k, v in result.timings.items()))
    _out.print(table)


def _write_report(analysis: Any, cfg: Settings, directory: Path, formats: list[str] | None) -> None:
    """Render a report, or exit 1 explaining why not.

    Imported here rather than at module scope: the report layer pulls in Jinja2
    and (optionally) WeasyPrint, and a `compare` that writes no report should not
    pay for them or fail because they are missing.
    """
    from sciforensics.pipeline import CopyMoveAnalysis

    try:
        from sciforensics.report import render_copy_move, render_pair
    except ImportError as exc:  # pragma: no cover - environment-dependent
        _err.print(
            f"[red]error:[/] the report extra is not installed ({exc}).\n"
            'Install it with: pip install -e ".[report]"'
        )
        raise typer.Exit(1) from exc

    wanted = tuple(formats) if formats else None
    valid = {"pdf", "html", "png", "json"}
    if wanted and not set(wanted) <= valid:
        _err.print(f"[red]error:[/] --format must be one of {sorted(valid)}, got {list(wanted)}")
        raise typer.Exit(1)

    render = render_copy_move if isinstance(analysis, CopyMoveAnalysis) else render_pair
    try:
        written = render(analysis, cfg, directory, formats=wanted)
    except Exception as exc:
        raise _fail(exc, "report rendering failed") from exc

    for warning in written.warnings:
        # A missing WeasyPrint is a downgrade, not a failure: the HTML is written
        # and prints to PDF from any browser.
        _err.print(f"  [magenta]warning:[/] {warning}")
    if written.primary is not None:
        _out.print(f"\n[bold]Report:[/] {written.primary}")


def _fail(exc: Exception, what: str) -> typer.Exit:
    _err.print(f"[red]{what}:[/] {exc}")
    _log.debug("%s", what, exc_info=True)
    return typer.Exit(1)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
@app.command()
def compare(
    left: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    right: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    config: Path | None = _ConfigOpt,
    set_: Optional[list[str]] = _SetOpt,
    device: str | None = _DeviceOpt,
    force_local: bool = typer.Option(
        False,
        "--force-local",
        help="Run the local stage even if the global distance did not trigger it.",
    ),
    attribution: bool | None = typer.Option(
        None, "--attribution/--no-attribution", help="Override attribution.enabled."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Also show evidence contributions and the audit block."
    ),
    report: Path | None = _ReportOpt,
    formats: Optional[list[str]] = _FormatOpt,
    as_json: bool = _JsonOpt,
    out: Path | None = _OutOpt,
    log_level: str = _LogLevelOpt,
    log_format: str = _LogFormatOpt,
) -> None:
    """Compare two images for reuse or manipulation."""
    cfg = _bootstrap(config, set_, log_level, log_format)
    pipeline = _build_pipeline(cfg, device)
    try:
        analysis = pipeline.compare(
            left, right, force_local=force_local, attribution=attribution
        )
    except Exception as exc:
        raise _fail(exc, "compare failed") from exc

    _render_scan(analysis.result, verbose=verbose)
    _emit(analysis.result, as_json=as_json, out=out)
    if report is not None:
        _write_report(analysis, cfg, report, formats)


@app.command()
def cmfd(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    config: Path | None = _ConfigOpt,
    set_: Optional[list[str]] = _SetOpt,
    device: str | None = _DeviceOpt,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Also show evidence contributions and the audit block."
    ),
    report: Path | None = _ReportOpt,
    formats: Optional[list[str]] = _FormatOpt,
    as_json: bool = _JsonOpt,
    out: Path | None = _OutOpt,
    log_level: str = _LogLevelOpt,
    log_format: str = _LogFormatOpt,
) -> None:
    """Search one image for cloned (copy-move) regions."""
    cfg = _bootstrap(config, set_, log_level, log_format)
    pipeline = _build_pipeline(cfg, device)
    try:
        analysis = pipeline.copy_move(image)
    except Exception as exc:
        # Includes PipelineError when copy_move.enabled is false -- a deliberate
        # refusal, not a crash, and its message says so.
        raise _fail(exc, "copy-move search failed") from exc

    _render_copy_move(analysis.result, verbose=verbose)
    _emit(analysis.result, as_json=as_json, out=out)
    if report is not None:
        _write_report(analysis, cfg, report, formats)


@app.command(name="config")
def config_cmd(
    config: Path | None = _ConfigOpt,
    set_: Optional[list[str]] = _SetOpt,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of YAML."),
    log_level: str = _LogLevelOpt,
    log_format: str = _LogFormatOpt,
) -> None:
    """Print the fully resolved configuration.

    The answer to "which threshold is actually in effect?", after defaults, the
    user file, ``SCIFORENSICS_*`` and ``--set`` have all been layered.
    """
    cfg = _bootstrap(config, set_, log_level, log_format)
    payload = cfg.model_dump(mode="json")
    if as_json:
        _out.print_json(json.dumps(payload))
        return
    try:
        import yaml

        _out.print(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    except ImportError:  # pragma: no cover
        _out.print_json(json.dumps(payload))


@app.command()
def serve(
    config: Path | None = _ConfigOpt,
    set_: Optional[list[str]] = _SetOpt,
    host: str | None = typer.Option(None, "--host", help="Overrides api.host."),
    port: int | None = typer.Option(None, "--port", help="Overrides api.port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code change (dev only)."),
    log_level: str = _LogLevelOpt,
    log_format: str = _LogFormatOpt,
) -> None:
    """Run the HTTP API backing the web demo."""
    cfg = _bootstrap(config, set_, log_level, log_format)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - environment-dependent
        _err.print(
            f"[red]error:[/] the api extra is not installed ({exc}).\n"
            'Install it with: pip install -e ".[api]"'
        )
        raise typer.Exit(1) from exc

    bind_host = host or cfg.api.host
    bind_port = port or cfg.api.port
    _out.print(f"[bold]SciForensics API[/] http://{bind_host}:{bind_port}  (docs at /docs)")

    if reload:
        # Reload needs an import string rather than an object, and the reloader
        # re-imports in a fresh process, so config has to come from the
        # environment rather than the Settings we just built.
        uvicorn.run(
            "sciforensics.api.app:create_app",
            factory=True,
            host=bind_host,
            port=bind_port,
            reload=True,
        )
        return

    from sciforensics.api import create_app

    uvicorn.run(create_app(cfg), host=bind_host, port=bind_port, log_level=log_level.lower())


@app.command()
def version(
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Print version and provenance."""
    from sciforensics import audit as audit_mod

    info = {
        "version": __version__,
        "tool_version": audit_mod.tool_version(),
        "git_commit": audit_mod.git_commit(),
        "git_dirty": audit_mod.git_is_dirty(),
        "default_config": str(default_config_path()),
        "python": sys.version.split()[0],
    }
    if as_json:
        _out.print_json(json.dumps(info))
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    for key, value in info.items():
        table.add_row(key, "-" if value is None else str(value))
    _out.print(table)


def main() -> None:
    """Console-script shim, for ``python -m sciforensics.cli`` parity."""
    app()


if __name__ == "__main__":
    main()
