"""Report rendering, and bug 7's regression.

**Bug 7.** The prototype drew its verdict with ``cv2.putText`` onto a fixed
420 px canvas. A full result put its last wrapped line at baseline y=415, so the
closing words were sliced off the bottom of *every* report it produced —
"scientific imagery" was cut in half in the committed outputs.

The test asserts the property rather than a pixel measurement: the complete
verdict text, the rejection explanation and the disclaimer must all appear in
the rendered output. Flowed HTML cannot clip, so this holds for any string
length in any language, which a canvas-size assertion would not.

Torch-free: the templates are driven directly with structural stand-ins, so the
text path is covered in a bare environment. ``render_pair`` itself needs a real
``PairAnalysis`` and is exercised by ``test_pipeline.py``.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "sciforensics" / "report" / "templates"

# The longest realistic rejection prose in RejectionReason.explanation. If any
# text clips, it clips here first.
LONG_EXPLANATION = (
    "Inliers collapsed onto too few distinct keypoints: many-to-one matches inflate the "
    "inlier count without adding independent geometric constraints."
)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _image(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        path=Path(f"inputs/{name}"),
        sha256="a" * 64,
        width=1920,
        height=1080,
        was_downscaled=True,
        analysed_at=(2048, 1152),
    )


def _scan_result() -> types.SimpleNamespace:
    """A worst case: refused geometry, a flip, notes, and warnings all present."""
    reason = types.SimpleNamespace(
        value="degenerate_correspondences", explanation=LONG_EXPLANATION
    )
    geometry = types.SimpleNamespace(
        verified=False,
        rejection_reason=reason,
        method="magsac",
        inlier_count=121,
        inlier_ratio=0.52,
        distinct_inliers=12,
        reproj_rms=3.4,
        inlier_spread=0.08,
        matched_area_fraction=0.02,
        condition_number=1.9e4,
        transform=types.SimpleNamespace(
            rotation_deg=45.2,
            scale=1.2,
            scale_x=1.2,
            scale_y=1.2,
            anisotropic=False,
            flip=True,
            sheared=False,
            shear_deg=0.0,
            translation=(12.5, -8.0),
            model="full",
        ),
    )
    return types.SimpleNamespace(
        left=_image("mountains.jpg"),
        right=_image("mountains_manipulated.jpg"),
        verdict=types.SimpleNamespace(label="Likely reused or manipulated"),
        confidence=0.87,
        calibrated=False,
        notes=("Only the global stage produced usable evidence.",),
        warnings=("Input exceeded max_dimension; analysed at reduced resolution.",),
        global_evidence=types.SimpleNamespace(
            distance=0.42,
            distance_threshold=1.0,
            similarity=0.78,
            embedding_dim=128,
            triggered_local=True,
        ),
        keypoints=types.SimpleNamespace(left=2000, right=12, detector="orb"),
        matches=types.SimpleNamespace(
            raw=980,
            ratio_passed=231,
            good=231,
            ratio=0.75,
            mutual_nn=True,
            distinct_left=231,
            distinct_right=12,
            is_injective=False,
        ),
        geometry=geometry,
        timings={"global": 0.31, "local": 1.2},
        contributions=(),
    )


def _render_pair_html(**overrides: object) -> str:
    result = _scan_result()
    context = {
        "title": "Image Comparison",
        "css": (TEMPLATES / "base.css").read_text(encoding="utf-8"),
        "result": result,
        "g": result.global_evidence,
        "k": result.keypoints,
        "m": result.matches,
        "geo": result.geometry,
        "t": result.geometry.transform,
        "overlays": {},
        "contributions": [],
        "audit": {"tool_version": "0.2.0.dev0", "git_commit": "bcb29bc"},
        "verdict_class": "is-alert",
    }
    context.update(overrides)
    return _env().get_template("report.html.j2").render(**context)


def test_verdict_text_is_never_clipped() -> None:
    """Bug 7. The whole verdict must survive into the output, not most of it."""
    html = _render_pair_html()
    assert "Likely reused or manipulated" in html
    # The full closing sentence of the disclaimer -- the prototype's canvas cut
    # this class of trailing prose off entirely.
    assert "absence of a finding is not proof of originality" in html.lower()


def test_rejection_reason_is_explained_in_full() -> None:
    """A named gate, in prose. "Not verified" is a different finding to this."""
    assert LONG_EXPLANATION in _render_pair_html()


def test_uncalibrated_score_is_labelled() -> None:
    """Until B5 fits a calibrator, the report must not imply a probability."""
    html = _render_pair_html()
    assert "monotone ranking value" in html
    assert "not a calibrated" in html or "not a probability" in html


def test_calibrated_result_drops_the_caveat() -> None:
    result = _scan_result()
    result.calibrated = True
    html = _render_pair_html(result=result, g=result.global_evidence)
    assert "monotone ranking value" not in html


def test_flip_is_reported_when_present() -> None:
    """Bugs 2 and 3: this row was unreachable in the prototype."""
    assert "Present" in _render_pair_html()


def test_non_injective_matching_is_flagged() -> None:
    """Bug 4: 231 matches over 12 distinct points must not read as clean."""
    assert "Not injective" in _render_pair_html()


def test_missing_optional_overlays_do_not_break_rendering() -> None:
    """StrictUndefined is on, so an absent overlay key must be handled, not assumed.

    Attribution is legitimately dropped when the map is degenerate, and the
    local stage may not run at all, so every overlay is optional.
    """
    html = _render_pair_html(overlays={})
    assert "Measurements" in html


def test_copymove_template_renders_without_regions() -> None:
    """"No clones found" must be distinguishable from "the detector was off"."""
    result = types.SimpleNamespace(
        image=_image("base_cells_1.png"),
        summary_line="No cloned regions found",
        verdict=types.SimpleNamespace(label="No evidence of reuse found"),
        confidence=0.04,
        calibrated=False,
        notes=(),
        warnings=(),
        evidence=types.SimpleNamespace(
            detected=False, cluster_count=3, regions=(), self_matches=64, keypoints=1800
        ),
        contributions=(),
        timings={},
    )
    html = _env().get_template("copymove.html.j2").render(
        title="Copy-Move Analysis",
        css=(TEMPLATES / "base.css").read_text(encoding="utf-8"),
        result=result,
        ev=result.evidence,
        overlays={},
        contributions=[],
        audit=None,
        verdict_class="is-clean",
    )
    assert "No cloned regions found" in html
    # The funnel gap: 3 candidate clusters, 0 verified regions.
    assert "failed per-region geometric verification" in html


def test_stylesheet_has_no_fixed_height_in_the_text_path() -> None:
    """Bug 7 structurally: a fixed height anywhere in the prose path can clip."""
    css = (TEMPLATES / "base.css").read_text(encoding="utf-8")
    for selector in (".verdict", ".caveat", "footer.disclaimer"):
        assert selector in css
    # The only heights in the sheet belong to bar graphics, never to text
    # containers. `max-height` on a text block would reintroduce the bug.
    assert "max-height" not in css
