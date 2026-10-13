"""Phase A contracts: asynchronous jobs, real reflection matching, report artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sciforensics.api.app import create_app
from sciforensics.api.evidence import viewer_payload
from sciforensics.config import Settings
from sciforensics.global_match.backbone import EmbeddingNet
from sciforensics.global_match.embed import Embedder
from sciforensics.pipeline import Pipeline
from sciforensics.report.render import render_pair
from sciforensics.types import Stage
from tests.helpers import textured_image


def make_pipeline(cfg: Settings) -> Pipeline:
    return Pipeline(
        cfg,
        device="cpu",
        embedder=Embedder(cfg.global_match, image=cfg.image, device="cpu", model=EmbeddingNet()),
    )


def wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in {"complete", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_reflected_image_verifies_in_submitted_coordinates(cfg: Settings, tmp_path: Path) -> None:
    image = textured_image()
    left, right = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(left), image)
    cv2.imwrite(str(right), cv2.flip(image, 1))
    progress: list[Stage] = []
    analysis = make_pipeline(cfg).compare(
        left, right, force_local=True, attribution=False, progress=progress.append
    )
    geometry = analysis.result.geometry
    assert geometry is not None and geometry.verified
    assert geometry.transform is not None and geometry.transform.flip
    assert geometry.transform.scale == pytest.approx(1, abs=0.01)
    assert geometry.method == "ransac"
    assert progress == [Stage.GLOBAL, Stage.LOCAL, Stage.GEOMETRY, Stage.FUSION]
    corr = analysis.correspondences
    assert corr is not None
    mask = analysis.verification.inlier_mask if analysis.verification else np.zeros(len(corr), bool)
    assert np.max(np.abs(corr.src[mask, 0] + corr.dst[mask, 0] - (image.shape[1] - 1))) < 1
    payload = viewer_payload(analysis, limit=20)
    assert len(payload["matches"]) == 20
    assert set(payload["hulls"]) == {"left", "right"}
    assert payload["aspects"]["left"] == pytest.approx(image.shape[1] / image.shape[0])


def test_background_job_serves_evidence_assets_and_pdf(
    cfg: Settings, image_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def pdf(html: str, destination: Path, base_url: Path) -> None:
        destination.write_bytes(b"%PDF-test")

    monkeypatch.setattr("sciforensics.report.render._render_pdf", pdf)
    app = create_app(cfg)
    app.state.pipeline_state["pipeline"] = make_pipeline(cfg)
    with TestClient(app) as client:
        response = client.post(
            "/v1/compare?background=true",
            files={
                "left": ("a.png", image_pair[0].read_bytes(), "image/png"),
                "right": ("b.png", image_pair[1].read_bytes(), "image/png"),
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = wait_for_job(client, job_id)
        assert job["status"] == "complete", job
        assert job["result"]["geometry"]["verified"]
        assert job["report_format"] == "pdf"
        assert job["viewer"]["matches"]
        assert "directory" not in job
        with client.websocket_connect(f"/v1/jobs/{job_id}/progress") as socket:
            assert socket.receive_json()["status"] == "complete"
        report = client.get(f"/v1/jobs/{job_id}/report")
        assert report.headers["content-type"] == "application/pdf"
        assert report.content.startswith(b"%PDF")
        for overlay in job["overlays"]:
            assert client.get(f"/v1/jobs/{job_id}/assets/{overlay['file']}").status_code == 200


def test_failed_background_job_releases_admission(
    cfg: Settings, image_pair: tuple[Path, Path]
) -> None:
    class Broken:
        def compare(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("intentional failure")

    one = cfg.model_copy(update={"api": cfg.api.model_copy(update={"max_concurrent_jobs": 1})})
    app = create_app(one)
    app.state.pipeline_state["pipeline"] = Broken()
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/v1/compare?background=true",
                files={
                    "left": ("a.png", image_pair[0].read_bytes(), "image/png"),
                    "right": ("b.png", image_pair[1].read_bytes(), "image/png"),
                },
            )
            assert response.status_code == 202
            assert wait_for_job(client, response.json()["job_id"])["status"] == "failed"


def test_example_assets_are_packaged_and_unknown_paths_are_rejected(cfg: Settings) -> None:
    with TestClient(create_app(cfg)) as client:
        items = client.get("/v1/examples").json()["examples"]
        assert any(item["kind"] == "negative" for item in items)
        assert any(item["mode"] == "cmfd" for item in items)
        for item in items:
            assert client.get(item["left"]).status_code == 200
            if item["right"]:
                assert client.get(item["right"]).status_code == 200
        assert client.get("/v1/examples/unknown/left").status_code == 404


def test_png_report_is_a_real_image(
    cfg: Settings, image_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    analysis = make_pipeline(cfg).compare(*image_pair, attribution=False)
    written = render_pair(analysis, cfg, tmp_path / "report", formats=("png",))
    contact = tmp_path / "report/report.png"
    assert contact in written.images
    raster = cv2.imread(str(contact))
    assert raster is not None and raster.shape[0] > 400
    assert not written.warnings


def test_reload_preserves_explicit_demo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    import uvicorn
    from typer.testing import CliRunner

    from sciforensics.cli import app as cli_app
    from sciforensics.config import load_config

    captured: list[Settings] = []

    def run(app: Any, **kwargs: Any) -> None:
        assert kwargs["reload"] is True
        captured.append(Settings.model_validate_json(os.environ["SCIFORENSICS_SERVE_SETTINGS"]))

    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.delenv("SCIFORENSICS_SERVE_SETTINGS", raising=False)
    response = CliRunner().invoke(
        cli_app, ["serve", "--reload", "--set", "copy_move.nn_ratio=0.91"]
    )
    assert response.exit_code == 0, response.output
    assert captured[0].copy_move.nn_ratio == 0.91
    assert "SCIFORENSICS_SERVE_SETTINGS" not in os.environ
    assert (
        load_config(
            env={"SCIFORENSICS_WEIGHTS_URL": "https://example.com/weights.pth"}
        ).image.embed_resize
        == "letterbox"
    )
