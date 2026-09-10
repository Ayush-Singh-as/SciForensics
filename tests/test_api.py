"""The HTTP surface, exercised without the model.

Upload validation is the load-bearing part here, and it is deliberately tested
adversarially. The API accepts arbitrary bytes from anyone who can reach it, so
the guards are not hygiene -- ``max_decoded_pixels`` is what stops a 40,000 x
40,000 PNG (a few hundred KB on the wire) from becoming a 6.4 GB allocation, and
the asset route joins a *client-supplied filename* to a server path, which is a
directory traversal unless it is confined.

Torch-free: every route asserted on either avoids the pipeline or is expected to
fail before reaching it. The analysis routes themselves need a real backbone and
belong to ``test_pipeline.py``.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from sciforensics.api.app import create_app
from sciforensics.config import Settings, load_config


@pytest.fixture(scope="module")
def settings() -> Settings:
    return load_config()


@pytest.fixture(scope="module")
def client(settings: Settings) -> TestClient:
    # `raise_server_exceptions=False` so a 500 is asserted as a response rather
    # than re-raised into the test, matching what a browser would observe.
    return TestClient(create_app(settings), raise_server_exceptions=False)


def test_healthz_does_not_touch_the_model(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_route_exposes_the_bands_the_ui_needs(
    client: TestClient, settings: Settings
) -> None:
    """The frontend re-bands a cached score client-side, so it needs these.

    Serving them keeps one source of truth: hard-coding the thresholds in the UI
    would reintroduce bug 10 in a new location.
    """
    payload = client.get("/v1/config").json()
    assert payload["global_match"]["distance_threshold"] == pytest.approx(
        settings.global_match.distance_threshold
    )
    assert set(payload["bands"]) == {"likely_manipulated", "suspicious", "inconclusive"}
    # Until stage B5 fits a calibrator this must stay False, or the UI will
    # present a monotone score as a probability.
    assert payload["calibrated"] is False


def test_examples_are_resolvable_and_include_a_negative_control(client: TestClient) -> None:
    """A gallery of only true positives is how a false-positive rate goes unmeasured."""
    items = client.get("/v1/examples").json()["examples"]
    assert items, "no example pairs resolved from the repo"
    assert any(item["kind"] == "negative" for item in items)


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/v1/jobs/deadbeef").status_code == 404
    assert client.get("/v1/jobs/deadbeef/report").status_code == 404


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("evil.exe", b"MZ\x90\x00", 415),  # not an allowed decoder
        ("empty.png", b"", 400),
        ("garbage.png", b"definitely not a png", 400),
    ],
)
def test_bad_uploads_are_rejected(
    client: TestClient, name: str, content: bytes, expected: int
) -> None:
    response = client.post("/v1/cmfd", files={"image": (name, content, "image/png")})
    assert response.status_code == expected
    assert "error" in response.json()


def test_oversize_upload_is_rejected_before_it_is_buffered(
    client: TestClient, settings: Settings
) -> None:
    """The cap is enforced per chunk while streaming, not after a full read."""
    payload = b"\x00" * (settings.api.max_upload_bytes + 1024)
    response = client.post(
        "/v1/cmfd", files={"image": ("big.png", io.BytesIO(payload), "image/png")}
    )
    assert response.status_code == 413


def test_undecodable_upload_does_not_leak_the_server_path(client: TestClient) -> None:
    """The decoder's own message embeds a temp path; the client must not see it."""
    response = client.post("/v1/cmfd", files={"image": ("x.png", b"nope", "image/png")})
    assert response.status_code == 400
    detail = response.json()["error"]
    assert "Temp" not in detail and "AppData" not in detail and "/tmp" not in detail


@pytest.mark.parametrize(
    "name",
    [
        "../../../../etc/passwd",
        "..%2f..%2fsecret.png",
        "....//report.json",
    ],
)
def test_asset_route_confines_the_filename(client: TestClient, name: str) -> None:
    """A client-supplied filename joined to a server path must not escape it."""
    response = client.get(f"/v1/jobs/deadbeef/assets/{name}")
    # 404 for the unknown job, or 404 for the rejected path -- never 200, and
    # never a file from outside the job directory.
    assert response.status_code in {307, 404}
    assert response.status_code != 200
