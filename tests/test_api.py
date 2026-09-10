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
from pathlib import Path

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


# ---------------------------------------------------------------------------
# C3 -- limits that were configured but enforced by nothing
# ---------------------------------------------------------------------------
def test_readyz_distinguishes_alive_from_able_to_serve(client: TestClient) -> None:
    """A container can be alive but unable to serve, and the difference matters.

    `/healthz` deliberately never touches the model -- a healthcheck that ran an
    inference would mark the container unhealthy during a slow first request,
    exactly when it is working. `/readyz` is where load state belongs, so an
    orchestrator can stop routing without restarting.
    """
    payload = client.get("/readyz").json()
    assert payload["status"] in {"ready", "warming"}
    assert "model_loaded" in payload
    assert payload["capacity"] >= 1


def test_metrics_are_prometheus_text(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    for counter in (
        "sciforensics_requests_total",
        "sciforensics_rejected_total",
        "sciforensics_failed_total",
        "sciforensics_completed_total",
        "sciforensics_jobs_held",
    ):
        assert f"# TYPE {counter}" in body, counter
        assert f"\n{counter} " in f"\n{body}", counter


def test_rejected_uploads_still_count_as_requests(client: TestClient) -> None:
    """Metrics must reflect load, including work that was refused.

    A counter that only advanced on success would under-report exactly when the
    service is struggling.
    """
    before = int(
        next(
            line.split()[1]
            for line in client.get("/metrics").text.splitlines()
            if line.startswith("sciforensics_requests_total ")
        )
    )
    client.post("/v1/cmfd", files={"image": ("x.png", b"nope", "image/png")})
    after = int(
        next(
            line.split()[1]
            for line in client.get("/metrics").text.splitlines()
            if line.startswith("sciforensics_requests_total ")
        )
    )
    assert after == before + 1


def test_job_ttl_expires_records_and_removes_their_directories(tmp_path: Path) -> None:
    """`api.job_ttl_seconds` existed in config and was read by nothing.

    The API advertised a retention policy it did not have, and every job
    directory survived until the entry cap evicted it -- a slow disk leak, not
    an expiry. Both halves are asserted: the record goes, and so do the files.
    """
    from sciforensics.api.app import _Store

    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "overlay.png").write_bytes(b"pixels")

    store = _Store(ttl_seconds=0.0)
    store.put("abc", {"directory": str(directory)})

    assert store.get("abc") is None, "an expired job must not be served"
    assert not directory.exists(), "expiring a job must delete its files too"


def test_job_within_ttl_is_retained(tmp_path: Path) -> None:
    from sciforensics.api.app import _Store

    directory = tmp_path / "fresh"
    directory.mkdir()
    store = _Store(ttl_seconds=3600.0)
    store.put("abc", {"directory": str(directory)})

    assert store.get("abc") is not None
    assert directory.exists()


def test_entry_cap_evicts_oldest_and_cleans_up(tmp_path: Path) -> None:
    from sciforensics.api.app import _Store

    store = _Store(max_entries=2, ttl_seconds=3600.0)
    directories = []
    for index in range(3):
        directory = tmp_path / f"j{index}"
        directory.mkdir()
        directories.append(directory)
        store.put(f"job{index}", {"directory": str(directory)})

    assert store.get("job0") is None, "the oldest entry should have been evicted"
    assert not directories[0].exists()
    assert store.get("job2") is not None
    assert len(store) == 2


def test_capacity_overload_returns_503_rather_than_queueing() -> None:
    """`api.max_concurrent_jobs` was configured and enforced by nothing.

    The service accepted unbounded work and every request queued on the pipeline
    lock until the client gave up -- indistinguishable from a hang. Overload must
    produce an immediate 503 carrying `Retry-After`, which a client can act on.

    Driven by occupying the app's own semaphore (published on `app.state`).
    Racing real concurrent analyses would need a loaded model and would end up
    testing the test's threading rather than the admission rule.
    """
    app = create_app(load_config(overrides=["api.max_concurrent_jobs=1"]))
    client = TestClient(app, raise_server_exceptions=False)

    assert app.state.admission.acquire(blocking=False) is True
    try:
        response = client.post("/v1/cmfd", files={"image": ("x.png", b"not an image", "image/png")})
    finally:
        app.state.admission.release()

    assert response.status_code == 503, response.text
    # The header was set at the raise site and silently dropped by the uniform
    # error handler until `exc.headers` was forwarded.
    assert response.headers.get("Retry-After") == "10"
    assert "capacity" in response.json()["error"]


def test_capacity_released_after_a_request() -> None:
    """A refused-upload path must not leak its slot, or the service wedges."""
    client = TestClient(
        create_app(load_config(overrides=["api.max_concurrent_jobs=1"])),
        raise_server_exceptions=False,
    )
    for _ in range(3):
        response = client.post("/v1/cmfd", files={"image": ("x.png", b"not an image", "image/png")})
        # 400, not 503: the slot from the previous call was returned.
        assert response.status_code == 400, response.text
