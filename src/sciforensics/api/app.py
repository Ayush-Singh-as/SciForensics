"""FastAPI service backing the web demo.

**The design constraint that shapes this module:** endpoints return *structured
evidence*, never baked-in pixels. Overlays are served as separate image URLs and
every number the frontend shows comes from the same ``ScanResult`` the CLI and
the PDF render from. That is what makes the threshold sliders honest -- the UI
re-bands a cached score client-side rather than asking the server to re-run a
pipeline it already ran, so moving a slider cannot silently change the evidence
underneath it.

**Concurrency.** ``Pipeline`` is explicitly not thread-safe: attribution runs a
backward pass that mutates ``.grad`` on captured activations, so two concurrent
``compare`` calls would interleave gradients between pairs. FastAPI runs sync
endpoint functions in a thread pool, so the pipeline is guarded by a lock. A
forensic result that is quietly wrong because two requests shared a gradient
buffer is far worse than a request that waits.

But waiting must be *bounded*. ``api.max_concurrent_jobs`` is enforced by an
admission semaphore that refuses with **503 + Retry-After** rather than letting
requests pile up on the pipeline lock until clients time out -- which is
indistinguishable from a hang, and was the behaviour before stage C3 (the
setting was configured and read by nothing).

**Operational surface.** ``/healthz`` is liveness and never touches the model:
a healthcheck that ran an inference would mark the container unhealthy during a
slow first request, exactly when it is working. ``/readyz`` carries load state
-- model loaded, jobs held, capacity -- so an orchestrator can stop routing
without restarting. ``/metrics`` is Prometheus text. Job results expire on
``api.job_ttl_seconds`` and their directories are deleted with them.

**Uploads are hostile until proven otherwise.** Size is capped before the body
is read into memory, and decoded pixel count is capped before allocation --
a 40,000x40,000 PNG compresses to a few hundred KB and would otherwise become a
6.4 GB allocation. Bug 14 capped the analysis resolution; this caps the decode.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from sciforensics import __version__
from sciforensics.config import Settings, load_config
from sciforensics.demo import ASSET_ROOT, EXAMPLES
from sciforensics.runtime import get_logger, setup_logging

_log = get_logger(__name__)

# Decoders we accept. An allow-list rather than a deny-list: OpenCV will happily
# attempt formats with a far worse parser history than these.
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class _Store:
    """In-process result store.

    Deliberately not Redis. This backs a demo whose job queue is a lock and
    whose lifetime is one container; a real deployment gets a real queue in
    stage C3, and pretending otherwise here would be scaffolding for a
    requirement that does not exist yet.

    ``ponytail: in-process dict, bounded by max_entries -- swap for Redis when
    the API runs more than one worker.``
    """

    max_entries: int = 64
    #: Seconds a job survives. Enforced here rather than merely configured:
    #: `api.job_ttl_seconds` existed in `configs/default.yaml` and was read by
    #: nothing, so the API advertised a retention policy it did not have and
    #: leaked every job directory until the entry cap evicted it.
    ttl_seconds: float = 3600.0
    _items: dict[str, dict[str, Any]] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = {**value, "created": time.monotonic()}
            self._order.append(key)
            self._expire_locked()
            while len(self._order) > self.max_entries:
                self._drop_locked(self._order.pop(0))

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            self._expire_locked()
            return self._items.get(key)

    def update(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if key in self._items:
                self._items[key] = {**self._items[key], **value, "created": time.monotonic()}

    def _expire_locked(self) -> None:
        """Drop anything past its TTL. Called under `_lock`.

        The bound is ``<=`` rather than ``<`` so a TTL of zero expires
        immediately. With a strict ``<`` the cutoff equals the creation stamp of
        a just-stored entry, so ``ttl_seconds=0`` retained everything forever --
        the opposite of what it reads as, and the value a test or a
        cache-disabling deployment would reach for first.
        """
        cutoff = time.monotonic() - self.ttl_seconds
        stale = [
            key
            for key in self._order
            if self._items.get(key, {}).get("created", 0) <= cutoff
            and self._items.get(key, {}).get("status") not in {"queued", "running"}
        ]
        for key in stale:
            self._order.remove(key)
            self._drop_locked(key)

    def _drop_locked(self, key: str) -> None:
        record = self._items.pop(key, None)
        directory = (record or {}).get("directory")
        if directory:
            # Uploads and overlays live on disk; forgetting the key without
            # removing them is a slow disk leak rather than an expiry.
            shutil.rmtree(directory, ignore_errors=True)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def create_app(cfg: Settings | None = None) -> FastAPI:
    """Build the ASGI app.

    A factory rather than a module-level singleton so tests can construct one
    with a CPU-pinned config and an injected pipeline without touching global
    state or loading a 35 MB checkpoint.
    """
    snapshot = os.environ.get("SCIFORENSICS_SERVE_SETTINGS")
    settings = cfg or (Settings.model_validate_json(snapshot) if snapshot else load_config())
    setup_logging(level=settings.runtime.log_level, fmt=settings.runtime.log_format)

    executor = ThreadPoolExecutor(max_workers=settings.api.max_concurrent_jobs)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
        yield
        executor.shutdown(wait=True)
        shutil.rmtree(workdir, ignore_errors=True)

    app = FastAPI(
        title="SciForensics",
        version=__version__,
        summary="Forensic detection of image reuse and manipulation in scientific figures.",
        docs_url="/docs",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    store = _Store(ttl_seconds=float(settings.api.job_ttl_seconds))
    workdir = Path(tempfile.mkdtemp(prefix="sciforensics-api-"))
    # One pipeline, one lock. See the module docstring.
    state: dict[str, Any] = {"pipeline": None}
    app.state.pipeline_state = state
    pipeline_lock = threading.Lock()

    # `api.max_concurrent_jobs` was configured and enforced by nothing, so the
    # service accepted unbounded work and every request queued on the pipeline
    # lock until the client timed out -- indistinguishable from a hang. A
    # semaphore turns overload into an immediate, honest 503.
    admission = threading.BoundedSemaphore(settings.api.max_concurrent_jobs)
    metrics: dict[str, int] = {"requests": 0, "rejected": 0, "failed": 0, "completed": 0}
    metrics_lock = threading.Lock()

    # Published on `app.state` so a test can occupy a slot and assert the
    # overload response. The alternative -- monkeypatching `threading` to
    # capture the object -- is fragile and does not typecheck, and racing real
    # analyses would test the test's threading rather than the admission rule.
    app.state.admission = admission
    app.state.store = store

    def _count(name: str) -> None:
        with metrics_lock:
            metrics[name] = metrics.get(name, 0) + 1

    @contextmanager
    def _admit() -> Iterator[None]:
        """Reject rather than queue when the service is already saturated."""
        _count("requests")
        if not admission.acquire(blocking=False):
            _count("rejected")
            raise HTTPException(
                503,
                f"server is at capacity ({settings.api.max_concurrent_jobs} concurrent "
                "analyses); retry shortly",
                headers={"Retry-After": "10"},
            )
        try:
            yield
        finally:
            admission.release()

    def pipeline() -> Any:
        """Construct the pipeline on first use, so startup does not block on torch."""
        if state["pipeline"] is None:
            from sciforensics.pipeline import Pipeline
            from sciforensics.runtime import seed_everything

            seed_everything(settings.runtime.seed)
            state["pipeline"] = Pipeline(settings)
        return state["pipeline"]

    # ---------------------------------------------------------------- helpers
    def _check_upload(upload: UploadFile) -> None:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(
                415,
                f"unsupported image type {suffix or '(none)'}; "
                f"expected one of {sorted(_ALLOWED_SUFFIXES)}",
            )

    def _save(upload: UploadFile, destination: Path) -> Path:
        """Stream to disk, enforcing the size cap as we go.

        Chunked rather than ``upload.read()``: reading first and checking after
        is exactly the allocation the cap exists to prevent.
        """
        _check_upload(upload)
        limit = settings.api.max_upload_bytes
        total = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            while chunk := upload.file.read(1 << 20):
                total += len(chunk)
                if total > limit:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(413, f"upload exceeds {limit} bytes")
                handle.write(chunk)
        if total == 0:
            destination.unlink(missing_ok=True)
            raise HTTPException(400, "uploaded file is empty")
        _check_decoded_size(destination)
        return destination

    def _check_decoded_size(path: Path) -> None:
        """Reject decompression bombs before anything allocates the pixels."""
        from sciforensics.io.images import probe_dimensions

        try:
            width, height = probe_dimensions(path)
        except Exception as exc:
            # The underlying message embeds the server-side temp path, which is
            # no use to the client and discloses the filesystem layout. Log it,
            # return a generic reason.
            _log.info("rejected undecodable upload: %s", exc)
            raise HTTPException(400, "file is not a decodable image") from exc
        if width * height > settings.api.max_decoded_pixels:
            raise HTTPException(
                413,
                f"image decodes to {width}x{height} = {width * height} pixels, "
                f"over the {settings.api.max_decoded_pixels} limit",
            )

    @contextmanager
    def _job() -> Iterator[tuple[str, Path]]:
        job_id = uuid.uuid4().hex[:16]
        directory = workdir / job_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            yield job_id, directory
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def _render(analysis: Any, directory: Path) -> dict[str, Any]:
        """Write overlays and return the evidence payload the frontend consumes."""
        from sciforensics.api.evidence import viewer_payload
        from sciforensics.pipeline import CopyMoveAnalysis
        from sciforensics.report.render import render_copy_move, render_pair

        render = render_copy_move if isinstance(analysis, CopyMoveAnalysis) else render_pair
        written = render(analysis, settings, directory, formats=("html", "json", "pdf"))
        # Descriptors, not bare filenames: the caption and trust level were
        # already computed in `overlays.py` for the PDF, and sending them keeps
        # the web demo from labelling an overlay differently from the report of
        # the same analysis. `file` is the asset-route filename.
        return {
            "result": analysis.result.model_dump(mode="json"),
            "overlays": [
                {
                    "name": name,
                    "file": Path(descriptor["href"]).name,
                    "caption": descriptor["caption"],
                    "trust": descriptor["trust"],
                }
                for name, descriptor in written.overlays.items()
            ],
            "warnings": list(written.warnings),
            "viewer": viewer_payload(analysis, limit=settings.report.max_match_lines),
            "report_format": "pdf" if written.pdf else "html",
        }

    def _submit(kind: str, uploads: list[UploadFile]) -> JSONResponse:
        _count("requests")
        if not admission.acquire(blocking=False):
            _count("rejected")
            raise HTTPException(
                503, "server is at capacity; retry shortly", headers={"Retry-After": "10"}
            )
        try:
            with _job() as (job_id, directory):
                paths = [
                    _save(
                        upload,
                        directory / f"input{index}{Path(upload.filename or '').suffix.lower()}",
                    )
                    for index, upload in enumerate(uploads)
                ]
                store.put(
                    job_id, {"directory": str(directory), "status": "queued", "stage": "queued"}
                )
                executor.submit(_run, job_id, directory, kind, paths)
        except Exception:
            admission.release()
            raise
        return JSONResponse(
            {"job_id": job_id, "status": "queued", "stage": "queued"}, status_code=202
        )

    def _run(job_id: str, directory: Path, kind: str, paths: list[Path]) -> None:
        from sciforensics.types import Stage

        def progress(stage: Stage) -> None:
            store.update(job_id, {"status": "running", "stage": stage.value})

        try:
            with pipeline_lock:
                store.update(job_id, {"status": "running", "stage": "loading"})
                runner = pipeline()
                analysis = (
                    runner.compare(*paths, force_local=True, progress=progress)
                    if kind == "compare"
                    else runner.copy_move(paths[0], progress=progress)
                )
                store.update(job_id, {"stage": "report"})
                payload = _render(analysis, directory)
            store.update(job_id, {**payload, "status": "complete", "stage": "complete"})
            _count("completed")
        except Exception:
            _log.exception("background analysis failed")
            store.update(
                job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "error": "Analysis failed. Check the API logs and checkpoint configuration.",
                },
            )
            _count("failed")
        finally:
            admission.release()

    # --------------------------------------------------------------- routes
    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness. Deliberately does not touch the model, so it stays cheap.

        A healthcheck that ran an inference would mark the container unhealthy
        during a slow first request, which is precisely when it is working.
        """
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        """Readiness: is the model actually loaded and is there capacity?

        Distinct from `/healthz` on purpose. A container can be alive but unable
        to serve -- still loading a 35 MB checkpoint, or saturated -- and an
        orchestrator needs to stop routing traffic to it without restarting it.
        """
        loaded = state["pipeline"] is not None
        return {
            "status": "ready" if loaded else "warming",
            "model_loaded": loaded,
            "jobs_held": len(store),
            "capacity": settings.api.max_concurrent_jobs,
        }

    @app.get("/metrics")
    def read_metrics() -> Response:
        """Prometheus text exposition.

        Hand-rolled rather than pulling in `prometheus-client`: four counters
        and a gauge do not justify a dependency, and the text format is stable
        and trivially correct. Swap it in if labels or histograms are ever
        needed.
        """
        with metrics_lock:
            snapshot = dict(metrics)
        lines = [
            "# HELP sciforensics_requests_total Analysis requests received.",
            "# TYPE sciforensics_requests_total counter",
            f"sciforensics_requests_total {snapshot['requests']}",
            "# HELP sciforensics_rejected_total Requests refused at capacity.",
            "# TYPE sciforensics_rejected_total counter",
            f"sciforensics_rejected_total {snapshot['rejected']}",
            "# HELP sciforensics_failed_total Analyses that raised.",
            "# TYPE sciforensics_failed_total counter",
            f"sciforensics_failed_total {snapshot['failed']}",
            "# HELP sciforensics_completed_total Analyses that produced a result.",
            "# TYPE sciforensics_completed_total counter",
            f"sciforensics_completed_total {snapshot['completed']}",
            "# HELP sciforensics_jobs_held Job results currently retained.",
            "# TYPE sciforensics_jobs_held gauge",
            f"sciforensics_jobs_held {len(store)}",
        ]
        body = "\n".join(lines) + "\n"
        return Response(body, media_type="text/plain; version=0.0.4")

    @app.get("/v1/config")
    def read_config() -> dict[str, Any]:
        """The thresholds and bands in effect.

        The frontend needs these to re-band a cached score client-side; serving
        them keeps the sliders' defaults in sync with the backend instead of
        hard-coding the numbers in two places (bug 10's failure mode, in a new
        location).
        """
        return {
            "global_match": {
                "distance_threshold": settings.global_match.distance_threshold,
                "local_trigger_distance": settings.global_match.local_trigger_distance,
                "similarity_temperature": settings.global_match.similarity_temperature,
            },
            "geometry": {
                "min_inliers": settings.geometry.min_inliers,
                "min_inlier_ratio": settings.geometry.min_inlier_ratio,
            },
            "bands": settings.fusion.bands.model_dump(mode="json"),
            "calibrated": settings.fusion.calibrator is not None,
        }

    @app.get("/v1/examples")
    def examples() -> dict[str, Any]:
        return {
            "examples": [
                {
                    **item,
                    "left": f"/v1/examples/{item['id']}/left",
                    "right": f"/v1/examples/{item['id']}/right" if item["right"] else None,
                }
                for item in EXAMPLES
            ]
        }

    @app.get("/v1/examples/{example_id}/{side}")
    def example_asset(example_id: str, side: str) -> FileResponse:
        item = next((e for e in EXAMPLES if e["id"] == example_id), None)
        if item is None or side not in {"left", "right"} or not item[side]:
            raise HTTPException(404, "unknown example")
        return FileResponse(ASSET_ROOT / item[side], filename=item[side])

    @app.post("/v1/compare")
    def compare(
        left: UploadFile = File(...), right: UploadFile = File(...), background: bool = False
    ) -> Any:
        if background:
            return _submit("compare", [left, right])
        with _admit(), _job() as (job_id, directory):
            left_path = _save(left, directory / f"left{Path(left.filename or '').suffix.lower()}")
            right_path = _save(
                right, directory / f"right{Path(right.filename or '').suffix.lower()}"
            )
            with pipeline_lock:
                try:
                    analysis = pipeline().compare(left_path, right_path, force_local=True)
                except Exception as exc:
                    _count("failed")
                    _log.exception("compare failed")
                    raise HTTPException(500, f"analysis failed: {exc}") from exc
                payload = _render(analysis, directory)
                _count("completed")

            store.put(
                job_id,
                {"directory": str(directory), "status": "complete", "stage": "complete", **payload},
            )
            return {"job_id": job_id, "status": "complete", **payload}

    @app.post("/v1/cmfd")
    def cmfd(image: UploadFile = File(...), background: bool = False) -> Any:
        if background:
            return _submit("cmfd", [image])
        with _admit(), _job() as (job_id, directory):
            path = _save(image, directory / f"input{Path(image.filename or '').suffix.lower()}")
            with pipeline_lock:
                try:
                    analysis = pipeline().copy_move(path)
                except Exception as exc:
                    _count("failed")
                    _log.exception("copy-move failed")
                    raise HTTPException(500, f"analysis failed: {exc}") from exc
                payload = _render(analysis, directory)
                _count("completed")

            store.put(
                job_id,
                {"directory": str(directory), "status": "complete", "stage": "complete", **payload},
            )
            return {"job_id": job_id, "status": "complete", **payload}

    @app.get("/v1/jobs/{job_id}")
    def read_job(job_id: str) -> dict[str, Any]:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(404, "unknown or expired job")
        return {k: v for k, v in record.items() if k != "directory"}

    @app.websocket("/v1/jobs/{job_id}/progress")
    async def job_progress(socket: WebSocket, job_id: str) -> None:
        origin = socket.headers.get("origin")
        if origin and origin not in settings.api.cors_origins:
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            while True:
                record = store.get(job_id)
                if record is None:
                    await socket.send_json({"status": "failed", "error": "unknown or expired job"})
                    break
                await socket.send_json({k: record.get(k) for k in ("status", "stage", "error")})
                if record.get("status") in {"complete", "failed"}:
                    break
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            pass
        finally:
            with suppress(RuntimeError, WebSocketDisconnect):
                await socket.close()

    @app.get("/v1/jobs/{job_id}/assets/{name}")
    def read_asset(job_id: str, name: str) -> FileResponse:
        """Serve one overlay PNG.

        ``name`` is resolved and confined to the job directory: a demo that
        accepts a filename from the client and joins it to a path is a directory
        traversal waiting to happen.
        """
        record = store.get(job_id)
        if record is None:
            raise HTTPException(404, "unknown or expired job")
        directory = (Path(record["directory"]) / "assets").resolve()
        target = (directory / name).resolve()
        if directory not in target.parents or not target.is_file():
            raise HTTPException(404, "unknown asset")
        return FileResponse(target)

    @app.get("/v1/jobs/{job_id}/report")
    def read_report(job_id: str) -> FileResponse:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(404, "unknown or expired job")
        directory = Path(record["directory"])
        for candidate in ("report.pdf", "report.html"):
            path = directory / candidate
            if path.is_file():
                return FileResponse(path, filename=f"sciforensics-{job_id}-{candidate}")
        raise HTTPException(404, "no report was rendered for this job")

    @app.exception_handler(HTTPException)
    def _http_error(request: Request, exc: HTTPException) -> JSONResponse:  # noqa: ARG001
        # `request` is unused but required: Starlette calls every exception
        # handler with (request, exc), so dropping it is a TypeError at runtime.
        #
        # `exc.headers` must be forwarded. Without it this handler silently
        # discarded the `Retry-After` on a 503, so a client told to back off had
        # nothing to back off by -- the header was set at the raise site and
        # thrown away one layer later.
        return JSONResponse(
            {"error": exc.detail, "status": exc.status_code},
            exc.status_code,
            headers=exc.headers,
        )

    return app
