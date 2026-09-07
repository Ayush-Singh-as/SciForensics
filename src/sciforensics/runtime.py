"""Device resolution, seeding and logging setup.

``torch`` is imported lazily inside the functions that need it. Importing it at
module scope costs roughly a second, which is a second added to every
``sciforensics --help`` and to every test that only touches configuration.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["get_logger", "resolve_device", "seed_everything", "setup_logging"]

_LOGGER_NAME = "sciforensics"


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a config string into a concrete :class:`torch.device`.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. An explicit request for
    an unavailable backend is an error rather than a silent CPU fallback: a
    benchmark that quietly ran 40x slower on CPU while reporting GPU numbers is
    worse than one that refused to start.
    """
    import torch

    normalized = spec.strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"runtime.device={spec!r} was requested but CUDA is not available. "
                "Use 'cpu', or 'auto' to pick the best available backend."
            )
        device = torch.device(normalized)
        index = 0 if device.index is None else device.index
        count = torch.cuda.device_count()
        if index >= count:
            raise RuntimeError(
                f"runtime.device={spec!r} requests CUDA device {index}, but only {count} "
                f"device(s) are visible."
            )
        return device

    if normalized == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError(f"runtime.device={spec!r} was requested but MPS is not available.")
        return torch.device("mps")

    raise ValueError(
        f"unrecognised runtime.device={spec!r}; expected one of "
        "'auto', 'cpu', 'cuda', 'cuda:N', 'mps'."
    )


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int | None, *, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches.

    Covers ``random``, ``numpy``, ``torch`` (CPU and all CUDA devices) and
    OpenCV, which keeps its own generator used by ``RANSAC``/``MAGSAC``. Missing
    the OpenCV one is why robust-estimator results used to shift slightly
    between otherwise identical runs.

    ``deterministic`` additionally requests deterministic cuDNN kernels and
    sets ``CUBLAS_WORKSPACE_CONFIG``, which cuBLAS requires for reproducible
    GEMMs. ``warn_only=True`` keeps operators that have no deterministic
    implementation working instead of raising.
    """
    if seed is None:
        return

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import cv2

        cv2.setRNGSeed(seed)
    except Exception:  # pragma: no cover - OpenCV is a hard dependency
        get_logger().debug("could not seed the OpenCV RNG", exc_info=True)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch builds
            get_logger().debug("torch.use_deterministic_algorithms unavailable", exc_info=True)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """Minimal structured formatter for container/API deployments."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach anything callers passed via `extra=`, skipping LogRecord's own
        # attributes so the payload stays small and predictable.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_RESERVED_LOG_KEYS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
    | {"message", "asctime", "taskName"}
)


def setup_logging(
    level: str = "INFO",
    fmt: Literal["console", "json"] = "console",
    *,
    stream: Any = None,
) -> logging.Logger:
    """Configure the ``sciforensics`` logger and return it.

    Idempotent: repeated calls replace the handler rather than stacking them, so
    a CLI command that loads config twice does not double every log line.
    Attaches to the package logger only -- the root logger is left alone so
    importing ``sciforensics`` inside someone else's application does not
    hijack their logging.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if fmt == "json":
        new_handler: logging.Handler = logging.StreamHandler(stream or sys.stderr)
        new_handler.setFormatter(_JsonFormatter())
    else:
        try:
            from rich.logging import RichHandler

            new_handler = RichHandler(
                rich_tracebacks=True, show_path=False, omit_repeated_times=False
            )
            new_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        except ImportError:  # pragma: no cover - rich is a hard dependency
            new_handler = logging.StreamHandler(stream or sys.stderr)
            new_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
            )

    new_handler.setLevel(level.upper())
    logger.addHandler(new_handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger.

    Pass ``__name__`` from a submodule; the ``sciforensics.`` prefix is stripped
    so log lines read ``sciforensics.local_match.geometry`` rather than being
    doubled.
    """
    if name is None or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    suffix = name.removeprefix(f"{_LOGGER_NAME}.")
    return logging.getLogger(f"{_LOGGER_NAME}.{suffix}")
