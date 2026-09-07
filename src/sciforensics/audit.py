"""Provenance capture for report audit blocks.

A forensic result that cannot be tied to the exact code, configuration and
inputs that produced it is not evidence, it is an anecdote. Every report
therefore carries a :class:`~sciforensics.types.AuditBlock` built here.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from sciforensics.runtime import get_logger
from sciforensics.types import AuditBlock

__all__ = [
    "build_audit_block",
    "file_sha256",
    "git_commit",
    "git_is_dirty",
    "tool_version",
    "utc_timestamp",
]

_log = get_logger(__name__)

#: 1 MiB. Large enough that hashing a 40 MB checkpoint is a handful of reads,
#: small enough not to spike memory on constrained API workers.
_HASH_CHUNK = 1 << 20


@lru_cache(maxsize=1)
def tool_version() -> str:
    """Installed distribution version, or a dev marker outside an install."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("sciforensics")
    except PackageNotFoundError:  # pragma: no cover - only when run from a raw checkout
        return "0.0.0+unknown"


def _git(*args: str) -> str | None:
    """Run a git command inside the package's repository, or return ``None``.

    Returns ``None`` rather than raising for every realistic failure -- not a
    checkout, git absent, a submodule oddity -- because provenance is
    best-effort metadata and must never take down an analysis.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@lru_cache(maxsize=1)
def git_commit() -> str | None:
    """Full SHA of ``HEAD``, or ``None`` outside a git checkout."""
    return _git("rev-parse", "HEAD")


@lru_cache(maxsize=1)
def git_is_dirty() -> bool | None:
    """Whether tracked files differ from ``HEAD``.

    A dirty tree means the recorded commit does not fully describe the code that
    ran, so the report says so explicitly instead of implying reproducibility it
    cannot offer.
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None
    return bool(status)


def file_sha256(path: str | Path) -> str:
    """Streaming SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    """SHA-256 of an in-memory payload (used for API uploads)."""
    return hashlib.sha256(data).hexdigest()


def utc_timestamp() -> str:
    """Current time as a second-resolution ISO-8601 UTC string."""
    # `datetime.UTC` is 3.11+; this package supports 3.10.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dependency_versions() -> tuple[str | None, str | None]:
    """``(torch, opencv)`` versions, tolerating either being unavailable."""
    torch_version: str | None = None
    cv_version: str | None = None
    try:
        import torch

        torch_version = torch.__version__
    except Exception:  # pragma: no cover
        _log.debug("could not determine the torch version", exc_info=True)
    try:
        import cv2

        cv_version = cv2.__version__
    except Exception:  # pragma: no cover
        _log.debug("could not determine the OpenCV version", exc_info=True)
    return torch_version, cv_version


def build_audit_block(
    *,
    config_fingerprint: str,
    device: str,
    seed: int | None = None,
    weights_path: str | Path | None = None,
    weights_sha256: str | None = None,
) -> AuditBlock:
    """Assemble the provenance record embedded in every report.

    ``weights_sha256`` may be passed directly (the API keeps it cached across
    requests); otherwise it is computed from ``weights_path``. Hashing is
    skipped silently if the file is missing so that a metadata-only run -- for
    example ``sciforensics compare --no-global`` -- still produces a report.
    """
    if weights_sha256 is None and weights_path is not None:
        try:
            weights_sha256 = file_sha256(weights_path)
        except OSError:
            _log.warning("could not hash weights at %s; audit block will omit it", weights_path)

    torch_version, cv_version = _dependency_versions()

    return AuditBlock(
        tool_version=tool_version(),
        config_fingerprint=config_fingerprint,
        timestamp_utc=utc_timestamp(),
        device=device,
        git_commit=git_commit(),
        git_dirty=git_is_dirty(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        opencv_version=cv_version,
        weights_sha256=weights_sha256,
        seed=seed,
    )
