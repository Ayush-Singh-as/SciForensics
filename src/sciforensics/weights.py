"""Locating and verifying the pretrained checkpoint.

The 35 MB checkpoint is committed into this repository's git history, which is a
mistake I am not compounding: nothing in the package reads it by hard-coded path.
:func:`ensure` resolves it through an explicit precedence chain and verifies the
SHA-256 before handing it to :func:`~sciforensics.global_match.load_backbone`, so
a truncated download, a half-written file or the wrong artifact entirely fails
here with a diagnosis rather than three layers down as a shape error.

Verification is not optional and it is not a warning. A checkpoint that loads but
is not the checkpoint the benchmark numbers were measured with would make every
reported figure meaningless while everything continued to run, which is the
failure mode this whole refactor exists to eliminate.

**On the download path.** :data:`RELEASE_URL` is ``None`` until the artifact is
published as a GitHub Release asset, and while it is ``None`` a missing checkpoint
raises with instructions rather than attempting a fetch. That is deliberate:
a resolver that silently proceeds without weights, or that invents a plausible URL,
is worse than one that says exactly which file it wants and where to put it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from sciforensics.audit import file_sha256
from sciforensics.runtime import get_logger

__all__ = [
    "EXPECTED_SHA256",
    "RELEASE_URL",
    "WeightsError",
    "cache_dir",
    "ensure",
    "verify",
]

_log = get_logger(__name__)

#: SHA-256 of the checkpoint every number in ``benchmarks/`` was measured with.
#: Recorded from the copy in this repository's ``models/`` directory; stage B2
#: trains a replacement and will add a second entry rather than overwrite this
#: one, so a report can always name which weights produced it.
EXPECTED_SHA256 = "befec5b863580ab6dfa1fa3005b37e447f54d79e7d16e90fe3fe5f2dc71a3c6f"

#: Size in bytes, used only to make a truncated file's error message concrete.
EXPECTED_BYTES = 35_358_123

#: Pinned release asset. ``None`` until published -- see the module docstring.
RELEASE_URL: str | None = None

#: Filename used in the repository, the cache and (eventually) the release asset.
FILENAME = "weights.pth"

#: Environment variable holding an explicit path to a checkpoint, which takes
#: precedence over the repository copy and the cache. Set it to run against a
#: locally retrained model without editing configuration.
ENV_VAR = "SCIFORENSICS_WEIGHTS"

_DOWNLOAD_CHUNK = 1 << 20


class WeightsError(RuntimeError):
    """Raised when the checkpoint cannot be located, fetched or verified."""


def cache_dir() -> Path:
    """Directory downloads are cached in.

    Honours ``XDG_CACHE_HOME`` so a container can point it at a mounted volume
    and avoid re-downloading on every start.
    """
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "sciforensics"


def _repo_copy() -> Path:
    """The checkpoint committed alongside the source tree, if present."""
    return Path(__file__).resolve().parents[2] / "models" / FILENAME


def verify(path: Path, *, expected: str | None = EXPECTED_SHA256) -> str:
    """Hash ``path`` and confirm it matches ``expected``.

    Returns the digest so a caller can record it in the audit block. Pass
    ``expected=None`` to hash a locally retrained checkpoint without a known
    digest -- the digest still lands in the report, which is what makes a run
    against custom weights distinguishable from a run against the shipped ones.

    Raises
    ------
    WeightsError
        If the digest does not match. The message includes both digests and the
        file size, because a size far from :data:`EXPECTED_BYTES` says
        "truncated download" while a matching size with a different digest says
        "different model".
    """
    digest = file_sha256(path)
    if expected is not None and digest != expected:
        size = path.stat().st_size
        raise WeightsError(
            f"checkpoint at {path} does not match the expected artifact.\n"
            f"  expected sha256 {expected}\n"
            f"  actual   sha256 {digest}\n"
            f"  size {size:,} bytes (expected {EXPECTED_BYTES:,})\n"
            "Delete the file and re-fetch it. Loading it anyway would produce "
            "distances that cannot be compared against any published number."
        )
    return digest


def _download(url: str, destination: Path) -> None:
    """Fetch ``url`` to ``destination`` atomically.

    Downloads to a temporary file in the same directory and renames on success,
    so an interrupted fetch cannot leave a truncated file that later looks
    cached. The rename is atomic within a filesystem, which is why the temporary
    file is not placed in ``/tmp``.

    The scheme is checked rather than assumed. ``urlopen`` will happily open
    ``file://``, so a future edit that set :data:`RELEASE_URL` to a local path
    would turn "fetch and verify the pinned artifact" into "copy whatever is
    there" -- and the digest check would then be the only thing standing between
    that and a silent substitution.
    """
    if not url.lower().startswith("https://"):
        raise WeightsError(f"refusing to fetch weights over a non-HTTPS URL: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _log.info("fetching weights from %s", url)
    handle, temp_name = tempfile.mkstemp(dir=destination.parent, suffix=".partial")
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as sink, urlopen(url) as source:
            shutil.copyfileobj(source, sink, _DOWNLOAD_CHUNK)
        temp.replace(destination)
    except Exception as exc:
        temp.unlink(missing_ok=True)
        raise WeightsError(f"could not download weights from {url}: {exc}") from exc


def ensure(
    configured: Path | None = None,
    *,
    expected: str | None = EXPECTED_SHA256,
    allow_download: bool = True,
) -> Path:
    """Return a verified path to the checkpoint.

    Resolution order, first hit wins:

    1. ``configured`` -- ``global_match.weights`` from the configuration.
    2. ``$SCIFORENSICS_WEIGHTS``.
    3. ``models/weights.pth`` beside the source tree.
    4. The download cache, and failing that :data:`RELEASE_URL`.

    An explicitly configured path that does not exist is an error rather than a
    fall-through to the next candidate. Silently ignoring the path someone asked
    for and running against a different model instead is precisely how a
    benchmark comes to report numbers from weights nobody selected.

    Parameters
    ----------
    expected
        Digest to require. Defaults to :data:`EXPECTED_SHA256`; pass ``None``
        when pointing at a retrained checkpoint.
    allow_download
        Set ``False`` in tests and offline environments so a missing file raises
        immediately instead of reaching for the network.
    """
    if configured is not None:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise WeightsError(
                f"global_match.weights points at {path}, which does not exist. "
                "Remove the setting to fall back to automatic resolution."
            )
        verify(path, expected=expected)
        return path

    env = os.environ.get(ENV_VAR)
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise WeightsError(f"${ENV_VAR} points at {path}, which does not exist")
        verify(path, expected=expected)
        return path

    for candidate in (_repo_copy(), cache_dir() / FILENAME):
        if candidate.is_file():
            verify(candidate, expected=expected)
            _log.debug("using weights at %s", candidate)
            return candidate

    cached = cache_dir() / FILENAME
    if RELEASE_URL is None:
        raise WeightsError(
            "no checkpoint found and no release URL is pinned in this build.\n"
            f"Place {FILENAME} (sha256 {EXPECTED_SHA256[:16]}..., "
            f"{EXPECTED_BYTES:,} bytes) at one of:\n"
            f"  {_repo_copy()}\n"
            f"  {cached}\n"
            f"or set ${ENV_VAR} to its path."
        )
    if not allow_download:
        raise WeightsError(
            f"no checkpoint at {cached} and downloading is disabled; "
            f"set ${ENV_VAR} or pass allow_download=True"
        )

    _download(RELEASE_URL, cached)
    verify(cached, expected=expected)
    return cached
