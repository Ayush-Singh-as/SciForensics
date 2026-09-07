"""SciForensics — forensic detection of image reuse and manipulation in scientific figures.

Nothing heavyweight is imported at package scope: ``torch`` and ``cv2`` are
pulled in by the submodules that need them, so ``import sciforensics`` and
``sciforensics --help`` stay fast.

The public entry points are :func:`sciforensics.config.load_config` and the
pipeline functions in :mod:`sciforensics.pipeline`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("sciforensics")
except PackageNotFoundError:  # pragma: no cover - raw checkout without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
