"""Shared fixtures.

The suite runs on CPU with synthetic inputs only, so it stays fast and has no
dependency on the 35 MB weights file or on any dataset. Anything that needs real
weights or real data is marked (``weights`` / ``data``) and deselected by default
in CI.

Builders live in :mod:`tests.helpers` so test modules can import them directly;
this module only wires them up as fixtures.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sciforensics.config import Settings, load_config
from tests.helpers import SEED, textured_image


@pytest.fixture(scope="session")
def cfg() -> Settings:
    """The shipped default configuration, with env overrides disabled.

    ``use_env=False`` matters: without it a developer's ``SCIFORENSICS_*`` shell
    variables would silently change what the tests assert.
    """
    return load_config(use_env=False)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def weights_path() -> Path:
    """Path to the pretrained checkpoint, skipping the test if it is absent.

    Paired with the ``weights`` marker: the marker documents the dependency and
    lets CI deselect the whole group, while the skip keeps a developer who simply
    hasn't fetched the 35 MB artifact from seeing a failure that says nothing about
    their change.

    Deliberately *not* routed through ``sciforensics.weights.ensure()``. A test that
    downloads on demand would turn an offline run into a network failure, and a
    regression guard that only works with connectivity is not one.
    """
    path = Path(__file__).resolve().parents[1] / "models" / "weights.pth"
    if not path.is_file():
        pytest.skip(f"pretrained weights not present at {path}")
    return path


@pytest.fixture
def texture() -> np.ndarray:
    return textured_image()


@pytest.fixture
def image_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A textured image and a byte-identical copy of it, written to disk as PNG."""
    image = textured_image()
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    cv2.imwrite(str(left), image)
    cv2.imwrite(str(right), image)
    return left, right
