"""FastAPI service. Stage A4.

``create_app`` is a factory rather than a module-level ``app`` so that tests and
``uvicorn --factory`` can build one with an explicit config, and so importing
this package does not construct a pipeline as a side effect.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    if name == "create_app":
        from sciforensics.api.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
