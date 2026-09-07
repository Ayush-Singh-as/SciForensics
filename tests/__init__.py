"""Test package.

Made a package (rather than a bare directory of scripts) so shared helpers are
importable as ``tests.helpers`` under pytest's ``importlib`` import mode, and so
mypy can type-check the suite alongside the library.
"""
