"""Compatibility namespace; prefer importing the top-level ``package`` API."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


# Internal modules historically use top-level imports such as ``data`` and
# ``spatial``. The compatibility namespace reproduces the supported repository
# root import mode, then delegates to the same single public ``package`` module.
_repository_root = str(Path(__file__).resolve().parent)
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)
_package = import_module("package")

__all__ = list(_package.__all__)


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_package, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
