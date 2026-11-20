"""Unified namespace, file readers, and CLI for pythonic_biotools."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import List

from .registry import (
    ALIAS_TO_CANONICAL,
    BUNDLES,
    CANONICAL_SPECS,
    COPYLEFT_PACKAGES,
    MIT_PACKAGES,
    PACKAGE_IMPORT_MAP,
    PACKAGE_SPECS,
    PackageSpec,
    check_bundle_imports,
    enable_workspace_imports,
    get_repo_root,
    install_bundle_editable,
    load_package,
    resolve_bundle_packages,
)

__version__ = "0.2.0"

_SUBMODULES = frozenset(("io", "cli", "registry"))

__all__ = [
    "__version__",
    "io",
    "cli",
    "registry",
    "ALIAS_TO_CANONICAL",
    "BUNDLES",
    "CANONICAL_SPECS",
    "COPYLEFT_PACKAGES",
    "MIT_PACKAGES",
    "PACKAGE_IMPORT_MAP",
    "PACKAGE_SPECS",
    "PackageSpec",
    "check_bundle_imports",
    "enable_workspace_imports",
    "get_repo_root",
    "install_bundle_editable",
    "load_package",
    "resolve_bundle_packages",
    *sorted(CANONICAL_SPECS.keys()),
]


def __getattr__(name: str) -> ModuleType:
    """PEP 562 lazy attribute loader for `io`, `cli`, and all bioinformatics packages."""
    if name in _SUBMODULES:
        mod = importlib.import_module(f"pythonic_biotools.{name}")
        globals()[name] = mod
        return mod

    if name in ALIAS_TO_CANONICAL:
        canonical = ALIAS_TO_CANONICAL[name]
        mod = load_package(name)
        globals()[canonical] = mod
        globals()[name] = mod
        return mod

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """List eagerly bound symbols, submodules, and lazy bioinformatics package names."""
    return sorted(
        set(globals().keys())
        | set(_SUBMODULES)
        | set(CANONICAL_SPECS.keys())
        | {"xenofilterR", "pythonic_bayesprism", "pythonic_biostats"}
    )
