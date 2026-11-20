"""Backward-compatible workspace CLI shim re-exporting `pythonic_biotools.registry` and `pythonic_biotools.cli`."""

from __future__ import annotations

from pythonic_biotools.cli import build_parser, main
from pythonic_biotools.registry import (
    ALIAS_TO_CANONICAL,
    BUNDLES,
    CANONICAL_SPECS,
    COPYLEFT_PACKAGES,
    MIT_PACKAGES,
    PACKAGE_CANONICAL_ATTR,
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

__all__ = [
    "ALIAS_TO_CANONICAL",
    "BUNDLES",
    "CANONICAL_SPECS",
    "COPYLEFT_PACKAGES",
    "MIT_PACKAGES",
    "PACKAGE_CANONICAL_ATTR",
    "PACKAGE_IMPORT_MAP",
    "PACKAGE_SPECS",
    "PackageSpec",
    "build_parser",
    "check_bundle_imports",
    "enable_workspace_imports",
    "get_repo_root",
    "install_bundle_editable",
    "load_package",
    "main",
    "resolve_bundle_packages",
]


if __name__ == "__main__":
    raise SystemExit(main())
