"""Package metadata, license classification, bundle definitions, and lazy loader."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

PACKAGE_IMPORT_MAP: Dict[str, str] = {
    "bayesprism": "pythonic_bayesprism",
    "biostats": "pythonic_biostats",
    "bisque": "bisque",
    "buysetest": "buysetest",
    "cause": "cause",
    "chord": "chord",
    "chromvar": "chromvar",
    "cibersortx": "cibersortx",
    "cismr": "cismr",
    "cohort-curation": "cohort_curation",
    "colliderbias": "colliderbias",
    "coloc": "coloc",
    "colocproptest": "colocproptest",
    "conipher": "conipher",
    "csaw": "csaw",
    "cside": "cside",
    "ctwas": "ctwas",
    "decoupler-ulm": "decoupler_ulm",
    "deqms": "deqms",
    "discover": "discover",
    "dndscv": "dndscv",
    "dss": "dss",
    "evam": "evam",
    "facets": "facets",
    "fgwas": "fgwas",
    "figr": "figr",
    "fishhook": "fishhook",
    "genomicsem": "genomicsem",
    "grapple": "grapple",
    "hdl": "hdl",
    "hyprcoloc": "hyprcoloc",
    "ichorcna": "ichorcna",
    "indexevent": "indexevent",
    "jmbayes2": "jmbayes2",
    "lava": "lava",
    "lhcmr": "lhcmr",
    "limma": "limma",
    "logistf": "logistf",
    "mashr": "mashr",
    "mmsig": "mmsig",
    "monalisa": "monalisa",
    "mrapss": "mrapss",
    "mrbee": "mrbee",
    "mrclust": "mrclust",
    "mrhorse": "mrhorse",
    "mrlap": "mrlap",
    "mrpresso": "mrpresso",
    "mrraps": "mrraps",
    "mtcojo": "mtcojo",
    "music": "music",
    "mutationtimer": "mutationtimer",
    "mvcml": "mvcml",
    "nebula": "nebula",
    "numbat": "numbat",
    "purecn": "purecn",
    "qdnaseq": "qdnaseq",
    "rose": "rose",
    "saturn": "saturn",
    "scarhrd": "scarhrd",
    "shatterseek": "shatterseek",
    "slopehunter": "slopehunter",
    "smr-heidi": "smr_heidi",
    "survidinri": "survidinri",
    "survrm2": "survrm2",
    "sva": "sva",
    "timeroc": "timeroc",
    "tracksig": "tracksig",
    "twosamplemr": "twosamplemr",
    "variancepartition": "variancepartition",
    "xenofilterR": "xenofilterR",
}

COPYLEFT_PACKAGES: Dict[str, str] = {
    "cause": "GPL-3.0-or-later",
    "evam": "AGPL-3.0-or-later",
    "mrpresso": "GPL-3.0-or-later",
    "mrraps": "GPL-3.0-or-later",
    "mtcojo": "GPL-3.0-or-later",
    "smr-heidi": "GPL-3.0-or-later",
    "timeroc": "GPL-2.0-or-later",
}

PACKAGE_CANONICAL_ATTR: Dict[str, str] = {
    pkg_dir: (
        "bayesprism"
        if pkg_dir == "bayesprism"
        else "biostats"
        if pkg_dir == "biostats"
        else "xenofilterr"
        if pkg_dir == "xenofilterR"
        else pkg_dir.replace("-", "_")
    )
    for pkg_dir in PACKAGE_IMPORT_MAP
}


@dataclass(frozen=True)
class PackageSpec:
    """Metadata descriptor for one of the 70 pythonic_biotools packages."""

    pkg_dir: str
    mod_name: str
    canonical_attr: str
    license: str
    is_copyleft: bool

    @property
    def canonical(self) -> str:
        """Canonical snake_case submodule attribute name under `pythonic_biotools`."""
        return self.canonical_attr


PACKAGE_SPECS: Dict[str, PackageSpec] = {
    pkg_dir: PackageSpec(
        pkg_dir=pkg_dir,
        mod_name=mod_name,
        canonical_attr=PACKAGE_CANONICAL_ATTR[pkg_dir],
        license=COPYLEFT_PACKAGES.get(pkg_dir, "MIT"),
        is_copyleft=(pkg_dir in COPYLEFT_PACKAGES),
    )
    for pkg_dir, mod_name in PACKAGE_IMPORT_MAP.items()
}

CANONICAL_SPECS: Dict[str, PackageSpec] = {
    spec.canonical_attr: spec for spec in PACKAGE_SPECS.values()
}

MIT_PACKAGES: Tuple[str, ...] = tuple(
    sorted(pkg_dir for pkg_dir, spec in PACKAGE_SPECS.items() if not spec.is_copyleft)
)

ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _spec in PACKAGE_SPECS.values():
    ALIAS_TO_CANONICAL[_spec.canonical_attr] = _spec.canonical_attr
    ALIAS_TO_CANONICAL[_spec.mod_name] = _spec.canonical_attr
    ALIAS_TO_CANONICAL[_spec.pkg_dir] = _spec.canonical_attr


BUNDLES: Dict[str, Tuple[str, ...]] = {
    "oncology": (
        "xenofilterR",
        "facets",
        "purecn",
        "qdnaseq",
        "ichorcna",
        "chord",
        "scarhrd",
        "shatterseek",
        "tracksig",
        "mmsig",
        "mutationtimer",
        "conipher",
        "dndscv",
        "numbat",
        "discover",
        "fishhook",
        "evam",
        "cohort-curation",
    ),
    "statgen": (
        "coloc",
        "hyprcoloc",
        "colocproptest",
        "smr-heidi",
        "ctwas",
        "cismr",
        "hdl",
        "lava",
        "genomicsem",
        "fgwas",
        "twosamplemr",
        "mrpresso",
        "mrraps",
        "mrhorse",
        "mrapss",
        "mrbee",
        "mvcml",
        "cause",
        "mrclust",
        "lhcmr",
        "mrlap",
        "grapple",
        "slopehunter",
        "indexevent",
        "colliderbias",
        "mtcojo",
        "mashr",
    ),
    "singlecell": (
        "music",
        "bayesprism",
        "bisque",
        "cibersortx",
        "cside",
        "nebula",
        "saturn",
        "numbat",
        "chromvar",
        "figr",
        "decoupler-ulm",
    ),
    "epigenomics": (
        "dss",
        "csaw",
        "monalisa",
        "chromvar",
        "figr",
        "rose",
        "fishhook",
    ),
    "proteomics": (
        "deqms",
        "limma",
        "sva",
        "variancepartition",
        "mashr",
    ),
    "survival": (
        "biostats",
        "logistf",
        "timeroc",
        "survidinri",
        "buysetest",
        "survrm2",
        "jmbayes2",
    ),
}
BUNDLES["all"] = tuple(sorted(PACKAGE_IMPORT_MAP.keys()))
BUNDLES["mit"] = MIT_PACKAGES


def get_repo_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the pythonic_biotools workspace root."""
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parent.parent


def resolve_bundle_packages(bundle: str = "all") -> Tuple[str, ...]:
    """Return ordered directory names for a named bundle."""
    if bundle not in BUNDLES:
        valid = ", ".join(sorted(BUNDLES))
        raise ValueError(f"Unknown bundle {bundle!r}; expected one of: {valid}")
    return BUNDLES[bundle]


def enable_workspace_imports(
    bundle: str = "all",
    repo_root: str | Path | None = None,
) -> List[str]:
    """Add all subpackage directories in `bundle` to `sys.path` in-memory."""
    root = get_repo_root(repo_root)
    added: List[str] = []
    for pkg_dir_name in resolve_bundle_packages(bundle):
        pkg_path = str(root / pkg_dir_name)
        if (root / pkg_dir_name).is_dir() and pkg_path not in sys.path:
            sys.path.insert(0, pkg_path)
            added.append(pkg_dir_name)
    return added


def load_package(
    name: str,
    repo_root: str | Path | None = None,
) -> ModuleType:
    """Resolve and import a subpackage by canonical attribute or alias."""
    canonical = ALIAS_TO_CANONICAL.get(name)
    if canonical is None:
        raise AttributeError(f"module 'pythonic_biotools' has no attribute {name!r}")

    spec = CANONICAL_SPECS[canonical]

    # 1. Try unified wheel / package-dir namespace first
    ns_mod_name = f"pythonic_biotools.{canonical}"
    try:
        return importlib.import_module(ns_mod_name)
    except ModuleNotFoundError as exc:
        if exc.name not in (ns_mod_name, "pythonic_biotools"):
            raise

    # 2. Fall back to workspace checkout or standalone subpackage install
    root = get_repo_root(repo_root)
    pkg_dir_path = root / spec.pkg_dir
    if (pkg_dir_path / spec.mod_name / "__init__.py").is_file():
        pkg_dir_str = str(pkg_dir_path)
        if pkg_dir_str not in sys.path:
            sys.path.insert(0, pkg_dir_str)

    try:
        return importlib.import_module(spec.mod_name)
    except ModuleNotFoundError as exc:
        if exc.name != spec.mod_name:
            raise

    # 3. Clear install guidance for copyleft packages in MIT-wheel environments
    if spec.is_copyleft:
        raise ImportError(
            f"Package {spec.pkg_dir!r} ({spec.license}) is copyleft-licensed and is "
            "not bundled in the MIT-only 'pythonic-biotools' wheel. "
            "Install it separately with:\n"
            f'  pip install "git+https://github.com/gillanilab/pythonic_biotools.git#subdirectory={spec.pkg_dir}"'
        )

    raise ImportError(
        f"Could not import {canonical!r} (neither 'pythonic_biotools.{canonical}' "
        f"nor standalone {spec.mod_name!r} is installed)."
    )


def check_bundle_imports(
    bundle: str = "all",
    repo_root: str | Path | None = None,
) -> Dict[str, bool]:
    """Import every package in `bundle` and verify PEP 561 `py.typed` presence."""
    root = get_repo_root(repo_root)
    enable_workspace_imports(bundle=bundle, repo_root=root)
    status: Dict[str, bool] = {}
    for pkg_dir_name in resolve_bundle_packages(bundle):
        spec = PACKAGE_SPECS[pkg_dir_name]
        try:
            mod = importlib.import_module(spec.mod_name)
        except ModuleNotFoundError:
            mod = load_package(spec.canonical_attr, repo_root=root)
        mod_file = Path(getattr(mod, "__file__", ""))
        py_typed = mod_file.parent / "py.typed"
        status[pkg_dir_name] = mod is not None and py_typed.is_file()
    return status


def install_bundle_editable(
    bundle: str = "all",
    repo_root: str | Path | None = None,
    no_deps: bool = False,
) -> int:
    """Editable-install (`pip install -e`) every package in `bundle`."""
    root = get_repo_root(repo_root)
    pkgs = resolve_bundle_packages(bundle)
    cmd = [sys.executable, "-m", "pip", "install"]
    if no_deps:
        cmd.append("--no-deps")
    for pkg in pkgs:
        cmd.extend(["-e", str(root / pkg)])
    res = subprocess.run(cmd, cwd=str(root), check=False)
    return int(res.returncode)
