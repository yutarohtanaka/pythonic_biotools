"""Workspace bundle installer and import helper for pythonic_biotools."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def get_repo_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the pythonic_biotools workspace root."""
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parent


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


def check_bundle_imports(
    bundle: str = "all",
    repo_root: str | Path | None = None,
) -> Dict[str, bool]:
    """Import every package in `bundle` and verify PEP 561 `py.typed` presence."""
    root = get_repo_root(repo_root)
    enable_workspace_imports(bundle=bundle, repo_root=root)
    status: Dict[str, bool] = {}
    for pkg_dir_name in resolve_bundle_packages(bundle):
        mod_name = PACKAGE_IMPORT_MAP[pkg_dir_name]
        mod = importlib.import_module(mod_name)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pythonic-biotools",
        description="Manage and inspect standalone packages in pythonic_biotools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List packages in a bundle.")
    p_list.add_argument(
        "--bundle",
        default="all",
        choices=sorted(BUNDLES),
        help="Domain bundle to inspect.",
    )

    p_check = sub.add_parser(
        "check-imports",
        help="Verify imports and py.typed markers across a bundle.",
    )
    p_check.add_argument(
        "--bundle",
        default="all",
        choices=sorted(BUNDLES),
        help="Domain bundle to verify.",
    )

    p_install = sub.add_parser(
        "install",
        help="Editable-install all standalone packages in a bundle.",
    )
    p_install.add_argument(
        "bundle",
        choices=sorted(BUNDLES),
        help="Domain bundle to install.",
    )
    p_install.add_argument(
        "--no-deps",
        action="store_true",
        help="Pass --no-deps to pip when scientific stack is already installed.",
    )

    args = parser.parse_args(argv)
    if args.command == "list":
        pkgs = resolve_bundle_packages(args.bundle)
        print(f"Bundle '{args.bundle}' ({len(pkgs)} packages):")
        for p in pkgs:
            print(f"  {p:20s} -> import {PACKAGE_IMPORT_MAP[p]}")
        return 0

    if args.command == "check-imports":
        status = check_bundle_imports(bundle=args.bundle)
        bad = [k for k, ok in status.items() if not ok]
        print(f"Verified {len(status) - len(bad)}/{len(status)} packages in '{args.bundle}'.")
        if bad:
            print("Failed py.typed or import check:", ", ".join(bad))
            return 1
        return 0

    if args.command == "install":
        return install_bundle_editable(bundle=args.bundle, no_deps=args.no_deps)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
