"""
Broad GDAC Firehose GISTIC2 Level 4 significant focal CNA peak downloader & parser,
plus AACR Project GENIE targeted-sequencing gene-panel coverage masking.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
import io
from pathlib import Path
import re
import tarfile
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .hgnc_and_cellosaurus import (
    SettingsMode,
    _normalize_entrez_id,
    _validate_settings,
    harmonize_hgnc_symbols,
    repair_excel_mangled_genes,
)

FIREHOSE_ROOT = "https://gdac.broadinstitute.org/runs/analyses__latest/data"


def _safe_extract_tar_member(tar: tarfile.TarFile, member_name: str, dest_root: Path) -> Path:
    """Safely extract a single regular file from `tar` under `dest_root` (Python 3.10+ compatible)."""
    root_resolved = dest_root.resolve()
    dest_path = (dest_root / member_name).resolve()
    if not dest_path.is_relative_to(root_resolved):
        raise ValueError(f"Unsafe path traversal detected in tar member: {member_name!r}")

    info = tar.getmember(member_name)
    if not info.isfile():
        raise ValueError(f"Refusing to extract non-regular tar member: {member_name!r}")

    src = tar.extractfile(info)
    if src is None:
        raise ValueError(f"Unable to read tar member bytes: {member_name!r}")
    with src:
        data = src.read()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)
    return dest_path


def download_gistic_peak_files(
    study_code: str,
    run_date: str = "20160128",
    cache_dir: str | Path = "data/firehose_cache",
    tumor_type_code: str = "TP",
) -> tuple[Path, Path]:
    """Download and cache `amp_genes.conf_99.txt` and `del_genes.conf_99.txt` from Broad Firehose.

    Parameters
    ----------
    study_code : str
        TCGA / Firehose study code (e.g., `'LUAD'`, `'SKCM'`, `'LAML'`).
    tumor_type_code : str, default `'TP'`
        Firehose sample type suffix (`'TP'` primary solid tumor, `'TM'` metastatic e.g. SKCM, `'TB'` blood e.g. LAML).
    """
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    archive_dir = f"gdac.broadinstitute.org_{study_code}-{tumor_type_code}.CopyNumber_Gistic2.Level_4.{run_date}00.0.0"
    amp_path = cdir / archive_dir / "amp_genes.conf_99.txt"
    del_path = cdir / archive_dir / "del_genes.conf_99.txt"
    if amp_path.exists() and del_path.exists():
        return amp_path, del_path

    url = f"{FIREHOSE_ROOT}/{study_code}/{run_date}/{archive_dir}.tar.gz"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in (f"{archive_dir}/amp_genes.conf_99.txt", f"{archive_dir}/del_genes.conf_99.txt"):
            _safe_extract_tar_member(tar, member, cdir)
    return amp_path, del_path


def parse_gistic_peak_genes(path: str | Path) -> dict[str, set[str]]:
    """Parse GISTIC2 transposed `amp_genes`/`del_genes.conf_*.txt` into `{cytoband: {gene, ...}}`."""
    with open(path, "r") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if len(rows) < 5:
        return {}
    cytobands = [cb for cb in rows[0][1:] if cb]
    peaks: dict[str, set[str]] = {cb: set() for cb in cytobands}
    for row in rows[4:]:
        for i, cb in enumerate(cytobands):
            cell = row[i + 1] if i + 1 < len(row) else ""
            if cell:
                gene = cell.split("|")[0].strip("[]").strip()
                if gene:
                    peaks[cb].add(gene)
    return peaks


def cross_check_gistic_panel(
    panel_genes: Iterable[str],
    amp_peaks: dict[str, set[str]],
    del_peaks: dict[str, set[str]],
) -> pd.DataFrame:
    """Cross-reference candidate genes against significant GISTIC2 focal amplification/deletion cytobands."""
    amp_all = set().union(*amp_peaks.values()) if amp_peaks else set()
    del_all = set().union(*del_peaks.values()) if del_peaks else set()
    rows = []
    for g in panel_genes:
        amp_cb = sorted(cb for cb, gs in amp_peaks.items() if g in gs)
        del_cb = sorted(cb for cb, gs in del_peaks.items() if g in gs)
        rows.append(
            {
                "gene": g,
                "in_significant_amp_peak": g in amp_all,
                "in_significant_del_peak": g in del_all,
                "amp_cytobands": ",".join(amp_cb),
                "del_cytobands": ",".join(del_cb),
            }
        )
    return pd.DataFrame(rows)


def _build_entrez_lookup(
    entrez_to_symbol: Mapping[int | str, str] | pd.Series | pd.DataFrame | str | Path | None,
) -> tuple[dict[str, str], pd.DataFrame | None]:
    if entrez_to_symbol is None:
        return {}, None

    if isinstance(entrez_to_symbol, (str, Path)):
        df = pd.read_csv(entrez_to_symbol, sep="\t", dtype=str)
    elif isinstance(entrez_to_symbol, pd.DataFrame):
        df = entrez_to_symbol.copy()
    else:
        items = entrez_to_symbol.items()
        lookup: dict[str, str] = {}
        for k, v in items:
            ek = _normalize_entrez_id(k)
            if ek is not None and v is not None and pd.notna(v):
                lookup[ek] = str(v).strip()
        return lookup, None

    hgnc_df = df if {"symbol", "status"}.issubset(df.columns) else None
    if "status" in df.columns:
        df = df.loc[df["status"] == "Approved"]

    e_col = "entrez_id" if "entrez_id" in df.columns else ("Entrez_Gene_Id" if "Entrez_Gene_Id" in df.columns else None)
    s_col = "symbol" if "symbol" in df.columns else ("Hugo_Symbol" if "Hugo_Symbol" in df.columns else None)
    lookup = {}
    if e_col is not None and s_col is not None:
        for _, row in df.dropna(subset=[e_col, s_col]).iterrows():
            ek = _normalize_entrez_id(row[e_col])
            if ek is not None:
                lookup[ek] = str(row[s_col]).strip()
    return lookup, hgnc_df


def _normalize_panel_gene_token(raw_tok: str, entrez_lookup: dict[str, str]) -> str:
    tok = raw_tok.strip()
    if tok.upper().startswith("ENTREZ:"):
        tok = tok.split(":", 1)[1].strip()
    # Normalize integer/float Entrez IDs (`7157`, `7157.0`)
    if re.fullmatch(r"\d+(?:\.0+)?", tok):
        norm_id = str(int(float(tok)))
        return entrez_lookup.get(norm_id, norm_id)
    return entrez_lookup.get(tok, tok)


def build_genie_panel_matrix(
    panel_dir_or_files: str | Path | Iterable[str | Path],
    entrez_to_symbol: Mapping[int | str, str] | pd.Series | pd.DataFrame | str | Path | None = None,
    *,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """Parse AACR Project GENIE / cBioPortal `data_gene_panel_*.txt` files into a binary `panels x genes` DataFrame.

    Supports both HUGO symbol (`gene_list:`) and Entrez Gene ID (`entrez_gene_list:` or numeric `gene_list:`)
    panel definitions. When `entrez_to_symbol` (a dict, Series, or HGNC DataFrame/TSV) is supplied, Entrez IDs
    are translated to canonical HUGO symbols. When `settings='R_default'`, legacy/Excel-mangled panel symbols
    (`1-Mar` -> `MARCHF1`, `2-APR` -> `PMAIP1`, `OCT4` -> `POU5F1`, `MARC1` -> `MTARC1`) are canonicalized
    automatically following `MSKCC-Epi-Bio/gnomeR`.
    """
    _validate_settings(settings)
    entrez_lookup, hgnc_df = _build_entrez_lookup(entrez_to_symbol)

    if isinstance(panel_dir_or_files, (str, Path)):
        p = Path(panel_dir_or_files)
        files = sorted(p.glob("data_gene_panel_*.txt")) if p.is_dir() else [p]
    else:
        files = [Path(f) for f in panel_dir_or_files]

    panel_to_genes: dict[str, set[str]] = {}
    all_genes: set[str] = set()
    for fp in files:
        lines = [ln.strip() for ln in fp.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        stable_id = fp.stem.replace("data_gene_panel_", "")
        genes: list[str] = []
        for ln in lines:
            if ln.startswith("stable_id:"):
                stable_id = ln.split(":", 1)[1].strip()
            elif ln.startswith("gene_list:") or ln.startswith("entrez_gene_list:"):
                raw = ln.split(":", 1)[1].strip()
                tokens = [g.strip() for g in re.split(r"[\t,\s]+", raw) if g.strip()]
                genes.extend(_normalize_panel_gene_token(tok, entrez_lookup) for tok in tokens)

        if settings == "R_default" and genes:
            genes = repair_excel_mangled_genes(genes)
            if hgnc_df is not None:
                genes = list(
                    harmonize_hgnc_symbols(
                        pd.Series(genes, dtype=object),
                        hgnc_df,
                        unmapped_as_na=False,
                        settings="R_default",
                    )
                )

        panel_to_genes[stable_id] = set(genes)
        all_genes.update(genes)

    sorted_genes = sorted(all_genes)
    sorted_panels = sorted(panel_to_genes)
    mat = np.zeros((len(sorted_panels), len(sorted_genes)), dtype=np.int8)
    g_idx = {g: j for j, g in enumerate(sorted_genes)}
    for i, pan in enumerate(sorted_panels):
        for g in panel_to_genes[pan]:
            mat[i, g_idx[g]] = 1
    return pd.DataFrame(mat, index=sorted_panels, columns=sorted_genes)


def mask_unprofiled_panel_genes(
    alteration_matrix: pd.DataFrame,
    sample_to_panel: pd.Series | dict[str, str],
    panel_coverage_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Set `alteration_matrix[sample, gene] = NaN` wherever `sample`'s sequencing panel did not assay `gene`."""
    out = alteration_matrix.astype(np.float64).copy()
    s2p = sample_to_panel if isinstance(sample_to_panel, dict) else sample_to_panel.to_dict()
    cov = panel_coverage_matrix.reindex(columns=out.columns, fill_value=0)

    for sample_id in out.index:
        panel_id = s2p.get(sample_id)
        if panel_id is None or panel_id not in cov.index:
            out.loc[sample_id, :] = np.nan
            continue
        covered = cov.loc[panel_id].to_numpy(dtype=bool)
        out.loc[sample_id, ~covered] = np.nan
    return out
