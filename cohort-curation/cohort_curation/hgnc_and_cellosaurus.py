"""
HGNC gene symbol harmonization, Excel date-mangled gene symbol repair (`MARCHF`, `SEPTIN`, `DELEC1`),
and streaming ExPASy Cellosaurus cell-line QC parser (DepMap / CVCL cross-contamination, MSI, doubling time).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd

EXCEL_MANGLED_GENE_MAP: dict[str, str] = {
    "1-DEC": "DELEC1",
    "15-SEP": "SELENOF",
    **{f"{i}-MAR": f"MARCHF{i}" for i in range(1, 12)},
    **{f"MARCH{i}": f"MARCHF{i}" for i in range(1, 12)},
    **{f"{i}-SEP": f"SEPTIN{i}" for i in range(1, 15)},
    **{f"SEPT{i}": f"SEPTIN{i}" for i in range(1, 15)},
}

_NUM = r"(\d+(?:\.\d+)?)"
_HOUR_RANGE = re.compile(rf"~?{_NUM}\s*(?:-|to)\s*{_NUM}\s*hours", re.IGNORECASE)
_HOUR_SINGLE = re.compile(rf"~?{_NUM}\s*hours", re.IGNORECASE)
_DAY_RANGE = re.compile(rf"~?{_NUM}\s*(?:-|to)\s*{_NUM}\s*days?\b", re.IGNORECASE)
_DAY_SINGLE = re.compile(rf"~?{_NUM}\s*days?\b", re.IGNORECASE)


def repair_excel_mangled_genes(
    symbols: Iterable[str] | pd.Index | pd.Series,
    strip_ensembl_prefix: bool = True,
) -> list[str]:
    """Repair spreadsheet date-corrupted gene symbols (`1-Mar` -> `MARCHF1`, `1-Sep` -> `SEPTIN1`, `1-Dec` -> `DELEC1`)
    and optionally strip `ENSG00000..._` prefixes from hybrid identifiers.
    """
    out: list[str] = []
    for raw in symbols:
        s = str(raw).strip()
        if strip_ensembl_prefix and s.upper().startswith("ENSG") and "_" in s:
            s = s.split("_", 1)[1].strip()
        s_up = s.upper()
        out.append(EXCEL_MANGLED_GENE_MAP.get(s_up, s))
    return out


def harmonize_hgnc_symbols(
    symbols: pd.Series,
    hgnc_table: str | Path | pd.DataFrame,
    entrez_ids: pd.Series | None = None,
) -> pd.Series:
    """Deterministically map platform/legacy gene symbols to current HGNC approved symbols.

    Precedence:
      1. Entrez Gene ID -> Approved HGNC symbol
      2. Repair Excel date-mangled symbols (`1-Mar` -> `MARCHF1`, etc.)
      3. Already an Approved HGNC symbol
      4. Unambiguous `prev_symbol` (1-to-1 mapping)
      5. Unambiguous `alias_symbol` that does not collide with an approved symbol
      6. Original symbol unchanged
    """
    if isinstance(hgnc_table, pd.DataFrame):
        h = hgnc_table.copy()
    else:
        h = pd.read_csv(
            hgnc_table,
            sep="\t",
            dtype=str,
            usecols=["symbol", "status", "prev_symbol", "alias_symbol", "entrez_id"],
        )
    h = h.loc[h["status"] == "Approved"].copy()
    approved = set(h["symbol"].dropna())
    entrez_map = h.dropna(subset=["entrez_id"]).set_index("entrez_id")["symbol"].to_dict()

    def _unique_col_map(col_name: str) -> dict[str, str]:
        m = h[["symbol", col_name]].dropna()
        if m.empty:
            return {}
        m = m.assign(k=m[col_name].astype(str).str.split("|")).explode("k")
        m["k"] = m["k"].str.strip()
        counts = m.groupby("k")["symbol"].nunique()
        uniq_keys = set(counts[counts == 1].index)
        m = m.loc[m["k"].isin(uniq_keys)].drop_duplicates("k")
        return m.set_index("k")["symbol"].to_dict()

    prev_map = _unique_col_map("prev_symbol")
    alias_map = {k: v for k, v in _unique_col_map("alias_symbol").items() if k not in approved}

    out = {}
    for idx, raw_sym in symbols.items():
        if entrez_ids is not None:
            e = entrez_ids.get(idx)
            if isinstance(e, str) and e.strip():
                e_first = e.split("///")[0].strip()
                if e_first in entrez_map:
                    out[idx] = entrez_map[e_first]
                    continue
        sym = repair_excel_mangled_genes([str(raw_sym)])[0]
        if sym in approved:
            out[idx] = sym
        elif sym in prev_map:
            out[idx] = prev_map[sym]
        elif sym in alias_map:
            out[idx] = alias_map[sym]
        else:
            out[idx] = sym
    return pd.Series(out, index=symbols.index, name=symbols.name)


def _parse_doubling_hours(cc_lines: list[str]) -> float | None:
    values: list[float] = []
    for line in cc_lines:
        for lo, hi in _HOUR_RANGE.findall(line):
            values.append((float(lo) + float(hi)) / 2.0)
        stripped = _HOUR_RANGE.sub("", line)
        for v in _HOUR_SINGLE.findall(stripped):
            values.append(float(v))
        for lo, hi in _DAY_RANGE.findall(line):
            values.append((float(lo) + float(hi)) / 2.0 * 24.0)
        day_stripped = _DAY_RANGE.sub("", line)
        for v in _DAY_SINGLE.findall(day_stripped):
            values.append(float(v) * 24.0)
    if not values:
        return None
    values.sort()
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2.0


def parse_cellosaurus(
    cellosaurus_txt_path: str | Path,
    depmap_ids: set[str] | None = None,
    cvcl_ids: set[str] | None = None,
    cell_line_names: set[str] | None = None,
) -> pd.DataFrame:
    """Single-pass streaming parser over `cellosaurus.txt` extracting QC metadata:
    CVCL accession, DepMap ID, problematic/contaminated cell line warnings, MSI status, and doubling time (hours).
    """
    filter_active = any(x is not None for x in (depmap_ids, cvcl_ids, cell_line_names))
    names_upper = {n.upper() for n in cell_line_names} if cell_line_names else set()

    def _iter_entries(path: Path):
        buf: list[str] = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("//"):
                    if buf:
                        yield buf
                    buf = []
                else:
                    buf.append(line.rstrip("\n"))
        if buf:
            yield buf

    rows = []
    for entry in _iter_entries(Path(cellosaurus_txt_path)):
        cell_name = next((l.split("   ", 1)[1].strip() for l in entry if l.startswith("ID   ")), None)
        cvcl = next((l.split("   ", 1)[1].strip() for l in entry if l.startswith("AC   ")), None)
        depmap_id = None
        for l in entry:
            if l.startswith("DR   DepMap;"):
                depmap_id = l.split(";", 1)[1].strip()
                break

        if filter_active:
            matched = (
                (depmap_ids is not None and depmap_id in depmap_ids)
                or (cvcl_ids is not None and cvcl in cvcl_ids)
                or (cell_line_names is not None and cell_name is not None and cell_name.upper() in names_upper)
            )
            if not matched:
                continue

        problematic = [l for l in entry if "Problematic cell line" in l]
        msi = [l for l in entry if l.startswith("CC   Microsatellite instability")]
        doubling = [l for l in entry if l.startswith("CC   Doubling time")]
        rows.append(
            {
                "cell_line_name": cell_name,
                "cvcl_accession": cvcl,
                "depmap_id": depmap_id,
                "is_problematic": bool(problematic),
                "problematic_comment": " | ".join(problematic) if problematic else None,
                "msi_comment": " | ".join(msi) if msi else None,
                "doubling_time_hours": _parse_doubling_hours(doubling),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "cell_line_name",
            "cvcl_accession",
            "depmap_id",
            "is_problematic",
            "problematic_comment",
            "msi_comment",
            "doubling_time_hours",
        ],
    )
