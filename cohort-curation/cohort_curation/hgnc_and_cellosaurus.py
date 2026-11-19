"""
HGNC gene symbol harmonization, Excel date-mangled gene symbol repair (`MARCHF`, `SEPTIN`, `DELEC1`),
and streaming ExPASy Cellosaurus cell-line QC parser (DepMap / CVCL cross-contamination, MSI, doubling time).
"""

from __future__ import annotations

import gzip
from pathlib import Path
import re
from typing import Iterable

import numpy as np
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


def _normalize_entrez_id(val: object) -> str | None:
    """Normalize string, integer, or float Entrez Gene IDs (`8085`, `8085.0`, `'8085 /// 12'`) to canonical string."""
    if val is None or not pd.notna(val):
        return None
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if not np.isfinite(val):
            return None
        return str(int(val)) if float(val).is_integer() else str(val)
    first = str(val).split("///")[0].strip()
    if not first or first.lower() == "nan":
        return None
    try:
        fval = float(first)
        if np.isfinite(fval) and fval.is_integer():
            return str(int(fval))
    except ValueError:
        pass
    return first


def harmonize_hgnc_symbols(
    symbols: pd.Series,
    hgnc_table: str | Path | pd.DataFrame,
    entrez_ids: pd.Series | None = None,
) -> pd.Series:
    """Deterministically map platform/legacy gene symbols to current HGNC approved symbols.

    Precedence:
      1. Entrez Gene ID -> Approved HGNC symbol (handles str, int, float, and `///`-delimited IDs)
      2. Preserve missing (`NaN` / `None`) symbols without stringifying to `'nan'`
      3. Repair Excel date-mangled symbols (`1-Mar` -> `MARCHF1`, etc.)
      4. Already an Approved HGNC symbol (exact or case-insensitive fallback)
      5. Unambiguous `prev_symbol` (1-to-1 mapping, exact or uppercase fallback)
      6. Unambiguous `alias_symbol` that does not collide with an approved symbol (exact or uppercase fallback)
      7. Original symbol unchanged
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
    approved = set(h["symbol"].dropna().astype(str))
    approved_upper: dict[str, str] = {}
    for sym_a in approved:
        approved_upper.setdefault(sym_a.upper(), sym_a)

    entrez_map: dict[str, str] = {}
    if "entrez_id" in h.columns:
        for _, row in h.dropna(subset=["entrez_id", "symbol"]).iterrows():
            ek = _normalize_entrez_id(row["entrez_id"])
            if ek is not None:
                entrez_map[ek] = str(row["symbol"])

    def _unique_col_maps(col_name: str) -> tuple[dict[str, str], dict[str, str]]:
        if col_name not in h.columns:
            return {}, {}
        m = h[["symbol", col_name]].dropna()
        if m.empty:
            return {}, {}
        m = m.assign(k=m[col_name].astype(str).str.split("|")).explode("k")
        m["k"] = m["k"].str.strip()
        m = m.loc[m["k"] != ""]
        if m.empty:
            return {}, {}

        counts = m.groupby("k")["symbol"].nunique()
        uniq_keys = set(counts[counts == 1].index)
        exact = m.loc[m["k"].isin(uniq_keys)].drop_duplicates("k").set_index("k")["symbol"].to_dict()

        m_up = m.assign(ku=m["k"].str.upper())
        counts_up = m_up.groupby("ku")["symbol"].nunique()
        uniq_up = set(counts_up[counts_up == 1].index)
        upper = m_up.loc[m_up["ku"].isin(uniq_up)].drop_duplicates("ku").set_index("ku")["symbol"].to_dict()
        return exact, upper

    prev_map, prev_map_upper = _unique_col_maps("prev_symbol")
    alias_raw, alias_raw_upper = _unique_col_maps("alias_symbol")
    alias_map = {k: v for k, v in alias_raw.items() if k not in approved}
    alias_map_upper = {ku: v for ku, v in alias_raw_upper.items() if ku not in approved_upper}

    out = {}
    for idx, raw_sym in symbols.items():
        if entrez_ids is not None and idx in entrez_ids.index:
            e_val = entrez_ids.loc[idx]
            e_norm = _normalize_entrez_id(e_val)
            if e_norm is not None and e_norm in entrez_map:
                out[idx] = entrez_map[e_norm]
                continue

        if raw_sym is None or not pd.notna(raw_sym):
            out[idx] = raw_sym
            continue

        sym = repair_excel_mangled_genes([str(raw_sym)])[0]
        sym_up = sym.upper()
        if sym in approved:
            out[idx] = sym
        elif sym_up in approved_upper:
            out[idx] = approved_upper[sym_up]
        elif sym in prev_map:
            out[idx] = prev_map[sym]
        elif sym_up in prev_map_upper:
            out[idx] = prev_map_upper[sym_up]
        elif sym in alias_map:
            out[idx] = alias_map[sym]
        elif sym_up in alias_map_upper:
            out[idx] = alias_map_upper[sym_up]
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
    """Single-pass streaming parser over `cellosaurus.txt[.gz]` extracting QC metadata:
    CVCL accession, DepMap ID, problematic/contaminated cell line warnings, MSI status, and doubling time (hours).
    """
    filter_active = any(x is not None for x in (depmap_ids, cvcl_ids, cell_line_names))
    names_upper = {n.upper() for n in cell_line_names} if cell_line_names else set()

    def _iter_entries(path: Path):
        opener = gzip.open if str(path).endswith(".gz") else open
        buf: list[str] = []
        in_entry = False
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("//"):
                    if buf:
                        yield buf
                    buf = []
                    in_entry = False
                else:
                    if not in_entry and line.startswith("ID   "):
                        in_entry = True
                    if in_entry:
                        buf.append(line.rstrip("\n"))
        if buf:
            yield buf

    rows = []
    for entry in _iter_entries(Path(cellosaurus_txt_path)):
        cell_name = next((l.split("   ", 1)[1].strip() for l in entry if l.startswith("ID   ")), None)
        cvcl = next((l.split("   ", 1)[1].strip() for l in entry if l.startswith("AC   ")), None)
        if cell_name is None and cvcl is None:
            continue

        synonyms_upper: set[str] = set()
        depmap_id = None
        for l in entry:
            if l.startswith("SY   "):
                for syn in l.split("   ", 1)[1].split(";"):
                    s_clean = syn.strip()
                    if s_clean:
                        synonyms_upper.add(s_clean.upper())
            elif depmap_id is None and l.startswith("DR   DepMap;"):
                depmap_id = l.split(";", 1)[1].split(";")[0].strip().rstrip(".")

        if filter_active:
            name_matched = cell_line_names is not None and (
                (cell_name is not None and cell_name.upper() in names_upper)
                or bool(synonyms_upper & names_upper)
            )
            matched = (
                (depmap_ids is not None and depmap_id in depmap_ids)
                or (cvcl_ids is not None and cvcl in cvcl_ids)
                or name_matched
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
