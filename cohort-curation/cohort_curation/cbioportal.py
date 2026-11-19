"""
Resilient cBioPortal REST API client for discrete GISTIC CNA matrices, arm-level
categorical CNA assays, and sample/patient clinical metadata harmonization.

Logs unmapped or absent genes explicitly rather than dropping them silently, and
preserves `NaN` for uncalled chromosome arms during binarization.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DEFAULT_API_ROOT = "https://www.cbioportal.org/api"
logger = logging.getLogger("cohort_curation.cbioportal")

_RETRY_STATUS_CODES = {502, 503, 504}


def _request_with_retry(
    method: str,
    path: str,
    api_root: str = DEFAULT_API_ROOT,
    max_retries: int = 4,
    backoff_seconds: float = 5.0,
    **kwargs: object,
) -> object:
    last_exc: requests.exceptions.HTTPError | None = None
    for attempt in range(1, max_retries + 1):
        resp = requests.request(method, f"{api_root.rstrip('/')}{path}", timeout=60, **kwargs)
        if resp.status_code not in _RETRY_STATUS_CODES:
            resp.raise_for_status()
            return resp.json()
        last_exc = requests.exceptions.HTTPError(f"{resp.status_code} from {path}", response=resp)
        logger.warning("Transient %s from %s (retry %d/%d)", resp.status_code, path, attempt, max_retries)
        if attempt < max_retries:
            time.sleep(backoff_seconds)
    assert last_exc is not None
    raise last_exc


def resolve_entrez_ids(hugo_symbols: list[str], api_root: str = DEFAULT_API_ROOT) -> dict[str, int]:
    """Map HUGO gene symbols to Entrez IDs via cBioPortal `/genes/fetch`, logging unmapped genes."""
    records = _request_with_retry(
        "POST",
        "/genes/fetch",
        api_root=api_root,
        json=hugo_symbols,
        params={"geneIdType": "HUGO_GENE_SYMBOL"},
    )
    symbol_to_entrez = {r["hugoGeneSymbol"]: int(r["entrezGeneId"]) for r in records}
    missing = sorted(set(hugo_symbols) - symbol_to_entrez.keys())
    if missing:
        logger.warning("Gene symbols not resolved by cBioPortal: %s", missing)
    return symbol_to_entrez


def fetch_discrete_cna(
    cna_profile_id: str,
    sample_list_id: str,
    hugo_symbols: list[str],
    api_root: str = DEFAULT_API_ROOT,
) -> pd.DataFrame:
    """Fetch a `samples x genes` discrete GISTIC CNA matrix (values in `{-2, -1, 0, 1, 2}`)."""
    symbol_to_entrez = resolve_entrez_ids(hugo_symbols, api_root=api_root)
    if not symbol_to_entrez:
        raise ValueError("No requested genes resolved to Entrez IDs.")
    entrez_to_symbol = {v: k for k, v in symbol_to_entrez.items()}

    records = _request_with_retry(
        "POST",
        f"/molecular-profiles/{cna_profile_id}/molecular-data/fetch",
        api_root=api_root,
        json={"entrezGeneIds": list(symbol_to_entrez.values()), "sampleListId": sample_list_id},
        params={"projection": "SUMMARY"},
    )
    if not records:
        raise ValueError(f"Empty CNA response for profile={cna_profile_id!r} sampleList={sample_list_id!r}.")

    df = pd.DataFrame.from_records(records)[["sampleId", "patientId", "entrezGeneId", "value"]]
    df["gene"] = df["entrezGeneId"].map(entrez_to_symbol)
    return df.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first")


def fetch_arm_level_cna(
    armlevel_profile_id: str,
    sample_list_id: str,
    arm_stable_ids: list[str],
    api_root: str = DEFAULT_API_ROOT,
) -> pd.DataFrame:
    """Fetch a `samples x arms` categorical arm-level CNA table (`Loss`, `Gain`, `Unchanged`)."""
    records = _request_with_retry(
        "POST",
        f"/generic_assay_data/{armlevel_profile_id}/fetch",
        api_root=api_root,
        json={"genericAssayStableIds": arm_stable_ids, "sampleListId": sample_list_id},
        params={"projection": "SUMMARY"},
    )
    if not records:
        raise ValueError(f"Empty arm-level CNA response for profile={armlevel_profile_id!r}.")
    df = pd.DataFrame.from_records(records)[["sampleId", "stableId", "value"]]
    return df.pivot_table(index="sampleId", columns="stableId", values="value", aggfunc="first")


def binarize_arm_events(
    arm_level_df: pd.DataFrame,
    event_definitions: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Binarize directional arm events `{event_name: (arm_col, positive_label)}` while keeping `NaN`s intact."""
    columns = {}
    for event_name, (arm_col, positive_value) in event_definitions.items():
        if arm_col not in arm_level_df.columns:
            raise ValueError(f"Column {arm_col!r} not in arm_level_df.")
        raw = arm_level_df[arm_col]
        binary = (raw == positive_value).astype(float)
        binary[raw.isna()] = np.nan
        columns[event_name] = binary
    return pd.DataFrame(columns, index=arm_level_df.index)


def fetch_sample_clinical_data(study_id: str, api_root: str = DEFAULT_API_ROOT) -> pd.DataFrame:
    """Fetch full sample-level clinical attribute matrix (`samples x attributes`)."""
    records = _request_with_retry(
        "GET",
        f"/studies/{study_id}/clinical-data",
        api_root=api_root,
        params={"clinicalDataType": "SAMPLE", "projection": "SUMMARY"},
    )
    df = pd.DataFrame.from_records(records)[["sampleId", "patientId", "clinicalAttributeId", "value"]]
    return df.pivot_table(index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first")


def fetch_clinical_attribute(
    study_id: str,
    attribute_id: str,
    clinical_data_type: str = "PATIENT",
    api_root: str = DEFAULT_API_ROOT,
) -> pd.Series:
    """Fetch a named clinical attribute indexed by `sampleId` (automatically joining patient-level attributes)."""
    records = _request_with_retry(
        "GET",
        f"/studies/{study_id}/clinical-data",
        api_root=api_root,
        params={"clinicalDataType": clinical_data_type, "attributeId": attribute_id, "projection": "SUMMARY"},
    )
    if not records:
        raise ValueError(f"Empty clinical-data response for study={study_id!r}, attribute={attribute_id!r}.")
    if clinical_data_type.upper() == "SAMPLE":
        return pd.DataFrame.from_records(records).set_index("sampleId")["value"]

    patient_values = pd.DataFrame.from_records(records)[["patientId", "value"]].set_index("patientId")["value"]
    samples = _request_with_retry("GET", f"/studies/{study_id}/samples", api_root=api_root, params={"projection": "SUMMARY"})
    s2p = pd.DataFrame.from_records(samples)[["sampleId", "patientId"]]
    s2p["value"] = s2p["patientId"].map(patient_values)
    return s2p.set_index("sampleId")["value"]
