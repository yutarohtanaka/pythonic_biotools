"""
Resilient cBioPortal REST API client for discrete GISTIC CNA matrices, arm-level
categorical CNA assays, and sample/patient clinical metadata harmonization.

Logs unmapped or absent genes explicitly rather than dropping them silently,
preserves diploid unaltered samples/genes as `0` in discrete GISTIC CNA matrices,
and preserves `NaN` for uncalled chromosome arms during binarization.
"""

from __future__ import annotations

from contextlib import nullcontext
import logging
import time

import numpy as np
import pandas as pd
import requests

DEFAULT_API_ROOT = "https://www.cbioportal.org/api"
logger = logging.getLogger("cohort_curation.cbioportal")

_RETRY_STATUS_CODES = {429, 502, 503, 504}


def _compute_retry_delay(resp: requests.Response | None, attempt: int, backoff_seconds: float) -> float:
    exp_delay = float(backoff_seconds) * (2.0 ** (attempt - 1))
    if resp is not None and resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                val = float(retry_after)
                if np.isfinite(val) and val >= 0.0:
                    return val
            except ValueError:
                pass
    return exp_delay


def _request_with_retry(
    method: str,
    path: str,
    api_root: str = DEFAULT_API_ROOT,
    max_retries: int = 4,
    backoff_seconds: float = 5.0,
    session: requests.Session | None = None,
    **kwargs: object,
) -> object:
    url = f"{api_root.rstrip('/')}{path}"
    last_exc: Exception | None = None
    ctx = nullcontext(session) if session is not None else requests.Session()
    with ctx as sess:
        for attempt in range(1, max_retries + 1):
            try:
                resp = sess.request(method, url, timeout=60, **kwargs)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                logger.warning(
                    "Transient connection/timeout error from %s (retry %d/%d): %s",
                    path,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(_compute_retry_delay(None, attempt, backoff_seconds))
                    continue
                raise

            if resp.status_code not in _RETRY_STATUS_CODES:
                resp.raise_for_status()
                return resp.json()

            last_exc = requests.exceptions.HTTPError(f"{resp.status_code} from {path}", response=resp)
            logger.warning("Transient %s from %s (retry %d/%d)", resp.status_code, path, attempt, max_retries)
            if attempt < max_retries:
                time.sleep(_compute_retry_delay(resp, attempt, backoff_seconds))

    assert last_exc is not None
    raise last_exc


def resolve_entrez_ids(
    hugo_symbols: list[str],
    api_root: str = DEFAULT_API_ROOT,
    chunk_size: int = 500,
    session: requests.Session | None = None,
) -> dict[str, int]:
    """Map HUGO gene symbols to Entrez IDs via chunked cBioPortal `/genes/fetch`, logging unmapped genes."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    uniq_symbols = [s for s in dict.fromkeys(hugo_symbols) if s]
    if not uniq_symbols:
        return {}

    symbol_to_entrez: dict[str, int] = {}
    ctx = nullcontext(session) if session is not None else requests.Session()
    with ctx as sess:
        for start in range(0, len(uniq_symbols), chunk_size):
            batch = uniq_symbols[start : start + chunk_size]
            records = _request_with_retry(
                "POST",
                "/genes/fetch",
                api_root=api_root,
                session=sess,
                json=batch,
                params={"geneIdType": "HUGO_GENE_SYMBOL"},
            )
            for r in records:  # type: ignore[union-attr]
                symbol_to_entrez[str(r["hugoGeneSymbol"])] = int(r["entrezGeneId"])

    missing = sorted(set(uniq_symbols) - symbol_to_entrez.keys())
    if missing:
        logger.warning("Gene symbols not resolved by cBioPortal: %s", missing)
    return symbol_to_entrez


def fetch_discrete_cna(
    cna_profile_id: str,
    sample_list_id: str,
    hugo_symbols: list[str],
    api_root: str = DEFAULT_API_ROOT,
    chunk_size: int = 500,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a complete `samples x genes` discrete GISTIC CNA matrix (values in `{-2, -1, 0, 1, 2}`).

    Queries `/sample-lists/{sample_list_id}/sample-ids` and reindexes `(sample_ids, resolved_symbols)`
    with `fill_value=0` so diploid unaltered samples and genes are preserved as `0` rather than dropped/`NaN`.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    ctx = nullcontext(session) if session is not None else requests.Session()
    with ctx as sess:
        sample_ids_raw = _request_with_retry(
            "GET",
            f"/sample-lists/{sample_list_id}/sample-ids",
            api_root=api_root,
            session=sess,
        )
        sample_ids = [str(s) for s in sample_ids_raw]  # type: ignore[union-attr]
        if not sample_ids:
            raise ValueError(f"Sample list {sample_list_id!r} contains no sample IDs.")

        symbol_to_entrez = resolve_entrez_ids(
            hugo_symbols, api_root=api_root, chunk_size=chunk_size, session=sess
        )
        if not symbol_to_entrez:
            raise ValueError("No requested genes resolved to Entrez IDs.")

        requested_resolved = [s for s in dict.fromkeys(hugo_symbols) if s in symbol_to_entrez]
        entrez_list = [symbol_to_entrez[s] for s in requested_resolved]
        entrez_to_symbol = {v: k for k, v in symbol_to_entrez.items()}

        all_records: list[dict] = []
        for start in range(0, len(entrez_list), chunk_size):
            batch_entrez = entrez_list[start : start + chunk_size]
            records = _request_with_retry(
                "POST",
                f"/molecular-profiles/{cna_profile_id}/molecular-data/fetch",
                api_root=api_root,
                session=sess,
                json={"entrezGeneIds": batch_entrez, "sampleListId": sample_list_id},
                params={"projection": "SUMMARY"},
            )
            if records:
                all_records.extend(records)  # type: ignore[arg-type]

    sample_index = pd.Index(sample_ids, name="sampleId")
    gene_columns = pd.Index(requested_resolved, name="gene")
    if not all_records:
        return pd.DataFrame(0, index=sample_index, columns=gene_columns, dtype=np.int64)

    df = pd.DataFrame.from_records(all_records)[["sampleId", "entrezGeneId", "value"]]
    df["gene"] = df["entrezGeneId"].astype(int).map(entrez_to_symbol)
    piv = df.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first")
    piv = piv.reindex(index=sample_index, columns=gene_columns, fill_value=0)
    return piv.fillna(0).astype(np.int64)


def fetch_arm_level_cna(
    armlevel_profile_id: str,
    sample_list_id: str,
    arm_stable_ids: list[str],
    api_root: str = DEFAULT_API_ROOT,
    chunk_size: int = 500,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a `samples x arms` categorical arm-level CNA table (`Loss`, `Gain`, `Unchanged`)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    all_records: list[dict] = []
    ctx = nullcontext(session) if session is not None else requests.Session()
    with ctx as sess:
        for start in range(0, len(arm_stable_ids), chunk_size):
            batch = arm_stable_ids[start : start + chunk_size]
            records = _request_with_retry(
                "POST",
                f"/generic_assay_data/{armlevel_profile_id}/fetch",
                api_root=api_root,
                session=sess,
                json={"genericAssayStableIds": batch, "sampleListId": sample_list_id},
                params={"projection": "SUMMARY"},
            )
            if records:
                all_records.extend(records)  # type: ignore[arg-type]

    if not all_records:
        raise ValueError(f"Empty arm-level CNA response for profile={armlevel_profile_id!r}.")
    df = pd.DataFrame.from_records(all_records)[["sampleId", "stableId", "value"]]
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


def fetch_sample_clinical_data(
    study_id: str,
    api_root: str = DEFAULT_API_ROOT,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch full sample-level clinical attribute matrix (`samples x attributes`)."""
    records = _request_with_retry(
        "GET",
        f"/studies/{study_id}/clinical-data",
        api_root=api_root,
        session=session,
        params={"clinicalDataType": "SAMPLE", "projection": "SUMMARY"},
    )
    df = pd.DataFrame.from_records(records)[["sampleId", "patientId", "clinicalAttributeId", "value"]]  # type: ignore[arg-type]
    return df.pivot_table(index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first")


def fetch_clinical_attribute(
    study_id: str,
    attribute_id: str,
    clinical_data_type: str = "PATIENT",
    api_root: str = DEFAULT_API_ROOT,
    session: requests.Session | None = None,
) -> pd.Series:
    """Fetch a named clinical attribute indexed by `sampleId` (automatically joining patient-level attributes)."""
    ctx = nullcontext(session) if session is not None else requests.Session()
    with ctx as sess:
        records = _request_with_retry(
            "GET",
            f"/studies/{study_id}/clinical-data",
            api_root=api_root,
            session=sess,
            params={"clinicalDataType": clinical_data_type, "attributeId": attribute_id, "projection": "SUMMARY"},
        )
        if not records:
            raise ValueError(f"Empty clinical-data response for study={study_id!r}, attribute={attribute_id!r}.")
        if clinical_data_type.upper() == "SAMPLE":
            return pd.DataFrame.from_records(records).set_index("sampleId")["value"]  # type: ignore[arg-type]

        patient_values = pd.DataFrame.from_records(records)[["patientId", "value"]].set_index("patientId")["value"]  # type: ignore[arg-type]
        samples = _request_with_retry(
            "GET",
            f"/studies/{study_id}/samples",
            api_root=api_root,
            session=sess,
            params={"projection": "SUMMARY"},
        )
    s2p = pd.DataFrame.from_records(samples)[["sampleId", "patientId"]]  # type: ignore[arg-type]
    s2p["value"] = s2p["patientId"].map(patient_values)
    return s2p.set_index("sampleId")["value"]
