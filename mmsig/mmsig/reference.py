"""
Canonical 96-trinucleotide SBS channel definitions, catalog builders, and
reference alignment utilities for `mmsig`.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

SettingsMode = Literal["default", "R_default"]

PYRIMIDINE_SUBS: tuple[str, ...] = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
BASES: tuple[str, ...] = ("A", "C", "G", "T")
COMPLEMENT: dict[str, str] = {"A": "T", "C": "G", "G": "C", "T": "A"}

CANONICAL_96_CONTEXTS: tuple[str, ...] = tuple(
    f"{b5}[{sub}]{b3}"
    for sub in PYRIMIDINE_SUBS
    for b5 in BASES
    for b3 in BASES
)

# Chemotherapy signatures associated with transcriptional strand asymmetry
# in Rustad et al. (Communications Biology 2021):
# - SBS31 / SBS35: platinum chemotherapy (guanine crosslinks -> C>A, C>T asymmetry)
# - SBS99 / SBS-MM1: melphalan chemotherapy (adenine adducts -> T>A, T>C asymmetry)
CHEMO_STRAND_SIGNATURES: dict[str, tuple[str, ...]] = {
    "SBS31": ("C>A", "C>T"),
    "SBS35": ("C>A", "C>T"),
    "SBS99": ("T>A", "T>C"),
    "SBS-MM1": ("T>A", "T>C"),
}


def validate_settings(settings: str) -> SettingsMode:
    """Validate `settings` mode is `'default'` or `'R_default'`."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"Unknown settings={settings!r}. Expected 'default' or 'R_default'."
        )
    return settings  # type: ignore[return-value]


def normalize_context_label(ref: str, alt: str, trinuc: str) -> str:
    """Convert `(ref, alt, trinuc)` into canonical pyrimidine `X[Y>Z]W` notation."""
    ref_u = ref.strip().upper()
    alt_u = alt.strip().upper()
    tri_u = trinuc.strip().upper()
    if len(tri_u) != 3 or ref_u not in COMPLEMENT or alt_u not in COMPLEMENT or ref_u == alt_u:
        raise ValueError(
            f"Invalid substitution ({ref!r}>{alt!r}) or trinucleotide ({trinuc!r})."
        )
    if tri_u[1] != ref_u:
        if COMPLEMENT.get(tri_u[1]) == ref_u:
            tri_u = "".join(COMPLEMENT[b] for b in reversed(tri_u))
        else:
            raise ValueError(
                f"Center base of trinucleotide {trinuc!r} does not match ref {ref!r}."
            )
    if ref_u in ("C", "T"):
        return f"{tri_u[0]}[{ref_u}>{alt_u}]{tri_u[2]}"
    # Reverse-complement purine reference base to pyrimidine representation
    rc_ref = COMPLEMENT[ref_u]
    rc_alt = COMPLEMENT[alt_u]
    rc_5 = COMPLEMENT[tri_u[2]]
    rc_3 = COMPLEMENT[tri_u[0]]
    return f"{rc_5}[{rc_ref}>{rc_alt}]{rc_3}"


def _extract_sub_tri_context(df: pd.DataFrame) -> pd.Index | None:
    """Extract `X[Y>Z]W` index if `sub` and `tri` metadata columns are present."""
    cols_lower = {str(c).lower(): c for c in df.columns}
    if "context" in cols_lower and df.shape[0] == 96:
        ctx_col = cols_lower["context"]
        return pd.Index(df[ctx_col].astype(str).tolist(), name="context")
    if "sub" in cols_lower and "tri" in cols_lower:
        sub_col = cols_lower["sub"]
        tri_col = cols_lower["tri"]
        subs = df[sub_col].astype(str).str.strip().str.upper()
        tris = df[tri_col].astype(str).str.strip().str.upper()
        labels = [
            f"{t[0]}[{s}]{t[2]}" if len(t) == 3 else f"{t}[{s}]"
            for s, t in zip(subs, tris)
        ]
        return pd.Index(labels, name="context")
    return None


def prepare_signature_reference(sig_ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a 96-channel x K signature reference DataFrame.

    Columns are normalized to sum to 1.0 over channels. Optional `sub`/`tri` or
    `context` metadata columns (as in upstream R `mmsig` signature tables) are
    converted to the row index automatically.
    """
    if not isinstance(sig_ref_df, pd.DataFrame):
        raise TypeError("sig_ref_df must be a pandas DataFrame.")
    df = sig_ref_df.copy()
    ctx_idx = _extract_sub_tri_context(df)
    drop_meta = [
        c for c in df.columns if str(c).lower() in ("sub", "tri", "context", "mutationtype")
    ]
    if ctx_idx is not None:
        df = df.drop(columns=drop_meta)
        df.index = ctx_idx
    elif drop_meta:
        if str(drop_meta[0]).lower() == "mutationtype":
            df.index = pd.Index(df[drop_meta[0]].astype(str).tolist(), name="context")
        df = df.drop(columns=drop_meta)

    if df.shape[1] == 0:
        raise ValueError("sig_ref_df contains no signature columns.")

    vals = df.to_numpy(dtype=float)
    if np.any(~np.isfinite(vals)) or np.any(vals < 0.0):
        raise ValueError("sig_ref_df entries must be finite and non-negative.")

    col_sums = vals.sum(axis=0)
    if np.any(col_sums <= 0.0):
        bad = [str(df.columns[i]) for i in np.where(col_sums <= 0.0)[0]]
        raise ValueError(f"Signature columns have zero sum: {bad}")

    vals_norm = vals / col_sums[np.newaxis, :]
    out = pd.DataFrame(vals_norm, index=df.index.copy(), columns=df.columns.astype(str))
    if set(out.index) == set(CANONICAL_96_CONTEXTS) and len(out.index) == 96:
        out = out.loc[list(CANONICAL_96_CONTEXTS)]
    return out


def build_96_catalog(
    muts_df: pd.DataFrame,
    reference_index: pd.Index | None = None,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Convert a mutation table or 96-channel catalog into a channel x sample DataFrame.

    Supports:
    1. A pre-aggregated channel x sample catalog (optionally with `sub`/`tri`/`context`
       metadata columns).
    2. A variant-level mutation table with `sample` plus either `context` (e.g. `A[C>T]G`)
       or `(ref, alt, trinuc)` columns.
    """
    validate_settings(settings)
    if not isinstance(muts_df, pd.DataFrame):
        raise TypeError("muts_df must be a pandas DataFrame.")
    if muts_df.empty:
        raise ValueError("muts_df cannot be empty.")

    cols_lower = {str(c).lower(): c for c in muts_df.columns}

    # Case 1: Variant-level table with a `sample` column
    if "sample" in cols_lower and (
        "context" in cols_lower
        or ("ref" in cols_lower and "alt" in cols_lower and ("trinuc" in cols_lower or "tri" in cols_lower))
    ):
        sample_col = cols_lower["sample"]
        samples = muts_df[sample_col].astype(str)
        if "context" in cols_lower:
            contexts = muts_df[cols_lower["context"]].astype(str).str.strip()
        else:
            ref_col = cols_lower["ref"]
            alt_col = cols_lower["alt"]
            tri_col = cols_lower.get("trinuc", cols_lower.get("tri"))
            assert tri_col is not None
            contexts = pd.Series(
                [
                    normalize_context_label(r, a, t)
                    for r, a, t in zip(
                        muts_df[ref_col].astype(str),
                        muts_df[alt_col].astype(str),
                        muts_df[tri_col].astype(str),
                    )
                ],
                index=muts_df.index,
            )

        target_index = (
            list(reference_index)
            if reference_index is not None
            else list(CANONICAL_96_CONTEXTS)
        )
        ct = pd.crosstab(contexts, samples)
        ct = ct.reindex(index=target_index, fill_value=0)
        # Preserve sample appearance order
        ordered_samples = list(dict.fromkeys(samples.tolist()))
        return ct.reindex(columns=ordered_samples, fill_value=0).astype(float)

    # Case 2: Pre-aggregated catalog (channels x samples)
    df = muts_df.copy()
    ctx_idx = _extract_sub_tri_context(df)
    drop_meta = [
        c for c in df.columns if str(c).lower() in ("sub", "tri", "context", "mutationtype")
    ]
    if ctx_idx is not None:
        df = df.drop(columns=drop_meta)
        df.index = ctx_idx
    elif drop_meta:
        if str(drop_meta[0]).lower() == "mutationtype":
            df.index = pd.Index(df[drop_meta[0]].astype(str).tolist(), name="context")
        df = df.drop(columns=drop_meta)

    if df.shape[1] == 0:
        raise ValueError("muts_df catalog contains no sample columns.")

    vals = df.to_numpy(dtype=float)
    if np.any(~np.isfinite(vals)) or np.any(vals < 0.0):
        raise ValueError("Catalog mutation counts must be finite and non-negative.")

    out = pd.DataFrame(vals, index=df.index.copy(), columns=df.columns.astype(str))
    if reference_index is not None:
        if set(out.index) == set(reference_index) and len(out.index) == len(reference_index):
            out = out.loc[list(reference_index)]
        elif len(out.index) != len(reference_index):
            raise ValueError(
                f"Catalog channel count ({len(out.index)}) does not match "
                f"signature reference ({len(reference_index)})."
            )
        else:
            out.index = reference_index.copy()
    return out


def make_synthetic_cosmic_reference(
    signatures: Sequence[str] = (
        "SBS1",
        "SBS2",
        "SBS5",
        "SBS8",
        "SBS13",
        "SBS35",
        "SBS99",
    ),
) -> pd.DataFrame:
    """
    Construct a deterministic, biologically realistic 96-channel SBS reference DataFrame.

    Motifs follow canonical COSMIC profiles:
    - `SBS1`: N[C>T]G spontaneous 5-methylcytosine deamination peaks
    - `SBS2`: T[C>T]A / T[C>T]T APOBEC cytidine deaminase peaks
    - `SBS5`: Flat clock-like background spectrum across substitutions
    - `SBS8`: Broad C>A and T>A late-replicating spectrum
    - `SBS13`: T[C>G]A / T[C>G]T / T[C>A]T APOBEC spectrum
    - `SBS31` / `SBS35`: Platinum C>T / C>A spectrum
    - `SBS99` / `SBS-MM1`: Melphalan T>A / T>C transcriptional damage spectrum
    """
    ctx = list(CANONICAL_96_CONTEXTS)
    rng = np.random.default_rng(20210401)
    cols: dict[str, np.ndarray] = {}

    for sig in signatures:
        v = np.full(96, 0.001, dtype=float)
        if sig == "SBS1":
            for i, c in enumerate(ctx):
                if "[C>T]G" in c:
                    v[i] = 0.22
                elif "[C>T]" in c:
                    v[i] = 0.004
        elif sig == "SBS2":
            for i, c in enumerate(ctx):
                if c in ("T[C>T]A", "T[C>T]T"):
                    v[i] = 0.42
                elif c.startswith("T[C>T]"):
                    v[i] = 0.05
        elif sig == "SBS5":
            v = 0.008 + 0.006 * rng.uniform(0.5, 1.5, size=96)
            for i, c in enumerate(ctx):
                if "[C>T]" in c or "[T>C]" in c:
                    v[i] *= 1.6
        elif sig == "SBS8":
            for i, c in enumerate(ctx):
                if "[C>A]" in c:
                    v[i] = 0.032
                elif "[T>A]" in c:
                    v[i] = 0.024
                else:
                    v[i] = 0.002
        elif sig == "SBS13":
            for i, c in enumerate(ctx):
                if c in ("T[C>G]A", "T[C>G]T"):
                    v[i] = 0.34
                elif c in ("T[C>A]A", "T[C>A]T"):
                    v[i] = 0.11
        elif sig in ("SBS31", "SBS35"):
            for i, c in enumerate(ctx):
                if c in ("C[C>T]C", "C[C>T]T", "C[C>A]C", "G[C>A]C"):
                    v[i] = 0.18 if sig == "SBS31" else 0.13
                elif "[C>A]" in c or "[C>T]" in c:
                    v[i] = 0.015
        elif sig in ("SBS99", "SBS-MM1"):
            for i, c in enumerate(ctx):
                if "[T>A]" in c:
                    v[i] = 0.035 if c.endswith("A") or c.endswith("T") else 0.02
                elif "[T>C]" in c:
                    v[i] = 0.028 if c.startswith("G") or c.startswith("A") else 0.014
        else:
            # Deterministic distinct profile hashed by signature name
            seed_val = sum((idx + 1) * ord(ch) for idx, ch in enumerate(sig))
            local_rng = np.random.default_rng(seed_val)
            v = local_rng.dirichlet(np.full(96, 0.35))
        cols[sig] = v / v.sum()

    return pd.DataFrame(cols, index=pd.Index(ctx, name="context"))
