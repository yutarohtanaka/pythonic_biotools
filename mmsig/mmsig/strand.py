"""
Transcriptional strand-bias asymmetry testing for chemotherapy signatures
(`SBS31`/`SBS35` platinum, `SBS99`/`SBS-MM1` melphalan) in `mmsig`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .reference import (
    CHEMO_STRAND_SIGNATURES,
    COMPLEMENT,
    SettingsMode,
    validate_settings,
)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values without external dependencies."""
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def _aggregate_variant_strand_df(strand_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate variant-level transcriptional strand annotations into counts."""
    cols_lower = {str(c).lower(): c for c in strand_df.columns}
    sample_col = cols_lower.get("sample")
    strand_col = cols_lower.get("strand", cols_lower.get("tx_strand"))
    if sample_col is None or strand_col is None:
        raise ValueError(
            "Variant-level strand_df requires 'sample' and 'strand' columns."
        )

    # Determine pyrimidine substitution class per variant
    if "sub" in cols_lower or "mut_type" in cols_lower:
        sub_col = cols_lower.get("sub", cols_lower.get("mut_type"))
        assert sub_col is not None
        subs = strand_df[sub_col].astype(str).str.strip().str.upper().tolist()
    elif "context" in cols_lower:
        ctx_col = cols_lower["context"]
        subs = [
            s[s.find("[") + 1 : s.find("]")] if ("[" in s and "]" in s) else s
            for s in strand_df[ctx_col].astype(str).str.strip().str.upper()
        ]
    elif "ref" in cols_lower and "alt" in cols_lower:
        refs = strand_df[cols_lower["ref"]].astype(str).str.strip().str.upper()
        alts = strand_df[cols_lower["alt"]].astype(str).str.strip().str.upper()
        subs = [
            f"{r}>{a}" if r in ("C", "T") else f"{COMPLEMENT.get(r, r)}>{COMPLEMENT.get(a, a)}"
            for r, a in zip(refs, alts)
        ]
    else:
        raise ValueError(
            "Variant-level strand_df must contain 'sub', 'context', or ('ref', 'alt')."
        )

    raw_strand = strand_df[strand_col].astype(str).str.strip().str.lower().tolist()
    norm_strand: list[str | None] = []
    for st in raw_strand:
        if st in ("t", "transcribed", "-", "-1", "template"):
            norm_strand.append("T")
        elif st in ("u", "untranscribed", "+", "+1", "coding", "nontranscribed"):
            norm_strand.append("U")
        else:
            norm_strand.append(None)

    work = pd.DataFrame(
        {
            "sample": strand_df[sample_col].astype(str).tolist(),
            "sub": subs,
            "strand_norm": norm_strand,
        }
    )
    work = work[work["strand_norm"].isin(["T", "U"])]

    records: list[dict[str, object]] = []
    samples = list(dict.fromkeys(strand_df[sample_col].astype(str).tolist()))
    for sample in samples:
        sub_s = work[work["sample"] == sample]
        for sig, target_subs in (
            ("SBS31", CHEMO_STRAND_SIGNATURES["SBS31"]),
            ("SBS35", CHEMO_STRAND_SIGNATURES["SBS35"]),
            ("SBS99", CHEMO_STRAND_SIGNATURES["SBS99"]),
        ):
            sub_sig = sub_s[sub_s["sub"].isin(target_subs)]
            n_t = int((sub_sig["strand_norm"] == "T").sum())
            n_u = int((sub_sig["strand_norm"] == "U").sum())
            records.append(
                {
                    "sample": sample,
                    "signature": sig,
                    "transcribed": n_t,
                    "untranscribed": n_u,
                }
            )
    return pd.DataFrame(records)


def check_strand_bias(
    strand_df: pd.DataFrame,
    alpha: float = 0.05,
    settings: SettingsMode = "default",
) -> pd.DataFrame:
    """
    Test transcriptional strand asymmetry via exact conditional Poisson / Binomial test.

    Given transcribed count $n_T \\sim \\text{Poisson}(\\lambda_T E_T)$ and untranscribed
    count $n_U \\sim \\text{Poisson}(\\lambda_U E_U)$, the exact conditional distribution
    of $n_T$ given $N = n_T + n_U$ under $H_0: \\lambda_T = \\lambda_U$ is
    $\\text{Binomial}(N, p_0)$ with $p_0 = E_T / (E_T + E_U)$ (default $0.5$, identical
    to R `stats::poisson.test(c(n_T, n_U), T = c(E_T, E_U))`).

    Parameters
    ----------
    strand_df : pd.DataFrame
        Either a summary count table with columns `sample`, `transcribed` (or `T`),
        `untranscribed` (or `U`), and optional `signature` / `opp_transcribed` /
        `opp_untranscribed`; or a variant-level table with `sample`, `strand`, and
        `sub` / `context` / (`ref`, `alt`).
    alpha : float, default 0.05
        Significance threshold on exact two-sided p-value (`p_value < alpha`).
    settings : {"default", "R_default"}, default "default"
        `"default"` computes Haldane-smoothed strand ratio `(n_T + 0.5) / (n_U + 0.5)`
        and Benjamini-Hochberg `q_value`; `"R_default"` computes exact unsmoothed
        `poisson.test` rate ratio `(n_T / E_T) / (n_U / E_U)` matching R `mmsig`.

    Returns
    -------
    pd.DataFrame
        Columns: `sample`, `signature`, `transcribed`, `untranscribed`, `total`,
        `expected_prop`, `strand_ratio`, `log2_ratio`, `p_value`, `q_value`, `significant`.
    """
    validate_settings(settings)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if not isinstance(strand_df, pd.DataFrame):
        raise TypeError("strand_df must be a pandas DataFrame.")
    if strand_df.empty:
        return pd.DataFrame(
            columns=[
                "sample",
                "signature",
                "transcribed",
                "untranscribed",
                "total",
                "expected_prop",
                "strand_ratio",
                "log2_ratio",
                "p_value",
                "q_value",
                "significant",
            ]
        )

    cols_lower = {str(c).lower(): c for c in strand_df.columns}
    t_candidates = ("transcribed", "t", "t_strand", "transcribed_count", "n_transcribed")
    u_candidates = ("untranscribed", "u", "u_strand", "untranscribed_count", "n_untranscribed")

    t_col = next((cols_lower[k] for k in t_candidates if k in cols_lower), None)
    u_col = next((cols_lower[k] for k in u_candidates if k in cols_lower), None)

    if t_col is None or u_col is None:
        df = _aggregate_variant_strand_df(strand_df)
        cols_lower = {str(c).lower(): c for c in df.columns}
        t_col = "transcribed"
        u_col = "untranscribed"
    else:
        df = strand_df.copy()

    sample_col = cols_lower.get("sample")
    sig_col = cols_lower.get("signature", cols_lower.get("sig", cols_lower.get("sub")))
    opp_t_col = cols_lower.get("opp_transcribed", cols_lower.get("e_t"))
    opp_u_col = cols_lower.get("opp_untranscribed", cols_lower.get("e_u"))

    rows: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        sample_name = str(row[sample_col]) if sample_col is not None else f"sample_{idx}"
        sig_name = str(row[sig_col]) if sig_col is not None else "SBS99"
        n_t_float = float(row[t_col])
        n_u_float = float(row[u_col])
        if n_t_float < 0 or n_u_float < 0:
            raise ValueError("Strand mutation counts must be non-negative.")
        n_t = int(round(n_t_float))
        n_u = int(round(n_u_float))
        total = n_t + n_u

        e_t = float(row[opp_t_col]) if opp_t_col is not None else 1.0
        e_u = float(row[opp_u_col]) if opp_u_col is not None else 1.0
        if e_t <= 0 or e_u <= 0:
            raise ValueError("Strand opportunity territories must be strictly positive.")
        p0 = e_t / (e_t + e_u)

        if total == 0:
            pval = 1.0
        else:
            pval = float(binomtest(k=n_t, n=total, p=p0, alternative="two-sided").pvalue)

        log2_ratio = float(np.log2(((n_t + 0.5) / e_t) / ((n_u + 0.5) / e_u)))
        if settings == "default":
            ratio = float(((n_t + 0.5) / e_t) / ((n_u + 0.5) / e_u))
        else:
            if n_u == 0:
                ratio = float("inf") if n_t > 0 else 1.0
            else:
                ratio = float((n_t / e_t) / (n_u / e_u))

        rows.append(
            {
                "sample": sample_name,
                "signature": sig_name,
                "transcribed": n_t,
                "untranscribed": n_u,
                "total": total,
                "expected_prop": p0,
                "strand_ratio": ratio,
                "log2_ratio": log2_ratio,
                "p_value": pval,
            }
        )

    out = pd.DataFrame(rows)
    out["q_value"] = _bh_fdr(out["p_value"].to_numpy(dtype=float))
    out["significant"] = out["p_value"] < alpha
    return out
