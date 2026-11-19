"""
`mmsig`: Python implementation of mmsig (Rustad et al., Communications Biology 2021)
for interpretable mutational signature refitting with backward cosine-similarity
pruning, multinomial bootstrap CIs, and transcriptional strand-bias confirmation.
"""

from __future__ import annotations

from .core import (
    MMSig,
    MMSigResult,
    cosine_similarity_vec,
    fit_multinomial_em_nnls,
    fit_simplex_ls,
    mm_fit_signatures,
    prune_signatures_single,
)
from .reference import (
    CANONICAL_96_CONTEXTS,
    CHEMO_STRAND_SIGNATURES,
    build_96_catalog,
    make_synthetic_cosmic_reference,
    normalize_context_label,
    prepare_signature_reference,
)
from .strand import check_strand_bias

__all__ = [
    "CANONICAL_96_CONTEXTS",
    "CHEMO_STRAND_SIGNATURES",
    "MMSig",
    "MMSigResult",
    "build_96_catalog",
    "check_strand_bias",
    "cosine_similarity_vec",
    "fit_multinomial_em_nnls",
    "fit_simplex_ls",
    "make_synthetic_cosmic_reference",
    "mm_fit_signatures",
    "normalize_context_label",
    "prepare_signature_reference",
    "prune_signatures_single",
]

__version__ = "0.1.0"
