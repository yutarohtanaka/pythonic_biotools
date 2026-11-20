"""
`facets`: Python implementation of FACETS (Shen & Seshan, Nucleic Acids Research 2016)
for allele-specific somatic copy number, tumor purity, ploidy, and clonal cellular
fraction estimation, paired with ASCAT equations (Van Loo et al., PNAS 2010).
"""

from __future__ import annotations

from .ascat import (
    AscatResult,
    ascat_baf,
    ascat_invert_cn,
    ascat_logr,
    compare_facets_ascat,
    fit_ascat,
    run_ascat,
)
from .core import run_facets
from .em import FacetsResult, em_cncf
from .io import FacetsConfig, validate_and_normalize_pileup
from .plotting import plot_facets
from .preprocess import (
    PreProcResult,
    compute_cnlr,
    compute_valor,
    lowess_1d,
    pre_proc_sample,
    preprocess_sample,
)
from .reference import (
    GRCH37_REFERENCE,
    GRCH38_REFERENCE,
    ChromArmReference,
    canonical_chrom,
    get_genome_reference,
)
from .segment import (
    SegmentedResult,
    cluster_segments,
    find_diploid_logr,
    hotelling_t2_cbs,
    proc_sample,
)

__all__ = [
    # Config & Reference
    "FacetsConfig",
    "ChromArmReference",
    "GRCH38_REFERENCE",
    "GRCH37_REFERENCE",
    "canonical_chrom",
    "get_genome_reference",
    "validate_and_normalize_pileup",
    # Result Dataclasses
    "PreProcResult",
    "SegmentedResult",
    "FacetsResult",
    "AscatResult",
    # Pipeline Functions
    "pre_proc_sample",
    "preprocess_sample",
    "proc_sample",
    "em_cncf",
    "run_facets",
    "plot_facets",
    # Normalization & Segmentation Primitives
    "lowess_1d",
    "compute_cnlr",
    "compute_valor",
    "hotelling_t2_cbs",
    "cluster_segments",
    "find_diploid_logr",
    # ASCAT Equations & Solver
    "ascat_logr",
    "ascat_baf",
    "ascat_invert_cn",
    "fit_ascat",
    "run_ascat",
    "compare_facets_ascat",
]

__version__ = "0.1.0"
