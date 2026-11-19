"""
End-to-end FACETS pipeline runner (`run_facets`).
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .em import FacetsResult, em_cncf
from .io import FacetsConfig
from .preprocess import pre_proc_sample
from .segment import proc_sample


def run_facets(
    pileup: pd.DataFrame,
    *,
    config: FacetsConfig | None = None,
    **kwargs,
) -> FacetsResult:
    """
    Execute the complete FACETS workflow end-to-end on a tumor-normal SNP pileup:
      1. `pre_proc_sample`: depth/het filtering, `snp_nbhd` thinning, GC LOWESS logR, Haldane logOR
      2. `proc_sample`: two-stage bivariate Hotelling-T2 CBS, clustering, `dipLogR`
      3. `em_cncf`: Gaussian + folded-normal EM for purity, ploidy, integer ASCN, LOH, and CCF
    """
    if config is None:
        cfg = FacetsConfig(**kwargs)
    elif kwargs:
        cfg = replace(config, **kwargs)
    else:
        cfg = config

    preproc = pre_proc_sample(pileup, config=cfg)
    segmented = proc_sample(preproc)
    return em_cncf(
        segmented,
        max_cn=cfg.max_cn,
        min_nhet=cfg.min_nhet,
        diploid_logr=cfg.diploid_logr,
        keep_low_nhet_lcn=cfg.keep_low_nhet_lcn,
    )
