"""
End-to-end FACETS pipeline runner (`run_facets`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pandas as pd

from .em import FacetsResult, em_cncf
from .io import FacetsConfig, validate_settings
from .preprocess import pre_proc_sample
from .segment import proc_sample


def run_facets(
    pileup: pd.DataFrame,
    *,
    config: FacetsConfig | None = None,
    settings: Literal["default", "R_default"] = "default",
    **kwargs,
) -> FacetsResult:
    """
    Execute the complete FACETS workflow end-to-end on a tumor-normal SNP pileup:
      1. `pre_proc_sample` (`preprocess_sample`): depth/het filtering, `snp_nbhd` thinning, GC LOWESS logR, logOR
      2. `proc_sample`: two-stage bivariate CBS, clustering, `dipLogR`
      3. `em_cncf`: EM for purity, ploidy, integer ASCN, LOH, and CCF

    Supports `settings="default"` (Pythonic Hotelling-T2 + Folded-Normal EM) and
    `settings="R_default"` (exact `mskcc/facets` v0.6.2 upstream parity).
    """
    validate_settings(settings)
    if config is None:
        cfg = FacetsConfig(settings=settings, **kwargs)
    else:
        eff_settings = (
            config.settings
            if settings == "default" and config.settings != "default"
            else settings
        )
        cfg = replace(config, settings=eff_settings, **kwargs)

    eff_mode = cfg.settings
    preproc = pre_proc_sample(pileup, config=cfg, settings=eff_mode)
    segmented = proc_sample(preproc, settings=eff_mode)
    return em_cncf(
        segmented,
        max_cn=cfg.max_cn,
        min_nhet=cfg.min_nhet,
        diploid_logr=cfg.diploid_logr,
        keep_low_nhet_lcn=cfg.keep_low_nhet_lcn,
        settings=eff_mode,
    )
