"""Unified CisMR drug-target sensitivity analysis wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .genius import MRGeniusResult, mr_genius, mr_genius_summary
from .pcgmm import PCGMMResult, mr_pcgmm, mr_pcivw
from .spi import MRSPIResult, mr_spi
from .utils import SettingsType, validate_settings


@dataclass
class CisMRReport:
    """Comprehensive cis-locus Mendelian Randomization sensitivity report."""

    pcgmm: PCGMMResult
    pcivw: PCGMMResult
    spi: MRSPIResult
    genius_summary: MRGeniusResult | None
    genius_individual: MRGeniusResult | None
    settings: SettingsType

    def to_dataframe(self) -> pd.DataFrame:
        """Return a standardized sensitivity summary DataFrame across all run methods."""
        rows: list[dict[str, object]] = [
            {
                "method": "PC-GMM",
                "estimate": self.pcgmm.estimate,
                "se": self.pcgmm.se,
                "ci_low": self.pcgmm.ci_low,
                "ci_high": self.pcgmm.ci_high,
                "pval": self.pcgmm.pval,
                "n_instruments": self.pcgmm.n_pcs,
                "f_stat": self.pcgmm.f_stat,
                "het_stat": self.pcgmm.j_stat,
                "het_pval": self.pcgmm.j_pval,
            },
            {
                "method": "PC-IVW",
                "estimate": self.pcivw.estimate,
                "se": self.pcivw.se,
                "ci_low": self.pcivw.ci_low,
                "ci_high": self.pcivw.ci_high,
                "pval": self.pcivw.pval,
                "n_instruments": self.pcivw.n_pcs,
                "f_stat": self.pcivw.f_stat,
                "het_stat": self.pcivw.j_stat,
                "het_pval": self.pcivw.j_pval,
            },
            {
                "method": "MR-SPI",
                "estimate": self.spi.estimate,
                "se": self.spi.se,
                "ci_low": self.spi.ci_low,
                "ci_high": self.spi.ci_high,
                "pval": self.spi.pval,
                "n_instruments": self.spi.n_valid,
                "f_stat": self.spi.f_stat,
                "het_stat": self.spi.het_stat,
                "het_pval": self.spi.het_pval,
            },
        ]
        if self.genius_summary is not None:
            rows.append(
                {
                    "method": "MR-GENIUS (summary_vqtl)",
                    "estimate": self.genius_summary.estimate,
                    "se": self.genius_summary.se,
                    "ci_low": self.genius_summary.ci_low,
                    "ci_high": self.genius_summary.ci_high,
                    "pval": self.genius_summary.pval,
                    "n_instruments": self.genius_summary.n_instruments,
                    "f_stat": self.genius_summary.het_f_stat,
                    "het_stat": self.genius_summary.j_stat,
                    "het_pval": self.genius_summary.j_pval,
                }
            )
        if self.genius_individual is not None:
            rows.append(
                {
                    "method": "MR-GENIUS (individual)",
                    "estimate": self.genius_individual.estimate,
                    "se": self.genius_individual.se,
                    "ci_low": self.genius_individual.ci_low,
                    "ci_high": self.genius_individual.ci_high,
                    "pval": self.genius_individual.pval,
                    "n_instruments": self.genius_individual.n_instruments,
                    "f_stat": self.genius_individual.het_f_stat,
                    "het_stat": self.genius_individual.j_stat,
                    "het_pval": self.genius_individual.j_pval,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"CisMR Drug-Target Sensitivity Suite (settings={self.settings!r})",
            self.pcgmm.summary(),
            self.pcivw.summary(),
            self.spi.summary(),
        ]
        if self.genius_summary is not None:
            lines.append(self.genius_summary.summary())
        if self.genius_individual is not None:
            lines.append(self.genius_individual.summary())
        return "\n".join(lines)


class CisMR:
    """Unified cis-Mendelian Randomization drug-target sensitivity runner.

    Executes correlated-instrument ``PC-GMM``, ``PC-IVW``, ``MR-SPI`` plurality
    voting, and optionally ``MR-GENIUS`` when summary vQTL variance effects
    (``v_gx``) or individual-level data (``G, X, Y``) are supplied.
    """

    def __init__(
        self,
        thres: float = 0.99,
        robust: bool = True,
        alpha: float = 0.05,
        max_iter: int = 50,
        settings: SettingsType = "default",
    ) -> None:
        self.settings = validate_settings(settings)
        if not (0.0 < thres <= 1.0):
            raise ValueError(f"thres must be in (0, 1], got {thres}.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.thres = float(thres)
        self.robust = bool(robust)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.report_: CisMRReport | None = None

    def fit(
        self,
        bx: np.ndarray | Sequence[float],
        sx: np.ndarray | Sequence[float],
        by: np.ndarray | Sequence[float],
        sy: np.ndarray | Sequence[float],
        ld_R: np.ndarray,
        nx: int | float,
        ny: int | float | None = None,
        v_gx: np.ndarray | Sequence[float] | None = None,
        se_vgx: np.ndarray | Sequence[float] | None = None,
        G: np.ndarray | None = None,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        covariates: np.ndarray | None = None,
    ) -> CisMRReport:
        """Run the cis-locus Mendelian Randomization sensitivity table."""
        pcgmm_res = mr_pcgmm(
            bx=bx,
            sx=sx,
            by=by,
            sy=sy,
            ld_R=ld_R,
            nx=nx,
            ny=ny,
            thres=self.thres,
            robust=self.robust,
            max_iter=self.max_iter,
            alpha=self.alpha,
            settings=self.settings,
        )
        pcivw_res = mr_pcivw(
            bx=bx,
            sx=sx,
            by=by,
            sy=sy,
            ld_R=ld_R,
            nx=nx,
            thres=self.thres,
            robust=self.robust,
            alpha=self.alpha,
            settings=self.settings,
        )
        spi_res = mr_spi(
            bx=bx,
            sx=sx,
            by=by,
            sy=sy,
            ld_R=ld_R,
            n=nx,
            alpha=self.alpha,
            settings=self.settings,
        )
        genius_sum = (
            mr_genius_summary(
                bx=bx,
                sx=sx,
                by=by,
                sy=sy,
                v_gx=v_gx,
                se_vgx=se_vgx,
                ld_R=ld_R,
                alpha=self.alpha,
                settings=self.settings,
            )
            if v_gx is not None
            else None
        )
        genius_ind = (
            mr_genius(
                G=G,
                X=X,
                Y=Y,
                covariates=covariates,
                alpha=self.alpha,
                settings=self.settings,
            )
            if (G is not None and X is not None and Y is not None)
            else None
        )
        self.report_ = CisMRReport(
            pcgmm=pcgmm_res,
            pcivw=pcivw_res,
            spi=spi_res,
            genius_summary=genius_sum,
            genius_individual=genius_ind,
            settings=self.settings,
        )
        return self.report_

    def __call__(
        self,
        bx: np.ndarray | Sequence[float],
        sx: np.ndarray | Sequence[float],
        by: np.ndarray | Sequence[float],
        sy: np.ndarray | Sequence[float],
        ld_R: np.ndarray,
        nx: int | float,
        ny: int | float | None = None,
        v_gx: np.ndarray | Sequence[float] | None = None,
        se_vgx: np.ndarray | Sequence[float] | None = None,
        G: np.ndarray | None = None,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        covariates: np.ndarray | None = None,
    ) -> CisMRReport:
        return self.fit(
            bx=bx,
            sx=sx,
            by=by,
            sy=sy,
            ld_R=ld_R,
            nx=nx,
            ny=ny,
            v_gx=v_gx,
            se_vgx=se_vgx,
            G=G,
            X=X,
            Y=Y,
            covariates=covariates,
        )
