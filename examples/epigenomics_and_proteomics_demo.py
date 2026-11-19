"""End-to-end epigenomics (DSS + monaLisa) and proteomics (DEqMS) demo."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pythonic_biotools_cli as biotools

biotools.enable_workspace_imports("all", repo_root=REPO_ROOT)

from deqms import median_sweeping, output_result, spectra_count_ebayes
from dss import call_dmr, dml_test
from monalisa import calc_binned_motif_enrichment, pfm_to_pwm


def main() -> None:
    rng = np.random.default_rng(123)
    pos = np.arange(1000, 1000 + 50 * 40, 40)
    ctrl, trt = [], []
    for _ in range(3):
        n_cov = rng.integers(25, 45, size=50)
        ctrl.append(pd.DataFrame({"chr": "chr1", "pos": pos, "N": n_cov, "X": rng.binomial(n_cov, 0.20)}))
        p_t = np.where((pos >= 1400) & (pos <= 1850), 0.80, 0.20)
        trt.append(pd.DataFrame({"chr": "chr1", "pos": pos, "N": n_cov, "X": rng.binomial(n_cov, p_t)}))

    dmls = dml_test(ctrl, trt, settings="default")
    dmrs = call_dmr(dmls, p_threshold=1e-3, minCG=3, minlen=50)
    print(f"[DSS] Detected {len(dmrs)} DMR(s)")

    pfm = np.array([[2, 92, 2, 92, 92, 2], [2, 2, 2, 2, 2, 92], [94, 4, 2, 4, 4, 4], [2, 2, 94, 2, 2, 2]], float)
    enr = calc_binned_motif_enrichment(
        ["TCGATAACTCGATAAC" * 4] * 20 + ["TCGCAGCGTCGCAGCG" * 4] * 20,
        ["hyper"] * 20 + ["hypo"] * 20,
        pwms={"GATAAC": pfm_to_pwm(pfm)},
    )
    print("[monaLisa] Binned motif enrichment:\n", enr.to_dataframe()[["motif", "bin", "log2enr", "neg_log10_padj"]])

    psm_rows = []
    for p_idx in range(30):
        n_psms = 10 if p_idx < 8 else 2
        shift = 1.5 if p_idx == 0 else 0.0
        for _ in range(n_psms):
            vals = np.concatenate([rng.normal(20.0, 0.25, 3), rng.normal(20.0 + shift, 0.25, 3)])
            row = {"protein": f"PROT_{p_idx}"}
            row.update({f"S{k}": vals[k] for k in range(6)})
            psm_rows.append(row)
    swept = median_sweeping(pd.DataFrame(psm_rows), group_col="protein")
    fit = spectra_count_ebayes(swept.abundance, design=np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]]), count=swept.count)
    top = output_result(fit, coef=1)
    print("[DEqMS] Top protein:", top.iloc[0][["logFC", "count", "sca.t", "sca.P.Value"]].to_dict())


if __name__ == "__main__":
    main()
