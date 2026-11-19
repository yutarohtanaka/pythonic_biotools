"""End-to-end drug-target cis-MR, proportional coloc, cTWAS, and LAVA demo."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pythonic_biotools_cli as biotools

biotools.enable_workspace_imports("statgen", repo_root=REPO_ROOT)

from cismr import mr_pcgmm, mr_spi
from colocproptest import run_coloc_prop_test
from ctwas import impute_expr_z, susie_rss_group_prior
from lava import process_locus, run_pcor


def main() -> None:
    rng = np.random.default_rng(7)
    m, n = 30, 50_000
    idx = np.arange(m)
    ld_R = 0.72 ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(ld_R)

    b_x_true = np.zeros(m)
    b_x_true[[4, 16]] = [0.20, 0.15]
    theta = 0.40
    se = np.full(m, 1.0 / np.sqrt(n))
    bx = ld_R @ b_x_true + se * (L @ rng.standard_normal(m))
    by = ld_R @ (theta * b_x_true) + se * (L @ rng.standard_normal(m))

    pcgmm = mr_pcgmm(bx, se, by, se, ld_R=ld_R, nx=n, ny=n, settings="default")
    spi = mr_spi(bx, se, by, se, ld_R=ld_R, n=n, settings="default")
    print(f"[cisMR] PC-GMM theta={pcgmm.theta:.3f} (SE={pcgmm.se:.3f}), MR-SPI={spi.estimate:.3f}")

    prop = run_coloc_prop_test({"beta": bx, "se": se}, {"beta": by, "se": se}, LD=ld_R, settings="default")
    print(f"[colocPropTest] eta_hat={prop.eta_hat:.3f}, reject_proportionality={prop.reject_proportionality}")

    gwas_z = pd.DataFrame({"id": [f"rs{i}" for i in range(m)], "z": by / se})
    imp = impute_expr_z(gwas_z, ld_R, [{"gene_id": "TARGET_GENE", "snps": ["rs4", "rs16"], "weights": [0.20, 0.15]}])
    susie = susie_rss_group_prior(imp.z_joint, imp.R_joint, imp.is_gene, 0.15, 0.005, 0.05, 0.005, n_samples=n)
    print(f"[cTWAS] Causal gene PIP={float(susie.pip[imp.is_gene][0]):.3f}")

    z_df = pd.DataFrame({"X": bx / se, "Y": by / se, "Z": L @ rng.standard_normal(m)})
    locus = process_locus(z_df, ld_R, sample_sizes={"X": n, "Y": n, "Z": n})
    pcor = run_pcor(locus, pair=("X", "Y"), cond_traits=["Z"], n_sim=2000)
    print(f"[LAVA] Partial local r_g(X, Y | Z)={float(pcor['pcor'].iloc[0]):.3f} (p={float(pcor['p'].iloc[0]):.3e})")


if __name__ == "__main__":
    main()
