from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import polygamma
from limma import limma_top_table, moderated_lm_contrast, moderated_ttest, trigamma_inverse

def test_trigamma_inverse_and_moderated_models():
    y_true = np.array([0.25, 1.5, 5.0, 25.0])
    np.testing.assert_allclose(trigamma_inverse(polygamma(1, y_true)), y_true, rtol=1e-6)

    rng = np.random.default_rng(42)
    n1, n2, n_genes = 6, 6, 200
    X = rng.normal(0.0, 1.0, size=(n1 + n2, n_genes))
    group = np.array([False] * n1 + [True] * n2)
    X[group, :5] += 3.5

    res_t = moderated_ttest(X, group, feature_names=[f"G{i}" for i in range(n_genes)])
    design = np.column_stack([np.ones(n1 + n2), group.astype(float)])
    res_lm = moderated_lm_contrast(X, design, contrast=[0.0, 1.0], feature_names=[f"G{i}" for i in range(n_genes)])
    np.testing.assert_allclose(res_t.t_stat, res_lm.t_stat, atol=1e-10)
    top = limma_top_table(pd.DataFrame(X, columns=[f"G{i}" for i in range(n_genes)]), group)
    assert set(top.index[:5]) == {"G0", "G1", "G2", "G3", "G4"}
