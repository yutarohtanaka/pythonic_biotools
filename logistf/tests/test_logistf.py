from __future__ import annotations
import numpy as np
from logistf import fit_firth_logistic

def test_firth_logistic_handles_complete_separation():
    x = np.array([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
    y = (x > 0).astype(int)
    res = fit_firth_logistic(x, y, add_intercept=True, coef_names=["biomarker"])
    assert np.all(np.isfinite(res.beta))
    assert 0.5 < res.beta[1] < 10.0
