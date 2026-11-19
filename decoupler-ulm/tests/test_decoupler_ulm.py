from __future__ import annotations
import numpy as np
import pandas as pd
from decoupler_ulm import project_gep_programs

def test_project_gep_programs():
    rng = np.random.default_rng(5)
    genes = [f"G{i}" for i in range(60)]
    w_df = pd.DataFrame({"Prog1": np.r_[np.ones(15), np.zeros(45)], "Prog2": np.r_[np.zeros(45), np.ones(15)]}, index=genes)
    expr_df = pd.DataFrame({
        "S1": np.r_[rng.uniform(50, 120, 15), rng.uniform(1, 5, 45)],
        "S2": np.r_[rng.uniform(1, 5, 45), rng.uniform(50, 120, 15)],
        "S3": rng.uniform(5, 10, 60),
    }, index=genes)
    scores = project_gep_programs(expr_df, w_df, is_log_transformed=False, method="ulm_tstat")
    assert scores.loc["S1", "Prog1"] > 3.0 and scores.loc["S2", "Prog2"] > 3.0
