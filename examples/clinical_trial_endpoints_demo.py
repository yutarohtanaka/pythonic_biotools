"""End-to-end clinical trial endpoints demo (timeROC, survIDINRI, survRM2, BuyseTest)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pythonic_biotools_cli as biotools

biotools.enable_workspace_imports("survival", repo_root=REPO_ROOT)

from buysetest import Endpoint, buyse_test
from survidinri import idi_inf
from survrm2 import rmst2
from timeroc import time_roc


def main() -> None:
    rng = np.random.default_rng(99)
    n = 120
    arm = np.repeat([0, 1], n // 2)
    stage = rng.integers(0, 2, size=n).astype(float)
    genomic_score = rng.normal(0.0, 1.0, size=n)

    t_event = rng.exponential(scale=np.exp(0.6 * arm - 0.8 * genomic_score - 0.35 * stage) * 6.0)
    t_cens = rng.uniform(4.0, 15.0, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)

    roc = time_roc(T=time, delta=event, marker=genomic_score, cause=1, times=[3.0, 5.0])
    print(f"[timeROC] AUC(t=3,5)={np.round(roc.auc_1, 3)}")

    idi = idi_inf(time, event, stage[:, None], np.column_stack([stage, genomic_score]), t0=5.0, npert=80)
    print(f"[survIDINRI] IDI={idi.m1_idi.est:.3f} (p={idi.m1_idi.pval:.3e}), cNRI={idi.m2_nri.est:.3f}")

    rmst = rmst2(time, event, arm, tau=8.0)
    print(f"[survRM2] RMST diff (Arm1 - Arm0) at tau=8.0: {rmst.rmst_diff_10.est:.3f} (p={rmst.rmst_diff_10.pval:.3e})")

    df = pd.DataFrame({"arm": arm, "os": time, "os_ev": event, "tox": rng.binomial(1, np.where(arm == 1, 0.15, 0.35))})
    bt = buyse_test(
        df,
        treatment="arm",
        endpoints=[Endpoint("os", "tte", "os_ev", threshold=0.5), Endpoint("tox", "binary", operator="lower")],
    )
    tbl = bt.to_dataframe()
    print(f"[BuyseTest] Composite Win Ratio={float(tbl['win_ratio'].iloc[-1]):.3f}, Net Benefit={float(tbl['net_benefit'].iloc[-1]):.3f}")


if __name__ == "__main__":
    main()
