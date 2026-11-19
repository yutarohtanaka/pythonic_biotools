# `evam`: Python Implementations of Published Evolutionary Accumulation Models (`EvAM-Tools`)

Standalone Python implementations and wrappers for published cross-sectional cancer progression models:
- **Oncogenetic Trees (`fit_ot`)**: Desper et al. (*J Comput Biol* 1999) and Szabo & Boucher (`Oncotree`).
- **Disjunctive & Conjunctive Bayesian Networks (`fit_oncobn`)**: Nicol et al. (*PLOS Comput Biol* 2021, `OncoBN`).
- **Hidden Conjunctive Bayesian Networks (`fit_cbn`)**: Gerstung et al. (*Bioinformatics* 2009, `ct-cbn`).
- **Mutual Hazard Networks (`fit_mhn`, `rates_from_theta`, `order_from_theta`)**: Schill et al. (*Bioinformatics* 2020, `MHN`).
- **Exact & Monte-Carlo CTMC Path Ordering (`ctmc_order_exact`, `ctmc_order_mc`)**: Diaz-Uriarte & Herrera-Nieto (*Bioinformatics* 2022, `EvAM-Tools`).

## License

Derivative translations of `EvAM-Tools` (Diaz-Uriarte & Herrera-Nieto, 2022) in `ot.py`, `oncobn.py`, and `cbn.py` are distributed under **AGPL-3.0-or-later**.
