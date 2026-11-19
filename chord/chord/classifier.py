"""
CHORD Random Forest classifier, synthetic pan-cancer HRD archetype generator,
and HRD / BRCA1-type / BRCA2-type prediction (`CHORDClassifier`, `chord_predict`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .channels import (
    ALL_CHORD_CHANNELS,
    INDEL_CHANNELS,
    INDEL_MH_DEL_CHANNELS,
    INDEL_NONE_CHANNELS,
    INDEL_REP_DEL_CHANNELS,
    INDEL_REP_INS_CHANNELS,
    R_SV_CHANNEL_ALIASES,
    SBS96_CHANNELS,
    SV_CHANNELS,
)
from .config import CHORDConfig, SettingsMode, get_chord_config, validate_settings

CLASS_LABELS: tuple[str, ...] = ("HRP", "BRCA1", "BRCA2")
_CACHED_ENSEMBLES: dict[SettingsMode, RandomForestClassifier] = {}


@dataclass(frozen=True)
class CHORDResult:
    """
    Prediction container returned by `chord_predict` and `CHORDClassifier.predict`.
    Provides DataFrame access (`to_dataframe()`, `res["p_hrd"]`) as well as scalar
    convenience attributes for single-sample predictions (or Series for cohorts).
    """

    predictions: pd.DataFrame
    settings: SettingsMode

    def _col_or_scalar(self, col: str) -> Any:
        s = self.predictions[col]
        if len(s) == 1:
            val = s.iloc[0]
            if isinstance(val, (np.floating, float)):
                return float(val)
            return str(val)
        return s

    @property
    def p_BRCA1(self) -> float | pd.Series:
        return self._col_or_scalar("p_BRCA1")

    @property
    def p_BRCA2(self) -> float | pd.Series:
        return self._col_or_scalar("p_BRCA2")

    @property
    def p_hrd(self) -> float | pd.Series:
        return self._col_or_scalar("p_hrd")

    @property
    def p_hrp(self) -> float | pd.Series:
        return self._col_or_scalar("p_hrp")

    @property
    def hr_status(self) -> str | pd.Series:
        return self._col_or_scalar("hr_status")

    @property
    def hrd_type(self) -> str | pd.Series:
        return self._col_or_scalar("hrd_type")

    @property
    def remarks(self) -> str | pd.Series:
        return self._col_or_scalar("remarks")

    def to_dataframe(self) -> pd.DataFrame:
        return self.predictions.copy()

    def summary(self) -> str:
        n = len(self.predictions)
        n_hrd = int((self.predictions["hr_status"] == "HR_deficient").sum())
        n_b1 = int((self.predictions["hrd_type"] == "BRCA1_type").sum())
        n_b2 = int((self.predictions["hrd_type"] == "BRCA2_type").sum())
        return (
            f"CHORDResult (settings={self.settings}, samples={n})\n"
            f"  HR_deficient: {n_hrd} (BRCA1_type={n_b1}, BRCA2_type={n_b2})\n"
            f"  HR_proficient: {int((self.predictions['hr_status'] == 'HR_proficient').sum())}"
        )

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, key: Any) -> Any:
        return self.predictions[key]


def align_contexts_df(contexts_df: pd.DataFrame) -> pd.DataFrame:
    """Align a context count DataFrame to `ALL_CHORD_CHANNELS`, filling missing channels with 0."""
    if not isinstance(contexts_df, pd.DataFrame) or contexts_df.empty:
        raise ValueError("`contexts_df` must be a non-empty pandas DataFrame.")

    df = contexts_df.copy()
    if "sample" in df.columns and df.index.name != "sample":
        df = df.set_index("sample")

    # Rename legacy R CHORD SV column aliases if present
    rename_map = {
        col: R_SV_CHANNEL_ALIASES[col]
        for col in df.columns
        if col in R_SV_CHANNEL_ALIASES and R_SV_CHANNEL_ALIASES[col] not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    out = pd.DataFrame(0.0, index=df.index, columns=list(ALL_CHORD_CHANNELS))
    shared = [c for c in ALL_CHORD_CHANNELS if c in df.columns]
    if not shared:
        raise ValueError(
            "`contexts_df` contains none of the expected CHORD SBS/Indel/SV channels."
        )
    out[shared] = np.maximum(df[shared].to_numpy(dtype=np.float64), 0.0)
    return out


def compute_chord_features(aligned_df: pd.DataFrame, cfg: CHORDConfig) -> np.ndarray:
    """
    Transform raw 132-channel counts into CHORD normalized feature matrix:
    - `"R_default"`: exact within-class (`snv`, `indel`, `sv`) relative abundance
      (`x / sum(x)`, `0` if class sum is `0`) + linear class sub-aggregates.
    - `"default"`: Laplace-smoothed within-class relative abundance + class
      sub-aggregates + 3 log1p burden features.
    """
    mat = aligned_df.to_numpy(dtype=np.float64)
    n_sbs = len(SBS96_CHANNELS)
    n_ind = len(INDEL_CHANNELS)
    n_sv = len(SV_CHANNELS)

    sbs = mat[:, :n_sbs]
    ind = mat[:, n_sbs : n_sbs + n_ind]
    sv = mat[:, n_sbs + n_ind :]

    sbs_sum = np.sum(sbs, axis=1, keepdims=True)
    ind_sum = np.sum(ind, axis=1, keepdims=True)
    sv_sum = np.sum(sv, axis=1, keepdims=True)

    alpha = cfg.laplace_alpha
    if alpha > 0.0:
        sbs_rel = np.where(
            sbs_sum > 0, (sbs + alpha) / (sbs_sum + n_sbs * alpha), 0.0
        )
        ind_rel = np.where(
            ind_sum > 0, (ind + alpha) / (ind_sum + n_ind * alpha), 0.0
        )
        sv_rel = np.where(
            sv_sum > 0, (sv + alpha) / (sv_sum + n_sv * alpha), 0.0
        )
    else:
        sbs_rel = np.divide(sbs, sbs_sum, out=np.zeros_like(sbs), where=sbs_sum > 0)
        ind_rel = np.divide(ind, ind_sum, out=np.zeros_like(ind), where=ind_sum > 0)
        sv_rel = np.divide(sv, sv_sum, out=np.zeros_like(sv), where=sv_sum > 0)

    # Exact relative abundance sub-aggregates (invariant to total mutation scale in R_default)
    ind_col_idx = {c: i for i, c in enumerate(INDEL_CHANNELS)}
    sv_col_idx = {c: i for i, c in enumerate(SV_CHANNELS)}

    mh_2_5_cols = [ind_col_idx[f"del.mh.bimh.{k}"] for k in range(2, 6)]
    mh_all_cols = [ind_col_idx[c] for c in INDEL_MH_DEL_CHANNELS]
    rep_cols = [ind_col_idx[c] for c in INDEL_REP_DEL_CHANNELS + INDEL_REP_INS_CHANNELS]
    none_cols = [ind_col_idx[c] for c in INDEL_NONE_CHANNELS]

    dup_short_cols = [sv_col_idx["DUP_1e3_1e4"], sv_col_idx["DUP_1e4_1e5"]]
    del_short_cols = [sv_col_idx["DEL_1e3_1e4"], sv_col_idx["DEL_1e4_1e5"]]
    dup_all_cols = [sv_col_idx[c] for c in SV_CHANNELS if c.startswith("DUP_")]
    del_all_cols = [sv_col_idx[c] for c in SV_CHANNELS if c.startswith("DEL_")]

    # Use unsmoothed exact class proportions for summary ratios so zero counts stay exact
    ind_exact = np.divide(ind, ind_sum, out=np.zeros_like(ind), where=ind_sum > 0)
    sv_exact = np.divide(sv, sv_sum, out=np.zeros_like(sv), where=sv_sum > 0)
    sbs_exact = np.divide(sbs, sbs_sum, out=np.zeros_like(sbs), where=sbs_sum > 0)

    rel_mh_2_5 = np.sum(ind_exact[:, mh_2_5_cols], axis=1, keepdims=True)
    rel_mh_all = np.sum(ind_exact[:, mh_all_cols], axis=1, keepdims=True)
    rel_rep = np.sum(ind_exact[:, rep_cols], axis=1, keepdims=True)
    rel_none = np.sum(ind_exact[:, none_cols], axis=1, keepdims=True)

    rel_dup_short = np.sum(sv_exact[:, dup_short_cols], axis=1, keepdims=True)
    rel_del_short = np.sum(sv_exact[:, del_short_cols], axis=1, keepdims=True)
    rel_dup_all = np.sum(sv_exact[:, dup_all_cols], axis=1, keepdims=True)
    rel_del_all = np.sum(sv_exact[:, del_all_cols], axis=1, keepdims=True)
    dup_minus_del_short = rel_dup_short - rel_del_short

    # Normalized Shannon entropy of SBS96 (flat SBS3 -> ~0.97+, peaked SBS1/APOBEC -> <0.85)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_log_p = np.where(sbs_exact > 1e-12, -sbs_exact * np.log(sbs_exact), 0.0)
    sbs_entropy = np.sum(p_log_p, axis=1, keepdims=True) / np.log(float(n_sbs))

    parts = [
        sbs_rel,
        ind_rel,
        sv_rel,
        rel_mh_2_5,
        rel_mh_all,
        rel_rep,
        rel_none,
        rel_dup_short,
        rel_del_short,
        rel_dup_all,
        rel_del_all,
        dup_minus_del_short,
        sbs_entropy,
    ]
    if cfg.include_log_burden:
        parts.append(np.log1p(np.hstack([sbs_sum, ind_sum, sv_sum])))

    return np.hstack(parts)


def simulate_chord_archetypes(
    n_per_class: int = 160,
    seed: int = 2020,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Deterministically synthesize pan-cancer mutational catalogs encoding Nguyen et al.
    (Nature Communications 2020) `HRP`, `BRCA1`-type, and `BRCA2`-type archetypes.
    """
    rng = np.random.default_rng(seed)
    n_sbs = len(SBS96_CHANNELS)
    n_ind = len(INDEL_CHANNELS)
    n_sv = len(SV_CHANNELS)

    # Build SBS probability templates
    # 1) SBS3 (broad flat HRD spectrum with slight C>A / C>G / T>A enrichment)
    sbs3_p = np.ones(n_sbs, dtype=np.float64)
    for i, ch in enumerate(SBS96_CHANNELS):
        if "[C>G]" in ch or "[C>A]" in ch or "[T>A]" in ch:
            sbs3_p[i] = 1.35
    sbs3_p /= np.sum(sbs3_p)

    # 2) Clock-like SBS1 + SBS5 (strong N[C>T]G CpG deamination + T>C transitions)
    sbs1_5_p = np.full(n_sbs, 0.35, dtype=np.float64)
    for i, ch in enumerate(SBS96_CHANNELS):
        if "[C>T]G" in ch:
            sbs1_5_p[i] = 9.0
        elif "[C>T]" in ch or "[T>C]" in ch:
            sbs1_5_p[i] = 1.6
    sbs1_5_p /= np.sum(sbs1_5_p)

    # 3) APOBEC SBS2 + SBS13 (T[C>T]N and T[C>G]N)
    sbs_apobec_p = np.full(n_sbs, 0.25, dtype=np.float64)
    for i, ch in enumerate(SBS96_CHANNELS):
        if ch.startswith("T[C>T]") or ch.startswith("T[C>G]"):
            sbs_apobec_p[i] = 8.5
    sbs_apobec_p /= np.sum(sbs_apobec_p)

    ind_map = {c: i for i, c in enumerate(INDEL_CHANNELS)}
    sv_map = {c: i for i, c in enumerate(SV_CHANNELS)}

    rows: list[np.ndarray] = []
    labels: list[str] = []
    sample_names: list[str] = []

    # --- Class 0: HRP ---
    for k in range(n_per_class):
        tot_sbs = int(rng.integers(400, 15_000))
        tot_ind = int(rng.integers(25, 1_200))
        tot_sv = int(rng.integers(0, 180))

        subtype = k % 3
        if subtype == 0:
            sbs_prob = rng.dirichlet(40.0 * sbs1_5_p + 0.15)
        elif subtype == 1:
            sbs_prob = rng.dirichlet(35.0 * sbs_apobec_p + 15.0 * sbs1_5_p + 0.15)
        else:
            sbs_prob = rng.dirichlet(25.0 * sbs1_5_p + 0.4)

        ind_alpha = np.full(n_ind, 0.2, dtype=np.float64)
        for c in INDEL_REP_DEL_CHANNELS:
            ind_alpha[ind_map[c]] = 4.5
        for c in INDEL_REP_INS_CHANNELS:
            ind_alpha[ind_map[c]] = 3.0
        ind_alpha[ind_map["del.none"]] = 4.0
        ind_alpha[ind_map["ins.none"]] = 2.5
        ind_alpha[ind_map["del.mh.bimh.1"]] = 1.5
        # Very low bimh >= 2 in HRP
        for m_i in range(2, 6):
            ind_alpha[ind_map[f"del.mh.bimh.{m_i}"]] = 0.20
        ind_prob = rng.dirichlet(ind_alpha)

        sv_alpha = np.full(n_sv, 0.5, dtype=np.float64)
        # Scattered large SVs / inversions / translocations in HRP
        sv_alpha[sv_map["DEL_1e5_1e6"]] = 2.5
        sv_alpha[sv_map["DUP_1e6_1e7"]] = 2.5
        sv_alpha[sv_map["INV_1e5_1e6"]] = 2.5
        sv_alpha[sv_map["TRA"]] = 3.5
        sv_alpha[sv_map["DUP_1e3_1e4"]] = 0.25
        sv_alpha[sv_map["DUP_1e4_1e5"]] = 0.25
        sv_prob = rng.dirichlet(sv_alpha)

        vec = np.concatenate(
            [
                rng.multinomial(tot_sbs, sbs_prob),
                rng.multinomial(tot_ind, ind_prob),
                rng.multinomial(tot_sv, sv_prob) if tot_sv > 0 else np.zeros(n_sv, dtype=np.int64),
            ]
        )
        rows.append(vec)
        labels.append("HRP")
        sample_names.append(f"syn_HRP_{k}")

    # --- Class 1: BRCA1-type HRD ---
    for k in range(n_per_class):
        tot_sbs = int(rng.integers(1_200, 20_000))
        tot_ind = int(rng.integers(35, 1_500))
        tot_sv = int(rng.integers(25, 450))

        sbs_prob = rng.dirichlet(60.0 * sbs3_p + 10.0 * sbs1_5_p + 0.2)

        ind_alpha = np.full(n_ind, 0.4, dtype=np.float64)
        ind_alpha[ind_map["del.mh.bimh.1"]] = 2.0
        ind_alpha[ind_map["del.mh.bimh.2"]] = 6.5
        ind_alpha[ind_map["del.mh.bimh.3"]] = 7.5
        ind_alpha[ind_map["del.mh.bimh.4"]] = 5.5
        ind_alpha[ind_map["del.mh.bimh.5"]] = 4.5
        ind_alpha[ind_map["del.rep.len.1"]] = 2.5
        ind_alpha[ind_map["ins.rep.len.1"]] = 1.5
        ind_alpha[ind_map["del.none"]] = 2.0
        ind_prob = rng.dirichlet(ind_alpha)

        sv_alpha = np.full(n_sv, 0.35, dtype=np.float64)
        # Signature RS3: 1-100 kb tandem duplications
        sv_alpha[sv_map["DUP_1e3_1e4"]] = 11.0
        sv_alpha[sv_map["DUP_1e4_1e5"]] = 13.0
        sv_alpha[sv_map["DUP_0_1e3"]] = 2.0
        sv_alpha[sv_map["DEL_1e3_1e4"]] = 1.8
        sv_alpha[sv_map["DEL_1e4_1e5"]] = 1.8
        sv_alpha[sv_map["TRA"]] = 2.5
        sv_prob = rng.dirichlet(sv_alpha)

        vec = np.concatenate(
            [
                rng.multinomial(tot_sbs, sbs_prob),
                rng.multinomial(tot_ind, ind_prob),
                rng.multinomial(tot_sv, sv_prob),
            ]
        )
        rows.append(vec)
        labels.append("BRCA1")
        sample_names.append(f"syn_BRCA1_{k}")

    # --- Class 2: BRCA2-type HRD ---
    for k in range(n_per_class):
        tot_sbs = int(rng.integers(1_200, 20_000))
        tot_ind = int(rng.integers(35, 1_800))
        # Include both moderate/high-SV and low-SV BRCA2 tumors
        tot_sv = int(rng.integers(5, 35) if (k % 5 == 0) else rng.integers(25, 400))

        sbs_prob = rng.dirichlet(60.0 * sbs3_p + 10.0 * sbs1_5_p + 0.2)

        ind_alpha = np.full(n_ind, 0.35, dtype=np.float64)
        ind_alpha[ind_map["del.mh.bimh.1"]] = 2.0
        ind_alpha[ind_map["del.mh.bimh.2"]] = 7.5
        ind_alpha[ind_map["del.mh.bimh.3"]] = 9.0
        ind_alpha[ind_map["del.mh.bimh.4"]] = 7.0
        ind_alpha[ind_map["del.mh.bimh.5"]] = 6.0
        ind_alpha[ind_map["del.rep.len.1"]] = 2.0
        ind_alpha[ind_map["del.none"]] = 1.8
        ind_prob = rng.dirichlet(ind_alpha)

        sv_alpha = np.full(n_sv, 0.35, dtype=np.float64)
        # Signature RS5: 1-100 kb deletions WITHOUT short tandem duplications
        sv_alpha[sv_map["DEL_0_1e3"]] = 3.5
        sv_alpha[sv_map["DEL_1e3_1e4"]] = 12.0
        sv_alpha[sv_map["DEL_1e4_1e5"]] = 11.0
        sv_alpha[sv_map["DEL_1e5_1e6"]] = 2.5
        sv_alpha[sv_map["TRA"]] = 3.0
        sv_alpha[sv_map["DUP_1e3_1e4"]] = 0.15
        sv_alpha[sv_map["DUP_1e4_1e5"]] = 0.15
        sv_prob = rng.dirichlet(sv_alpha)

        vec = np.concatenate(
            [
                rng.multinomial(tot_sbs, sbs_prob),
                rng.multinomial(tot_ind, ind_prob),
                rng.multinomial(tot_sv, sv_prob),
            ]
        )
        rows.append(vec)
        labels.append("BRCA2")
        sample_names.append(f"syn_BRCA2_{k}")

    df = pd.DataFrame(np.vstack(rows), index=sample_names, columns=list(ALL_CHORD_CHANNELS))
    df.index.name = "sample"
    return df, labels


def _normalize_training_label(lbl: Any) -> str:
    s = str(lbl).strip()
    up = s.upper()
    if up in ("0", "HRP", "HR_PROFICIENT", "PROFICIENT", "NONE"):
        return "HRP"
    if up in ("1", "BRCA1", "BRCA1_TYPE", "BRCA1-TYPE"):
        return "BRCA1"
    if up in ("2", "BRCA2", "BRCA2_TYPE", "BRCA2-TYPE"):
        return "BRCA2"
    raise ValueError(
        f"Unrecognized training label {lbl!r}. Expected one of 'HRP', 'BRCA1', 'BRCA2'."
    )


def _get_pretrained_rf(settings: SettingsMode) -> RandomForestClassifier:
    if settings not in _CACHED_ENSEMBLES:
        cfg = get_chord_config(settings)
        syn_df, syn_labels = simulate_chord_archetypes(n_per_class=160, seed=2020)
        x_train = compute_chord_features(syn_df, cfg)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=2020,
        )
        rf.fit(x_train, syn_labels)
        _CACHED_ENSEMBLES[settings] = rf
    return _CACHED_ENSEMBLES[settings]


class CHORDClassifier:
    """
    Random-Forest HRD and BRCA1-type vs. BRCA2-type classifier following CHORD
    (Nguyen et al., Nature Communications 2020).

    Ships pre-trained on pan-cancer mutational signature archetypes and supports
    `.fit(contexts_df, labels)` on custom cohorts.
    """

    def __init__(
        self,
        settings: SettingsMode = "default",
        *,
        n_estimators: int = 200,
        random_state: int = 2020,
    ) -> None:
        self.settings: SettingsMode = validate_settings(settings)
        self.config: CHORDConfig = get_chord_config(self.settings)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.rf_model_: RandomForestClassifier = _get_pretrained_rf(self.settings)
        self.is_custom_fitted_: bool = False

    def fit(
        self,
        contexts_df: pd.DataFrame,
        labels: list[str] | np.ndarray | pd.Series,
    ) -> CHORDClassifier:
        """Fit a custom RandomForestClassifier on user-supplied CHORD context counts."""
        aligned = align_contexts_df(contexts_df)
        y_norm = [_normalize_training_label(v) for v in labels]
        if len(y_norm) != len(aligned):
            raise ValueError(
                f"Length of `labels` ({len(y_norm)}) must match `contexts_df` ({len(aligned)})."
            )
        x_mat = compute_chord_features(aligned, self.config)
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features="sqrt",
            random_state=self.random_state,
        )
        rf.fit(x_mat, y_norm)
        self.rf_model_ = rf
        self.is_custom_fitted_ = True
        return self

    def predict_proba(self, contexts_df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns `p_hrp`, `p_BRCA1`, `p_BRCA2`, `p_hrd`."""
        aligned = align_contexts_df(contexts_df)
        x_mat = compute_chord_features(aligned, self.config)
        raw_probs = self.rf_model_.predict_proba(x_mat)
        classes = list(self.rf_model_.classes_)

        n = len(aligned)
        p_hrp = np.zeros(n, dtype=np.float64)
        p_b1 = np.zeros(n, dtype=np.float64)
        p_b2 = np.zeros(n, dtype=np.float64)

        for idx, cls_name in enumerate(classes):
            if cls_name == "HRP":
                p_hrp = raw_probs[:, idx]
            elif cls_name == "BRCA1":
                p_b1 = raw_probs[:, idx]
            elif cls_name == "BRCA2":
                p_b2 = raw_probs[:, idx]

        p_hrd = np.clip(p_b1 + p_b2, 0.0, 1.0)
        return pd.DataFrame(
            {
                "p_BRCA1": p_b1,
                "p_BRCA2": p_b2,
                "p_hrd": p_hrd,
                "p_hrp": p_hrp,
            },
            index=aligned.index,
        )

    def predict(
        self,
        contexts_df: pd.DataFrame,
        *,
        hrd_cutoff: float = 0.5,
        min_indel_load: int = 50,
        msi_cutoff: int = 14_000,
    ) -> CHORDResult:
        """Predict HRD status, BRCA1/2 subtype, and QC remarks."""
        return chord_predict(
            contexts_df,
            rf_model=self.rf_model_,
            hrd_cutoff=hrd_cutoff,
            min_indel_load=min_indel_load,
            msi_cutoff=msi_cutoff,
            settings=self.settings,
        )


def chord_predict(
    contexts_df: pd.DataFrame,
    rf_model: CHORDClassifier | RandomForestClassifier | None = None,
    hrd_cutoff: float = 0.5,
    min_indel_load: int = 50,
    msi_cutoff: int = 14_000,
    settings: SettingsMode = "default",
) -> CHORDResult:
    """
    Predict pan-cancer HRD probability and BRCA1-type vs. BRCA2-type status from
    extracted CHORD context counts (`extract_sigs_chord` output).
    """
    mode = validate_settings(settings)
    cfg = get_chord_config(mode)
    if not (0.0 < float(hrd_cutoff) < 1.0):
        raise ValueError(f"`hrd_cutoff` must be in (0, 1), got {hrd_cutoff}.")

    aligned = align_contexts_df(contexts_df)

    if isinstance(rf_model, CHORDClassifier):
        clf_rf = rf_model.rf_model_
        feat_cfg = rf_model.config
    elif isinstance(rf_model, RandomForestClassifier):
        clf_rf = rf_model
        feat_cfg = cfg
    elif rf_model is None:
        clf_rf = _get_pretrained_rf(mode)
        feat_cfg = cfg
    else:
        raise ValueError(
            "`rf_model` must be a `CHORDClassifier`, `RandomForestClassifier`, or `None`."
        )

    x_mat = compute_chord_features(aligned, feat_cfg)
    raw_probs = clf_rf.predict_proba(x_mat)
    classes = list(clf_rf.classes_)

    n = len(aligned)
    p_hrp = np.zeros(n, dtype=np.float64)
    p_b1 = np.zeros(n, dtype=np.float64)
    p_b2 = np.zeros(n, dtype=np.float64)
    for idx, cls_name in enumerate(classes):
        if cls_name == "HRP":
            p_hrp = raw_probs[:, idx]
        elif cls_name == "BRCA1":
            p_b1 = raw_probs[:, idx]
        elif cls_name == "BRCA2":
            p_b2 = raw_probs[:, idx]

    p_hrd = np.clip(p_b1 + p_b2, 0.0, 1.0)

    rep_cols = list(INDEL_REP_DEL_CHANNELS) + list(INDEL_REP_INS_CHANNELS)
    rep_indels = aligned[rep_cols].sum(axis=1).to_numpy(dtype=np.float64)
    total_indels = aligned[list(INDEL_CHANNELS)].sum(axis=1).to_numpy(dtype=np.float64)
    total_svs = aligned[list(SV_CHANNELS)].sum(axis=1).to_numpy(dtype=np.float64)

    hr_status_list: list[str] = []
    hrd_type_list: list[str] = []
    remarks_list: list[str] = []

    for i in range(n):
        if rep_indels[i] > float(msi_cutoff):
            hr_status_list.append("MSI")
            hrd_type_list.append("none")
            remarks_list.append("MSI_detected")
            continue

        low_indel = total_indels[i] < float(min_indel_load)
        if low_indel and cfg.strict_low_indel_mask:
            hr_status_list.append("cannot_be_determined")
            hrd_type_list.append("cannot_be_determined")
            remarks_list.append("low_indel_load")
            continue

        if p_hrd[i] >= float(hrd_cutoff):
            status = "HR_deficient"
            subtype = "BRCA1_type" if p_b1[i] >= p_b2[i] else "BRCA2_type"
        else:
            status = "HR_proficient"
            subtype = "none"

        remark_parts: list[str] = []
        if low_indel:
            remark_parts.append("low_indel_load")
        if status == "HR_deficient" and total_svs[i] < 30:
            remark_parts.append("low_sv_load")

        hr_status_list.append(status)
        hrd_type_list.append(subtype)
        remarks_list.append(";".join(remark_parts))

    pred_df = pd.DataFrame(
        {
            "sample": aligned.index.astype(str),
            "p_BRCA1": p_b1,
            "p_BRCA2": p_b2,
            "p_hrd": p_hrd,
            "p_hrp": p_hrp,
            "hr_status": hr_status_list,
            "hrd_type": hrd_type_list,
            "remarks": remarks_list,
        },
        index=aligned.index,
    )
    return CHORDResult(predictions=pred_df, settings=mode)
