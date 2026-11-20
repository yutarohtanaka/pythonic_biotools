"""Unified command-line interface (`pythonic-biotools` / `pbt`) for pythonic_biotools."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import io as bio_io
from .registry import (
    BUNDLES,
    PACKAGE_IMPORT_MAP,
    check_bundle_imports,
    install_bundle_editable,
    load_package,
    resolve_bundle_packages,
)


def _write_tsv(df: pd.DataFrame, out_path: str | Path, *, index: bool = False) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)


def _read_delimited(path: str | Path) -> pd.DataFrame:
    return bio_io._read_table_skipping_prefixes(path, skip_prefixes=("#",), sep=None)


def _load_variants_any(path: str | Path) -> pd.DataFrame:
    name_low = Path(path).name.lower()
    if name_low.endswith((".vcf", ".vcf.gz")):
        return bio_io.read_vcf_variants(path)
    if name_low.endswith((".maf", ".maf.gz")):
        return bio_io.read_maf(path)
    df = _read_delimited(path)
    if any(c in df.columns for c in ("Start_Position", "Reference_Allele", "Tumor_Seq_Allele2")):
        return bio_io.read_maf(df)
    return df


def _run_scarhrd(args: argparse.Namespace) -> int:
    scarhrd = load_package("scarhrd")
    seg_df = bio_io.read_cn_segments(args.segments)
    if "sample_id" in seg_df.columns and seg_df["sample_id"].nunique() > 1:
        res_df = scarhrd.calculate_hrd_cohort(
            seg_df, genome=args.genome, settings=args.settings
        )
    else:
        res = scarhrd.calculate_hrd(
            seg_df,
            genome=args.genome,
            ploidy=args.ploidy,
            sample_id=args.sample_id,
            settings=args.settings,
        )
        res_df = res.to_dataframe()
    _write_tsv(res_df, args.out)
    return 0


def _run_ichorcna(args: argparse.Namespace) -> int:
    ichorcna = load_package("ichorcna")
    raw = bio_io._read_table_skipping_prefixes(
        args.bins, skip_prefixes=("@", "#"), sep="\t"
    )
    if "reads" in raw.columns and "gc" in raw.columns and "logR" not in raw.columns:
        bins_df = raw
    else:
        bins_df = bio_io.read_gatk_copy_ratios(raw)

    res = ichorcna.run_ichorcna(
        bins_df,
        max_iter=args.max_iter,
        low_tf_mode=args.low_tf_mode,
        settings=args.settings,
    )
    _write_tsv(res.segments_df, args.out_segments)
    _write_tsv(pd.DataFrame([res.summary()]), args.out_summary)
    return 0


def _run_purecn(args: argparse.Namespace) -> int:
    purecn = load_package("purecn")
    raw_cov = bio_io._read_table_skipping_prefixes(
        args.coverage, skip_prefixes=("@", "#"), sep="\t"
    )
    low_cols = {str(c).lower() for c in raw_cov.columns}
    if "seg_mean" in low_cols and "coverage" not in low_cols:
        cov_df = raw_cov
    elif low_cols & {"coverage", "counts", "count", "depth", "dp", "reads", "tumor_cov"}:
        cov_df = raw_cov
    else:
        cov_df = bio_io.read_gatk_copy_ratios(raw_cov)

    var_df = _load_variants_any(args.variants)
    res = purecn.run_purecn(
        tumor_cov_df=cov_df,
        variants_df=var_df,
        callable_mb=args.callable_mb,
        settings=args.settings,
    )
    _write_tsv(res.variants, args.out_variants)
    summary_df = pd.DataFrame(
        [
            {
                "purity": res.purity,
                "ploidy": res.ploidy,
                "tmb": float(res.tmb),
                "cin": float(res.cin),
                "loh_fraction": res.loh_fraction,
                "n_segments": int(len(res.segments)),
                "n_variants": int(len(res.variants)),
                "settings": res.settings,
            }
        ]
    )
    _write_tsv(summary_df, args.out_summary)
    if args.out_segments:
        _write_tsv(res.segments, args.out_segments)
    return 0


def _run_mutationtimer(args: argparse.Namespace) -> int:
    mutationtimer = load_package("mutationtimer")
    var_df = _load_variants_any(args.variants)
    seg_df = bio_io.read_cn_segments(args.segments)
    res = mutationtimer.mutation_time(
        var_df,
        seg_df,
        purity=args.purity,
        gender=args.gender,
        n_boot=args.n_boot,
        settings=args.settings,
    )
    _write_tsv(res.variants_df, args.out_variants)
    _write_tsv(res.segments_df, args.out_segments)
    return 0


def _run_dss(args: argparse.Namespace) -> int:
    dss = load_package("dss")
    ctrl_paths = [p.strip() for p in args.ctrl.split(",") if p.strip()]
    trt_paths = [p.strip() for p in args.trt.split(",") if p.strip()]
    if not ctrl_paths or not trt_paths:
        raise ValueError("Both --ctrl and --trt must provide at least one replicate file.")

    ctrl_bs = bio_io.read_bismark_cov(ctrl_paths)
    trt_bs = bio_io.read_bismark_cov(trt_paths)
    dml_df = dss.dml_test(ctrl_bs.frames, trt_bs.frames, settings=args.settings)
    if args.p_threshold is not None:
        dmr_df = dss.call_dmr(
            dml_df, p_threshold=args.p_threshold, settings=args.settings
        )
    else:
        dmr_df = dss.call_dmr(dml_df, settings=args.settings)
    _write_tsv(dml_df, args.out_dml)
    _write_tsv(dmr_df, args.out_dmr)
    return 0


def _build_group_design(groups_str: str) -> tuple[np.ndarray, int]:
    tokens = [g.strip() for g in groups_str.split(",") if g.strip()]
    if len(tokens) < 3:
        raise ValueError("--groups must specify at least 3 sample group assignments.")
    uniq = list(dict.fromkeys(tokens))
    if set(uniq) == {"0", "1"}:
        uniq = ["0", "1"]
    if len(uniq) < 2:
        raise ValueError("--groups must contain at least 2 distinct groups.")
    n = len(tokens)
    design = np.ones((n, len(uniq)), dtype=np.float64)
    for k, lvl in enumerate(uniq[1:], start=1):
        design[:, k] = np.array([1.0 if g == lvl else 0.0 for g in tokens], dtype=np.float64)
    return design, n


def _run_deqms(args: argparse.Namespace) -> int:
    deqms = load_package("deqms")
    design, n_samples = _build_group_design(args.groups)

    raw = bio_io._read_table_skipping_prefixes(args.input, skip_prefixes=(), sep="\t")
    cols = list(raw.columns)
    is_mq_or_diann = args.counts_col is None and (
        any(
            c in cols
            for c in (
                "Majority protein IDs",
                "Protein.Group",
                "Protein.Ids",
                "Razor + unique peptides",
            )
        )
        or any(str(c).startswith(("LFQ intensity ", "Reporter intensity ")) for c in cols)
    )

    if is_mq_or_diann:
        abundance_df, count_vec = bio_io.read_maxquant_diann(raw)
    else:
        cnt_col = args.counts_col
        if cnt_col is None:
            for cand in ("count", "psm_count", "peptides", "Peptides", "N.Sequences"):
                if cand in raw.columns:
                    cnt_col = cand
                    break

        id_col: str | None = None
        for cand_id in ("protein", "gene", "id", "Protein", "Gene"):
            if cand_id in raw.columns and cand_id != cnt_col:
                id_col = cand_id
                break
        if id_col is None:
            non_num = [
                c
                for c in raw.columns
                if c != cnt_col and not pd.api.types.is_numeric_dtype(raw[c])
            ]
            if non_num:
                id_col = str(non_num[0])

        if cnt_col is None:
            if id_col is not None and raw[id_col].duplicated().any():
                swept = deqms.median_sweeping(
                    raw, group_col=id_col, settings=args.settings
                )
                abundance_df = swept.abundance
                count_vec = swept.count
            else:
                raise ValueError(
                    "Proteomics TSV requires either --counts-col, a 'count'/'peptides' column, "
                    "PSM-level duplicate protein rows, or MaxQuant/DIA-NN format."
                )
        else:
            if cnt_col not in raw.columns:
                raise ValueError(f"--counts-col {cnt_col!r} not found in {cols}.")
            exclude = {cnt_col} | ({id_col} if id_col is not None else set())
            sample_cols = [c for c in raw.columns if c not in exclude]
            idx = (
                pd.Index(raw[id_col].astype(str), name="protein")
                if id_col is not None
                else pd.Index([f"PROT_{i + 1}" for i in range(len(raw))], name="protein")
            )
            abundance_df = raw[sample_cols].apply(pd.to_numeric, errors="raise")
            abundance_df.index = idx
            count_vec = pd.Series(
                pd.to_numeric(raw[cnt_col], errors="raise").to_numpy(dtype=np.float64),
                index=idx,
                name="count",
            )

    if abundance_df.shape[1] != n_samples:
        raise ValueError(
            f"Sample column count ({abundance_df.shape[1]}) does not match --groups length ({n_samples})."
        )

    fit = deqms.spectra_count_ebayes(
        abundance_df,
        design=design,
        count=count_vec,
        settings=args.settings,
    )
    out_df = deqms.output_result(fit, coef=1, settings=args.settings)
    _write_tsv(out_df, args.out, index=False)
    return 0


def _run_coloc(args: argparse.Namespace) -> int:
    coloc = load_package("coloc")
    t1 = bio_io.read_gwas_sumstats(args.trait1).drop_duplicates(subset=["snp"])
    t2 = bio_io.read_gwas_sumstats(args.trait2).drop_duplicates(subset=["snp"])
    merged = t1.merge(t2, on="snp", suffixes=("_1", "_2"))
    if merged.empty:
        raise ValueError("No overlapping SNPs found between --trait1 and --trait2.")

    beta2 = merged["beta_2"].to_numpy(dtype=np.float64).copy()
    has_alleles = (
        (merged["ea_1"] != "")
        & (merged["nea_1"] != "")
        & (merged["ea_2"] != "")
        & (merged["nea_2"] != "")
    )
    flipped = (
        has_alleles
        & (merged["ea_2"] == merged["nea_1"])
        & (merged["nea_2"] == merged["ea_1"])
    ).to_numpy()
    beta2[flipped] = -beta2[flipped]

    res = coloc.coloc_abf(
        beta1=merged["beta_1"].to_numpy(dtype=np.float64),
        se1=merged["se_1"].to_numpy(dtype=np.float64),
        beta2=beta2,
        se2=merged["se_2"].to_numpy(dtype=np.float64),
        snp_ids=merged["snp"].tolist(),
        p1=args.p1,
        p2=args.p2,
        p12=args.p12,
        dataset_type1=args.type1,
        dataset_type2=args.type2,
        settings=args.settings,
    )
    _write_tsv(pd.DataFrame([res.summary_dict()]), args.out)
    if args.out_snps:
        _write_tsv(res.snp_table(), args.out_snps)
    return 0


def _run_cismr(args: argparse.Namespace) -> int:
    cismr = load_package("cismr")
    df = _read_delimited(args.input)
    col_bx = bio_io._find_col(df, ("bx", "beta_exp", "beta_x"))
    col_sx = bio_io._find_col(df, ("sx", "se_exp", "se_x"))
    col_by = bio_io._find_col(df, ("by", "beta_out", "beta_y"))
    col_sy = bio_io._find_col(df, ("sy", "se_out", "se_y"))
    if None in (col_bx, col_sx, col_by, col_sy):
        raise ValueError(
            f"cisMR locus table must contain columns bx, sx, by, sy; got {list(df.columns)}."
        )

    ld_res = bio_io.read_plink_ld(args.ld)
    ld_mat = ld_res[0] if isinstance(ld_res, tuple) else ld_res

    v_col = bio_io._find_col(df, ("v_gx",))
    sv_col = bio_io._find_col(df, ("se_vgx",))
    runner = cismr.CisMR(settings=args.settings)
    report = runner.fit(
        bx=pd.to_numeric(df[col_bx], errors="raise").to_numpy(dtype=np.float64),  # type: ignore[index]
        sx=pd.to_numeric(df[col_sx], errors="raise").to_numpy(dtype=np.float64),  # type: ignore[index]
        by=pd.to_numeric(df[col_by], errors="raise").to_numpy(dtype=np.float64),  # type: ignore[index]
        sy=pd.to_numeric(df[col_sy], errors="raise").to_numpy(dtype=np.float64),  # type: ignore[index]
        ld_R=ld_mat,
        nx=args.nx,
        ny=args.ny,
        v_gx=(
            pd.to_numeric(df[v_col], errors="raise").to_numpy(dtype=np.float64)
            if v_col is not None
            else None
        ),
        se_vgx=(
            pd.to_numeric(df[sv_col], errors="raise").to_numpy(dtype=np.float64)
            if sv_col is not None
            else None
        ),
    )
    _write_tsv(report.to_dataframe(), args.out)
    return 0


def _run_survrm2(args: argparse.Namespace) -> int:
    survrm2 = load_package("survrm2")
    df = _read_delimited(args.input)
    for col in (args.time_col, args.status_col, args.arm_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {list(df.columns)}.")

    time_arr = pd.to_numeric(df[args.time_col], errors="raise").to_numpy(dtype=np.float64)
    status_arr = pd.to_numeric(df[args.status_col], errors="raise").to_numpy(dtype=np.int64)

    arm_ser = df[args.arm_col]
    if pd.api.types.is_numeric_dtype(arm_ser) and set(pd.unique(arm_ser.dropna())).issubset({0, 1}):
        arm_arr = arm_ser.to_numpy(dtype=np.int64)
    else:
        uniq = sorted(pd.unique(arm_ser.dropna()), key=str)
        if len(uniq) != 2:
            raise ValueError(f"Arm column {args.arm_col!r} must have 2 distinct levels; got {uniq}.")
        arm_arr = np.where(arm_ser == uniq[1], 1, 0).astype(np.int64)

    res = survrm2.rmst2(
        time=time_arr,
        status=status_arr,
        arm=arm_arr,
        tau=args.tau,
        settings=args.settings,
    )
    out_df = res.to_dataframe()
    out_df.insert(0, "tau", res.tau)
    out_df["rmst_arm1"] = res.arm1.rmst
    out_df["rmst_arm0"] = res.arm0.rmst
    _write_tsv(out_df, args.out)
    return 0


def _run_buysetest(args: argparse.Namespace) -> int:
    buysetest = load_package("buysetest")
    df = _read_delimited(args.input)
    ep = buysetest.Endpoint(
        args.tte_col,
        kind="tte",
        status_col=args.status_col,
        operator=args.operator,
        threshold=args.threshold,
    )
    res = buysetest.buyse_test(
        df,
        treatment=args.arm_col,
        endpoints=[ep],
        scoring_rule=args.scoring_rule,
        settings=args.settings,
    )
    _write_tsv(res.to_dataframe(), args.out)
    return 0


def _add_settings_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--settings",
        default="default",
        choices=["default", "R_default"],
        help="Execution mode: enhanced 'default' or strict upstream 'R_default' parity.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for `pythonic-biotools` / `pbt`."""
    parser = argparse.ArgumentParser(
        prog="pythonic-biotools",
        description="Unified package manager and pipeline CLI for pythonic_biotools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List packages in a domain bundle.")
    p_list.add_argument(
        "--bundle",
        default="all",
        choices=sorted(BUNDLES),
        help="Domain bundle to inspect.",
    )

    p_check = sub.add_parser(
        "check-imports",
        help="Verify imports and PEP 561 py.typed markers across a bundle.",
    )
    p_check.add_argument(
        "--bundle",
        default="all",
        choices=sorted(BUNDLES),
        help="Domain bundle to verify.",
    )

    p_install = sub.add_parser(
        "install",
        help="Editable-install all standalone packages in a bundle.",
    )
    p_install.add_argument(
        "bundle",
        choices=sorted(BUNDLES),
        help="Domain bundle to install.",
    )
    p_install.add_argument(
        "--no-deps",
        action="store_true",
        help="Pass --no-deps to pip when scientific stack is already installed.",
    )

    p_run = sub.add_parser(
        "run",
        help="Execute a file-in / TSV-out bioinformatics tool pipeline.",
    )
    run_sub = p_run.add_subparsers(dest="tool", required=True)

    # 1. scarhrd
    p_scar = run_sub.add_parser("scarhrd", help="Genomic scar HRD scoring (LOH + TAI + LST).")
    p_scar.add_argument("--segments", required=True, help="Allele-specific CN segments TSV/.cns.")
    p_scar.add_argument("--genome", default="GRCh38", help="Reference genome (GRCh38 or GRCh37).")
    p_scar.add_argument("--ploidy", type=float, default=None, help="Optional explicit tumor ploidy.")
    p_scar.add_argument("--sample-id", default="Sample_1", help="Sample identifier.")
    p_scar.add_argument("--out", required=True, help="Output HRD summary TSV path.")
    _add_settings_arg(p_scar)
    p_scar.set_defaults(_handler=_run_scarhrd)

    # 2. ichorcna
    p_ichor = run_sub.add_parser("ichorcna", help="ULP-WGS cfDNA tumor fraction & CNA HMM.")
    p_ichor.add_argument("--bins", required=True, help="Genomic bin copy-ratio or read-count TSV.")
    p_ichor.add_argument("--max-iter", type=int, default=50, help="Maximum EM iterations.")
    p_ichor.add_argument("--low-tf-mode", action="store_true", help="Enable low tumor-fraction mode.")
    p_ichor.add_argument("--out-segments", required=True, help="Output CNA segments TSV path.")
    p_ichor.add_argument("--out-summary", required=True, help="Output tumor-fraction summary TSV path.")
    _add_settings_arg(p_ichor)
    p_ichor.set_defaults(_handler=_run_ichorcna)

    # 3. purecn
    p_pure = run_sub.add_parser("purecn", help="Tumor-only purity/ploidy, somatic status, CCF & TMB.")
    p_pure.add_argument("--coverage", required=True, help="Tumor coverage intervals or copy-ratio TSV.")
    p_pure.add_argument("--variants", required=True, help="Variants file (.vcf/.vcf.gz, .maf, or TSV).")
    p_pure.add_argument("--callable-mb", type=float, default=None, help="Callable footprint in Mb.")
    p_pure.add_argument("--out-variants", required=True, help="Output classified variants TSV path.")
    p_pure.add_argument("--out-summary", required=True, help="Output purity/ploidy/TMB summary TSV path.")
    p_pure.add_argument("--out-segments", default=None, help="Optional output allele-specific segments TSV.")
    _add_settings_arg(p_pure)
    p_pure.set_defaults(_handler=_run_purecn)

    # 4. mutationtimer
    p_mt = run_sub.add_parser("mutationtimer", help="Evolutionary timing of somatic SNVs, gains & WGD.")
    p_mt.add_argument("--variants", required=True, help="Somatic variants (.vcf/.vcf.gz, .maf, or TSV).")
    p_mt.add_argument("--segments", required=True, help="Allele-specific CN segments TSV/.cns.")
    p_mt.add_argument("--purity", type=float, required=True, help="Tumor purity in (0, 1].")
    p_mt.add_argument("--gender", default="female", choices=["female", "male"], help="Sample sex.")
    p_mt.add_argument("--n-boot", type=int, default=200, help="Bootstrap iterations per gained segment.")
    p_mt.add_argument("--out-variants", required=True, help="Output timed variants TSV path.")
    p_mt.add_argument("--out-segments", required=True, help="Output timed CN segments TSV path.")
    _add_settings_arg(p_mt)
    p_mt.set_defaults(_handler=_run_mutationtimer)

    # 5. dss
    p_dss = run_sub.add_parser("dss", help="Beta-Binomial differential DNA methylation (DML & DMR).")
    p_dss.add_argument("--ctrl", required=True, help="Comma-separated control Bismark/bedGraph/TSV files.")
    p_dss.add_argument("--trt", required=True, help="Comma-separated treatment Bismark/bedGraph/TSV files.")
    p_dss.add_argument("--p-threshold", type=float, default=None, help="Optional DMR p-value threshold.")
    p_dss.add_argument("--out-dml", required=True, help="Output per-CpG DML statistics TSV path.")
    p_dss.add_argument("--out-dmr", required=True, help="Output called DMR regions TSV path.")
    _add_settings_arg(p_dss)
    p_dss.set_defaults(_handler=_run_dss)

    # 6. deqms
    p_deq = run_sub.add_parser("deqms", help="Peptide-count-moderated differential proteomics.")
    p_deq.add_argument("--input", required=True, help="Proteomics TSV, MaxQuant proteinGroups.txt, or DIA-NN pg_matrix.tsv.")
    p_deq.add_argument("--groups", required=True, help="Comma-separated group labels (e.g. 0,0,0,1,1,1).")
    p_deq.add_argument("--counts-col", default=None, help="Optional peptide/PSM count column name.")
    p_deq.add_argument("--out", required=True, help="Output differential abundance TSV path.")
    _add_settings_arg(p_deq)
    p_deq.set_defaults(_handler=_run_deqms)

    # 7. coloc
    p_coloc = run_sub.add_parser("coloc", help="Wakefield Approximate Bayes Factor colocalization.")
    p_coloc.add_argument("--trait1", required=True, help="Trait 1 GWAS/QTL summary statistics TSV.")
    p_coloc.add_argument("--trait2", required=True, help="Trait 2 GWAS/QTL summary statistics TSV.")
    p_coloc.add_argument("--p1", type=float, default=1e-4, help="Prior probability SNP associated with trait 1.")
    p_coloc.add_argument("--p2", type=float, default=1e-4, help="Prior probability SNP associated with trait 2.")
    p_coloc.add_argument("--p12", type=float, default=1e-5, help="Prior probability SNP shared causal.")
    p_coloc.add_argument("--type1", default="quant", choices=["quant", "cc"], help="Trait 1 study design.")
    p_coloc.add_argument("--type2", default="quant", choices=["quant", "cc"], help="Trait 2 study design.")
    p_coloc.add_argument("--out", required=True, help="Output colocalization posterior summary TSV path.")
    p_coloc.add_argument("--out-snps", default=None, help="Optional per-SNP SNP.PP.H4 table TSV path.")
    _add_settings_arg(p_coloc)
    p_coloc.set_defaults(_handler=_run_coloc)

    # 8. cismr
    p_cismr = run_sub.add_parser("cismr", help="Correlated cis-Mendelian Randomization sensitivity suite.")
    p_cismr.add_argument("--input", required=True, help="Locus TSV containing bx, sx, by, sy columns.")
    p_cismr.add_argument("--ld", required=True, help="Square LD correlation matrix TSV / .ld / .ld.gz.")
    p_cismr.add_argument("--nx", type=float, required=True, help="Exposure GWAS/QTL sample size.")
    p_cismr.add_argument("--ny", type=float, required=True, help="Outcome GWAS sample size.")
    p_cismr.add_argument("--out", required=True, help="Output cis-MR sensitivity summary TSV path.")
    _add_settings_arg(p_cismr)
    p_cismr.set_defaults(_handler=_run_cismr)

    # 9. survrm2
    p_rmst = run_sub.add_parser("survrm2", help="Two-arm Restricted Mean Survival Time (RMST/RMTL).")
    p_rmst.add_argument("--input", required=True, help="Survival clinical dataset TSV/CSV.")
    p_rmst.add_argument("--time-col", default="time", help="Follow-up time column name.")
    p_rmst.add_argument("--status-col", default="status", help="Event indicator column name (0/1).")
    p_rmst.add_argument("--arm-col", default="arm", help="Treatment arm column name (0/1 or 2 levels).")
    p_rmst.add_argument("--tau", type=float, default=None, help="Optional truncation time horizon tau.")
    p_rmst.add_argument("--out", required=True, help="Output RMST contrast table TSV path.")
    _add_settings_arg(p_rmst)
    p_rmst.set_defaults(_handler=_run_survrm2)

    # 10. buysetest
    p_buyse = run_sub.add_parser("buysetest", help="Generalized Pairwise Comparisons (Win Ratio & Net Benefit).")
    p_buyse.add_argument("--input", required=True, help="Clinical trial dataset TSV/CSV.")
    p_buyse.add_argument("--arm-col", default="arm", help="Treatment arm column name.")
    p_buyse.add_argument("--tte-col", required=True, help="Time-to-event endpoint column name.")
    p_buyse.add_argument("--status-col", required=True, help="Event status column name (1=event, 0=censored).")
    p_buyse.add_argument("--threshold", type=float, default=0.0, help="Clinically meaningful TTE threshold.")
    p_buyse.add_argument("--operator", default="higher", choices=["higher", "lower"], help="Favorable direction.")
    p_buyse.add_argument("--scoring-rule", default="Peron", choices=["Peron", "Gehan"], help="Pairwise TTE scoring rule.")
    p_buyse.add_argument("--out", required=True, help="Output Win Ratio / Net Benefit TSV path.")
    _add_settings_arg(p_buyse)
    p_buyse.set_defaults(_handler=_run_buysetest)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `pythonic-biotools` and `pbt` console scripts."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        pkgs = resolve_bundle_packages(args.bundle)
        print(f"Bundle '{args.bundle}' ({len(pkgs)} packages):")
        for p in pkgs:
            print(f"  {p:20s} -> import {PACKAGE_IMPORT_MAP[p]}")
        return 0

    if args.command == "check-imports":
        status = check_bundle_imports(bundle=args.bundle)
        bad = [k for k, ok in status.items() if not ok]
        print(f"Verified {len(status) - len(bad)}/{len(status)} packages in '{args.bundle}'.")
        if bad:
            print("Failed py.typed or import check:", ", ".join(bad))
            return 1
        return 0

    if args.command == "install":
        return install_bundle_editable(bundle=args.bundle, no_deps=args.no_deps)

    if args.command == "run":
        handler = getattr(args, "_handler", None)
        if handler is None:
            parser.error(f"Unknown tool {getattr(args, 'tool', None)!r}")
        return int(handler(args))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
