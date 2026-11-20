# R-to-Python Rosetta Stone Migration Guide

This guide translates idioms, data structures, and function calls from published R, Bioconductor, CRAN, C++, and Fortran packages directly into `pythonic_biotools`—eliminating `rpy2` bridges, fragile Conda R environments, and `GRanges`/`S4` serialization overhead.

---

## Core Data-Structure Translation (`R / Bioconductor` $\rightarrow$ `Python`)

| R / Bioconductor Object | `pythonic_biotools` Equivalent | Notes |
|---|---|---|
| `GenomicRanges::GRanges` | `pd.DataFrame` (`chrom`, `start`, `end`, ...) | Accepts both `"chr1"` and `"1"` chromosome naming conventions |
| `SummarizedExperiment` / `DGEList` | `pd.DataFrame` or `np.ndarray` (`features x samples`) | Gene/protein IDs in `.index`, sample IDs in `.columns` |
| `SingleCellExperiment` / `SpatialRNA` | `anndata.AnnData` (`cells/spots x genes`) | Metadata in `adata.obs`, spatial weights/coords in `adata.obsm` |
| `bsseq::BSseq` | `list[pd.DataFrame]` (`chr`, `pos`, `N`, `X`) or `dss.BSData` | Loaded directly from Bismark `.cov.gz` via `pythonic_biotools.io.read_bismark_cov` |
| `VariantAnnotation::VRanges` / MAF | `pd.DataFrame` (`chrom`, `pos`, `alt_count`, `total_depth`) | Loaded from VCF/MAF via `read_vcf_variants` / `read_maf` |
| `S4` / `list` return objects (`fit$...`, `slot(x, ...)`) | `@dataclass` result containers (`.to_dataframe()`, `.to_dict()`) | Fully PEP 561 typed for IDE autocomplete and `mypy`/`pyright` |

---

## When to Pass `settings="default"` vs. `settings="R_default"`

Every estimation function in `pythonic_biotools` accepts `settings: Literal["default", "R_default"] = "default"`:

- **Use `settings="default"` for new analyses and production pipelines**:
  Enables numerically stabilized log-space arithmetic (`logsumexp`), positive-semidefinite eigenvalue floors on out-of-sample LD matrices, coding-invariant SNP orientation, outlier-robust Empirical Bayes priors, and unfolded bootstrap standard errors near $\beta = 0$.
- **Use `settings="R_default"` when migrating legacy R pipelines or auditing exact numerical parity**:
  Activates the exact upstream R/C++/Fortran thresholds, tie-breaking conventions, spline/quadrature grids, degrees-of-freedom rules, and output column schemas (`98.1%` mean parity across all 70 packages; see [Upstream Parity & Scorecard](UPSTREAM_PARITY_AND_SCORECARD.md)).

---

## 1. Cancer Genomics, Liquid Biopsy & Signatures

### Function Translation Table

| Upstream R / CLI Tool | Upstream R Call | Python Import & Call | Key `R_default` Parity Behavior |
|---|---|---|---|
| **FACETS** (`mskcc/facets`) | `preProcSample() |> procSample() |> emcncf()` | `from facets import run_facets, run_ascat`<br>`run_facets(pileup_df, settings=...)` | `cbs2rank` uniform-rank bivariate CBS + `emcncf` surrogate weights |
| **PureCN** (`Bioc::PureCN`) | `runAbsoluteCN(normal.coverage.file, tumor.coverage.file, vcf.file)` | `from purecn import run_purecn`<br>`run_purecn(tumor_cov_df, variants_df, settings=...)` | `[0.15, 0.95]` purity grid, dbSNP/COSMIC somatic priors, `posterior >= 0.8` TMB gate |
| **QDNAseq + ACE** (`Bioc::QDNAseq`, `ACE`) | `correctBins() |> normalizeBins() |> segmentBins()`; `ACE::squaremodel()` | `from qdnaseq import QDNAseqPipeline, run_ace`<br>`QDNAseqPipeline(settings=...).run(bins_df, counts_df=...)` | `minMapq=37`, 3-pass 2D `(GC, mappability)` loess, unpenalized `[1.0, 5.0]` ACE ploidy grid |
| **ichorCNA** (`broadinstitute/ichorCNA`) | `runIchorCNA.R --WIG tumor.wig --ploidy "c(2,3)"` | `from ichorcna import run_ichorcna`<br>`run_ichorcna(bins_df, low_tf_mode=..., settings=...)` | `normal=(0.5..0.9)`, `txnE=0.9999999`, `txnStrength=10000`, subclonal genome filter |
| **scarHRD** (`sztup/scarHRD`) | `scar_score(seg, reference = "grch38", seqz = FALSE)` | `from scarhrd import calculate_hrd`<br>`calculate_hrd(cn_segments, genome="GRCh38", settings=...)` | Autosomes `chr1-22` only, `<3 Mb` left-to-right LST popping, `lst - 15.5 * ploidy` |
| **CHORD** (`UMCUGenetics/CHORD`) | `extractSigsChord()` + `chordPredict()` | `from chord import extract_sigs_chord, chord_predict`<br>`chord_predict(contexts_df, settings=...)` | Within-class relative normalization, `msi.cutoff=14000`, `indel.load.min=50` |
| **ShatterSeek** (`parklab/ShatterSeek`) | `shatterseek(SV.sample, CNV.sample, genome = "hg38")` | `from shatterseek import detect_chromothripsis`<br>`detect_chromothripsis(sv_df, cn_df, genome="GRCh38", settings=...)` | Single largest SV cluster per chr, `+-10 kb` TRA join linking, 3-state CN oscillation |
| **MutationTimeR** (`gerstung-lab/MutationTimeR`) | `mutationTime(vcf, bb, clusters, purity)` | `from mutationtimer import mutation_time, estimate_wgd_timing`<br>`mutation_time(variants_df, cn_segments_df, purity=...)` | Weak `1/K` Dirichlet prior, `p_clonal >= 0.5` / `p_early >= 0.5`, exact `piToTime` equations |
| **TrackSig** (`morrislab/TrackSig`) | `TrackSig(vcfFile, activeInSample, referenceSignatures, purity)` | `from tracksig import tracksig`<br>`tracksig(variants_df, sig_ref, purity=..., bin_size=100, settings=...)` | Strict `binSize=100` SNVs/bin, `activeInSample >= 0.05`, $\log T$ PELT BIC penalty |
| **mmsig** (`UM-Myeloma-Genomics/mmsig`) | `mm_fit_signatures(muts.input, sig.input, bootstrap = TRUE)` | `from mmsig import mm_fit_signatures`<br>`mm_fit_signatures(catalog_df, sig_ref_df, settings=...)` | Simplex least squares, `cos_sim_threshold=0.01`, `SBS1`/`SBS5` forcing, `n_boot=1000` |
| **CONIPHER** (`McGranahanLab/CONIPHER`) | `conipher_clustering()` + `conipher_treebuilding()` | `from conipher import run_conipher`<br>`run_conipher(mutations_df, purity_by_region=..., settings=...)` | `ccf_buffer=0.10` Sum/Crossing rules, truncal collapse at `CCF >= 0.85`, `min_cluster_size=5` |
| **dNdScv** (`im3sanger/dndscv`) | `dndscv(mutations, refdb = "hg38", cv = covs)` | `from dndscv import dndscv`<br>`dndscv(mutations_df, refcds, covariates=..., settings=...)` | `MASS::glm.nb` (`maxit=25`) alternating IRLS + Newton `theta.ml`, exact `sel_cv` ordering |
| **Numbat** (`kharchenkolab/numbat`) | `run_numbat(count_mat, lambdas_ref, df_allele)` | `from numbat import run_numbat`<br>`run_numbat(adata, allele_df=..., ref_expression=..., settings=...)` | 15-state `S15_STATES` joint HMM (with `bdel`), continuous $(\hat\phi,\hat\theta)$ MLE, `retest_bulks` |
| **DISCOVER** (`NKI-CCB/DISCOVER`) | `discover.matrix(events)` + `pairwise.discover.test()` | `from discover import fit_discover_background, pairwise_discover_test`<br>`pairwise_discover_test(events_df, settings=...)` | Inclusive `mid_p=False` Poisson-Binomial tails + exact `DBH` step-CDF discrete FDR |
| **fishHook** (`mskilab/fishHook`) | `Fish(hypotheses = tiles, events = muts, eligible = callable, covariates = ...)` | `from fishhook import fit_fishhook`<br>`fit_fishhook(regions_df, mutations_df, eligible_intervals=..., settings=...)` | Per-sample `idcap=1`, uniform randomized discrete p-values, all-point QQ slope |
| **XenofilteR** (`PeeperLab/XenofilteR`) | `XenofilteR(sample.list, destination.folder, MM_threshold = 4)` | `from xenofilterR import Xenofilter, run_xenofilter`<br>`run_xenofilter(graft_bam, host_bam, out_bam, settings=...)` | `.Count_MM = S + I + NM` (excluding `H`), `unmapped_penalty=8`, strict `< 4` per-mate gate |

### Side-by-Side Workflow: HRD Scar Scoring + Evolutionary Timing + Signature Trajectories

**Upstream R (`scarHRD` + `MutationTimeR` + `TrackSig` + `mmsig`)**
```r
library(scarHRD)
library(MutationTimeR)
library(TrackSig)
library(mmsig)

# 1. Genomic scar HRD score
hrd <- scar_score("tumor_segments.txt", reference = "grch38", seqz = FALSE)

# 2. Somatic SNV & WGD evolutionary timing
mt <- mutationTime(vcf, bb = cn_granges, purity = 0.80, n.boot = 200)
vcf <- addMutTime(vcf, mt$V)

# 3. CCF pseudotime signature trajectory & bootstrap signature refitting
traj <- TrackSig(vcfFile = "timed.vcf", purity = 0.80, binSize = 100,
                 referenceSignatures = cosmic_sigs)
mm <- mm_fit_signatures(muts.input = mut_96_mat, sig.input = cosmic_sigs,
                        cos_sim_threshold = 0.01, bootstrap = TRUE)
```

**Python (`pythonic_biotools`)**
```python
from scarhrd import calculate_hrd
from mutationtimer import mutation_time, estimate_wgd_timing
from tracksig import tracksig
from mmsig import mm_fit_signatures

# 1. Genomic scar HRD score (pass settings="R_default" for exact sztup/scarHRD rules)
hrd = calculate_hrd(cn_segments, genome="GRCh38", settings="R_default")
print(hrd.hrd_loh, hrd.tai, hrd.lst, hrd.hrd_sum, hrd.hrd_positive)

# 2. Somatic SNV & WGD evolutionary timing
mt = mutation_time(variants_df, cn_segments, purity=0.80, n_boot=200, settings="R_default")
wgd = estimate_wgd_timing(mt.segments_df, settings="R_default")
print(mt.variants_df[["chrom", "pos", "timing_class"]].head(), wgd.wgd_time)

# 3. CCF pseudotime signature trajectory & bootstrap signature refitting
traj = tracksig(variants_df, sig_ref_df, purity=0.80, bin_size=100, settings="R_default")
mm = mm_fit_signatures(catalog_df, sig_ref_df, cos_sim_threshold=0.01, settings="R_default")
print(traj.changepoints, mm.exposures)
```

---

## 2. Mendelian Randomization, Colocalization, Causal TWAS & Multivariate Genetics

### Function Translation Table

| Upstream R / CLI Tool | Upstream R Call | Python Import & Call | Key `R_default` Parity Behavior |
|---|---|---|---|
| **TwoSampleMR / MVMR** (`MRCIEU/TwoSampleMR`) | `harmonise_data()`; `mr(dat)`; `MVMR::ivw_mvmr()` | `from twosamplemr import harmonise_mr_data, run_mr_suite, multivariable_mr` | 2nd-order delta weights + `n_boot=1000` in weighted median; Student-$t_{K-P}$ in MVMR |
| **PC-GMM & MR-SPI** (`MendelianRandomization::mr_pcgmm`, `MR-SPI`) | `mr_pcgmm(mr_input, nx, ny, r = ld_R)`; `MR_SPI(betaX, betaY, ...)` | `from cismr import mr_pcgmm, mr_mvpcgmm, mr_spi, mr_genius_summary` | Unprojected `Phi0` PCA (`thres=0.99`), 2-step GMM with $t_{K-P}$, `sqrt(2*log(p))` voting |
| **MR-PRESSO** (`rondolab/MR-PRESSO`) | `mr_presso(BetaOutcome, BetaExposure, SdOutcome, SdExposure, data)` | `from mrpresso import mr_presso, mvmr_presso`<br>`mr_presso(by, bx, sy, sx, settings=...)` | `1/se_out^2` weighted LOO RSS, `nb_distribution=1000`, raw `count/K` empirical p-values |
| **mr.raps** (`qingyuanzhao/mr.raps`) | `mr.raps(b_exp, b_out, se_exp, se_out, over.dispersion = TRUE, loss.function = "huber")` | `from mrraps import mr_raps`<br>`mr_raps(bx, by, sx, sy, over_dispersion=True, loss_function="huber", settings=...)` | Enforces upstream `10 * median(se_out)` bound check on `tau2` roots |
| **MR-APSS** (`YangLabHKUST/MR-APSS`) | `MRAPSS(paras)` | `from mrapss import fit_mrapss`<br>`fit_mrapss(bx, sx, by, sy, ldscore=..., settings=...)` | 3-step Isserlis IRWLS bivariate LDSC, unregularized EM + L-BFGS-B, `pval = p_lrt` |
| **MRBEE** (`noahlorinczcomi/MRBEE`) | `MRBEE.IMRP(by, bX, byse, bXse, Rxy)` | `from mrbee import fit_mrbee, fit_mvmr_bee`<br>`fit_mrbee(bx, sx, by, sy, settings=...)` | Fixed $1/s_y^2$ pre-standardization, Huber RLM + `3*MAD` init, empirical IMRP residual variance |
| **CAUSE** (`jean997/cause`) | `est_cause_params(X, variants)` + `cause(X, param_ests, top_vars)` | `from cause import est_cause_params, cause`<br>`cause(data=df, param_ests=params, settings=...)` | Tail `2*max, 4*max` SD grid, $\operatorname{arctanh}(\rho)$ prior, `n_samples=1000` PSIS-LOO |
| **LHC-MR** (`LizaDarrous/lhcMR`) | `calculate_SP_ldsc()` + `lhc_mr()` | `from lhcmr import lhc_mr`<br>`lhc_mr(bx, sx, by, sy, ld_scores, nx=..., ny=..., settings=...)` | `z^2 > max(80, 0.001*N)` outlier filter, fixed Stage-1 LDSC intercepts, reflection swap check |
| **MRlap** (`n-mounier/MRlap`) | `MRlap(exposure = exp_df, outcome = out_df, ld = ...)` | `from mrlap import mrlap`<br>`mrlap(exposure_df, outcome_df, ld_scores=..., settings=...)` | Closed-form truncated bivariate Gaussian moments, `n_sim=1000` architecture resampling |
| **GRAPPLE** (`jingshuw/GRAPPLE`) | `grappleRobustEst(data, loss.function = "tukey", diagnosis = TRUE)` | `from grapple import grapple_robust_est, grapple_mvmr`<br>`grapple_robust_est(bx, sx, by, sy, settings=...)` | Tukey `k=3.0`, `supp_drop >= 1.0` multimodal profile-likelihood peak rule |
| **coloc** (`chr1swallace/coloc`) | `coloc.abf(dataset1, dataset2, p12 = 1e-5)` | `from coloc import coloc_abf`<br>`coloc_abf(beta1, se1, beta2, se2, p12=1e-5, settings=...)` | `prior_w=0.20` for `"cc"` datasets and automatic `adjust_prior` when $M p \ge 1$ |
| **HyPrColoc** (`jrs95/hyprcoloc`) | `hyprcoloc(betas, ses, trait.names = ..., snp.id = ...)` | `from hyprcoloc import hyprcoloc`<br>`hyprcoloc(betas, ses, trait_names=..., snp_ids=..., settings=...)` | Exact `rapid.hyprcoloc` regional $P_R$, 1-step $m-1$ subsets, `0.7` uniform prior gates |
| **colocPropTest** (`chr1swallace/colocPropTest`) | `run_proptest(d1, d2, LD = R)` | `from colocproptest import run_coloc_prop_test`<br>`run_coloc_prop_test(trait1, trait2, LD=ld_R, settings=...)` | Top-15 marginal $|Z|$ testers (`r^2 < 0.99`), BH-FDR pair test + `thr=0.80` LD-PC omnibus |
| **SMR-HEIDI** (`JianYang-Lab/smr`) | `smr --bfile ref --gwas-summary gwas.ma --beqtl-summary eqtl` | `from smr_heidi import smr_heidi_locus, smr_single, heidi_test`<br>`smr_heidi_locus(gwas_df, eqtl_df, ld_matrix, settings=...)` | 4th-order `est_cov_bxy` correction, degree-based LD pruning, Satterthwaite/Kuonen `pchisqsum` |
| **cTWAS** (`xinhe-lab/ctwas`) | `impute_expr_z()` + `ctwas_rss()` | `from ctwas import impute_expr_z, run_ctwas, susie_rss_group_prior`<br>`run_ctwas([imp_region], n_samples=N, settings=...)` | Stage-1 `thin=0.1` SNP subsampling with `1/thin` M-step scaling + `rerun_gene_PIP=0.8` |
| **HDL** (`zhenin/HDL`) | `HDL.rg(gwas1.df, gwas2.df, LD.path)` / `HDL.L()` | `from hdl import build_ld_eigen_panel, hdl_h2, hdl_rg, hdl_local`<br>`hdl_rg(z1, z2, panel, N1=..., N2=..., settings=...)` | `eigen.cut=0.99`, fixed univariate `b1=b2=1.0` intercepts, block jackknife SEs |
| **LAVA** (`josefin-werme/LAVA`) | `process.locus()` + `run.univ()` + `run.bivar()` + `run.pcor()` | `from lava import process_locus, run_univ, run_bivar, run_pcor, run_multireg` | `lim=0.99` LD PCA pruning, `univ_thresh=0.05` bivariate gate, adaptive `1e4->1e5` MC tails |
| **GenomicSEM** (`GenomicSEM/GenomicSEM`) | `ldsc()` + `commonfactor(covstruc)` / `usermodel(covstruc, model)` + `commonfactorGWAS()` | `from genomicsem import ldsc_covstruct, commonfactor, usermodel, commonfactor_gwas` | Unconstrained RAM variances, `diag(diag(S))` independence CFI, $Q_{\text{SNP}} \sim \chi^2_{k-1}$ |
| **fgwas** (`joepickrell/fgwas`) | `fgwas -i sumstats.gz -w enhancer -fine` | `from fgwas import fit_fgwas`<br>`fit_fgwas(df, annotations=["enhancer"], block_col="SEGNUMBER", fine_map=True, settings=...)` | `V=(0.1,)` in `-fine` mode, `K=5000` SNPs/window, exact `max_llk - 2.0` profile CI |
| **GCTA-mtCOJO** (`jianyangqt/gcta`) | `gcta64 --mtcojo-file traits.list --ref-ld-chr eur_w_ld_chr` | `from mtcojo import mtcojo, gsmr_meta`<br>`mtcojo(df, G=G_ref, ldsc=ldsc_dict, settings=...)` | Exact-string `A1`/`A2` allele matching and pairwise allele-frequency QC (`freq_thresh=0.2`) |
| **SlopeHunter** (`Osmahmoud/SlopeHunter`) | `hunt(dat, xbeta_col, xse_col, ybeta_col, yse_col)` | `from slopehunter import hunt`<br>`hunt(xbeta, xse, ybeta, yse, settings=...)` | Exact folded bootstrap SD `std(abs(bts), ddof=1)` matching upstream `hunt.R` |

### Side-by-Side Workflow: Drug-Target cis-MR + Causal TWAS + GenomicSEM

**Upstream R (`MendelianRandomization` + `colocPropTest` + `ctwas` + `GenomicSEM`)**
```r
library(MendelianRandomization)
library(colocPropTest)
library(ctwas)
library(GenomicSEM)

# 1. Correlated cis-MR + proportional colocalization
pcgmm <- mr_pcgmm(mr_input(bx = bx, bxse = sx, by = by, byse = sy, correlation = ld_R),
                  nx = 50000, ny = 50000, thres = 0.99)
prop  <- run_proptest(d1, d2, LD = ld_R)

# 2. Causal TWAS separating causal gene expression from non-coding LD hitchhikers
z_gene <- impute_expr_z(z_snp, weight = weights_db, ld_dirname = ld_dir)
ct_res <- ctwas_rss(z_gene, z_snp, ld_exprfs, ld_R_dir, n = 50000)

# 3. Multivariate structural equation model over LDSC genetic covariance
sem_fit <- usermodel(LDSCoutput, estimation = "DWLS", model = "F1 =~ T1 + T2 + T3")
```

**Python (`pythonic_biotools`)**
```python
from cismr import mr_pcgmm, mr_spi
from colocproptest import run_coloc_prop_test
from ctwas import impute_expr_z, run_ctwas
from genomicsem import GenomicCovStruct, usermodel

# 1. Correlated cis-MR (PC-GMM + MR-SPI) & proportional colocalization
pcgmm = mr_pcgmm(bx, sx, by, sy, ld_R=ld_R, nx=50_000, ny=50_000, settings="R_default")
spi = mr_spi(bx, sx, by, sy, ld_R=ld_R, n=50_000, settings="R_default")
prop = run_coloc_prop_test({"beta": bx, "se": sx}, {"beta": by, "se": sy}, LD=ld_R, settings="R_default")

# 2. Causal TWAS (Empirical Bayes group priors + joint gene/SNP SuSiE-RSS)
imp = impute_expr_z(gwas_z_df, ld_R, weights_list, settings="R_default")
ct_res = run_ctwas([imp], n_samples=50_000, settings="R_default")
print(ct_res.gene_df[["gene_id", "z", "susie_pip", "in_cs"]])

# 3. GenomicSEM Reticular Action Model in lavaan syntax
cov_struct = GenomicCovStruct(S=S_ldsc, V_S=V_ldsc, trait_names=["T1", "T2", "T3"])
sem_fit = usermodel(cov_struct, "F1 =~ T1 + T2 + T3", estimation="DWLS", settings="R_default")
print(sem_fit.modelfit, sem_fit.results_df)
```

---

## 3. Bulk & Spatial Deconvolution, Single-Cell Mixed Models & Isoform DTU

### Function Translation Table

| Upstream R Tool | Upstream R Call | Python Import & Call | Key `R_default` Parity Behavior |
|---|---|---|---|
| **MuSiC / MuSiC2** (`xuranw/MuSiC`) | `music_prop(bulk.mtx, sc.sce, clusters = "cellType", samples = "sampleID")` | `from music import music_prop, music2_prop`<br>`music_prop(bulk_adata, sc_adata, cluster_col="cellType", sample_col="sampleID", settings=...)` | Drops single-donor `Sigma=NaN` cell types; recomputes `music_basis` per `l_genes` in `music2_prop` |
| **BayesPrism** (`Danko-Lab/BayesPrism`) | `new.prism(reference, mixture, ...)` + `run.prism(prism)` | `from pythonic_bayesprism import Prism`<br>`bp = Prism(ref_adata, mix_adata, cell_type_labels="cell_type", settings=...)`<br>`bp.run()` | Populates posterior `bp.theta_cv` across thinned draws; multi-start KL-NMF in `learn_embedding_nmf` |
| **Bisque** (`cozygene/bisque`) | `ReferenceBasedDecomposition(bulk.eset, sc.eset)` | `from bisque import reference_based_decomposition, marker_based_decomposition` | Exact `SLSQP` simplex QP (`limSolve::lsei`), `old_cpm=True`, `use_overlap=True` |
| **CIBERSORTx** (Newman et al., 2019) | Web portal (`Create Signature Matrix`, `Impute Cell Fractions`, `B-mode`/`S-mode`, `HiRes`) | `from cibersortx import build_signature_matrix, cibersort, b_mode_adjust, s_mode_adjust, impute_gep_highres` | `>50%` within-phenotype non-zero expression gate, Welch `ttest`, barcode-sampled synthetic mini-bulks |
| **spacexr::CSIDE** (`dmcable/spacexr`) | `run.CSIDE.single(myRCTD, explanatory.variable)` / `run.CSIDE.nonparam(myRCTD, df = 15)` | `from cside import run_cside, run_cside_single, run_cside_nonparam`<br>`run_cside_single(adata_sp, explanatory_variable=..., settings=...)` | `cell_type_threshold=125`, `gene_threshold=5e-5`, `fdr=0.01`, `log_fc_thresh=0.4` |
| **NEBULA** (`lhe17/nebula`) | `group_cell(count, id, pred)` + `nebula(count, id, pred = design, offset = libsize)` | `from nebula import group_cell, nebula`<br>`nebula(adata_sc, id="donor_id", pred=design_df, offset=libsize, settings=...)` | Cohort-wide `"HL"` when `mfs < 30`, `kappa=800` LN+HL refit tier, 2nd-order Laplace at low counts |
| **satuRn** (`Bioc::satuRn`) | `fitDTU(sumExp, formula = ~ 0 + group)` + `testDTU(sumExp, contrasts = L)` | `from saturn import filter_dtu, fit_dtu, test_dtu, stagewise_dtu` | Unpenalized `quasibinomial` IRLS (`|beta|>15 -> NaN`), `squeezeVar(robust=False)`, Poisson-GLM `locfdr` |
| **mashr** (`stephenslab/mashr`) | `mash_set_data(Bhat, Shat)` + `cov_canonical()` + `cov_ed()` + `mash(data, Ulist)` | `from mashr import mash_set_data, cov_canonical, cov_pca, cov_ed, mash, get_pm, get_lfsr` | `1e-8` `mixSQP` active-set convergence and exact `[10, 1, ..., 1]` `"nullbiased"` prior |

### Side-by-Side Workflow: Spatial Cell-Type DE (`CSIDE`) + Isoform DTU (`satuRn`) + Multi-Condition Shrinkage (`mashr`)

**Upstream R (`spacexr` + `satuRn` + `mashr`)**
```r
library(spacexr)
library(satuRn)
library(mashr)

# 1. Spatial cell-type-specific differential expression along a microenvironment gradient
myRCTD <- run.CSIDE.single(myRCTD, explanatory.variable = hypoxia_score, doublet_mode = FALSE)

# 2. Quasi-binomial Empirical Bayes isoform DTU
sumExp <- fitDTU(sumExp, formula = ~ 1 + cond)
sumExp <- testDTU(sumExp, contrasts = matrix(c(0, 1), ncol = 1), sort = TRUE)

# 3. Multivariate adaptive shrinkage across conditions
mdata <- mash_set_data(Bhat, Shat)
mfit  <- mash(mdata, Ulist = cov_canonical(mdata))
```

**Python (`pythonic_biotools`)**
```python
from cside import run_cside_single
from saturn import filter_dtu, fit_dtu, test_dtu
from mashr import mash_set_data, cov_canonical, mash, get_pm, get_lfsr

# 1. Spatial cell-type-specific DE directly on AnnData (.obsm["cell_type_weights"])
cside_res = run_cside_single(adata_sp, explanatory_variable=adata_sp.obs["hypoxia"], settings="default")
print(cside_res.to_dataframe().head())

# 2. Quasi-binomial Empirical Bayes isoform DTU
filt = filter_dtu(tx_counts_df, tx2gene_df, min_count=10, min_samples=3)
dtu_fit = fit_dtu(filt.counts, filt.tx2gene, design_df, settings="default")
dtu_res = test_dtu(dtu_fit, contrast=[0.0, 1.0], settings="default")

# 3. Multivariate adaptive shrinkage across conditions
mdata = mash_set_data(Bhat, Shat, settings="default")
mfit = mash(mdata, cov_canonical(mdata), settings="default")
pm, lfsr = get_pm(mfit), get_lfsr(mfit)
```

---

## 4. Differential Expression, Quantitative Proteomics & Latent Confounding

### Function Translation Table

| Upstream R Tool | Upstream R Call | Python Import & Call | Key `R_default` Parity Behavior |
|---|---|---|---|
| **limma (`eBayes`)** (`Bioc::limma`) | `lmFit(expr, design) |> contrasts.fit(L) |> eBayes(trend = TRUE, robust = TRUE)` | `from limma import moderated_ttest, moderated_lm_contrast, limma_top_table` | 128-node Gauss–Legendre log-$F$ quadrature, 4-df natural spline trend, `df2_outlier`, Smyth `B` |
| **DEqMS** (`Bioc::DEqMS`) | `medianSweeping(psm_df, group_col = 2)` + `lmFit() |> eBayes() |> spectraCounteBayes(fit)` + `outputResult()` | `from deqms import median_sweeping, spectra_count_ebayes, output_result` | Degree-1 tricube `loess(span=0.75)` on $(\log_2\text{count}, \ln\sigma^2)$ and exact 11-column `outputResult` |
| **sva / svaseq** (`Bioc::sva`) | `num.sv(dat, mod, method = "be")`; `sva(dat, mod, mod0)`; `svaseq(counts, mod, mod0)` | `from sva import num_sv, sva, svaseq, f_pvalue` | Fixed `lambda=0.8` $\hat\pi_0$, 512-grid GCV spline `edge_lfdr`, strict `rank(-var) < vfilter` |
| **variancePartition / dream** (`Bioc::variancePartition`) | `fitExtractVarPartModel(expr, ~ cond + (1|donor), meta)`; `dream(expr, ~ cond + (1|donor), meta) |> eBayes()` | `from variancepartition import fit_extract_var_part_model, dream, eBayes, topTable` | ML default (`reml=False`) in `fit_extract_var_part_model`, `(N-1)/N` variance, `varFixedTotal` scaling |
| **logistf** (`CRAN::logistf`) | `logistf(y ~ x1 + x2, data = df, pl = TRUE)` | `from logistf import fit_firth_logistic`<br>`fit_firth_logistic(X, y, pl=True, settings=...)` | Augmented Newton–Fisher weights $w_i(1 + h_i)$ (`logistf.c`) with `max_iter=25, tol=1e-5` |

### Side-by-Side Workflow: PSM-to-Protein TMT/DIA Differential Abundance (`DEqMS`) & Mixed-Effects `dream`

**Upstream R (`DEqMS` + `variancePartition`)**
```r
library(DEqMS)
library(variancePartition)

# 1. PSM median sweeping & peptide-count-moderated Empirical Bayes
prot_mat <- medianSweeping(psm_df, group_col = 1)
fit <- lmFit(prot_mat, design)
fit$count <- psm_counts[rownames(prot_mat)]
fit_deq <- spectraCounteBayes(eBayes(fit))
res_deq <- outputResult(fit_deq, coef_col = 2)

# 2. Repeated-measures linear mixed model DE with Satterthwaite df
fit_dream <- dream(expr_mat, ~ condition + (1 | donor), meta_df)
fit_dream <- eBayes(fit_dream)
res_dream <- topTable(fit_dream, coef = "condition")
```

**Python (`pythonic_biotools`)**
```python
from deqms import median_sweeping, spectra_count_ebayes, output_result
from variancepartition import dream, topTable

# 1. PSM median sweeping & peptide-count-moderated Empirical Bayes
swept = median_sweeping(psm_df, group_col="protein", settings="R_default")
fit_deq = spectra_count_ebayes(swept.abundance, design=design_mat, count=swept.count, settings="R_default")
res_deq = output_result(fit_deq, coef=1, settings="R_default")

# 2. Repeated-measures linear mixed model DE with Satterthwaite df
fit_dream = dream(expr_df, "~ condition + (1 | donor)", data=meta_df, genes_as_rows=True, settings="R_default")
res_dream = topTable(fit_dream, coef="condition")
```

---

## 5. Epigenomics, Methylation & Censored Clinical Trial Survival

### Function Translation Table

| Upstream R Tool | Upstream R Call | Python Import & Call | Key `R_default` Parity Behavior |
|---|---|---|---|
| **DSS & DMRcate** (`Bioc::DSS`, `DMRcate`) | `DMLtest(BSobj, group1, group2, smoothing = TRUE)` + `callDMR(dmlTest)` / `dmrcate()` | `from dss import dml_test, call_dml, call_dmr, dml_fit_multifactor, dmrcate` | `smoothing=False` default in `dml_test`, `p_threshold=1e-5`, `dis_merge=50` rule when `dis_merge > minlen` |
| **csaw** (`Bioc::csaw`) | `windowCounts(bams)` + `normFactors()` + `glmQLFit()` + `mergeWindows()` + `combineTests()` | `from csaw import window_counts, filter_windows_global, norm_factors, glm_ql_fit_test, merge_windows, combine_tests` | `width=50`, `spacing=50`, `min_mapq=0`, non-robust `squeezeVar`, directional counts gated by Simes $q \le 0.05$ |
| **monaLisa** (`Bioc::monaLisa`) | `calcBinnedMotifEnrR(seqs, bins, pwmL)` + `randLassoStabSel(x = hits, y = delta)` | `from monalisa import pfm_to_pwm, calc_binned_motif_enrichment, rand_lasso_stab_sel` | `pseudocount.log2enr=8.0`, 8-bin GC + 16-dinucleotide iterative raking, `weakness=0.8, cutoff=0.8` |
| **chromVAR** (`Bioc::chromVAR`) | `getBackgroundPeaks()` + `computeDeviations(counts, annotations)` + `computeVariability()` | `from chromvar import get_background_peaks, compute_deviations, compute_variability` | `50x50` lattice 1-NN background, `expected < 1 -> NaN`, `n_boot=1000` variability bootstrap |
| **FigR** (`buenrostrolab/FigR`) | `runGenePeakcorr()` + `dorcJPlot()` + `getDORCScores()` + `runFigRGRN()` | `from figr import run_gene_peak_corr, call_dorcs, get_dorc_scores, run_figr_grn` | `>= 7` cis peaks for DORCs, `score_mode="kartha_union"`, `corr_p_method="z_across_tfs"`, `dorc_k_pool=30` |
| **ROSE** (`younglab/ROSE`) | `ROSE_main.py -g HG38 -i peaks.gff -r chip.bam -c input.bam -s 12500 -t 2500` | `from rose import call_rose_super_enhancers`<br>`call_rose_super_enhancers(peaks_df, stitch_bp=12500, tss_exclusion_bp=2500, settings=...)` | Drops `chrY`, strict full TSS containment, `>2`-TSS unstitching, `argmin(y - slope*x)` tangent cutoff |
| **timeROC** (`CRAN::timeROC`) | `timeROC(T = time, delta = status, marker = score, cause = 1, times = c(3, 5), iid = TRUE)` | `from timeroc import time_roc, compare_timeroc`<br>`time_roc(T=time, delta=event, marker=score, times=[3.0, 5.0], cause=1, settings=...)` | Upstream `timeROC_3.R` `auc_1`/`auc_2` naming convention, strict `T < t` case horizon, `ddof=1` IID SEs |
| **survIDINRI** (`CRAN::survIDINRI`) | `IDI.INF(indata = cbind(time, event), covs0, covs1, t0 = 5.0, npert = 300)` | `from survidinri import idi_inf`<br>`idi_inf(time, event, covs0, covs1, t0=5.0, npert=300, settings=...)` | Breslow Cox ties, Uno `[-1, 1]` **half-cNRI** (`0.5 * cNRI`) in `m2_nri`, empirical perturbation quantile CIs |
| **survRM2** (`CRAN::survRM2`) | `rmst2(time, status, arm, tau = 8.0, covariates = X)` | `from survrm2 import rmst2, rmst2reg`<br>`rmst2(time, status, arm, tau=8.0, covariates=X, settings=...)` | Strict KM `tau` support validation, exact `0` Greenwood boundary at $Y_k = d_k$, per-arm KM censoring in `rmst2reg` |
| **BuyseTest & WR** (`CRAN::BuyseTest`, `WR`) | `BuyseTest(arm ~ tte(os, status = ev, threshold = 0.5) + bin(tox, operator = "<0"), data = df)`; `WR::pwreg()` | `from buysetest import Endpoint, buyse_test, pwreg`<br>`buyse_test(df, treatment="arm", endpoints=[...], settings=...)` | 2nd-order H-decomposition variance (`order=2`), KM tail `uninf` preservation, `atanh`/`log` CIs |
| **JMbayes2** (`CRAN::JMbayes2`) | `lmeFit <- lme(y ~ obstime + arm, random = ~ obstime \| id)`; `coxFit <- coxph(...)`; `jm(coxFit, lmeFit, time_var = "obstime")`; `predict(..., process = "event")` | `from jmbayes2 import jm, predict_dynamic`<br>`jm_fit = jm(surv_df, long_df, surv_covs=["arm"], long_covs=["obstime", "arm"], settings=...)` | 15-node Gauss–Kronrod quadrature, quadratic penalized B-spline baseline log-hazard, MCMC tail `P` |

### Side-by-Side Workflow: Clinical Trial Endpoints (`survRM2` + `BuyseTest` + `JMbayes2`)

**Upstream R (`survRM2` + `BuyseTest` + `JMbayes2`)**
```r
library(survRM2)
library(BuyseTest)
library(nlme)
library(survival)
library(JMbayes2)

# 1. Restricted Mean Survival Time (RMST) under non-proportional hazards
rmst_res <- rmst2(time = df$os_time, status = df$os_event, arm = df$arm, tau = 8.0)

# 2. Hierarchical Composite Win Ratio & Net Benefit (Péron KM scoring)
bt_res <- BuyseTest(
  arm ~ tte(os_time, status = os_event, threshold = 0.5) + bin(tox_grade3, operator = "<0"),
  data = df, scoring.rule = "Peron"
)

# 3. Bayesian Joint Longitudinal Biomarker + Survival Model & Dynamic Prediction
lme_fit <- lme(y ~ obstime + arm, random = ~ obstime | id, data = long_df)
cox_fit <- coxph(Surv(time, event) ~ arm, data = surv_df)
jm_fit  <- jm(cox_fit, lme_fit, time_var = "obstime")
pred    <- predict(jm_fit, newdata = new_patient_df, process = "event", times = c(3.0, 5.0))
```

**Python (`pythonic_biotools`)**
```python
from survrm2 import rmst2
from buysetest import Endpoint, buyse_test
from jmbayes2 import jm, predict_dynamic

# 1. Restricted Mean Survival Time (RMST) difference & ratio at tau = 8.0
rmst_res = rmst2(df["os_time"], df["os_event"], df["arm"], tau=8.0, settings="R_default")
print(rmst_res.rmst_diff_10.est, rmst_res.rmst_diff_10.pval)

# 2. Hierarchical Composite Win Ratio & Net Benefit (Péron KM scoring)
bt_res = buyse_test(
    df,
    treatment="arm",
    endpoints=[
        Endpoint("os_time", kind="tte", status_col="os_event", threshold=0.5),
        Endpoint("tox_grade3", kind="binary", operator="lower"),
    ],
    settings="R_default",
)
print(bt_res.to_dataframe()[["endpoint", "win_ratio", "net_benefit", "win_ratio_pval"]])

# 3. Bayesian Joint Longitudinal Biomarker + Survival Model & Dynamic Prediction
jm_fit = jm(
    surv_df,
    long_df,
    surv_covs=["arm"],
    long_covs=["obstime", "arm"],
    n_iter=500,
    burn_in=150,
    settings="R_default",
)
dyn_pred = predict_dynamic(
    jm_fit,
    newdata_long_df=new_patient_df,
    t_landmark=1.5,
    horizons=[3.0, 5.0],
    surv_covs_row={"arm": 0.0},
    settings="R_default",
)
print(dyn_pred)
```

---

## Related Guides

- **[CLI, Pipeline Runners & File I/O Guide](CLI_AND_IO_GUIDE.md)** — Loading standard bioinformatics files (`MAF`, `VCF`, `Bismark`, `MaxQuant`/`DIA-NN`, `GWAS`, `PLINK LD`) and running `pbt run` in Nextflow/Snakemake.
- **[Use Cases by Biological & Statistical Problem](USE_CASES_BY_PROBLEM.md)** — Problem-oriented index across all 70 packages.
- **[Tool Comparison & Decision Matrix](TOOL_COMPARISON_AND_DECISION_MATRIX.md)** — Choosing the right statistical estimator within each domain.
- **[Upstream Parity & Dual Scorecard](UPSTREAM_PARITY_AND_SCORECARD.md)** — Complete 70-package parity audit (`settings="default"` vs. `settings="R_default"`).
