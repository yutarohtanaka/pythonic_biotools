# One-Line Installation, File Loaders (`pythonic_biotools.io`) & Pipeline CLI (`pbt`)

`pythonic_biotools` can be used in three ways depending on your workflow:

1. **Unified Collision-Safe Python Namespace (`import pythonic_biotools as pbt`)** — installs all 63 MIT-licensed libraries into a single `pip` package without colliding with unrelated PyPI package names.
2. **Command-Line Pipeline Runner (`pbt run <tool>`)** — `TSV`/`MAF`/`VCF`/`cov` file-in $\to$ `TSV` file-out commands ready for **Nextflow**, **Snakemake**, **WDL / Terra**, or **SLURM** bash scripts.
3. **Standalone Subpackage Installation (`#subdirectory=<pkg>`)** — install any individual library (including the 7 copyleft `GPL`/`AGPL` ports) in isolation.

---

## 1. Installation Options

### One-Line GitHub / Wheel Install (63 MIT Libraries + `io` + `pbt` CLI)
```bash
# Minimal core (numpy, scipy, pandas) + all 63 MIT libraries
pip install "git+https://github.com/gillanilab/pythonic_biotools.git"

# With domain extras (anndata, pysam, scikit-learn, matplotlib, requests)
pip install "pythonic-biotools[all] @ git+https://github.com/gillanilab/pythonic_biotools.git"
```

```python
import pythonic_biotools as pbt
from pythonic_biotools import purecn, ichorcna, ctwas, cismr, deqms, survrm2, io

# Every package supports both Pythonic defaults and strict upstream R parity:
res = survrm2.rmst2(time, status, arm, tau=12.0, settings="R_default")
fig = res.plot()
```

### Installing Individual Standalone or Copyleft Packages
```bash
# Install any single package by subdirectory (including GPL/AGPL ports)
pip install "git+https://github.com/gillanilab/pythonic_biotools.git#subdirectory=smr-heidi"
pip install "git+https://github.com/gillanilab/pythonic_biotools.git#subdirectory=mrpresso"
```

### Reproducible Docker & Conda Environments
```bash
# Conda environment
conda env create -f environment.yml
conda activate pythonic-biotools

# Docker container (entrypoint: pbt)
docker build -t pythonic-biotools .
docker run --rm -v "$PWD:/data" pythonic-biotools run scarhrd --segments /data/seg.tsv --out /data/hrd.tsv
```

---

## 2. Turnkey Bioinformatics File Readers (`pythonic_biotools.io`)

Skip manual column renaming when importing outputs from GATK, cBioPortal/TCGA, Mutect2/Strelka2, Bismark/MethylDackel, MaxQuant/DIA-NN, or PLINK/GWAS:

```python
from pythonic_biotools import io, purecn, dss, deqms, discover, cismr

# 1. GDC / cBioPortal / AACR GENIE MAF (.maf or .maf.gz)
maf_df = io.read_maf("cohort_mutations.maf.gz", keep_silent=False)
alt_mat = io.maf_to_binary_matrix(maf_df)  # {0, 1, NaN} matrix for discover / evam

# 2. Somatic or germline VCF (.vcf or .vcf.gz) -> GT, AD, DP, VAF, POPAF
vcf_df = io.read_vcf_variants("sample.mutect2.vcf.gz", pass_only=True)

# 3. GATK4 DenoisedCopyRatios / ModelSegments / CNVkit (.cnr, .cns, .seg)
bins_df = io.read_gatk_copy_ratios("sample.denoisedCR.tsv")
segs_df = io.read_cn_segments("sample.called.seg")

# 4. Bismark (.bismark.cov.gz) or MethylDackel CpG methylation coverage
bs = io.read_bismark_cov(["ctrl1.cov.gz", "ctrl2.cov.gz", "trt1.cov.gz", "trt2.cov.gz"])
dmls = dss.dml_test(bs.frames[:2], bs.frames[2:], settings="default")

# 5. MaxQuant proteinGroups.txt or DIA-NN pg_matrix.tsv (contaminants filtered, log2 + peptide counts)
abundance_df, peptide_counts = io.read_maxquant_diann("proteinGroups.txt")
fit = deqms.spectra_count_ebayes(abundance_df, design=design_mat, count=peptide_counts)

# 6. GWAS Summary Statistics (auto-detects SNP/A1/A2/BETA/OR/SE/P/N/EAF) & PLINK LD (.ld + .bim)
gwas_df = io.read_gwas_sumstats("ukb_trait.tsv.gz")
ld_mat, snp_ids = io.read_plink_ld("locus.ld.gz", bim_path="locus.bim")
```

---

## 3. Pipeline Command-Line Interface (`pbt` / `pythonic-biotools`)

Every `pbt run <tool>` subcommand accepts `--settings default` or `--settings R_default`:

```bash
# Inspect bundles and check environment
pbt list --bundle oncology
pbt check-imports --bundle all

# 1. Homologous Recombination Deficiency genomic scars (scarHRD)
pbt run scarhrd --segments sample.cns --genome GRCh38 --out sample.hrd.tsv --settings R_default

# 2. Ultra-low-pass WGS cfDNA tumor fraction & CNAs (ichorCNA)
pbt run ichorcna --bins sample.denoisedCR.tsv --out-segments segs.tsv --out-summary tfx.tsv

# 3. Tumor-only targeted panel purity/ploidy, somatic calling & TMB (PureCN)
pbt run purecn --coverage tumor_cov.tsv --variants tumor.vcf.gz --callable-mb 1.8 \
  --out-variants somatic_calls.tsv --out-summary purecn_summary.tsv

# 4. Somatic SNV & Whole-Genome Doubling molecular clock timing (MutationTimeR)
pbt run mutationtimer --variants tumor.vcf.gz --segments allele_cn.tsv --purity 0.72 \
  --out-variants timed_snvs.tsv --out-segments timed_gains.tsv

# 5. Bisulfite / EM-seq differential DNA methylation (DSS)
pbt run dss --ctrl c1.cov.gz,c2.cov.gz --trt t1.cov.gz,t2.cov.gz \
  --out-dml dml.tsv --out-dmr dmr.tsv

# 6. PSM/peptide-count-moderated differential proteomics (DEqMS)
pbt run deqms --input proteinGroups.txt --groups 0,0,0,1,1,1 --out deqms_results.tsv

# 7. Bayesian two-trait colocalization (coloc.abf) & correlated cis-MR (PC-GMM / MR-SPI)
pbt run coloc --trait1 eqtl.tsv --trait2 gwas.tsv --type2 cc --out coloc_summary.tsv
pbt run cismr --input cis_locus.tsv --ld locus.ld --nx 35000 --ny 180000 --out cismr.tsv

# 8. Clinical trial RMST (survRM2) & Win Ratio (BuyseTest)
pbt run survrm2 --input trial.tsv --time-col os_months --status-col event --arm-col trt --tau 24 --out rmst.tsv
pbt run buysetest --input trial.tsv --arm-col trt --tte-col os_months --status-col event --threshold 1.0 --out wr.tsv
```
