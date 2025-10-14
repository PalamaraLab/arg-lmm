# Efficient complex trait analyses using the Ancestral Recombination Graph (ARG)

This repository provides efficient computational tools for complex trait analysis using Ancestral Recombination Graphs (ARGs). We provide tools for heritability estimation based on the Haseman-Elston estimator and scalable linear mixed model (LMM) association testing through ARG-based leave-one-chromosome-out (LOCO) residual calculations, enabling large-scale studies with improved computational efficiency.

Requires arg-needle-lib:
> pip install arg-needle-lib

## 1. ARG-based estimation of heritability with Haseman-Elston (ARG-RHE)
> arg_rhe.py

### usage

`python arg_rhe.py --arg <path_to_argn_file> --pheno <path_to_phenotypes> --out <output_path> --mu <mutation_resampling_rate> --alpha <alpha> --mac <minimal_mac_to_include> --seed <seed>`

- `<path_to_argn_file>` should point to an ARG in `.argn` format inferred from `arg-needle` or `threads`.
- `<path_to_phenotypes>` is space-separated without header, with its first two columns containing FID, IID for the samples, and the rest being phenotypes (preferably mean-centred and normalised through RINT). The sample ordering must be exactly the same as the ARG file.
- `<output_path>` is the output file with estimated `h2_g` being the estimated h2 on the GRM random component. The row ordering is the same as the phenotype columns in `<path_to_phenotypes>`.
- `<mutation_resampling_rate>` (optional) is the rate at which to generate new mutations on the ARG. The default value is `1e-6`.
- `<alpha>` (optional) is the alpha normalising exponent where each entry of the genotype matrix is scaled by `(af*(1-af))**alpha` after mean-centring. The default value is `-1`.
- `<minimal_mac_to_include>` (optional) is the minimal MAC of the resampled mutations to be included in the analysis. The default value is `1`.
- `<seed>` (optional) is the random seed to ensure reproducibility. The default value is `42`.


### example

`python arg_rhe.py --arg demo_files/10kb_region_20k_samples.argn --pheno demo_files/h2_5e-03_alpha_-0.5.phenos --out demo_files/output.csv --mu 1e-6 --alpha -0.5 --mac 1 --seed 42`

## 2. ARG-based calculation of BLUP or LOCO residuals for association testing
> arg_lmm.py

### Outline
* Estimate genome-wide heritability with ARG-RHE (similar to part 1)
* Calculate leave-one-chromosome-out (LOCO) residuals for association testing
* Compute best linear unbiased predictors (BLUP)
* Thoughout, employ an efficient conjugate gradient solver that handles multiple phenotypes offering high scalability.

This python module can be used to calculate the leave-one-chromosome-out (LOCO) residuals to be used in within-ARG linear-mixed-model (LMM) association testing. This includes the estimation of total heritability using ARG-RHE, and a conjugate gradient iteration to solve the system $Vx=y$, where V is a LOCO covariance matrix. One feature is the calculation of the best linear unbiased predictor (BLUP), using the formula  $KV^{-1}y$, with $K$ representing $\frac{\sigma^2}{M}XX^\top$ and $V$ the genome-wide covariance matrix (as opposed to the LOCO matrices used in estimating the residuals).

### usage
```python arg_lmm.py --arg <list of paths to args> --pheno <path_to_phenotypes> -out <output_path> --blup --ncalib <num of snps to estimate gamma> --alpha <alpha>```

- `<list of paths to args>` filepaths to chromosome-specific ARGs as inferred by `arg-needle` or `threads`; at least two are needed.
- `<path_to_phenotypes>` is space-separated without header, with its first two columns containing FID, IID for the samples, and the rest being phenotypes (preferably mean-centred and normalised through RINT). The sample ordering must be exactly the same as the ARG file.
- `<blup>` flag to invoke the calculcation of the BLUP, which is not performed by default.
- `<output_path>` prefix for all output files (BLUP, LOCO residuals, calibration factors).
- `<alpha>` (optional) is the alpha normalising exponent where each entry of the genotype matrix is scaled by `(af*(1-af))**alpha` after mean-centring. The default value is `-1`.
- `<seed>` (optional) is the random seed to ensure reproducibility. The default value is `42`.
- `<ncalib>` (optional) the number of markers per chromosome to use during the estimation of the Grammar-gamma calibration factor.

### example
```python arg_lmm.py --arg demo_files/sims.N2000.chr{1..10}.arg --pheno demo_files/sims.N2000.phenotypes --out output/arg_loco.new --blup```

We provide simulated ARGs (one per hypothetical chromosome) and five phenotypes for 5,000 diploid samples in `demo_files`, generated after assuming 10 chromosomes and h2=0.25.

## Licenses

The `liu_sf` method is directly copied from `chiscore` under MIT License rather
than using PyPI due to a broken `chi2comb` dependency.

## Citation
> Zhu, Kalantzis, et al. (2025), "Fast variance component analysis using large-scale ancestral recombination graphs"
https://doi.org/10.1101/2024.08.31.610262