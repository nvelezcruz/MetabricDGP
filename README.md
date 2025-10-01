# MetabricDGP

Deep Gaussian Process (DGP) survival analysis pipeline applied to the METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset.

This code implements a variational Deep Gaussian Process framework for time-to-event prediction and risk stratification in ER+/HER2−, endocrine-treated breast cancer patients. It generates probability-based risk groups (Posterior-H grouping), estimates survival at clinical horizons (e.g., 60 and 120 months), and produces interpretability outputs such as feature importance and PC-to-gene projections.

## Features
- Deep Gaussian Process (DGP) survival modeling (gpytorch backend)
- Flexible likelihoods:
  - True-ELBO discrete-time model (default)
  - Cox partial likelihood pseudo-model (optional)
- Train-only PCA feature extraction with survival-guided PC selection
- Optional gene-set scores (ER and proliferation)
- Clinical covariates (age, tumor size, grade, menopausal state)
- Posterior-H grouping for risk stratification at 10 years
- Model interpretability:
  - Consensus feature importance 
  - Risk-oriented PC-to-gene loadings
  - Partial dependence (PD/ICE) curves
- Evaluation metrics: C-index, Somers’ Dxy, Integrated Brier Score (IPCW), calibration
- Biological checks (ER/proliferation correlations)
- Multi-seed reproducibility

## Requirements
- Python 3.9+
- PyTorch
- GPyTorch
- NumPy, SciPy, scikit-learn, pandas
- lifelines
- matplotlib

Install dependencies with:

    pip install torch gpytorch lifelines numpy scipy scikit-learn pandas matplotlib

## Inputs
- Clinical data:
  - data_clinical_patient.txt
  - data_clinical_sample.txt
- Expression data:
  - data_mrna_illumina_microarray.txt
- Optional whitelist of patient/sample IDs

Update the BASE and OUTDIR paths in the script to point to your local data directory.

## Usage
Run a single-seed training and evaluation:

    python MetabricDGP.py

Enable multiple seeds by toggling:

    RUN_MULTI_SEEDS = True

Outputs are written to:

    <BASE>/metabric_results/

## Outputs
- Risk prediction metrics: c_index.csv, c_dxy_bootstrap_test.csv
- Risk grouping: Kaplan-Meier plots (KM_*png), summary CSVs
- Interpretability:
  - Consensus importance plots (interpretability_consensus_*png)
  - PC-to-gene loadings (pc_gene_oriented_*csv, PNGs)
  - Partial dependence curves (PD_*png)
- Biological checks: correlation CSVs, grade3 enrichment, adjusted Cox results

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
