# Multimodal MCMC for Hierarchical Gaussian Mixture Models

This project compares several sampling methods for a simulated hierarchical Gaussian mixture model with multimodal posterior structure. Our main goal is to evaluate whether ensemble MCMC with teleporting walkers improves exploration and sampling efficiency relative to more standard approaches.

## Methods compared

- Gibbs sampling / Metropolis-within-Gibbs
- Parallel tempering
- Ensemble MCMC with teleporting walkers
- HMC / NUTS benchmark (via Stan or PyMC)

## Model

We simulate data from a hierarchical Gaussian mixture model with \(K\) latent mixture components. The model is designed to induce multimodality through label-switching and weak identification, making it a useful testbed for comparing samplers.

## Main evaluation metrics

- Effective sample size (ESS)
- \(\hat{R}\)
- Acceptance rates
- Autocorrelation / trace plots
- Runtime
- Mode visitation / switching frequency
- Total variation distance (TVD) to target or reference posterior approximation

## Repository structure

```text
.
├── configs/        # experiment configs
├── data/           # simulated datasets
├── docs/           # project notes and references
├── notebooks/      # exploratory notebooks
├── results/        # saved chains, tables, plots
├── scripts/        # command-line scripts
├── src/            # source code
│   ├── diagnostics/
│   ├── experiments/
│   ├── models/
│   ├── samplers/
│   ├── simulate/
│   ├── stan/
│   └── utils/
└── tests/          # unit tests