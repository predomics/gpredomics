# MCMC / Bayesian Inference

Markov Chain Monte Carlo (MCMC) is a Bayesian approach to feature selection and model estimation in Gpredomics. Unlike the optimization-based algorithms (GA, ACO, SA, Beam), MCMC samples from the posterior distribution of models, providing probabilistic statements about feature importance and coefficient uncertainty. It is combined with Sequential Backward Selection (SBS) for automatic dimensionality reduction.

## Overview

```
  All pre-selected features (p features)
           |
           v
  ┌─────────────────────────────────────────────┐
  │  MCMC Posterior Sampling (n_iter iterations) │
  │                                              │
  │  For each iteration:                         │
  │    1. Optimize betas (a, b, c) via Brent     │
  │    2. Sample betas from truncated normals     │
  │    3. Propose coef flips for each feature     │
  │    4. Accept/reject via Metropolis-Hastings   │
  │                                              │
  │  After burn-in: record posterior samples      │
  └─────────────────────────────────────────────┘
           |
           v
  Compute feature probabilities: P(+1), P(0), P(-1)
           |
           v
  Drop feature with highest P(neutral)
           |
           v
  ┌─────────────────────────────────────────────┐
  │  SBS: Repeat MCMC with p-1 features         │
  │        then p-2, p-3, ..., nmin features     │
  └─────────────────────────────────────────────┘
           |
           v
  Select optimal subset by log evidence
           |
           v
  Final MCMC run on optimal feature subset
```

The algorithm:
1. **Pre-select** features using statistical tests (same as other algorithms)
2. **Initialize** a model with all selected features, random coefficients in {-1, 0, +1}, and betas (a=1, b=-1, c=1)
3. **Run MCMC** for `n_iter` iterations:
   - Optimize each beta (a, b, c) using Brent's method, then sample from a truncated normal
   - For each feature, propose a coefficient change and accept/reject via Metropolis-Hastings
4. **Discard** the first `n_burn` iterations (burn-in period)
5. **Compute** posterior feature probabilities from post-burn-in samples
6. **SBS loop**: drop the feature with highest neutral probability, repeat MCMC with fewer features
7. **Select** the feature subset that maximizes log evidence
8. **Final run**: recompute full MCMC on the optimal subset

## Key Concepts

### Bayesian Logistic Regression

The MCMC approach models the probability of class membership using logistic regression with structured coefficients:

```
P(y=1 | x, model) = logistic(a * sum_pos + b * sum_neg + c)
```

Where:
- **sum_pos** = sum of feature values where coefficient = +1
- **sum_neg** = sum of feature values where coefficient = -1
- **a** (positive coefficient weight): constrained to be positive via truncated normal sampling
- **b** (negative coefficient weight): constrained to be negative via truncated normal sampling
- **c** (intercept): unconstrained, sampled from a normal distribution

This structure mirrors the BTR model languages but with continuous scaling factors (a, b, c) instead of fixed discrete coefficients.

### Metropolis-Hastings Sampling

The algorithm uses a two-phase sampling scheme within each iteration:

**Phase 1 -- Continuous parameters (betas):**
For each beta (a, b, c):
1. Find the mode via Brent optimization (minimizing negative log posterior)
2. Compute the local curvature (sigma) at the mode
3. Sample a new value from a truncated normal centered at the mode:
   - Beta a: truncated to positive values
   - Beta b: truncated to negative values
   - Beta c: unrestricted normal

**Phase 2 -- Discrete parameters (feature coefficients):**
For each feature:
1. Propose a new coefficient: current + random({2,3}) mod 3 - 1, cycling through {-1, 0, +1}
2. Compute the log posterior ratio between proposed and current states
3. Accept with probability min(1, exp(log_posterior_new - log_posterior_old))

### Log Posterior

The log posterior combines the likelihood and a Gaussian prior:

```
log P(model | data) = log_likelihood + log_prior

log_likelihood = sum_i [ y_i * log(sigma(z_i)) + (1-y_i) * log(sigma(-z_i)) ]
    where z_i = a * pos_i + b * neg_i + c

log_prior = -lambda * (a^2 + b^2 + c^2)
```

The `lambda` parameter controls regularization strength: higher lambda shrinks betas toward zero, preferring simpler models.

### Burn-in Period

The first `n_burn` iterations are discarded to allow the chain to converge to the stationary distribution. During burn-in, the sampler explores broadly; after burn-in, samples approximate the true posterior.

Only post-burn-in samples contribute to:
- Feature inclusion probabilities P(+1), P(0), P(-1)
- Beta coefficient means and variances
- Log posterior statistics

### Sequential Backward Selection (SBS)

SBS wraps the core MCMC to perform automatic feature selection:

1. Run MCMC with all p pre-selected features
2. Identify the feature with the highest P(neutral) -- the least important feature
3. Remove that feature and run MCMC again with p-1 features
4. Repeat until only `nmin` features remain
5. For each step, compute the log evidence: `log10(posterior_mean) - n_features * log10(3)`
6. Select the step with the highest log evidence as the optimal feature subset
7. Re-run MCMC on the optimal subset to produce the final posterior

### Coefficients: MCMC vs BTR

| Aspect | BTR (GA/ACO/SA/Beam) | MCMC |
|--------|---------------------|------|
| **Feature coefficients** | Discrete: {-1, 0, +1} | Discrete: {-1, 0, +1} (sampled) |
| **Scaling** | None (fixed at 1) | Continuous betas (a, b, c) |
| **Uncertainty** | None (point estimate) | Full posterior distribution |
| **Feature importance** | Frequency in FBM | Posterior inclusion probability |
| **Language** | Binary/Ternary/Ratio/Pow2 | Generic (single language) |

## Known Limitations

The MCMC implementation is currently in **beta** status with the following limitations:

- **Single-threaded**: the MCMC chain is inherently sequential; no parallelization across chains
- **Single language**: only the generic MCMC language is supported (no binary/ternary/ratio/pow2 distinction during sampling)
- **Single data type**: only one data type per run (if multiple are specified, only the first is used)
- **Memory-intensive**: SBS stores the full posterior trace for each feature count, which can be large for many features or iterations
- **Slow for many features**: each SBS step runs a full MCMC chain; with p features, this means p - nmin MCMC runs
- **No GPU support**: all computation is CPU-based
- **No cross-validation**: MCMC does not currently integrate with the CV framework

## Parameters

```yaml
mcmc:
  n_iter: 1000          # Total MCMC iterations per chain
  n_burn: 500           # Burn-in iterations to discard
  lambda: 0.1           # L2 regularization strength for betas
  nmin: 5               # Minimum features to keep in SBS (0 = no SBS)
```

### Tuning Guide

| Parameter | Effect of increasing | Recommended |
|-----------|---------------------|-------------|
| **n_iter** | Better posterior approximation, slower | 1000--10000 |
| **n_burn** | More burn-in discarded, must be < n_iter | 50% of n_iter |
| **lambda** | Stronger regularization, smaller betas | 0.01--1.0 |
| **nmin** | SBS stops earlier, keeps more features (0 = skip SBS) | 3--20 |

**Quick start**: Use `n_iter: 2000`, `n_burn: 1000`, `lambda: 0.1`. Set `nmin` to your target model size or 0 to run MCMC without SBS.

## When to Use MCMC

**Use MCMC when:**
- You need posterior uncertainty estimates for feature importance
- You want probabilistic feature inclusion/exclusion (not just point estimates)
- You need to quantify how confidently each feature contributes to prediction
- You want an automated feature elimination procedure (SBS)
- The number of pre-selected features is moderate (< 100)

**Don't use MCMC when:**
- Speed is critical (MCMC is significantly slower than GA/SA/Beam)
- You have many features (> 200 pre-selected) -- SBS will be very slow
- You need multi-language comparison (binary vs ternary vs ratio)
- You need GPU acceleration
- You need cross-validation integrated into the search

## Example

```yaml
general:
  algo: mcmc
  language: ter
  data_type: prev
  seed: 42
  fit: auc

mcmc:
  n_iter: 5000
  n_burn: 2500
  lambda: 0.1
  nmin: 5
```

For a quick exploratory run without SBS:

```yaml
mcmc:
  n_iter: 2000
  n_burn: 1000
  lambda: 0.1
  nmin: 0              # Disable SBS, run MCMC on all pre-selected features
```

## References

- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 21(6), 1087--1092.
- Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97--109.
- Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer.
- O'Hara, R. B. & Sillanpaa, M. J. (2009). A Review of Bayesian Variable Selection Methods. *Bayesian Analysis*, 4(1), 85--117.
- Prifti, E. et al. (2020). Interpretable and accurate prediction models for metagenomics data. *GigaScience*, 9(3), giaa010.

*Last updated: v0.9.0*
