# LASSO / Elastic Net

LASSO (Least Absolute Shrinkage and Selection Operator) is a mathematical optimization method that performs feature selection via L1 regularization. Unlike the heuristic metaheuristics (GA, ACO, SA, ILS), LASSO solves a convex optimization problem with a guaranteed global optimum for each regularization strength. It serves as a principled statistical baseline for comparison with BTR models.

## Overview

```
  Regularization path: alpha_max ──────────────────────▶ alpha_min

  ┌─────────────────────────────────────────────────────────────┐
  │  Features   α = α_max        α = mid          α = α_min    │
  │  ─────────  ──────────       ──────────       ──────────    │
  │  feat_1     ■                ■■■              ■■■■■■        │
  │  feat_2                      ■■               ■■■■          │
  │  feat_3                                       ■■            │
  │  feat_4                      ■                ■■■■■         │
  │  feat_5                                       ■             │
  │  feat_6                                                     │
  │  ...                                                        │
  │             ◀── sparse ──────────────────── dense ──▶       │
  │             few features,    moderate         many features, │
  │             high bias        trade-off        low bias       │
  └─────────────────────────────────────────────────────────────┘
     Coordinate descent solves each α using warm starts from
     the previous α, making the full path efficient.
```

The algorithm:
1. **Compute alpha_max** — the smallest regularization where all coefficients are zero
2. **Generate alpha path** — a log-spaced sequence from alpha_max to alpha_min
3. **For each alpha**, solve the penalized logistic regression using coordinate descent (warm-started from the previous solution)
4. **Select best alpha** by cross-validated or train/test performance
5. **Convert coefficients** to BTR model: non-zero features become model features, coefficient signs become ±1

## Key Concepts

### L1 Regularization (Sparsity)

The LASSO objective minimizes:

```
L(β) = -log_likelihood(β) + α × ||β||₁
```

- The L1 penalty `α × Σ|βⱼ|` drives coefficients exactly to zero (unlike L2/Ridge)
- Higher α = more sparsity (fewer features selected)
- At α_max, all coefficients are zero; at α = 0, the full unpenalized model
- This automatic feature selection is analogous to what heuristic BTR methods achieve through population dynamics

### Coordinate Descent

Each feature's coefficient is updated one at a time while holding all others fixed:

```
βⱼ ← soft_threshold(partial_residual_j, α) / (X_j^T X_j + α × l1_ratio)
```

The soft-thresholding operator naturally produces exact zeros, making coordinate descent the method of choice for L1-penalized problems.

### Warm Starts

When solving along the regularization path, each new α starts from the solution of the previous (slightly less regularized) α. This makes the full path nearly as cheap as solving for a single α — a key efficiency advantage.

### Elastic Net Mixing

The Elastic Net combines L1 and L2 penalties:

```
L(β) = -log_likelihood(β) + α × [l1_ratio × ||β||₁ + (1 - l1_ratio) × ||β||₂² / 2]
```

| l1_ratio | Behavior |
|----------|----------|
| 1.0 | Pure LASSO (maximum sparsity) |
| 0.5 | Equal L1 + L2 (grouped selection) |
| 0.0 | Pure Ridge (no sparsity, all features kept) |

Elastic Net is useful when features are correlated: LASSO arbitrarily picks one from a correlated group, while Elastic Net tends to keep or drop them together.

### Converting LASSO to BTR Models

LASSO coefficients are real-valued, but BTR models use discrete coefficients (±1 for ternary). The conversion process:

1. **Fit LASSO** at the selected alpha
2. **Identify non-zero coefficients** — these become the model's feature set
3. **Extract signs** — `sign(βⱼ)` becomes the BTR coefficient (+1 or -1)
4. **Evaluate** — the resulting ternary BTR model is scored using the standard BTR prediction and fitness functions

This means LASSO acts as a feature selector: it identifies *which* features matter and *which direction* they point, then hands off to the standard BTR evaluation pipeline. The resulting model is directly comparable to models found by GA, ACO, SA, or ILS.

## Comparison: LASSO vs Heuristic BTR

| Aspect | LASSO | GA / ACO / SA / ILS |
|--------|-------|---------------------|
| **Optimization** | Convex, global optimum | Heuristic, local optima |
| **Feature selection** | Automatic via L1 penalty | Population dynamics / perturbation |
| **Coefficients** | Real-valued (converted to ±1) | Native ±1 (ternary) |
| **Model size** | Controlled by α | Controlled by k_min/k_max + k_penalty |
| **Speed** | Very fast (coordinate descent) | Varies (SA fast, ACO slow) |
| **Interpretability** | Post-hoc (sign extraction) | Native BTR model |
| **Role** | Statistical baseline | Primary BTR discovery |

LASSO provides a mathematically rigorous reference point. If a heuristic method cannot outperform LASSO, it suggests the problem may not benefit from the discrete BTR structure. Conversely, when heuristic methods beat LASSO with much sparser models, it validates the BTR approach.

## Parameters

```yaml
lasso:
  n_alphas: 100                  # Number of alpha values along the regularization path
  alpha_min_ratio: 0.001         # Ratio alpha_min / alpha_max
  l1_ratio: 1.0                  # Elastic Net mixing (1.0 = pure LASSO)
  max_iterations: 1000           # Max coordinate descent iterations per alpha
  tolerance: 1e-4                # Convergence tolerance for coordinate descent
  k_min: 1                       # Min features (skip alphas producing fewer)
  k_max: 200                     # Max features (skip alphas producing more)
```

### Tuning Guide

| Parameter | Effect of increasing | Recommended range |
|-----------|---------------------|-------------------|
| **n_alphas** | Finer regularization path, more model sizes explored | 50–200 |
| **alpha_min_ratio** | Includes less regularized (denser) models | 0.0001–0.01 |
| **l1_ratio** | More sparsity (toward pure LASSO) | 0.5–1.0 |
| **max_iterations** | Better convergence for difficult problems | 500–5000 |
| **k_max** | Allows denser models to be considered | Dataset-dependent |

**Quick start**: The defaults work well for most datasets. For correlated features (e.g., closely related taxa in metagenomics), try `l1_ratio: 0.7` to enable Elastic Net grouping. For very sparse solutions, use `alpha_min_ratio: 0.01`.

## Benchmark

Qin2014 cirrhosis dataset (1980 features, 180 samples, ternary:prevalence):

| Method | Test AUC | FBM Test AUC | k | Time |
|--------|---------|-------------|---|------|
| **LASSO** | **0.853** | — | 177 | **0.1s** |
| SA | 0.911 | 0.892 | 48 | 0.3s |
| ILS | 0.813 | — | 22 | 0.05s |
| GA | 0.791 | 0.788 | 48 | 0.5s |
| ACO | 0.802 | 0.834 | 52 | 7.1s |

LASSO achieves strong AUC (0.853) but uses many more features (k=177) than the heuristic methods. This illustrates the typical trade-off: LASSO casts a wide net for predictive accuracy, while BTR heuristics find sparser, more interpretable models. The heuristic methods (especially SA) can surpass LASSO with far fewer features.

## When to Use LASSO

**Use LASSO when:**
- You need a statistical baseline to validate heuristic BTR results
- Fast initial screening of feature importance is needed
- You want to establish an upper bound on what linear methods can achieve
- The dataset has many correlated features (use Elastic Net variant)
- You need reproducible results with no stochastic variation

**Don't use LASSO when:**
- Model sparsity is critical (LASSO tends to select many features)
- You need native BTR coefficients (LASSO coefficients are converted, not native)
- FBM diversity or voting ensembles are needed
- Non-linear feature interactions are suspected

## Example

```yaml
general:
  algo: lasso
  language: ter
  data_type: prev
  seed: 42
  fit: auc
  k_penalty: 0.001

lasso:
  n_alphas: 100
  alpha_min_ratio: 0.001
  l1_ratio: 1.0
  max_iterations: 1000
  tolerance: 1e-4
  k_min: 1
  k_max: 200
```

## References

- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Zou, H. & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. *Journal of Statistical Software*, 33(1), 1–22.

*Last updated: v0.9.0*
