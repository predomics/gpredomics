# gpredomics Vignette — A Practical Tutorial

*Last updated: v0.9.0 — March 2026*

This vignette walks through a complete analysis using the **Qin2014 liver cirrhosis** metagenomics dataset (1,980 metagenomic species, 180 train / 30 test samples). By the end, you will know how to:

1. Prepare data and configure parameters
2. Run different optimization algorithms
3. Interpret the output (models, FBM, metrics)
4. Compare algorithms
5. Use cross-validation and voting ensemble
6. Extract feature importance

---

## 1. Installation

```bash
# Clone and build
git clone https://github.com/predomics/gpredomics.git
cd gpredomics
cargo build --release

# Verify
./target/release/gpredomics --help
```

The binary is at `target/release/gpredomics`. The sample dataset is in `samples/Qin2014/`.

---

## 2. Data Format

gpredomics expects **tab-separated** files with features in rows:

**X.tsv** (features × samples):
```
            sample1   sample2   sample3   ...
msp_0001    0.023     0.001     0.045     ...
msp_0002    0.000     0.012     0.003     ...
```

**y.tsv** (class labels):
```
sample1   1
sample2   0
sample3   1
```

- Class `0` = control (e.g., healthy)
- Class `1` = case (e.g., disease)
- Class `2` = unknown (excluded from training)

The Qin2014 dataset:
- `Xtrain.tsv`: 1,980 features × 180 samples
- `Ytrain.tsv`: 180 labels (97 class 0, 83 class 1)
- `Xtest.tsv`: 1,980 features × 30 samples
- `Ytest.tsv`: 30 labels (independent test set)

---

## 3. Your First Run — Beam Search

Create `param.yaml`:

```yaml
general:
  seed: 42
  algo: beam
  language: ter         # ternary: coefficients {-1, 0, +1}
  data_type: prev       # prevalence: presence/absence
  fit: auc              # optimize for AUC
  thread_number: 4

data:
  X: samples/Qin2014/Xtrain.tsv
  y: samples/Qin2014/Ytrain.tsv
  Xtest: samples/Qin2014/Xtest.tsv
  ytest: samples/Qin2014/Ytest.tsv
  features_in_rows: true
  feature_selection_method: wilcoxon
  feature_minimal_prevalence_pct: 10
  feature_maximal_adj_pvalue: 0.05

beam:
  method: LimitedExhaustive
  kmin: 1
  kmax: 10
```

Run:

```bash
./target/release/gpredomics --config param.yaml
```

### Understanding the Output

```
INFO [gpredomics::data] 340 features selected
```
From 1,980 features, 340 pass the prevalence (≥10%) and significance (p < 0.05) filters.

```
INFO [gpredomics::beam] Beam algorithm (LimitedExhaustive mode) computed 9 generations in 395ms
```
Beam explored all combinations from k=1 to k=10 in under a second.

```
FBM mean (n=23) - AUC 0.936/0.923 | accuracy 0.884/0.854 | sensitivity 0.842/0.875 | specificity 0.932/0.832
```
The **Family of Best Models** (FBM) contains 23 models whose AUC confidence intervals overlap. Metrics are shown as **Train/Test**. The FBM mean test AUC is **0.923**.

```
Model #1 Ternary:Prevalence [k=8] [fit:0.939] AUC 0.939/0.947
  Class 1: (msp_0610⁰ + msp_0844c⁰ + msp_0884⁰ + msp_1127⁰ + msp_1325⁰)
         - (msp_0063⁰ + msp_0151⁰ + msp_0478⁰) ≥ 1
```
The best model uses **8 features**. It predicts cirrhosis when the sum of 5 positive features minus 3 negative features exceeds 1. The `⁰` superscript indicates prevalence data type. This formula is directly interpretable by clinicians.

---

## 4. Trying Different Algorithms

### 4.1 Genetic Algorithm (GA) — Population-based evolution

```yaml
general:
  algo: ga
  language: bin,ter,ratio   # search across multiple languages
  data_type: raw,prev       # and multiple data types
  k_penalty: 0.0001         # penalize complexity

ga:
  population_size: 5000
  max_epochs: 200
```

The GA evolves 5,000 models over 200 generations using crossover and mutation. It searches across **multiple model languages** (binary, ternary, ratio) and **data types** (raw, prevalence) simultaneously — producing the most diverse FBM (240 models).

**Result**: Train 1.000 / Test 0.855, k=28, 12.5s

### 4.2 Simulated Annealing (SA) — Temperature-guided search

```yaml
general:
  algo: sa
  k_penalty: 0.0001

sa:
  initial_temperature: 1.0
  cooling_rate: 0.9995
  max_iterations: 10000
  k_min: 1
  k_max: 50
```

SA starts with a random model and makes small perturbations (add/remove/flip one feature). Worse solutions are accepted with decreasing probability as the temperature cools.

**Result**: Train 0.915 / Test 0.920, k=46, 0.6s — excellent test AUC!

### 4.3 LASSO — Mathematical baseline

```yaml
general:
  algo: lasso

lasso:
  alpha_min: 0.001
  alpha_max: 1.0
  n_alphas: 100
  l1_ratio: 1.0       # pure L1 (LASSO); 0.5 = Elastic Net
```

LASSO solves the optimal sparse linear model via coordinate descent. Continuous coefficients are converted to {-1, +1} for BTR models. The fastest algorithm and a strong mathematical baseline.

**Result**: Train 0.957 / Test 0.882, k=70, 0.1s

### 4.4 Iterated Local Search (ILS) — Fast greedy optimization

```yaml
general:
  algo: ils
  k_penalty: 0.0001

ils:
  max_iterations: 500
  perturbation_size: 3
  local_search_steps: 100
```

ILS alternates between greedy hill-climbing and random perturbation. Extremely fast.

**Result**: Train 0.996 / Test 0.880, k=38, 0.3s

### 4.5 Ant Colony Optimization (ACO) — Pheromone-guided construction

```yaml
general:
  algo: aco
  k_penalty: 0.0001

aco:
  n_ants: 500
  max_iterations: 200
```

Ants construct models feature-by-feature, guided by pheromone trails that encode collective experience.

**Result**: Train 0.986 / Test 0.800, k=55, 6.7s

### 4.6 MCMC — Bayesian posterior sampling

```yaml
general:
  algo: mcmc

mcmc:
  method: gibbs          # or "sbs" for backward selection
  n_iter: 5000
  n_burn: 2500
  p0: 0.1               # prior inclusion probability
```

Gibbs variable selection jointly samples feature inclusion and coefficients. Produces the sparsest models with posterior uncertainty estimates.

**Result (Gibbs)**: Train 0.914 / Test 0.679, k=14, 2.4s
**Result (SBS)**: Train 0.918 / Test 0.794, k=51, 3.2s

---

## 5. Algorithm Comparison

```
Algorithm      | Train  | Test   | k    | FBM  | Time
---------------|--------|--------|------|------|------
Beam           | 0.936  | 0.923  | 8    | 23   | 0.3s   ← best test AUC
SA             | 0.915  | 0.920  | 46   | 2    | 0.6s   ← close second
LASSO          | 0.957  | 0.882  | 70   | 1    | 0.1s   ← fastest
ILS            | 0.996  | 0.880  | 38   | 1    | 0.3s
GA             | 1.000  | 0.855  | 28   | 240  | 12.5s  ← most diverse
ACO            | 0.986  | 0.800  | 55   | 5    | 6.7s
MCMC (SBS)     | 0.918  | 0.794  | 51   | 25   | 3.2s
MCMC (Gibbs)   | 0.914  | 0.679  | 14   | 18   | 2.4s   ← sparsest
```

**How to choose:**
- **Best predictive performance**: Beam Search (small, exhaustive models)
- **Best speed/accuracy tradeoff**: SA or LASSO
- **Most model diversity**: GA (multi-language, 240 FBM models)
- **Mathematical baseline**: LASSO
- **Sparsest models**: MCMC Gibbs (k=14 with uncertainty)
- **Fastest**: LASSO (0.1s) or ILS (0.3s)

---

## 6. Cross-Validation

Add `cv: true` and configure folds:

```yaml
general:
  algo: beam
  cv: true

cv:
  outer_folds: 5         # 5-fold outer CV for performance estimation
  inner_folds: 5         # 5-fold inner CV for model selection
```

Cross-validation produces a more robust FBM by selecting models that perform consistently across folds. The output shows per-fold metrics and the inter-fold FBM.

---

## 7. Voting Ensemble

Enable voting to combine the FBM into a jury:

```yaml
voting:
  vote: true
  method: Majority       # or Consensus
  min_perf: 0.5          # minimum individual AUC for experts
  min_diversity: 5.0     # minimum Jaccard distance (%) between experts
  fbm_ci_alpha: 0.05     # confidence level for FBM selection
```

The **Jury** selects diverse, high-performing models from the FBM and combines their predictions through majority voting. Each expert votes independently, and the consensus prediction is often more robust than any single model.

### Expert Specialization

```yaml
voting:
  specialized: true
  specialized_sensitivity_threshold: 0.7
  specialized_specificity_threshold: 0.7
```

Experts can specialize: **PositiveSpecialists** (high sensitivity) vote only on positive predictions, **NegativeSpecialists** (high specificity) vote only on negative predictions, and **Balanced** experts vote on everything.

---

## 8. Feature Importance

```yaml
importance:
  compute_importance: true
  n_permutations_mda: 100    # permutations for Mean Decrease in Accuracy
```

gpredomics computes multiple importance types:

| Type | What it measures |
|------|-----------------|
| **MDA** | How much AUC drops when a feature is permuted |
| **Prevalence (Pop)** | % of FBM models containing the feature |
| **Prevalence (CV)** | % of CV folds where the feature appears |
| **Coefficient** | Mean absolute coefficient across models |

The output shows the top features ranked by importance:

```
Top features by importance:
  1. msp_1127  MDA: 0.082  Prevalence: 87%  Direction: +
  2. msp_0610  MDA: 0.071  Prevalence: 83%  Direction: +
  3. msp_0063  MDA: 0.058  Prevalence: 74%  Direction: -
  ...
```

---

## 9. Saving and Loading Experiments

### Save

```yaml
general:
  save_exp: results/my_experiment
```

This saves the full experiment (parameters, data, models, importance, voting) as a MessagePack binary (`results/my_experiment.msgpack`).

### Load and Evaluate on New Data

```bash
./target/release/gpredomics \
  --load results/my_experiment.msgpack \
  --x-test new_data/X.tsv \
  --y-test new_data/y.tsv
```

### Export CSV Report

```yaml
general:
  csv_report: true
  save_exp: results/my_experiment
```

Generates `results/my_experiment_report.csv` with all model metrics in a structured format for downstream analysis.

---

## 10. GPU Acceleration

```yaml
general:
  gpu: true

gpu:
  memory_policy: Adaptive   # Strict, Adaptive, or Performance
  max_total_memory_mb: 256
  fallback_to_cpu: true
```

GPU acceleration (Metal on macOS, Vulkan on Linux) speeds up fitness evaluation for large populations (GA, ACO). The `fallback_to_cpu: true` option ensures the analysis still runs if no GPU is available.

---

## 11. Complete Example — Full Pipeline

```yaml
general:
  seed: 42
  algo: ga
  language: ter
  data_type: prev
  fit: auc
  k_penalty: 0.0001
  thread_number: 4
  cv: true
  gpu: false
  save_exp: results/qin2014_full

data:
  X: samples/Qin2014/Xtrain.tsv
  y: samples/Qin2014/Ytrain.tsv
  Xtest: samples/Qin2014/Xtest.tsv
  ytest: samples/Qin2014/Ytest.tsv
  features_in_rows: true
  feature_selection_method: wilcoxon
  feature_minimal_prevalence_pct: 10
  feature_maximal_adj_pvalue: 0.05

ga:
  population_size: 5000
  max_epochs: 200

cv:
  outer_folds: 5

voting:
  vote: true
  method: Majority
  min_perf: 0.5
  min_diversity: 5.0

importance:
  compute_importance: true
  n_permutations_mda: 100
```

This runs the complete pipeline: feature selection → GA optimization → 5-fold CV → voting ensemble → feature importance → save experiment + CSV report.

---

## 12. Tips and Best Practices

### Choosing Parameters

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `k_penalty` | 0.0001 | Increase (0.001) for sparser models, decrease (0) for max AUC |
| `feature_minimal_prevalence_pct` | 10 | Lower for rare features, higher to reduce noise |
| `population_size` (GA) | 5000 | Larger = more diverse, slower. 1000 is fine for quick runs |
| `max_epochs` (GA) | 200 | Early stopping kicks in when best model doesn't improve |
| `n_ants` (ACO) | 100 | 500 for thorough search, 50 for quick runs |
| `cooling_rate` (SA) | 0.995 | Slower (0.9995) = more thorough, faster (0.99) = quicker |

### When to Use Each Algorithm

- **Exploratory analysis**: Start with **Beam** (fast, small models, best test AUC) or **LASSO** (instant baseline)
- **Production model**: Run **GA** with CV + voting for the most robust ensemble
- **Feature discovery**: Use **MCMC Gibbs** for posterior inclusion probabilities
- **Quick screening**: **ILS** or **SA** (sub-second results)
- **Algorithm comparison**: Run all 7 and compare (see Section 5)

### Reproducibility

gpredomics is fully deterministic with a fixed seed. The same `seed`, parameters, and data always produce identical results, regardless of platform or thread count.

---

## References

1. Prifti, E. et al. (2020). Interpretable and accurate prediction models for metagenomics data. *GigaScience*, 9(3). [doi:10.1093/gigascience/giaa010](https://doi.org/10.1093/gigascience/giaa010)
2. Qin, N. et al. (2014). Alterations of the human gut microbiome in liver cirrhosis. *Nature*, 513(7516). [doi:10.1038/nature13568](https://doi.org/10.1038/nature13568)
