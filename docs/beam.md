# Beam Search

Beam Search is a systematic, level-by-level exploration method available in Gpredomics. Unlike stochastic algorithms (GA, ACO, SA), Beam Search deterministically enumerates feature combinations at increasing model sizes, pruning the search space at each level by retaining only the best-performing models. This guarantees thorough coverage of small models and produces highly interpretable results.

## Overview

```
  Features: [f1, f2, f3, f4, f5, ...]

  k=1    (f1)  (f2)  (f3)  (f4)  (f5)   <- all 1-feature models
           |     |     |     |     |
           v     v     v     v     v
         [  Evaluate & select FBM  ]
                    |
                    v
  k=2  (f1,f2) (f1,f3) (f2,f3) ...      <- expand from best k=1
           |       |       |
           v       v       v
         [  Evaluate & select FBM  ]
                    |
                    v
  k=3  (f1,f2,f3) (f1,f2,f5) ...        <- expand from best k=2
           |           |
           v           v
         [  Evaluate & select FBM  ]
                    |
                    v
         ...continue to k_stop...
```

The algorithm:
1. **Initialize** all feature combinations of size `k_start` (typically 1)
2. **Evaluate** every model's fitness (AUC, MCC, etc.) on the dataset
3. **Select** the Family of Best Models (FBM) at the current level using a confidence-interval criterion or top percentage
4. **Expand** the best models to size k+1 using one of two methods:
   - *LimitedExhaustive*: generate all C(n, k+1) combinations from selected features
   - *ParallelForward*: extend each best model by adding one feature
5. **Prune** features if the combinatorial space exceeds `max_nb_of_models`
6. **Repeat** steps 2--5 until `k_stop` is reached or no valid models remain

## Key Concepts

### Beam Methods

Gpredomics implements two beam search strategies:

| Method | Construction | Scalability | Model diversity |
|--------|-------------|-------------|-----------------|
| **LimitedExhaustive** | All C(n, k) combinations from selected features | Smaller feature sets (<100 features after pruning) | Maximum -- every valid combination tested |
| **ParallelForward** | Extend each best model by one feature | Large feature sets | Lower -- inherits parent structure |

#### LimitedExhaustive (combinatorial)

At each level k, generates **all** k-combinations from the features that appear in the FBM of the previous level. This is equivalent to the `terbeam` approach in legacy Predomics. It guarantees that no promising combination is missed within the retained feature set, but the number of combinations grows as C(n, k), requiring pruning when too many features survive.

#### ParallelForward (incremental)

Takes the best models from level k-1 and extends each by adding one new feature from the retained set. This produces fewer candidates than exhaustive enumeration but scales better when the number of features is large. It tends to produce larger models since it inherits the structure of parent models.

### Beam Width and FBM Selection

At each level, the "beam width" is controlled by `best_models_criterion`:

- **If > 1.0** (e.g., 10.0): interpreted as a **percentage** -- keep the top N% of models. Features are then selected using frequency-based filtering (features appearing in at least 1% of best models, or in the top 10% of models).
- **If <= 1.0** (e.g., 0.05): interpreted as an **alpha level** for the FBM confidence interval -- keep all models whose fitness falls within the binomial CI of the best model at significance level alpha.

The FBM approach is more principled: it automatically adapts the beam width based on how clearly the best model separates from the rest, while percentage-based selection gives direct control.

### Feature Pruning

When the number of retained features would produce more combinations than `max_nb_of_models`, the algorithm prunes features intelligently:

1. **Compute** the maximum number of features n such that C(n, k) <= `max_nb_of_models`
2. **Rank** features by their statistical significance (from feature selection)
3. **Balance** positive and negative features (for ternary/ratio languages):
   - Split features by class association (positive vs negative coefficient)
   - Take approximately n/2 from each class, adjusting if one class has fewer features
4. **For binary language**: keep only positive-associated features (top n by significance)

This ensures that even with aggressive pruning, both directions of association are represented.

### Language Handling

Beam search handles BTR languages with special care:

- **k=1**: Ratio models are converted to binary (positive) or ternary (negative) since a ratio requires both numerator and denominator. Ternary models with only positive features are converted to binary.
- **k>1**: Ternary models must have at least one negative feature; ratio models must have both positive and negative features. Invalid models are rejected.
- **Coefficient signs** are determined by the feature's class association from the initial statistical test, not randomly assigned.

## Comparison with Other Algorithms

| Aspect | GA | ACO | SA | Beam |
|--------|-----|-----|-----|------|
| **Search type** | Stochastic, recombinative | Stochastic, constructive | Stochastic, perturbative | Deterministic, exhaustive |
| **Completeness** | No guarantee | No guarantee | No guarantee | Exhaustive within beam |
| **Best for k** | Medium to large | Medium to large | Any | Small (k < 15) |
| **Diversity** | Explicit (population) | Natural (pheromone) | Implicit (temperature) | Guaranteed (all combinations) |
| **Speed (small k)** | Moderate | Slow | Fast | Very fast |
| **Speed (large k)** | Fast | Moderate | Fast | Exponentially slow |
| **GPU support** | Yes | No | No | Yes |

## Parameters

```yaml
beam:
  method: LimitedExhaustive    # LimitedExhaustive or ParallelForward
  k_start: 1                   # Starting model size (first level)
  k_stop: 50                   # Maximum model size (stop level)
  best_models_criterion: 0.05  # FBM alpha (<= 1.0) or top percentage (> 1.0)
  fbm_ci_method: Binomial      # CI method for FBM selection
  max_nb_of_models: 500000     # Max combinations per level (0 = unlimited)
```

### Tuning Guide

| Parameter | Effect of increasing | Recommended |
|-----------|---------------------|-------------|
| **k_start** | Skip trivially small models | 1 (default) |
| **k_stop** | Explore larger models (exponential cost) | 5--30 depending on features |
| **best_models_criterion** | Wider beam (> 1.0: more %; <= 1.0: looser CI) | 0.05 or 10.0 |
| **max_nb_of_models** | More combinations tested, slower | 100000--1000000 |

**Quick start**: Use `LimitedExhaustive` with `max_nb_of_models: 500000` for datasets with <500 pre-selected features. Switch to `ParallelForward` for larger feature sets or when k_stop is large.

## When to Use Beam Search

**Use Beam when:**
- You want guaranteed coverage of all small models (k < 10--15)
- Interpretability is critical and you need the simplest possible model
- You want to compare all 1-feature, 2-feature, ... k-feature models systematically
- You need reproducible results without stochastic variation
- The feature space after pre-selection is manageable (<500 features)

**Don't use Beam when:**
- You need large models (k > 20) -- combinatorial explosion makes this impractical
- The feature space is very large and pre-selection cannot reduce it sufficiently
- You want to explore diverse model structures quickly (use GA or ACO instead)
- Runtime budget is tight and k_stop is large

## Example

```yaml
general:
  algo: beam
  language: ter
  data_type: prev
  seed: 42
  fit: auc
  k_penalty: 0.001

beam:
  method: LimitedExhaustive
  k_start: 1
  k_stop: 15
  best_models_criterion: 0.05
  max_nb_of_models: 500000
```

For a quick exploratory run with large feature sets:

```yaml
beam:
  method: ParallelForward
  k_start: 1
  k_stop: 30
  best_models_criterion: 10.0   # top 10% of models
  max_nb_of_models: 100000
```

## References

- Rubin, T. N. (2010). Feature Selection with Beam Search. *Proceedings of the NIPS Workshop on Feature Selection*.
- Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. (Chapter 3: beam search as a bounded-width breadth-first search)
- Lowerre, B. T. (1976). *The HARPY Speech Recognition System*. Ph.D. Thesis, Carnegie Mellon University. (original beam search)
- Prifti, E. et al. (2020). Interpretable and accurate prediction models for metagenomics data. *GigaScience*, 9(3), giaa010. (terbeam in legacy Predomics)

*Last updated: v0.9.0*
