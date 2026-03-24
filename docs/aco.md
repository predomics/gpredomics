# Ant Colony Optimization

Ant Colony Optimization (ACO) is one of the optimization methods available in Gpredomics, alongside the Genetic Algorithm, Beam Search, and MCMC approaches. It is a constructive metaheuristic inspired by the foraging behavior of ants, where solutions are built feature-by-feature guided by collective pheromone memory and heuristic information.

## Overview

The ACO implemented in Gpredomics follows a Max-Min Ant System (MMAS) variant:
1. **Initialization**: Initialize pheromone trails to maximum values on all features
2. **Construction**: Each ant builds a model by selecting features probabilistically
3. **Evaluation**: Compute fitness scores for all constructed models
4. **Local search**: Greedily remove features from top models if it improves penalized fitness
5. **Pheromone update**: Evaporate pheromone, then deposit on features used by best models
6. **Iteration**: Repeat 2–5 until convergence or stopping criteria are met

### Key differences from GA

| Aspect | GA | ACO |
|--------|-----|-----|
| **Approach** | Recombinative (crossover + mutation) | Constructive (build from scratch) |
| **Memory** | Implicit (population carries good features) | Explicit (pheromone matrix) |
| **Diversity** | Requires forced diversity mechanisms | Natural (probabilistic construction) |
| **Model size** | Controlled by mutation balance | Controlled by k_max + local search |
| **Speed** | Faster per generation (parallel fitness) | Slower but more diverse exploration |

## Key Concepts

### Pheromone Matrix

The pheromone matrix stores learned desirability for each (feature, coefficient sign) pair:

- **τ[feature][+1]**: Pheromone for using this feature with a positive coefficient
- **τ[feature][-1]**: Pheromone for using this feature with a negative coefficient

High pheromone means many good models have used this feature with this sign. The pheromone matrix is the collective memory of the colony — it encodes which features are consistently useful across diverse solutions.

#### MMAS Bounds

The Max-Min Ant System clamps pheromone values to [τ_min, τ_max]:
- **τ_max** prevents any single feature from dominating (avoids premature convergence)
- **τ_min** ensures every feature retains a minimum selection probability (maintains exploration)

Pheromone is initialized to τ_max at the start, encouraging maximum exploration in early iterations.

### Heuristic Information (η)

Each feature has a heuristic desirability based on its statistical significance from feature selection:

- For **p-value methods** (Student's t, Wilcoxon): `η[i] = 1 / (p_value + ε)` — lower p-value = higher desirability
- For **Bayesian Fisher**: `η[i] = |log_bayes_factor|` — higher factor = higher desirability

This biases ants toward statistically significant features without excluding others entirely.

### Solution Construction

Each ant constructs a complete model through the following steps:

1. **Choose model size k**: Uniformly sampled from [k_min, k_max]

2. **Select k features** using roulette wheel selection (without replacement):
   - Selection probability: `P(feature i) = τ[i]^α × η[i]^β / Σ(τ[j]^α × η[j]^β)`
   - α controls pheromone influence (higher = more exploitation)
   - β controls heuristic influence (higher = more bias toward significant features)
   - Features are removed from the candidate set after selection (no duplicates)

3. **Assign coefficients** based on language and pheromone:
   - **Binary**: Always +1
   - **Ternary/Ratio**: Sign chosen probabilistically based on pheromone `τ[i][+1]` vs `τ[i][-1]`, biased by the feature's class association
   - **Pow2**: Sign from class association, magnitude randomly sampled as 2^p where p ∈ {0, 1, 2, 3}

4. **Quality control**: Same as GA — remove stillborn (invalid coefficient patterns) and out-of-bounds (k outside range) individuals

### Local Search

After construction and fitness evaluation, the top 5 models undergo a greedy local search:

1. For each feature in the model:
   - Temporarily remove the feature
   - Recompute AUC
   - Compare penalized fitness: `fit - k_penalty × k`
   - If the sparser model has equal or better penalized fitness, keep the removal
2. This reduces model complexity without sacrificing performance

Local search is what allows ACO to produce models as sparse as GA despite the constructive approach's tendency toward larger models.

### Pheromone Update

After all ants have been evaluated:

1. **Evaporation**: All pheromone values decay by factor (1 - ρ):
   - `τ[i] = max(τ_min, τ[i] × (1 - ρ))`
   - Higher ρ = faster forgetting = more exploration
   - Lower ρ = longer memory = more exploitation

2. **Iteration-best deposit**: The best ant of the current iteration deposits pheromone on its features:
   - `τ[feature][sign] += fitness / k`
   - Normalized by k to avoid biasing toward larger models

3. **Global-best deposit**: The best-ever ant deposits additional pheromone:
   - `τ[feature][sign] += elite_weight × fitness / k`
   - `elite_weight` controls how strongly the global best influences future construction

4. **Clamping**: All values are clamped to [τ_min, τ_max]

### Stopping Criteria

The algorithm terminates when any of the following conditions is met:

1. **Maximum iterations reached**: The number of iterations exceeds `max_iterations`
2. **Best model age limit**: After `min_iterations`, if the best model hasn't improved for `max_age_best_model` iterations
3. **Manual interruption**: User sends a stop signal (Ctrl+C or SIGHUP/SIGTERM)

## Parameters

```yaml
aco:
  n_ants: 100               # Number of ants (models constructed) per iteration
  max_iterations: 200        # Maximum number of iterations
  min_iterations: 10         # Minimum iterations before early stopping is allowed
  alpha: 1.0                 # Pheromone importance exponent (α)
  beta: 2.0                  # Heuristic importance exponent (β)
  rho: 0.1                   # Pheromone evaporation rate (ρ ∈ [0,1])
  tau_min: 0.01              # Minimum pheromone value (MMAS lower bound)
  tau_max: 1.0               # Maximum pheromone value (MMAS upper bound)
  elite_weight: 2.0          # Extra pheromone deposit weight for global-best ant
  k_min: 1                   # Minimum number of features per model
  k_max: 200                 # Maximum number of features per model (0 = no limit)
  max_age_best_model: 10     # Early stopping: max iterations without improvement
```

### Parameter Tuning Guide

| Parameter | Effect of increasing | Recommended range |
|-----------|---------------------|-------------------|
| **n_ants** | Better exploration, slower per iteration | 50–500 |
| **alpha** | More exploitation (follow pheromone) | 0.5–2.0 |
| **beta** | More bias toward significant features | 1.0–5.0 |
| **rho** | Faster forgetting, more exploration | 0.05–0.3 |
| **tau_min** | Higher floor prevents feature extinction | 0.001–0.1 |
| **elite_weight** | Stronger influence of global best | 1.0–5.0 |
| **k_max** | Allows larger models (use with k_penalty) | Dataset-dependent |

**Typical starting configuration**: The defaults work well for most metagenomics datasets. For very large feature spaces (>5000 features), increase `n_ants` to 200–500 and `beta` to 3.0 to strengthen the heuristic guidance.

## Parallelization

Ant construction is parallelized via rayon:
- Each ant receives a deterministic seed derived from the master RNG
- Construction is independent across ants (no shared mutable state)
- Fitness evaluation reuses the existing parallel population evaluation

This ensures both performance and reproducibility.

## Reproducibility

All randomness is controlled by the `seed` parameter. The master RNG generates per-ant seeds deterministically, so results are identical across runs with the same seed, regardless of thread count.

## When to Use ACO vs GA

**Use ACO when:**
- You want to explore a wider variety of model structures
- FBM diversity is important (ACO naturally produces more diverse models)
- You suspect the GA is converging prematurely to a local optimum
- You want an alternative perspective on feature importance (via the pheromone matrix)

**Use GA when:**
- Speed is the priority (GA is typically 2–5x faster)
- You need very sparse models (GA's mutation pressure naturally reduces k)
- You have a good initial population to start from
- Cross-validation mode is needed (ACO + CV is supported but less tested)

**Use both together**: Run GA and ACO on the same data, then compare FBMs. Features that appear in both algorithms' top models are strong candidates. The voting ensemble can combine experts from both runs.

## Example

```yaml
general:
  algo: aco
  language: ter
  data_type: prev
  seed: 42
  fit: auc
  k_penalty: 0.001

aco:
  n_ants: 200
  max_iterations: 100
  min_iterations: 20
  alpha: 1.0
  beta: 2.0
  rho: 0.1
  k_min: 1
  k_max: 50
  max_age_best_model: 15

voting:
  vote: true
  fbm_ci_alpha: 0.05
```

## References

- Dorigo, M. & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
- Stützle, T. & Hoos, H. (2000). Max-Min Ant System. *Future Generation Computer Systems*, 16(8), 889–914.
- Al-Ani, A. (2005). Feature subset selection using ant colony optimization. *Expert Systems with Applications*, 36(8), 11198–11204.

*Last updated: v0.8.3*
