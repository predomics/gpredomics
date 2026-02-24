# Genetic Algorithm

The genetic algorithm (GA) is one of the three main optimization methods available in Gpredomics, alongside the Beam search and MCMC approaches. It is particularly suited for exploring large feature spaces and discovering diverse, high-performing models through evolutionary processes.

## Overview

The GA implemented in Gpredomics follows a classic evolutionary strategy:
1. **Initialization**: Generate a diverse initial population of models
2. **Evaluation**: Compute fitness scores for all individuals
3. **Selection**: Choose parents based on fitness and diversity criteria
4. **Reproduction**: Create offspring through crossover and mutation
5. **Iteration**: Repeat 2-4 until convergence or stopping criteria are met

This approach is well-suited for feature selection and model identification in high-dimensional spaces where exhaustive search is computationally prohibitive. The GA naturally balances exploration (discovering new model structures) and exploitation (refining promising candidates).

## Key Concepts

### Population Structure

The population is divided into ecological **niches** based on model language (Binary, Ternary, Ratio, Pow2) and data type (raw, prev, log). This ensures that all individual types are represented throughout evolution, preventing premature convergence to a single approach. See [population.md](population.md) for details on niche selection and diversity management.

### Population Initialization

The initial population is generated following a stratified random approach to ensure diversity across all model types:

1. **Stratification by niche**: The target population size is divided equally among all combinations of language (Binary, Ternary, Ratio, Pow2) and data type (raw, prevalence, log). Each niche generates a sub-population independently.

2. **Random feature selection**: For each individual, a random number `k` of features is first chosen uniformly between `k_min` and `k_max` (or between 1 and the total number of available features if bounds are not specified).

3. **Feature sampling**: Once `k` is determined, exactly `k` features are randomly selected from the available feature set:
   - **Uniform sampling** (default): Each feature has an equal probability of being selected
   - **Weighted sampling** (if prior weights are provided): Features are sampled with probabilities proportional to their `prior_weight` values, allowing domain knowledge to guide initialization

4. **Coefficient assignment**: Each selected feature receives a coefficient based on the model language and its feature class:
   - **Binary**: All coefficients are set to +1
   - **Ternary/Ratio**: Coefficients are set to +1 if the feature class is positive (numerator), -1 if negative (denominator)
   - **Pow2**: Coefficients are initialized to +4 or -4 (depending on feature class), allowing subsequent doubling/halving during mutation

5. **Quality control**: After generation, individuals are filtered to remove:
   - **Stillborn models**: Models that cannot discriminate between classes (e.g., ternary/ratio with all coefficients of the same sign, or empty models with k=0)
   - **Out-of-bounds models**: Models with k outside the specified range
   - **Duplicate models**: Identical individuals (same feature set and coefficients) are removed based on hash comparison

If stillborn or out-of-bounds individuals are detected, the generation process is repeated for that niche until the target sub-population size is reached with valid individuals only.

### Evolutionary Operators

#### Parent Selection

Three complementary selection mechanisms work together to choose parents for the next generation:

- **Elite selection** (`select_elite_pct`): The top-performing models are automatically retained as parents. This ensures that the best solutions are never lost and provides strong selection pressure. Lower values (e.g., 2%) create more elitist populations that converge faster but may lose diversity.

- **Niche selection** (`select_niche_pct`): For each combination of language and data type, the best models are selected independently. This preserves diversity and ensures all model types compete fairly. For example, with 4 niches and `select_niche_pct: 20`, approximately 5% of the population is selected per niche.

- **Random selection** (`select_random_pct`): A proportion of parents is chosen randomly from the remaining population, split equally among all language/data_type combinations. This introduces genetic diversity and prevents the algorithm from getting stuck in local optima.

Note: The same individual may be selected by multiple mechanisms (e.g., the overall best is also the best in its niche). The actual number of parents may exceed the sum of the percentages due to overlapping selections. Duplicate removal occurs later, after crossover and mutation, when the new population is formed.

#### Crossover

Crossover creates new individuals (children) by combining features from two randomly selected parents. The process:

1. Two parents are chosen from the parent pool
2. One parent is designated as the "main parent" (determines language and data type)
3. For each feature present in either parent:
   - A parent is randomly selected (50% probability for parent 1, 50% for parent 2)
   - If the selected parent possesses that feature, it is inherited by the child
   - If the inherited feature's coefficient comes from a different language than the child's language, the coefficient is converted appropriately
   - Features not possessed by the randomly selected parent are not inherited
4. The child's `k` (feature count) and parent hashes are recorded

This mechanism allows beneficial feature combinations to propagate while maintaining valid model structures.

##### Inter-language Coefficient Conversion

When a child inherits features from parents of different languages, coefficients must be converted to maintain semantic consistency:

**Conversion rules**:
- **To Binary**: All coefficients become +1 (sign information is lost)
- **To Ternary/Ratio** (from Pow2): Coefficients become +1 if positive, -1 if negative (magnitude information is lost)
- **To Ternary/Ratio** (from Binary): No conversion needed (Binary +1 is compatible with Ternary/Ratio ±1)
- **To Pow2** (from Binary/Ternary/Ratio): No conversion needed (±1 coefficients are valid Pow2 values)

**Conversion necessity**:
- Same language → same language: **Never converts**
- Binary parent → any child: **Never converts** (Binary +1 is universally compatible)
- Ternary/Ratio parent → any child: **Never converts** (±1 coefficients are universally compatible)
- Any parent → Binary child: **Always converts** (except Binary→Binary, already covered by rule 1)
- Pow2 parent → Ternary/Ratio child: **Always converts** (must reduce to ±1)

#### Mutation

After crossover, a percentage of children (`mutated_children_pct`) undergo mutation. Individuals to mutate are randomly sampled without replacement from all children. For each selected child, a subset of features (`mutated_features_pct` of all available features) is randomly sampled for potential mutation.

The mutation behavior depends on the model language:

**Binary mutation**: Each selected feature is first removed if present. Then, with probability `mutation_non_null_chance_pct / 100`, it is added with coefficient +1.

**Ternary and Ratio mutation**: Each selected feature is first removed if present, then:
- Added with coefficient +1 (probability `mutation_non_null_chance_pct / 200`)
- Added with coefficient -1 (probability `mutation_non_null_chance_pct / 200`)
- Remains absent (remaining probability)

Note: The total probability of adding the feature is `mutation_non_null_chance_pct / 100`.

**Pow2 mutation**: More complex, allowing coefficient doubling and halving in addition to addition/removal. Each feature can be:
- Set to +1 (probability p/2)
- Set to -1 (probability p/2)
- Doubled (probability p, if current value ≠ 0 and |value| < 64)
- Halved (probability p, if current value ≠ 0 and |value| > 1)
- Removed (remaining probability)

where p = `mutation_non_null_chance_pct / 100`.

Mutation introduces random variation essential for exploring new regions of the feature space and escaping local optima.

#### Stillborn and Out-of-Bounds Removal

After crossover and mutation, invalid individuals are filtered out in two steps:

1. **Stillborn removal**: An individual is considered stillborn if:
   - For ternary/ratio languages: all coefficients have the same sign (model cannot discriminate between classes)
   - For binary/pow2 languages: no features are present (k = 0)

2. **Out-of-bounds removal**: An individual is removed if its number of features (k) is outside the allowed range:
   - Too small: k < `k_min` (when `k_min` > 0)
   - Too large: k > `k_max` (when `k_max` > 0)

Invalid children are discarded, and the crossover/mutation process continues until the target population size is reached. This ensures that even when mutations create models outside the allowed complexity range, they are filtered out to maintain models within the specified bounds.

### Fitness Evaluation

Each individual's fitness is evaluated on the data (or on cross-validation folds if inner CV is enabled or on a subset if random sampling is enabled). The fitness combines a performance metric (AUC, sensitivity, MCC, etc.) with penalties for model complexity and overfitting. See [individual.md](individual.md) for details on fitness computation.

### Stopping Criteria

The algorithm terminates when any of the following conditions is met:

1. **Maximum epochs reached**: The number of generations exceeds `max_epochs`
2. **Best model age limit**: After `min_epochs`, if the best model hasn't changed for `max_age_best_model` generations
3. **Manual interruption**: User sends a stop signal (Ctrl+C or SIGHUP/SIGTERM)

Upon termination, the final population's metrics are computed on the full training dataset (if random sampling was used during optimization).

### Diversity Management

When enabled (`forced_diversity_pct` > 0), forced diversity prevents the population from becoming too homogeneous. At every `forced_diversity_epochs` generation, the parent pool is filtered to maintain diversity:

1. Parents are grouped by **language family**: Linear models (Binary, Ternary, Pow2) form one group, Ratio models another (when `considere_niche` is true)
2. Within each group, individuals are sorted by fitness (best first)
3. The best individual is always retained
4. For each subsequent candidate, it is retained only if its signed Jaccard dissimilarity with **all already-selected individuals** exceeds `forced_diversity_pct / 100`

This ensures that retained parents are sufficiently different from each other, preventing premature convergence while preserving the best solutions.

### Random Sampling

Random sampling can reduce overfitting by preventing models from memorizing the full training set. A random subset of samples, `random_sampling_pct`, is created each `random_sampling_epochs` and is used to compute the fit.

Note that:
- Final metrics are computed on the full dataset
- GPU acceleration is disabled when random sampling is active (incompatible with pre-allocated buffers)
- Displayed fit values reflect performance on sampled data, not the full set

## Reproducibility

All randomness in the GA is controlled by the `seed` parameter (located in the `general` section of `param.yaml`). This seed initializes a ChaCha8 pseudo-random number generator (RNG) that governs all stochastic processes:

- Population initialization (feature selection and coefficient assignment)
- Parent selection (random selection component)
- Crossover (parent pairing and feature inheritance)
- Mutation (feature and individual selection, coefficient changes)
- Random sampling (if enabled)

Using the same seed with identical parameters and data guarantees perfectly reproducible results across runs, even when GPU acceleration is enabled (the RNG is initialized **after** GPU setup to ensure deterministic behavior).

*Last updated: v0.7.7*