# Simulated Annealing

Simulated Annealing (SA) is a single-solution metaheuristic inspired by the annealing process in metallurgy. Unlike population-based methods (GA, ACO), SA maintains and refines a single model by proposing small modifications and accepting or rejecting them based on a temperature-controlled probability.

## Overview

```
       High T (exploration)              Low T (exploitation)
    ┌─────────────────────┐          ┌─────────────────────┐
    │  ●                  │          │                     │
    │     ●   ●           │          │                     │
    │  ●         ●        │          │          ★          │
    │        ●      ●     │          │                     │
    │     ●     ●         │          │                     │
    └─────────────────────┘          └─────────────────────┘
      Accepts worse solutions          Converges to optimum
      to escape local optima           (only improvements)
```

The algorithm:
1. **Initialize** a random model
2. **Propose** a small modification (add/remove/swap feature, flip sign)
3. **Evaluate** the new model's fitness
4. **Accept** if better; if worse, accept with probability `e^(-ΔE/T)`
5. **Cool** the temperature: `T *= cooling_rate`
6. **Repeat** until temperature drops below minimum or max iterations reached

## Key Concepts

### Metropolis Acceptance Criterion

The probability of accepting a worse solution is:

```
P(accept) = e^(-(fit_current - fit_neighbor) / T)
```

- At **high temperature**: P ≈ 1 — almost all moves accepted (random walk)
- At **low temperature**: P ≈ 0 — only improvements accepted (hill climbing)
- This gradual transition from exploration to exploitation avoids getting trapped in local optima

### Neighborhood Moves

Each iteration, one of four moves is randomly selected:

| Move | Effect | When |
|------|--------|------|
| **Add** | Insert a random feature from available set | k < k_max |
| **Remove** | Drop a random feature from model | k > k_min |
| **Swap** | Replace one feature with another | Always (when both possible) |
| **Flip sign** | Reverse coefficient sign (±1) | Ternary/Ratio/Pow2 only |

Each move produces a "neighbor" that differs from the current solution by exactly one feature change.

### Cooling Schedule

The temperature follows a geometric cooling schedule:

```
T(i+1) = T(i) × cooling_rate
```

| cooling_rate | Behavior | Iterations to reach 0.001 from 1.0 |
|-------------|----------|-------------------------------------|
| 0.999 | Slow cooling, thorough search | ~6900 |
| 0.995 | Moderate | ~1380 |
| 0.99 | Fast cooling, quick convergence | ~690 |

## Comparison with Other Algorithms

```
                    Population-based              Single-solution
                    ┌──────────────┐              ┌──────────────┐
  Constructive      │     ACO      │              │              │
                    │  (pheromone) │              │              │
                    └──────────────┘              └──────────────┘

  Recombinative     │      GA      │              │              │
                    │  (crossover) │              │              │
                    └──────────────┘              └──────────────┘

  Perturbative      │              │              │      SA      │
                    │              │              │  (annealing) │
                    └──────────────┘              └──────────────┘
```

| Aspect | GA | ACO | SA |
|--------|-----|-----|-----|
| **Search type** | Many solutions, recombine | Many solutions, construct | One solution, perturb |
| **Diversity** | Explicit (population) | Natural (pheromone) | Implicit (temperature) |
| **Parallelism** | High (population) | Medium (ants) | Low (sequential) |
| **Strengths** | Broad exploration | Feature discovery | Deep local refinement |
| **Weaknesses** | Premature convergence | Slow, large k | Only one model |

## Parameters

```yaml
sa:
  initial_temperature: 1.0     # Starting temperature
  cooling_rate: 0.999           # T *= this each iteration
  min_temperature: 0.001        # Stop when T drops below this
  max_iterations: 10000         # Maximum iterations
  snapshot_interval: 100        # Log progress every N iterations
  k_min: 1                     # Min features per model
  k_max: 50                    # Max features per model
```

### Tuning Guide

| Parameter | Effect of increasing | Recommended |
|-----------|---------------------|-------------|
| **initial_temperature** | More initial exploration | 0.5–2.0 |
| **cooling_rate** | Slower cooling, more thorough | 0.995–0.9999 |
| **max_iterations** | Longer search | 5000–50000 |
| **k_max** | Allows larger models | Dataset-dependent |

**Quick start**: The defaults work well for most datasets. For difficult problems, increase `max_iterations` and use `cooling_rate: 0.9995`.

## Benchmark

Qin2014 cirrhosis dataset (1980 features, 180 samples, ternary:prevalence):

| Method | Test AUC | FBM Test AUC | k | Time |
|--------|---------|-------------|---|------|
| **SA** | **0.911** | **0.892** | 48 | **0.3s** |
| GA | 0.791 | 0.788 | 48 | 0.5s |
| ACO | 0.802 | 0.834 | 52 | 7.1s |

SA found the best test AUC in the least time on this dataset. Results vary by dataset — SA excels when the search space has clear structure that local perturbations can exploit.

## When to Use SA

**Use SA when:**
- You want the fastest possible result
- The search space is smooth (nearby models have similar fitness)
- You need a single best model (not a diverse FBM)
- As a refinement step after GA/ACO (use their best model as SA initial solution)

**Don't use SA when:**
- You need FBM diversity (SA produces mostly one good model)
- The search space is very rugged (many isolated optima)
- You want to explore multiple languages/data types simultaneously

## Example

```yaml
general:
  algo: sa
  language: ter
  data_type: prev
  seed: 42
  fit: auc
  k_penalty: 0.001

sa:
  initial_temperature: 1.0
  cooling_rate: 0.999
  max_iterations: 10000
  k_min: 1
  k_max: 30
```

## References

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598), 671–680.
- Černý, V. (1985). Thermodynamical approach to the traveling salesman problem: An efficient simulation algorithm. *Journal of Optimization Theory and Applications*, 45(1), 41–51.
- Metropolis, N., et al. (1953). Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 21(6), 1087–1092.

*Last updated: v0.8.3*
