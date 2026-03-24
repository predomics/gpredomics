# Iterated Local Search

Iterated Local Search (ILS) is a single-solution metaheuristic that alternates between local search and perturbation to efficiently explore the space of BTR models. Unlike Simulated Annealing which uses random acceptance, ILS always runs a full local search to a local optimum, then escapes via controlled perturbation.

## Overview

```
    ┌──────────────────────────────────────────────────────────┐
    │                   ILS Main Loop                          │
    │                                                          │
    │   ┌─────────┐    ┌────────────┐    ┌─────────────────┐   │
    │   │ Initial  │───▶│  Local     │───▶│  Local Optimum  │   │
    │   │ Solution │    │  Search    │    │  s*             │   │
    │   └─────────┘    └────────────┘    └────────┬────────┘   │
    │                                             │            │
    │                       ┌─────────────────────┘            │
    │                       ▼                                  │
    │               ┌──────────────┐                           │
    │               │  Accept /    │◀──────────────────┐       │
    │               │  Reject      │                   │       │
    │               └──────┬───────┘                   │       │
    │                      │                           │       │
    │                      ▼                           │       │
    │               ┌──────────────┐    ┌────────────┐ │       │
    │               │  Perturbate  │───▶│  Local     │─┘       │
    │               │  (kick)      │    │  Search    │         │
    │               └──────────────┘    └────────────┘         │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

The algorithm:
1. **Initialize** a random model
2. **Local search** — greedily improve by single-feature moves until no improvement is possible
3. **Perturbate** — apply random multi-feature changes to escape the local optimum
4. **Local search** again — descend to a new local optimum
5. **Accept** the new solution if it meets the acceptance criterion
6. **Repeat** steps 3–5 until the iteration budget is exhausted

## Key Concepts

### Local Search (Greedy Descent)

The local search phase explores the immediate neighborhood of the current solution by trying single-feature moves:

| Move | Effect | When |
|------|--------|------|
| **Add** | Insert a random feature from available set | k < k_max |
| **Remove** | Drop a feature from model | k > k_min |
| **Swap** | Replace one feature with another | Always (when both possible) |
| **Flip sign** | Reverse coefficient sign (±1) | Ternary/Ratio/Pow2 only |

At each step, the best improving move is applied. The search continues until no single move improves the penalized fitness — the solution is then at a local optimum.

### Perturbation (Kick)

The perturbation phase applies multiple random changes simultaneously to escape the current local optimum without completely destroying the solution structure:

- **Perturbation strength** controls how many features are changed (typically 2–5 moves)
- Too weak: local search returns to the same optimum
- Too strong: equivalent to random restart (loses accumulated structure)
- The right balance preserves good features while opening new search regions

### Acceptance Criterion

After perturbation and local search yield a new local optimum, ILS decides whether to continue from the new solution or revert:

- **Better-or-equal**: Accept the new solution only if its fitness is at least as good as the current best (default, most conservative)
- This ensures the search always operates from the best known local optimum while using perturbation to explore neighboring basins of attraction

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

  Perturbative      │              │              │   SA / ILS   │
                    │              │              │  (local srch)│
                    └──────────────┘              └──────────────┘
```

| Aspect | GA | ACO | SA | ILS |
|--------|-----|-----|-----|-----|
| **Search type** | Many solutions, recombine | Many solutions, construct | One solution, random walk | One solution, local optima |
| **Local search** | None (relies on selection) | Greedy pruning on top-5 | Implicit (low temperature) | Explicit (full descent) |
| **Escape mechanism** | Crossover + mutation | Pheromone evaporation | Temperature (accept worse) | Perturbation (kick) |
| **Strengths** | Broad exploration | Feature discovery | Smooth landscapes | Fast, deep local refinement |
| **Weaknesses** | Premature convergence | Slow, large k | No guaranteed local optimum | Less diversity |

## Parameters

```yaml
ils:
  max_iterations: 1000           # Total ILS iterations (perturbation cycles)
  ls_max_iterations: 100         # Max iterations within each local search phase
  perturbation_strength: 3       # Number of random moves per perturbation
  k_min: 1                       # Min features per model
  k_max: 50                      # Max features per model
  snapshot_interval: 100         # Log progress every N iterations
```

### Tuning Guide

| Parameter | Effect of increasing | Recommended range |
|-----------|---------------------|-------------------|
| **max_iterations** | More perturbation cycles, broader exploration | 500–5000 |
| **ls_max_iterations** | Deeper local search per cycle | 50–500 |
| **perturbation_strength** | Larger jumps, more exploration (risk losing structure) | 2–5 |
| **k_max** | Allows larger models | Dataset-dependent |

**Quick start**: The defaults work well for most datasets. For problems where SA gets stuck in local optima, try ILS with `perturbation_strength: 3` and `max_iterations: 2000`. ILS is typically the fastest algorithm to reach a good local optimum.

## Benchmark

Qin2014 cirrhosis dataset (1980 features, 180 samples, ternary:prevalence):

| Method | Test AUC | FBM Test AUC | k | Time |
|--------|---------|-------------|---|------|
| **ILS** | **0.813** | — | 22 | **0.05s** |
| SA | 0.911 | 0.892 | 48 | 0.3s |
| GA | 0.791 | 0.788 | 48 | 0.5s |
| ACO | 0.802 | 0.834 | 52 | 7.1s |

ILS finds a compact model (k=22) extremely quickly. While its test AUC is lower than SA on this dataset, the model is significantly sparser. ILS is the fastest algorithm and a strong choice when model simplicity is valued.

## When to Use ILS

**Use ILS when:**
- You want the fastest possible convergence to a local optimum
- Model sparsity is important (ILS local search naturally prunes features)
- The search space has well-defined local optima connected by perturbation paths
- You need a quick baseline before running slower population-based methods

**Don't use ILS when:**
- You need FBM diversity (ILS produces a single model)
- The search space is very flat (local search has few improving moves)
- You want broad global exploration (use GA or ACO instead)
- You suspect many equally good but structurally different solutions exist

## Example

```yaml
general:
  algo: ils
  language: ter
  data_type: prev
  seed: 42
  fit: auc
  k_penalty: 0.001

ils:
  max_iterations: 1000
  ls_max_iterations: 100
  perturbation_strength: 3
  k_min: 1
  k_max: 30
```

## References

- Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). Iterated Local Search. In F. Glover & G. A. Kochenberger (Eds.), *Handbook of Metaheuristics* (pp. 320–353). Springer.
- Stützle, T. (1998). *Local Search Algorithms for Combinatorial Problems: Analysis, Improvements, and New Applications*. PhD thesis, TU Darmstadt.

*Last updated: v0.9.0*
