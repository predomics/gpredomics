# Grid Search

Grid search allows you to sweep over multiple parameter values in a single YAML file. Gpredomics loads data once and runs every combination serially, producing independent results for each.

## Quick Start

Add a `grid` section at the top level of your `param.yaml`:

```yaml
general:
  algo: ga
  seed: 42
  k_penalty: 0.0001          # default for non-grid runs

ga:
  population_size: 5000

grid:
  - path: "general.k_penalty"
    values: [0.0001, 0.001, 0.01]
  - path: "ga.population_size"
    values: [1000, 5000]
```

This produces 3 x 2 = **6 runs**, one for each combination of `k_penalty` and `population_size`. The base values in `general` / `ga` are overridden by each grid combination.

## YAML Format

Each grid axis is an entry with two fields:

| Field    | Type   | Description |
|----------|--------|-------------|
| `path`   | string | Dotted path to the parameter: `section.parameter` (e.g., `ga.population_size`, `general.seed`) |
| `values` | list   | List of values to sweep. Must be valid for the target parameter type. |

```yaml
grid:
  - path: "general.k_penalty"
    values: [0.0001, 0.001, 0.01]
  - path: "ga.max_epochs"
    values: [50, 100, 200]
  - path: "general.seed"
    values: [42, 123, 999]
```

The Cartesian product of all axes is computed: if the three axes above have 3, 3, and 3 values respectively, that yields 27 runs.

## How It Works

1. The YAML file is parsed as a raw structure.
2. The `grid` section is extracted and removed.
3. All combinations of axis values are computed (Cartesian product).
4. For each combination, the base YAML is patched at the specified paths, then deserialized into a validated `Param`.
5. Data (training and test) is loaded **once** from the paths in the `data` section.
6. Each combination is run serially through the same pipeline as a normal run (including cross-validation, voting, importance, etc.).
7. Results are displayed and saved independently for each run.

## Valid Paths

Any parameter accessible in `param.yaml` can be used. The path follows the YAML structure with a dot separator:

- `general.seed`, `general.algo`, `general.k_penalty`, `general.fit`, ...
- `ga.population_size`, `ga.max_epochs`, `ga.mutated_features_pct`, ...
- `beam.k_stop`, `beam.best_models_criterion`, ...
- `mcmc.n_iter`, `mcmc.lambda`, `mcmc.p0`, ...
- `aco.n_ants`, `aco.rho`, `aco.alpha`, ...
- `sa.initial_temperature`, `sa.cooling_rate`, ...
- `lasso.l1_ratio`, `lasso.n_alphas`, ...
- `data.holdout_ratio`, `data.feature_minimal_prevalence_pct`, ...
- `cv.outer_folds`, `cv.overfit_penalty`, ...

## Output

### Grid index CSV

A `<timestamp>_grid.csv` file is written at the start of the grid, listing every combination:

```csv
run,tag,general.k_penalty,ga.population_size
1,grid_1in6,0.0001,1000
2,grid_2in6,0.0001,5000
3,grid_3in6,0.001,1000
4,grid_4in6,0.001,5000
5,grid_5in6,0.01,1000
6,grid_6in6,0.01,5000
```

This makes it easy to cross-reference results with the parameter values used.

### Run tagging

Each grid run is tagged `grid_NinM` (e.g., `grid_1in6`, `grid_2in6`, ...). This tag appears in:

- Log messages: `Grid run 1/6 [grid_1in6]`
- Saved experiment filenames (if `save_exp` is set): `2026-04-01_12-00-00_grid_1in6_exp.mp`
- CSV report filenames (if `csv_report` is enabled): `2026-04-01_12-00-00_grid_1in6_csvr.csv`

## No Grid Section

If no `grid` section is present, Gpredomics behaves exactly as before: a single run with the parameters as written. There is no `grid: true` flag to set; the mere presence of the `grid` list activates grid mode.

## Signal Handling

You can interrupt a grid search with Ctrl+C. The current run will finish (or be interrupted at the next iteration), and remaining grid combinations will be skipped.

## Example: Comparing Algorithms

```yaml
general:
  seed: 42
  fit: auc

data:
  X: "samples/Qin2014/Xtrain.tsv"
  y: "samples/Qin2014/Ytrain.tsv"
  Xtest: "samples/Qin2014/Xtest.tsv"
  ytest: "samples/Qin2014/Ytest.tsv"

grid:
  - path: "general.algo"
    values: ["ga", "beam", "sa"]
  - path: "general.k_penalty"
    values: [0.0001, 0.001]
```

This runs 6 experiments: GA, beam, and SA, each with two penalty values, all on the same loaded dataset.
