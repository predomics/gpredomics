# Using Gpredomics

This guide covers how to run gpredomics, manage experiments, and interpret results.

## Running Gpredomics

### Basic execution

From the directory containing your `param.yaml`:

```bash
cargo run --release
```

Or with a custom configuration:

```bash
cargo run --release -- --config my_config.yaml
```

The tool will load parameters, run the selected algorithm ([GA](ga.md), [Beam](beam.md), [ACO](aco.md), or MCMC), display results, and optionally save the experiment.

### Command-line options

**Training mode** (default):
```bash
gpredomics --config <PATH>
```
Runs a new experiment with the specified configuration file.

**Display mode**:
```bash
gpredomics --load <EXPERIMENT_FILE>
```
Loads and displays results from a saved experiment.

**Evaluation mode**:
```bash
gpredomics --load <EXPERIMENT_FILE> --evaluate --x-test <X_PATH> --y-test <Y_PATH>
```
Evaluates a saved experiment on new test data.

**Export parameters**:
```bash
gpredomics --load <EXPERIMENT_FILE> --export-params <OUTPUT_YAML>
```
Extracts the parameter configuration from a saved experiment.

**CSV performance report**:
```bash
gpredomics --config param.yaml --csv-report
```
Exports a `<timestamp>_csvr.csv` file containing performance metrics for the best model, the Family of Best Models (FBM, averaged), and the voting jury (if enabled). Can also be activated in `param.yaml`:

```yaml
general:
  csv_report: true
```

The CSV includes all classification metrics (AUC, fit, accuracy, sensitivity, specificity, F1, MCC, PPV, NPV, G-mean, rejection rate) for both train and test, plus all experiment parameters as individual named columns. If the file already exists with a matching header, new rows are appended.

### Algorithm selection

The `general.algo` parameter selects the optimization algorithm. Available algorithms:

- **ga** — [Genetic Algorithm](ga.md) (default)
- **beam** — [Beam Search](beam.md)
- **aco** — [Ant Colony Optimization](aco.md)
- **mcmc** — Markov Chain Monte Carlo

Example configuration for ACO:

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
```

### Signal handling

Press `Ctrl+C` once for a graceful stop—the current epoch completes and results are saved. Press again to force exit.

You can also send process signals:
```bash
kill -1 <PID>   # SIGHUP: graceful stop
kill -15 <PID>  # SIGTERM: graceful stop
```

## Managing Experiments

### Saving experiments

Set `general.save_exp` in your parameters to automatically save results:

```yaml
general:
  save_exp: "myexperiment.mp"
```

The file is saved with a timestamp prefix: `2025-12-15_14-30-45_myexperiment.mp`

### File formats

Three formats are supported:

- **MessagePack** (`.mp`, `.msgpack`): Recommended. Compact, preserves precision, R and Rust compatible.
- **JSON** (`.json`): Human-readable but may lose decimal precision.
- **Bincode** (`.bin`, `.bincode`): Most compact, Rust-only.

If you provide a path without extension, gpredomics tries all formats when loading.

### What's in an experiment

A saved experiment contains:
- Training and test [data](data.md) (features, labels, annotations)
- Parameter configuration
- Final [population](population.md) of models
- Intermediate populations (if `keep_trace: true`)
- Cross-validation fold assignments (in CV mode)
- Feature importance (if computed)
- [Voting](voting.md) jury (if enabled)
- Execution metadata (version, timestamp, duration)

### Loading and evaluating

Load an experiment to view results:
```bash
gpredomics --load results/experiment_2025-12-15.mp
```

Evaluate on new data:
```bash
gpredomics --load results/experiment_2025-12-15.mp \
           --evaluate \
           --x-test new_data/Xtest.tsv \
           --y-test new_data/ytest.tsv
```

This computes performance metrics on the new dataset using the trained models.

## Reading Results

### Terminal output structure

Results are displayed in sections:

**Header**: Experiment ID, gpredomics version, algorithm used, execution time

**Final population**: Best [individuals](individual.md) ranked by performance
  - Rank, number of features ($k$)
  - Language and data type
  - Fit (AUC, accuracy, sensitivity, specificity)
  - Feature composition

**Feature importance** (if enabled): Top contributing features with their importance scores and scope (individual, population, or cross-fold)

**Voting analysis** (if enabled): Jury composition, voting method, ensemble metrics, and per-sample predictions

*Last updated: v0.8.3*