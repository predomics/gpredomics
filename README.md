# gpredomics

gpredomics is a Rust rewrite of Predomics for learning interpretable BTR‑style models with discrete coefficients across binary, ternary, ratio, and pow2 languages on raw, prevalence, and log‑transformed data, with optional GPU acceleration via wgpu backends.

## Features

- Interpretable languages: binary (subset sum), ternary (−1/0/1 algebraic sum), ratio (sum positive over sum negative), pow2 (ternary with powers of two coefficients).
- Data encodings: raw values, prevalence via epsilon thresholding, and log transforms with epsilon flooring for numerical stability.
- Optimizers: Genetic Algorithm (ga2 Predomics style), Beam search (combinatorial and incremental), and MCMC with Sequential Backward Selection (beta).
- Fitness targets: AUC, specificity, sensitivity, MCC, F1-score and G means with optional penalties on model size (k_penalty) and false‑rates (fr_penalty).
- Cross‑validation: stratified folds, Family of Best Models extraction, and OOB permutation importance aggregation across folds.
- GPU acceleration: wgpu‑based scoring with configurable memory policy and safe CPU fallback when device limits are reached.

## Installation

Install a recent Rust toolchain and build in release mode for performance on CPU and GPU.

`curl –proto '=https' –tlsv1.2 -sSf https://sh.rustup.rs | sh`

At the root of this repository, compile gpredomics: 

`cargo build --release`

## Use 

The executable loads param.yaml from the current working directory on startup.
This configuration file contains information about the inputs and experiments to be launched. 

To launch gpredomics, simply type: 

`cargo run --release`

### Data format

Below are minimal TSV schemas that match the loader’s expectations.

X.tsv : features by rows and samples by columns; first column contains feature names, subsequent columns contain numeric values per sample.
| feature |	sample_a | sample_b	| sample_c
| :-- | :-- | :-- | :-- 
| feature_1 | 0.10 | 0.20 | 0.30
| feature_2 | 0.00 | 0.05 | 0.10
| feature_3 | 0.90 | 0.80 | 0.70

y.tsv: two‑column TSV mapping sample to class; header line is ignored; classes: 0 (negative), 1 (positive), 2 (unknown, ignored in metrics).
| sample | class
| :-- | :-- 
| sample_a| 0
| sample_b| 1
| sample_c| 1

### CLI

CLI commands can be specified to reload a saved experiment or evaluate a new dataset using the models selected during the experiment:

- Default run: execute the binary in a directory that contains param.yaml; the program initializes logging and dispatches GA/Beam/MCMC according to general.algo.
- Reload and display: use --load <experiment.(json|mp|bin)> to deserialize a saved Experiment; the format is auto‑detected at load time. 
- Evaluate on external data: combine --load with --evaluate and provide --x-test and --y-test to score the saved run on a new dataset.

#### Examples

Flags are defined with clap:

```bash
# default execution (param.yaml in CWD)
gpredomics

# reload a saved experiment and print results
gpredomics --load 2025-01-01_12-00-00_run.msgpack

# evaluate on an external test set
gpredomics --load 2025-01-01_12-00-00_run.msgpack \
  --evaluate --x-test /path/X_test.tsv --y-test /path/y_test.tsv
```

Note that `--evaluate` requires `--load` and needs `--x-test` and `--y-test`.
Termination signals are handled for clean shutdown.

### GpredomicsR

Please note that there is a R package, [GpredomicsR](https://github.com/predomics/gpredomicsR), which provides an R interface around the gpredomics engine for ecosystem integration and workflows in R.

## About Gpredomics

## Algorithms

- Genetic Algorithm: evaluates and evolves a population with selection, crossover, mutation, and GPU‑accelerated scoring when general.gpu is true.
- Beam search (beta): supports combinatorial and incremental modes and is currently under active development.
- MCMC (beta): explores models probabilistically, enforcing single data_type per run, and retains the best log‑likelihood SBS trajectory.

## Cross‑validation and importance

- CV creates stratified folds, runs the chosen algorithm per training split, and reports training/validation metrics, collecting per‑fold populations.
- OOB permutation importance is computed on Family of Best Models and aggregated across folds by mean or median.

## Reproducibility

- Runs are controlled by general.seed for reproducibility and use rayon threading and optional GPU for performance with deterministic intent where applicable.

*Many details will also be included in the next release!*

## GPU support

The supported GPUs are:
- Apple Silicons (Metal),
- All GPU supported by Vulkan.

### Apple Metal
For Apple, Metal is supported out of the box, however you need a recent version of LLVM, more recent at least than the default one. Here is the procedure:

You're supposed to already have the developpers tools (installed with `xcode-select --install`, which you need anyway for Rust). 

The recommanded procedure is to use Homebrew (at least for LLVM, and probably for rustup).

```sh
brew install rustup llvm
# the following line is only required if you had already installed Rust with the https://rustup.sh site
mv ~/.cargo ~/.cargo.backup # optionnally remove the line in the .zshrc that load the .cargo/env environment file
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
echo << EOF >> ~/.zshrc
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
EOF
rustup default nightly
rustup update
```
Then build normally with `cargo build --release`

### Linux
For Linux, you *must* install Vulkan:
```sh
sudo apt install vulkan-tools libvulkan1 
```

For Nvidia cards, you will need also a driver, so for instance:
```sh
sudo apt install libnvidia-gl-550-server nvidia-driver-550 nvidia-utils-550
```

Check with `vulkaninfo` that your card is correctly detected.

NB under Linux, I always do a fully optimized build, but that is not mandatory, a simple `cargo build --release` is enough:
```sh
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release
```