# Gpredomics - Rapid, Interpretable and Accurate machine learning for omics data

[![Version](https://img.shields.io/badge/version-0.7.7-orange.svg)](https://github.com/predomics/gpredomics/releases)
[![Rust](https://img.shields.io/badge/Rust-1.89+-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()
[![GPU](https://img.shields.io/badge/GPU-Metal%20%7C%20Vulkan-green.svg)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-blue)](https://github.com/predomics/gpredomics/blob/main/CITATION.cff)
[![Rust CI](https://github.com/predomics/gpredomics/actions/workflows/rust.yml/badge.svg)](https://github.com/predomics/gpredomics/actions/workflows/rust.yml)
![GitHub bugs](https://img.shields.io/github/issues/predomics/gpredomics/bug)

**Gpredomics** is a high-performance Rust implementation of the Predomics framework for discovering interpretable predictive signatures in omics data (metagenomics, microbiome, metabolomics). It learns Binary/Ternary/Ratio (BTR) models with discrete coefficients {-1, 0, 1} for maximum interpretability, making it ideal for clinical diagnostics and biomarker discovery.

<div style="text-align: center;">
  <img src="docs/logo_predomics.png" alt="Official Predomics logo" width="300" height="300">
</div>

### Features

- Interpretable languages: binary (subset sum), ternary (−1/0/1 algebraic sum), ratio (sum positive over sum negative), pow2 (ternary with powers of two coefficients).
- Data encodings: raw values, prevalence via epsilon thresholding, and log transforms with epsilon flooring for numerical stability.
- Optimizers: Genetic Algorithm (ga2 Predomics style), Beam search (LimitedExhaustive and ParallelForward), and MCMC with Sequential Backward Selection (beta).
- Fitness targets: AUC, specificity, sensitivity, MCC, F1-score and G means with optional penalties on model size (k_penalty) and false‑rates (fr_penalty).
- Confidence interval for classification threshold, allowing to discover divisive models and to avoid uncertain classifications.
- Cross‑validation: stratified folds, Family of Best Models extraction, and MAD permutation importance aggregation across folds.
- GPU acceleration: wgpu‑based scoring with configurable memory policy and safe CPU fallback when device limits are reached.

## Table of Contents

- [Quick start](#quick-start)
- Configuration:
  - [Usage](docs/use.md)
  - [Data management](docs/data.md) 
  - [Parameters](docs/param.md) (*coming soon*)
  - [Cross-validation](docs/cv.md)
- Concepts:
  - [Individual](docs/individual.md)
  - [Population](docs/population.md)
  - [Rejection](docs/rejection.md)
- Algorithms:
  - [Genetic Algorithm](docs/ga.md) (*coming soon*)
  - [Beam Search](docs/beam.md) (*coming soon*)
  - [MCMC](docs/mcmc.md) (*coming soon*)
- To go further:
  - [Differences with legacy Predomics](docs/legacy.md) (*coming soon*)
  - [Technical documentation](docs/dev.md)
- You may be interested in:
  - [GpredomicsR, an R package using Gpredomics](https://github.com/predomics/gpredomicsR)
  - [Legacy Predomics](https://github.com/predomics/predomicspkg/tree/master)
  - [Multiclass classification via Predomics](https://github.com/UMMISCO/predomicsmc)

## Quick start

### Installation

Install a recent Rust toolchain and build in release mode for performance on CPU and GPU.

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

At the root of this repository, compile gpredomics: 

`cargo build --release`

If you want the binary to embed a git hash in logs and experiment metadata, set `GPREDOMICS_GIT_SHA` at build time. If it is not set, it defaults to `unknown`.

Example:

`GPREDOMICS_GIT_SHA=$(git rev-parse --short HEAD) cargo build --release`

### Use 

The executable loads param.yaml from the current working directory on startup.
This configuration file contains information about the inputs and experiments to be launched. 

To launch gpredomics, simply type: 

`cargo run --release`

### Data format

Below are minimal TSV schemas that match the loader’s expectations.

`X.tsv` : features by rows and samples by columns; first column contains feature names, subsequent columns contain numeric values per sample.
| feature |	sample_a | sample_b	| sample_c
| :-- | :-- | :-- | :-- 
| feature_1 | 0.10 | 0.20 | 0.30
| feature_2 | 0.00 | 0.05 | 0.10
| feature_3 | 0.90 | 0.80 | 0.70

`y.tsv`: two‑column TSV mapping sample to class; header line is ignored; classes: 0 (negative), 1 (positive), 2 (unknown, ignored in metrics).
| sample | class
| :-- | :-- 
| sample_a| 0
| sample_b| 1
| sample_c| 1

A new parameter now allows you to accept a transposed X format, which is standard in ML. To do this, set `features_in_rows` to `false` in param.yaml. 

### CLI

CLI commands can be specified to reload a saved experiment or evaluate a new dataset using the models selected during the experiment:

- Default run: execute the binary in a directory that contains `param.yaml`; the program initializes logging and dispatches GA/Beam/MCMC according to general.algo.
- Specific configuration: execute the binary using another configuration file using --config config.yaml.
- Reload and display: use --load <experiment.(json|mp|bin)> to deserialize a saved Experiment; the format is auto‑detected at load time. 
- Evaluate on external data: combine --load with --evaluate and provide --x-test and --y-test to score the saved run on a new dataset.

#### Examples

Flags are defined with clap:

```bash
# default execution (param.yaml in CWD)
gpredomics

# specific config
gpredomics --config ./path/config.yaml

# reload a saved experiment and print results
gpredomics --load 2025-01-01_12-00-00_run.msgpack

# evaluate on an external test set
gpredomics --load 2025-01-01_12-00-00_run.msgpack \
  --evaluate --x-test /path/X_test.tsv --y-test /path/y_test.tsv
```

Note that `--evaluate` requires `--load` and needs `--x-test` and `--y-test`.
Termination signals are handled for clean shutdown.

### GPU support

The supported GPUs are:
- Apple Silicons (Metal),
- All GPU supported by Vulkan.

#### Apple Metal

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

#### Linux
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

## Citation

If you use **Gpredomics** in your research, please cite it as follows:

**Original Method:**
> Prifti, E., Chevaleyre, Y., Hanczar, B., Belda, E., Danchin, A., Clément, K., & Zucker, J. D. (2020). Interpretable and accurate prediction models for metagenomics data. *GigaScience*, 9(3), giaa010. [https://doi.org/10.1093/gigascience/giaa010](https://doi.org/10.1093/gigascience/giaa010)

**Software:**
> Lesage, L., de Lahondès, R., Puller, V., & Prifti, E. (2025). *Gpredomics* (Version 0.7.7). GMT Science / IRD. [https://github.com/predomics/gpredomics](https://github.com/predomics/gpredomics)

*A `CITATION.cff` file is available in this repository. If you use GitHub, you can use the **"Cite this repository"** option in the "About" section to export this citation in BibTeX or APA formats.*

## Contact

If you have any questions, comments, or have found a bug, please contact us at the following address:

- Email: contact@predomics.com
- GitHub Issues: [Gpredomics Issues](https://github.com/predomics/gpredomics/issues)

*Last updated: v0.7.7*
