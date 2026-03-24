# Gpredomics

### Rapid, Interpretable and Accurate Machine Learning for Omics Data

[![Version](https://img.shields.io/badge/version-0.9.0-blue.svg)](https://github.com/predomics/gpredomics/releases)
[![Rust](https://img.shields.io/badge/Rust-1.89+-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()
[![GPU](https://img.shields.io/badge/GPU-Metal%20%7C%20Vulkan-green.svg)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-blue)](https://github.com/predomics/gpredomics/blob/main/CITATION.cff)
[![Rust CI](https://github.com/predomics/gpredomics/actions/workflows/rust.yml/badge.svg)](https://github.com/predomics/gpredomics/actions/workflows/rust.yml)
![GitHub bugs](https://img.shields.io/github/issues/predomics/gpredomics/bug)

<div align="center">
  <img src="docs/logo_predomics.png" alt="Official Predomics logo" width="200">
</div>

**Gpredomics** discovers sparse, interpretable predictive signatures from high-dimensional omics data. Unlike black-box models, gpredomics learns models with **discrete coefficients** ({-1, 0, +1}) that clinicians can read, verify, and trust. Built in Rust for speed, with GPU acceleration and a full-featured web interface.

> *"Is this patient's microbiome associated with disease? Which species matter, and in which direction?"*
> Gpredomics answers this with models like: **score = Bacteroides + Faecalibacterium - Enterococcus**

---

## Why Gpredomics?

| Challenge | Gpredomics Solution |
|-----------|-------------------|
| Black-box ML models can't be trusted in clinical settings | **Interpretable BTR models** with discrete coefficients |
| Feature selection on 10,000+ features is slow | **7 optimization algorithms** from 0.05s to 7s |
| Overfitting on small cohorts (n=50-500) | **Cross-validation**, threshold CI, k-penalty, voting ensemble |
| Need to compare approaches systematically | **LASSO baseline** + 6 metaheuristics + SOTA sklearn classifiers |
| Reproducibility across runs | **Deterministic** execution with fixed seed (BTreeMap-based) |

## Performance at a Glance

Benchmark on **Qin2014** cirrhosis dataset (1,980 features, 180 samples, ternary:prevalence):

| Algorithm | Test AUC | Model Size (k) | Time |
|-----------|---------|----------------|------|
| **Simulated Annealing** | **0.911** | 48 | 0.3s |
| LASSO / Elastic Net | 0.853 | 70 | 0.1s |
| Iterated Local Search | 0.813 | **22** | **0.05s** |
| Ant Colony Optimization | 0.802 | 52 | 7.1s |
| Genetic Algorithm | 0.791 | 48 | 0.5s |

All algorithms produce models that a biologist can read: `score = species_A + species_B - species_C`

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         predomicsapp-web             │
                    │     Vue.js frontend + FastAPI        │
                    │   (visualization, batch runs, ...)   │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │          gpredomicspy                │
                    │     Python bindings (PyO3)           │
                    │  clinical integration, benchmarks    │
                    └──────────────┬──────────────────────┘
                                   │
  ┌────────────────────────────────▼────────────────────────────────┐
  │                        gpredomics (Rust)                        │
  │                                                                 │
  │  ┌─────────┐ ┌──────┐ ┌──────┐ ┌─────┐ ┌────┐ ┌───────┐ ┌───┐│
  │  │   GA    │ │ Beam │ │ ACO  │ │ SA  │ │ILS │ │ LASSO │ │MCM││
  │  │ evolve  │ │search│ │ ants │ │annea│ │local│ │coord. │ │Bay││
  │  └────┬────┘ └──┬───┘ └──┬───┘ └──┬──┘ └─┬──┘ └───┬───┘ └─┬─┘│
  │       └─────────┴────────┴────────┴──────┴────────┴───────┘   │
  │                    Individual (BTR model)                       │
  │              features: {species_A: +1, species_C: -1}          │
  │                                                                 │
  │  ┌──────────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐ │
  │  │ Feature      │  │ Voting / │  │ Importance │  │    GPU    │ │
  │  │ Selection    │  │  Jury    │  │   (MDA)    │  │  (wgpu)   │ │
  │  └──────────────┘  └──────────┘  └───────────┘  └───────────┘ │
  └─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Model Languages
- **Binary**: subset sum — `score = A + B + C` (presence/absence)
- **Ternary**: algebraic sum — `score = A + B - C` (directional)
- **Ratio**: `score = (A + B) / (C + D)` (relative abundance)
- **Pow2**: ternary with powers of two — `score = 4A + 2B - 8C` (weighted)

### 7 Optimization Algorithms
| Algorithm | Type | Strength |
|-----------|------|----------|
| [Genetic Algorithm](docs/ga.md) | Population, recombinative | Broad exploration, diverse FBM |
| [Beam Search](docs/beam.md) | Systematic, exhaustive | Guaranteed coverage of small k |
| [Ant Colony Optimization](docs/aco.md) | Population, constructive | Feature discovery via pheromone |
| [Simulated Annealing](docs/sa.md) | Single-solution, perturbative | Deep refinement, best AUC |
| [Iterated Local Search](docs/ils.md) | Single-solution, perturbative | Fastest, sparsest models |
| [LASSO / Elastic Net](docs/lasso.md) | Direct optimization | Mathematical baseline |
| [MCMC / Bayesian](docs/mcmc.md) | Bayesian posterior | Uncertainty quantification |

### Fitness Functions
**Classification**: AUC, sensitivity, specificity, MCC, F1, PPV, NPV, G-mean
**Regression**: Spearman rank correlation, RMSE, Mutual Information

### Advanced Capabilities
- **Cross-validation**: stratified K-fold with inner/outer folds and overfitting penalty
- **Voting ensemble**: jury of diverse experts with specialization (class-restricted voting)
- **Threshold confidence intervals**: bootstrap CI for robust classification boundaries
- **Feature importance**: MDA permutation + prevalence + coefficient analysis
- **GPU acceleration**: wgpu-based scoring (Metal on macOS, Vulkan on Linux)
- **Full determinism**: same seed = identical results across runs

---

## Quick Start

### Install

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build gpredomics
git clone https://github.com/predomics/gpredomics.git
cd gpredomics
cargo build --release
```

### Run

```bash
# Place your param.yaml, X.tsv, and y.tsv in the working directory
cargo run --release

# Or specify a config file
./target/release/gpredomics --config path/to/param.yaml
```

### Minimal param.yaml

```yaml
general:
  algo: ga          # ga, beam, aco, sa, ils, lasso, mcmc
  language: ter     # bin, ter, ratio, pow2
  data_type: prev   # raw, prev, log
  fit: auc          # auc, mcc, f1_score, spearman, rmse
  seed: 42
  k_penalty: 0.001

data:
  X: X_train.tsv
  y: y_train.tsv
  Xtest: X_test.tsv
  ytest: y_test.tsv

ga:
  population_size: 5000
  max_epochs: 200

voting:
  vote: true
  fbm_ci_alpha: 0.05
```

### Data Format

**X.tsv** (features × samples):
| feature | sample_a | sample_b | sample_c |
|---------|----------|----------|----------|
| species_1 | 0.10 | 0.20 | 0.30 |
| species_2 | 0.00 | 0.05 | 0.10 |

**y.tsv** (sample → class):
| sample | class |
|--------|-------|
| sample_a | 0 |
| sample_b | 1 |

Set `features_in_rows: false` for samples × features format (standard ML layout).

---

## Ecosystem

| Component | Description | Link |
|-----------|-------------|------|
| **gpredomics** | Rust core engine (this repo) | [GitHub](https://github.com/predomics/gpredomics) |
| **gpredomicspy** | Python bindings (PyO3) + clinical integration | [GitHub](https://github.com/predomics/gpredomicspy) |
| **predomicsapp-web** | Full-stack web application (FastAPI + Vue.js) | [GitHub](https://github.com/predomics/predomicsapp-web) |
| **GpredomicsR** | R package interface | [GitHub](https://github.com/predomics/gpredomicsR) |
| **Legacy Predomics** | Original R implementation | [GitHub](https://github.com/predomics/predomicspkg) |

## Documentation

- **Getting Started**: [Usage](docs/use.md) · [Data management](docs/data.md) · [Cross-validation](docs/cv.md)
- **Concepts**: [Individual](docs/individual.md) · [Population](docs/population.md) · [Rejection](docs/rejection.md)
- **Algorithms**: [GA](docs/ga.md) · [Beam](docs/beam.md) · [ACO](docs/aco.md) · [SA](docs/sa.md) · [ILS](docs/ils.md) · [LASSO](docs/lasso.md) · [MCMC](docs/mcmc.md)
- **Development**: [Technical docs](docs/dev.md)

## CLI Reference

```bash
# Default run (reads param.yaml in current directory)
gpredomics

# Custom config
gpredomics --config experiment.yaml

# Generate CSV performance report
gpredomics --config experiment.yaml --csv-report

# Reload and display saved experiment
gpredomics --load 2025-01-01_run.msgpack

# Evaluate saved model on new data
gpredomics --load 2025-01-01_run.msgpack \
  --evaluate --x-test X_test.tsv --y-test y_test.tsv
```

## GPU Support

| Platform | Backend | Setup |
|----------|---------|-------|
| **macOS (Apple Silicon)** | Metal | `brew install rustup llvm` then build normally |
| **Linux (NVIDIA)** | Vulkan | `sudo apt install vulkan-tools libvulkan1 nvidia-driver-550` |

Set `general.gpu: true` in param.yaml. Falls back to CPU automatically if GPU is unavailable.

## Citation

If you use **Gpredomics** in your research, please cite:

**Original Method:**
> Prifti, E., Chevaleyre, Y., Hanczar, B., Belda, E., Danchin, A., Clément, K., & Zucker, J. D. (2020). Interpretable and accurate prediction models for metagenomics data. *GigaScience*, 9(3), giaa010. [https://doi.org/10.1093/gigascience/giaa010](https://doi.org/10.1093/gigascience/giaa010)

**Software:**
> Lesage, L., de Lahondès, R., Puller, V., & Prifti, E. (2025). *Gpredomics* (Version 0.9.0). GMT Science / IRD. [https://github.com/predomics/gpredomics](https://github.com/predomics/gpredomics)

A `CITATION.cff` file is available — use GitHub's **"Cite this repository"** button for BibTeX/APA export.

## Contact

- Issues: [GitHub Issues](https://github.com/predomics/gpredomics/issues)
- Email: contact@predomics.com

*Last updated: v0.9.0*
