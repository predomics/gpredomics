<div align="center">

# Gpredomics

### Rapid, Interpretable and Accurate Machine Learning for Omics Data

<img src="docs/logo_predomics.png" alt="Predomics" width="180">

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/predomics/gpredomics/releases)
[![Rust](https://img.shields.io/badge/Rust-1.89+-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()
[![GPU](https://img.shields.io/badge/GPU-Metal%20%7C%20Vulkan-green.svg)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-yellow)](https://github.com/predomics/gpredomics/blob/main/CITATION.cff)
[![Rust CI](https://github.com/predomics/gpredomics/actions/workflows/rust.yml/badge.svg)](https://github.com/predomics/gpredomics/actions/workflows/rust.yml)

**Discover sparse, interpretable predictive signatures from high-dimensional omics data.**
<br>
Models with **discrete coefficients** {-1, 0, +1} that clinicians can read, verify, and trust.

<br>

*"Is this patient's microbiome associated with disease? Which species matter, and in which direction?"*

```
score = Bacteroides + Faecalibacterium - Enterococcus
```

[Get Started](#-quick-start) · [Documentation](#-documentation) · [Web App](https://github.com/predomics/predomicsapp) · [Python Bindings](https://github.com/predomics/gpredomicspy)

</div>

---

## Why Gpredomics?

<table>
<tr>
<td width="50%">

**The Problem**

- Black-box ML models can't be trusted in clinical settings
- Feature selection on 10,000+ features is computationally expensive
- Small cohorts (n=50-500) are prone to overfitting
- No standard way to compare optimization approaches
- Results vary between runs

</td>
<td width="50%">

**Our Solution**

- **Interpretable BTR models** — discrete coefficients a clinician can read
- **7 optimization algorithms** — from 0.05s to 7s per run
- **Cross-validation + voting ensemble** — robust to overfitting
- **LASSO baseline** + 6 metaheuristics + SOTA sklearn classifiers
- **Fully deterministic** — same seed = identical results

</td>
</tr>
</table>

---

## Benchmark

> Qin2014 cirrhosis dataset — 1,980 features, 180 samples, ternary:prevalence

| | Algorithm | Test AUC | Train AUC | k | FBM | Time | Type |
|---|-----------|:-------:|:---------:|:--:|:---:|:----:|------|
| 🥇 | **Beam Search** | **0.923** | 0.936 | **8** | 23 | 0.3s | Systematic, exhaustive |
| 🥈 | **Simulated Annealing** | **0.920** | 0.915 | 46 | 2 | 0.6s | Single-solution, perturbative |
| 🥉 | **LASSO / Elastic Net** | 0.882 | 0.957 | 70 | 1 | **0.1s** | Direct optimization |
| | Iterated Local Search | 0.880 | 0.996 | 38 | 1 | 0.3s | Single-solution, greedy |
| | Genetic Algorithm | 0.855 | 1.000 | 28 | **240** | 12.5s | Population, multi-language |
| | Ant Colony Optimization | 0.800 | 0.986 | 55 | 5 | 6.7s | Population, constructive |
| | MCMC (SBS) | 0.794 | 0.918 | 51 | 25 | 3.2s | Bayesian, backward selection |
| | MCMC (Gibbs) | 0.679 | 0.914 | **14** | 18 | 2.4s | Bayesian, joint variable selection |

> All models are human-readable: `score = species_A + species_B - species_C ≥ threshold`

---

## Architecture

```
 ┌───────────────────────────────────────────────────────────────────┐
 │                      predomicsapp-web                             │
 │                  Vue.js frontend + FastAPI                        │
 │              visualization · batch runs · export                  │
 └─────────────────────────────┬─────────────────────────────────────┘
                               │
 ┌─────────────────────────────▼─────────────────────────────────────┐
 │                        gpredomicspy                               │
 │                   Python bindings (PyO3)                          │
 │            clinical integration · sklearn benchmarks              │
 └─────────────────────────────┬─────────────────────────────────────┘
                               │
 ┌─────────────────────────────▼─────────────────────────────────────┐
 │                       gpredomics (Rust)                           │
 │                                                                   │
 │  ┌──────┐ ┌──────┐ ┌─────┐ ┌────┐ ┌─────┐ ┌───────┐ ┌────────┐    │
 │  │  GA  │ │ Beam │ │ ACO │ │ SA │ │ ILS │ │ LASSO │ │  MCMC  │    │
 │  └──┬───┘ └──┬───┘ └──┬──┘ └─┬──┘ └──┬──┘ └───┬───┘ └───┬────┘    │
 │     └────────┴────────┴──────┴───────┴────────┴─────────┘         │
 │                    Individual (BTR model)                         │
 │           features: { species_A: +1, species_C: -1 }              │
 │                                                                   │
 │  ┌──────────────┐ ┌──────────┐ ┌──────────-─┐ ┌───────────────┐   │
 │  │   Feature    │ │  Voting  │ │ Importance │ │      GPU      │   │
 │  │  Selection   │ │   Jury   │ │   (MDA)    │ │    (wgpu)     │   │
 │  └──────────────┘ └──────────┘ └──────────-─┘ └───────────────┘   │
 └───────────────────────────────────────────────────────────────────┘
```

---

## Key Features

<details open>
<summary><b>Model Languages</b></summary>

| Language | Formula | Use case |
|----------|---------|----------|
| **Binary** | `A + B + C ≥ t` | Presence/absence biomarkers |
| **Ternary** | `A + B - C ≥ t` | Directional associations |
| **Ratio** | `(A + B) / (C + D)` | Relative abundance ratios |
| **Pow2** | `4A + 2B - 8C ≥ t` | Weighted importance |

</details>

<details open>
<summary><b>7 Optimization Algorithms</b></summary>

| Algorithm | Approach | Best for | Docs |
|-----------|----------|----------|------|
| Genetic Algorithm | Population · recombinative | Broad exploration, diverse FBM | [📖](docs/ga.md) |
| Beam Search | Systematic · exhaustive | Guaranteed small-k coverage | [📖](docs/beam.md) |
| Ant Colony (ACO) | Population · constructive | Feature discovery via pheromone | [📖](docs/aco.md) |
| Simulated Annealing | Single-solution · perturbative | Best AUC, deep refinement | [📖](docs/sa.md) |
| Iterated Local Search | Single-solution · perturbative | Fastest, sparsest models | [📖](docs/ils.md) |
| LASSO / Elastic Net | Direct · coordinate descent | Mathematical baseline | [📖](docs/lasso.md) |
| MCMC / Bayesian | Posterior sampling | Uncertainty quantification | [📖](docs/mcmc.md) |

</details>

<details>
<summary><b>Fitness Functions</b></summary>

**Classification:** AUC, sensitivity, specificity, MCC, F1, PPV, NPV, G-mean

**Regression:** Spearman rank correlation, RMSE, Mutual Information

</details>

<details>
<summary><b>Advanced Capabilities</b></summary>

- **Cross-validation** — stratified K-fold with inner/outer folds and overfitting penalty
- **Voting ensemble** — jury of diverse experts with class-specialized voting
- **Threshold confidence intervals** — bootstrap CI for robust classification boundaries
- **Feature importance** — MDA permutation + prevalence + coefficient sign analysis
- **GPU acceleration** — wgpu-based scoring (Metal on macOS, Vulkan on Linux)
- **CSV performance report** — all metrics for best model, FBM, and jury
- **Full determinism** — BTreeMap-based evaluation, same seed = identical results

</details>

---

## Quick Start

### Install

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/predomics/gpredomics.git
cd gpredomics
cargo build --release
```

### Run

```bash
# Default (reads param.yaml in current directory)
./target/release/gpredomics

# Custom config
./target/release/gpredomics --config experiment.yaml

# With CSV report
./target/release/gpredomics --config experiment.yaml --csv-report
```

### Minimal `param.yaml`

```yaml
general:
  algo: sa              # ga, beam, aco, sa, ils, lasso, mcmc
  language: ter         # bin, ter, ratio, pow2
  data_type: prev       # raw, prev, log
  fit: auc              # auc, mcc, f1_score, spearman, rmse
  seed: 42
  k_penalty: 0.001

data:
  X: X_train.tsv
  y: y_train.tsv
  Xtest: X_test.tsv
  ytest: y_test.tsv
```

<details>
<summary><b>Data format</b></summary>

**X.tsv** — features × samples (or set `features_in_rows: false` for samples × features):
| feature | sample_a | sample_b | sample_c |
|---------|----------|----------|----------|
| species_1 | 0.10 | 0.20 | 0.30 |
| species_2 | 0.00 | 0.05 | 0.10 |

**y.tsv** — sample → class (0=negative, 1=positive, 2=unknown):
| sample | class |
|--------|-------|
| sample_a | 0 |
| sample_b | 1 |

</details>

<details>
<summary><b>CLI Reference</b></summary>

```bash
# Reload saved experiment
gpredomics --load 2025-01-01_run.msgpack

# Evaluate on external test set
gpredomics --load 2025-01-01_run.msgpack \
  --evaluate --x-test X_test.tsv --y-test y_test.tsv
```

</details>

<details>
<summary><b>GPU Setup</b></summary>

| Platform | Backend | Setup |
|----------|---------|-------|
| macOS (Apple Silicon) | Metal | `brew install rustup llvm` then build normally |
| Linux (NVIDIA) | Vulkan | `sudo apt install vulkan-tools libvulkan1 nvidia-driver-550` |

Enable with `general.gpu: true` in param.yaml. Falls back to CPU automatically.

</details>

---

## Ecosystem

| | Component | Description |
|---|-----------|-------------|
| ⚙️ | [**gpredomics**](https://github.com/predomics/gpredomics) | Rust core engine (this repo) |
| 🐍 | [**gpredomicspy**](https://github.com/predomics/gpredomicspy) | Python bindings + clinical integration |
| 🌐 | [**predomicsapp**](https://github.com/predomics/predomicsapp) | Full-stack web application |
| 📊 | [**GpredomicsR**](https://github.com/predomics/gpredomicsR) | R package interface |
| 📦 | [**Legacy Predomics**](https://github.com/predomics/predomicspkg) | Original R implementation |

---

## Documentation

| | Topic | |
|---|-------|---|
| 🚀 | **Getting Started** | [Usage](docs/use.md) · [Data](docs/data.md) · [Cross-validation](docs/cv.md) |
| 🧬 | **Concepts** | [Individual](docs/individual.md) · [Population](docs/population.md) · [Rejection](docs/rejection.md) |
| ⚡ | **Algorithms** | [GA](docs/ga.md) · [Beam](docs/beam.md) · [ACO](docs/aco.md) · [SA](docs/sa.md) · [ILS](docs/ils.md) · [LASSO](docs/lasso.md) · [MCMC](docs/mcmc.md) |
| 🔧 | **Development** | [Technical docs](docs/dev.md) |

---

## Citation

If you use **Gpredomics** in your research, please cite:

> Prifti, E., Chevaleyre, Y., Hanczar, B., Belda, E., Danchin, A., Clément, K., & Zucker, J. D. (2020). **Interpretable and accurate prediction models for metagenomics data.** *GigaScience*, 9(3), giaa010. [doi:10.1093/gigascience/giaa010](https://doi.org/10.1093/gigascience/giaa010)

> Lesage, L., de Lahondès, R., Puller, V., & Prifti, E. (2025). **Gpredomics** (v0.9.0). GMT Science / IRD. [github.com/predomics/gpredomics](https://github.com/predomics/gpredomics)

<sub>A `CITATION.cff` is available — use GitHub's "Cite this repository" button for BibTeX/APA export.</sub>

---

<div align="center">

**Issues** · [github.com/predomics/gpredomics/issues](https://github.com/predomics/gpredomics/issues)
<br>
**Contact** · contact@predomics.com

<sub>Last updated: v0.9.0</sub>

</div>
