# Individual

## Concept

In Gpredomics, an `Individual` represents a unique predictive model, defined by a specific subset of features, their coefficients, and associated parameters. Individuals are the fundamental units manipulated by algorithms such as genetic algorithms, beam search, and ant colony optimization (ACO).

Currently, each model is a binary classification model, dependent on a threshold. Gpredomics is specifically designed for interpretable and easy-to-use models.

Let's take the example of a fictive target disease called *Lightheadedness*, causing patients' heads to transform into giant light bulbs. The disease is mainly associated with three features called `L`, `U` and `X`. A Gpredomics equation that characterises this relationship could therefore be: 

    L + U + X ≥ threshold

This equation is quite simple: if the sum of `L`, `U` and `X` is equal to or greater than the given threshold, the sample is classified as having *Lightheadedness*, otherwise it is classified as healthy. 

### Model languages

Legacy Predomics more precisely allows three types of languages, i.e. model formulae, called **BTR**. Each one is designed to describe a particular ecological relation:

**Binary | Cooperation / Cumulative effect | Coefficient ∈ {0, 1}**

    L + U + X ≥ threshold

**Ternary: Competition / Predation | Coefficient ∈ {-1, 0, 1}**

    L + U + X - (D + A + R + K) ≥ threshold

**Ratio: Imbalance | Coefficient ∈ {-1, 0, 1}**

    (L + U + X) / (D + A + R + K + ε) ≥ threshold

Gpredomics introduces a new language called Pow2, which is a variant of Ternary where each feature can be raised to a power of two: 

**Pow2 | Coefficient ∈ {-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64}:**

    L + U + 2 X - (64 D + 4 A + 2 R + K) ≥ threshold

See [Predomics' original paper](https://doi.org/10.1093/gigascience/giaa010) for more details on BTR languages. 

### Model data types


Each model is additionally characterized by a data type:

| Data Type | Description         | Displayed equation (bin example)                  | Implemented non-ratio (bin/ter/pow2) equation                                   | Implemented ratio equation                                                                          |
|-----------|---------------------|----------------------------------------|--------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `raw`       | Raw data            | $L + U + F \geq \text{threshold}$      | $\sum_i C_i F_i \geq \text{threshold}$                    | $\frac{\sum_{i \in pos} C_i F_i}{\sum_{j \in neg} \lvert C_j \rvert F_j + \epsilon} \geq \text{threshold}$   |
| `prev`      | Binarized data      | $L^0 + U^0 + F^0 \geq \text{threshold}$ | $\sum_i C_i \mathbb{I}(F_i > \epsilon) \geq \text{threshold}$ | $\frac{\sum_{i \in pos} C_i \mathbb{I}(F_i > \epsilon)}{\sum_{j \in neg} \lvert C_j \rvert \mathbb{I}(F_j > \epsilon) + \epsilon} \geq \text{threshold}$ |
| `log`       | Natural log of data | $\ln(L × U × F) \geq \text{threshold}$  | $\sum_i C_i (\ln F_i - \ln \epsilon) \geq \text{threshold}$ | $\sum_{i \in pos} |C_i| \ln\left(\frac{F_i}{\epsilon}\right) - \sum_{j \in neg} |C_j| \ln\left(\frac{F_j}{\epsilon}\right) - \epsilon \geq \text{threshold}$ $ |
| `zscore`    | Standardized (z-score) | $\tilde{L} + \tilde{U} + \tilde{F} \geq \text{threshold}$ | $\sum_i C_i \tilde{F_i} \geq \text{threshold}$ where $\tilde{F_i} = \frac{F_i - \mu_i}{\sigma_i}$ | $\frac{\sum_{i \in pos} C_i \tilde{F_i}}{\sum_{j \in neg} \lvert C_j \rvert \tilde{F_j} + \epsilon} \geq \text{threshold}$ |

Here, $F_i$ represents Features and $C_i$ their coefficients.

$\epsilon$ is a small positive constant used to avoid division by zero or logarithm of zero. Its default value is 1e-5 (0.00001), configurable via `datatype_epsilon` in the YAML configuration.

For the `zscore` data type, each feature is standardized using **training-data statistics** (mean $\mu_i$ and standard deviation $\sigma_i$ computed on the training set). These statistics are automatically propagated to test data to avoid data leakage. Features with zero variance are assigned $\sigma = 1$ to prevent division by zero. Aliases accepted in config: `zscore`, `z`, `standardized`.

## Struct layout

An `Individual` contains the model definition, a universal fitness score, and nested classification metrics.

### Individual fields

| Field        | Type                          | Description                                                     |
|--------------|-------------------------------|-----------------------------------------------------------------|
| `features`   | `BTreeMap<usize, i8>`         | Feature indices mapped to their coefficients. Uses `BTreeMap` (not `HashMap`) for deterministic iteration order. |
| `k`          | `usize`                       | Number of variables used in the model.                          |
| `language`   | `u8`                          | Language of the model (bin, ter, ratio, pow2).                  |
| `data_type`  | `u8`                          | Data type of the model (raw, prev, log, zscore).                        |
| `epsilon`    | `f64`                         | Epsilon value used during score calculation.                    |
| `fit`        | `f64`                         | Universal penalized objective (see [Fitness](#fitness) below).  |
| `cls`        | `ClassificationMetrics`       | Nested classification metrics (see below).                      |
| `epoch`      | `usize`                       | Iteration that produced this model.                             |
| `parents`    | `Option<Vec<u64>>`            | Parent hashes in the generational context.                      |
| `hash`       | `u64`                         | Identifier hash of the model.                                   |
| `betas`      | `Option<Betas>`               | Beta coefficients (MCMC individuals only).                      |

> **Note on `features`:** Prior to v0.8.3, `features` was a `HashMap<usize, i8>`. It is now a `BTreeMap<usize, i8>` so that feature iteration order is deterministic across runs and platforms, which matters for reproducibility of serialised models and score computation.

### ClassificationMetrics

The `ClassificationMetrics` struct groups all metrics that are specific to binary classification. It is stored in `Individual.cls`.

| Field         | Type                     | Description                                               |
|---------------|--------------------------|-----------------------------------------------------------|
| `auc`          | `f64`                    | Area Under the ROC Curve on the training set.             |
| `threshold`    | `f64`                    | Decision threshold used for binary classification.        |
| `threshold_ci` | `Option<ThresholdCI>`    | Confidence interval for the threshold and rejection rate. |
| `sensitivity`  | `f64`                    | True Positive Rate on the training set.                   |
| `specificity`  | `f64`                    | True Negative Rate on the training set.                   |
| `accuracy`     | `f64`                    | Accuracy on the training set.                             |
| `additional`   | `AdditionalMetrics`      | Additional derived metrics (see below).                   |

### AdditionalMetrics

The `AdditionalMetrics` struct holds extra classification metrics that are computed on demand. It is stored in `Individual.cls.additional`.

| Field      | Type            | Description                                    |
|------------|-----------------|------------------------------------------------|
| `mcc`       | `Option<f64>`   | Matthews Correlation Coefficient.              |
| `f1_score`  | `Option<f64>`   | Harmonic mean of Precision and Sensitivity.    |
| `npv`       | `Option<f64>`   | Negative Predictive Value — TN / (TN + FN).   |
| `ppv`       | `Option<f64>`   | Positive Predictive Value — TP / (TP + FP).   |
| `g_mean`    | `Option<f64>`   | Geometric mean of Sensitivity and Specificity. |

### Accessing metrics — quick reference

```rust
// Universal objective (always on Individual)
individual.fit

// Core classification metrics
individual.cls.auc
individual.cls.threshold
individual.cls.sensitivity
individual.cls.specificity
individual.cls.accuracy
individual.cls.threshold_ci

// Additional metrics (Option values)
individual.cls.additional.mcc
individual.cls.additional.f1_score
individual.cls.additional.npv
individual.cls.additional.ppv
individual.cls.additional.g_mean
```

## Threshold optimization

To compute the best threshold according to the individual formulae and the data, Gpredomics calculates the area under the curve of the model. Then, it computes the best threshold according to the fit targeted metrics : 

- for AUC, the best threshold is the threshold maximizing the Youden Maxima : 

        Specificity + Sensitivity - 1

- for asymetric target (specificity vs sensitivity), the best threshold is the threshold maximizing : 

        (target + (antagonist * *fr_penalty*)) / (1+*fr_penalty*)

- for others, the best threshold is the threshold maximizing the targeted metrics.

### Threshold Confidence Interval

As medicine requires divisive model, Gpredomics allows since 0.7.3 to surround the threshold with a empirical confidence intervale. For more details, consult the associated documentation: [rejection.md](rejection.md).

## Fitness

To assess how well a model fits the data, an individual is also characterised by a fitness metric, linked to a performance metric minus penalties. 

### Fitness metrics

Gpredomics allows several fit metrics:

| Metric         | Description                        | Formula                                                                 |
|----------------|------------------------------------|-------------------------------------------------------------------------|
| `auc`            | Area Under the Curve               | Calculated from ROC curve                                               |
| `sensitivity`    | True Positive Rate                 | TP / (TP + FN)                                                          |
| `specificity`    | True Negative Rate                 | TN / (TN + FP)                                                          |
| `g_mean` | Geometric mean, balance between Sens. & Spec.      | $\sqrt{\text{Sensitivity} \times \text{Specificity}}$                  |
| `mcc`            | Matthews Correlation Coefficient   | $\frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ |
| `f1_score`       | Harmonic mean of Prec. & Sens.     | $2 \times \frac{\text{Precision} \times \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}}$ |

Where:
- TP: True Positives
- TN: True Negatives
- FP: False Positives
- FN: False Negatives
- Precision = TP / (TP + FP)

This diversity of metrics allows for diverse exploration of the research space. 
Certain metrics, notably MCC and G-mean, are preferable in the case of unbalanced datasets. 

### Fit penalties

Several fit penalties can be applied on the fit to take specific constraints into account during the model selection at each iteration.

| param                      | Description                                                                                   | Equation                                                                                   | When to use it                                              |
|----------------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `k_penalty`                  | Penalty proportional to the number of variables used in the model.                            | fit -= k * `k_penalty`                                                                      | To favor simpler models and avoid overfitting               |
| `bias_penalty`               | Penalizes models with specificity/sensitivity < 0.5.                                          | fit -= (1.0 - bad_metrics) * `bias_penalty`                                                  | To avoid unbalanced models (very low specificity/sensitivity) |
| `threshold_ci_penalty`       | Penalizes according to rejection rate if a [threshold confidence interval](rejection.md) exists.               | fit -= rejection_rate * `threshold_ci_penalty`                                               | When you want to limit the rejection rate or control threshold uncertainty |
| `overfit_penalty`            | Penalizes overfitting during [cross-validation](cv.md#inner-cross-validation).                                                | fit -= mean(fit on k-1 folds - abs(delta with last fold)) * `overfit_penalty`                | To avoid overfitting detected during cross-validation       |
| `feature_penalty`            | Feature-specific penalty, defined in [annotations](data.md#feature-annotations).                        | fit -= Σ(`feature_penalty`)                                                                  | To penalize the use of specific variables according to their annotation |


## Individual feature importance

Beyond global statistics, Gpredomics can evaluate the importance of variables within a specific individual model. This is achieved using a **Mean Decrease Accuracy** (MDA) approach via permutations.

For each feature present in the individual, the algorithm randomly shuffles the feature's values across samples $N$ times (preserving the distribution but breaking the correlation with the target). It then measures the drop in AUC compared to the baseline. A significant drop indicates that the feature was crucial for the prediction. Conversely, a zero or negative importance suggests the feature - taken alone - is useless or noise. This metric allows the algorithm to perform pruning, automatically removing non-contributing features to simplify the model.

Please note that this feature allows you to evaluate the importance of a **single feature**, but does not allow you to evaluate its importance in conjunction with other features. In a metagenomic context, it can therefore be used to isolate features whose mere presence or absence is decisive, but it cannot capture group phenomena. 

*Last updated: v0.8.3*