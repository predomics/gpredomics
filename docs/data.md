# Dealing with Gpredomics data

## Format

Gpredomics data must be provided in `.tsv`, `.txt`, `.tab`, or `.csv` format in order to be correctly interpreted by the Rust binary.

The X and y files must be structured as follows: 

`X.tsv` : features by rows and samples by columns; first column contains feature names, subsequent columns contain numeric values per sample.
| feature |	sample_a | sample_b	| sample_c
| :-- | :-- | :-- | :-- 
| feature_1 | 0.10 | 0.20 | 0.30
| feature_2 | 0.00 | 0.05 | 0.10
| feature_3 | 0.90 | 0.80 | 0.70

`y.tsv`: two‑column TSV mapping sample to class; the first line (header) is ignored; classes: 0 (negative), 1 (positive), 2 (unknown, ignored in metrics).
| sample | class
| :-- | :-- 
| sample_a| 0
| sample_b| 1

Note: The X file can be transposed (samples as rows) if the `features_in_rows` parameter is set to false. 
Gpredomics automatically aligns samples between X and Y files based on their IDs.

It is possible to specify an external test set via the `Xtest` and `ytest` parameters. If no test set is provided, a split holdout can be generated automatically in Gpredomics according to the value of the `holdout_ratio` parameter. This split is stratified first by class, then by the `stratify_by` variable (if specified, see below) to ensure a good balanced representation in the training and test sets.

## Feature preselection and filtering

Before running the optimization algorithms (GA, Beam, or MCMC), Gpredomics performs a statistical preselection step to reduce the feature space and focus on the most discriminative variables. This process applies multiple filtering criteria and statistical tests to evaluate each feature's association with the target classes.

### Filtering criteria

All features are subject to the following preliminary filters, regardless of the statistical method used:

#### Prevalence-based filtering

Features must exhibit sufficient prevalence (non-zero presence) in at least one of the two classes to be considered informative. The `feature_minimal_prevalence_pct` parameter (default: 10%) defines the minimum percentage of samples in which a feature must be present.

For each feature, prevalence is calculated separately for each class:
- **Class 0 prevalence**: proportion of class 0 samples where the feature value is non-zero
- **Class 1 prevalence**: proportion of class 1 samples where the feature value is non-zero

A feature is **filtered out** if both class prevalences fall below the threshold. This criterion ensures that features appearing in too few samples across both classes are excluded from further analysis.

**Example**: With `feature_minimal_prevalence_pct: 10`:
```
- Feature A: 5% prevalence in class 0, 8% in class 1 → Excluded (both < 10%)
- Feature B: 5% prevalence in class 0, 15% in class 1 → Retained (class 1 ≥ 10%)
- Feature C: 12% prevalence in class 0, 20% in class 1 → Retained (both ≥ 10%)
```

#### Mean value filtering

To avoid selecting features with negligible abundance, Gpredomics filters out features whose mean value is too low across both classes. The `feature_minimal_feature_value` parameter (default: 0.0) sets this threshold.

For each feature, the mean is computed independently for each class:
- **Class 0 mean**: average feature value across all class 0 samples
- **Class 1 mean**: average feature value across all class 1 samples

A feature is **filtered out** if both class means fall below the threshold. This filter is particularly useful for abundance data (e.g., microbiome relative abundances, gene expression) where very low average values may represent noise or artifacts.

**Example**: With `feature_minimal_feature_value: 0.001`:
```
- Feature A: mean = 0.0005 in class 0, mean = 0.0008 in class 1 → Excluded (both < 0.001)
- Feature B: mean = 0.0005 in class 0, mean = 0.002 in class 1 → Retained (class 1 ≥ 0.001)
- Feature C: mean = 0.003 in class 0, mean = 0.005 in class 1 → Retained (both ≥ 0.001)
```

### Statistical testing methods

After applying the prevalence and mean filters, Gpredomics evaluates whether each remaining feature shows a statistically significant association with one of the two classes. Three statistical methods are available, selected via the `feature_selection_method` parameter:

#### Wilcoxon rank-sum test (default)

The **Wilcoxon rank-sum test** is a non-parametric test that compares the distributions of a feature across the two classes without assuming normality. It is particularly well-suited for:

- Sparse data with many zero values
- Log-normal or skewed distributions
- Data with outliers
- Small sample sizes

**Method**: `feature_selection_method: wilcoxon`

The test ranks all feature values across both classes and computes the U statistic. A normal approximation with continuity correction is used to derive a two-tailed p-value. Features with p-values below the threshold are retained and assigned to the class with the higher mean.

#### Student's t-test

The **Student's t-test** is a parametric test that compares the means of two normally distributed populations. It assumes:
- Normally distributed feature values within each class
- Equal variances across classes (pooled variance estimate)

**Method**: `feature_selection_method: studentt`

The test computes a t-statistic based on the difference between class means, pooled standard deviation, and sample sizes. Degrees of freedom are calculated as $n_0 + n_1 - 2$, and a two-tailed p-value is derived from the Student's t-distribution.

#### Bayesian Fisher's exact test

The **Bayesian Fisher's exact test** treats features as binary (present/absent) and evaluates the association between feature presence and class membership. This method:
- Constructs a 2×2 contingency table: (present/absent) × (class 0/class 1)
- Computes Bayes factors comparing the evidence for association vs. independence

**Method**: `feature_selection_method: bayesian_fisher`

A Bayes factor (BF) is calculated as the ratio of p-values from one-sided Fisher's tests:
$$\text{BF} = \frac{P(\text{greater})}{P(\text{less})}$$

Features are selected if $|\log_{10}(\text{BF})| \geq$ `feature_minimal_log_abs_bayes_factor` (default: 2.0, corresponding to BF ≥ 100 or BF ≤ 0.01).

**Recommended for**: Binary or presence/absence data (e.g., gene presence in metagenomics, binary clinical features).

### Multiple testing correction

For Wilcoxon and Student's t-test methods, Gpredomics applies **Benjamini-Hochberg (BH) false discovery rate (FDR) correction** to adjust p-values for multiple comparisons. This correction controls the expected proportion of false positives among significant features.

Only features with **adjusted p-values** below `feature_maximal_adj_pvalue` are retained. 
Note that Bayesian Fisher does not use FDR correction; instead, it relies on the Bayes factor threshold.

### Feature ranking and selection

After statistical testing and multiple testing correction:

1. Features are ranked by their statistical significance:
   - **Wilcoxon/Student's t-test**: sorted by ascending adjusted p-value (most significant first)
   - **Bayesian Fisher**: sorted by descending absolute log Bayes factor (strongest evidence first)

2. Each significant feature is assigned a **class label**:
   - Class 0: if the feature is significantly higher in class 0
   - Class 1: if the feature is significantly higher in class 1

3. If `max_features_per_class` > 0, only the top N features per class are retained. This parameter allows limiting the feature space for computational efficiency or to focus on the most discriminative markers.

### Configuration example

```yaml
# Feature preselection parameters
feature_selection_method: wilcoxon           # Statistical test method
feature_minimal_prevalence_pct: 15.0         # Minimum 15% prevalence in at least one class
feature_minimal_feature_value: 0.0001        # Minimum mean value of 0.0001
feature_maximal_adj_pvalue: 0.05             # Adjusted p-value threshold (FDR-corrected)
max_features_per_class: 100                  # Keep top 100 features per class
```

### Best practices

- **Start permissive**: Begin with relaxed thresholds (e.g., `feature_maximal_adj_pvalue: 0.5`) to avoid excluding potentially informative features too early.
- **Adapt to data type**: Use Wilcoxon for sparse/skewed data, Student's t-test for normal data, and Bayesian Fisher for binary data.
- **Monitor feature counts**: Check the number of selected features per class in the log output. If too few features pass the filters, consider relaxing `feature_minimal_prevalence_pct` or `feature_maximal_adj_pvalue`.
- **Prevalence matters**: For microbiome data, a prevalence threshold of 10-20% is typically appropriate to exclude rare taxa.
- **Mean filtering**: Adjust `feature_minimal_feature_value` based on your data scale. For relative abundances, 0.0001 (0.01%) can help remove noise.

## Data annotations

### Feature annotations

It is possible to provide Gpredomics with specific annotations for features via a .tsv file whose path is specified in the `feature_annotations` parameter. This file may contain `prior_weight`, `feature_penalty`, and additional tags. It must be structured as follows: 

| feature |	prior_weight | feature_penalty | order | taste
| :-- | :-- | :-- | :-- | :-- 
| Penicillium_camemberti | 1 | 0.01| Eurotiales | yummy
| Rhizopus_stolonifer | 2 | 0.5 | Mucorales | yuck
| Penicillium_glaucum | 1 | 0.01 | Eurotiales | yummy


Tags columns (e.g., order, taste): Used only to enrich the final results display and help the user identify potential biological patterns.
`prior_weight`: Influences the generation of the initial population in the genetic algorithm. Features with higher weights have a higher probability of being selected in randomly generated individuals.
`feature_penalty`: Allows the addition of a custom "soft" penalty per variable. The penalization is obtained by calculating the weighted average (based on absolute feature coefficients) of the penalty values for all selected features. Specifically, a cost proportional to this average is subtracted from the model's fitness. If penalties are too powerful or not powerful enough, it is possible to modify the weight using the `user_feature_penalties_weight` parameter.

### Sample annotations

It is also possible to provide Gpredomics with annotations associated with samples (e.g., hospital, batch), allowing folds to be stratified according to both class and metadata. For more information, see the [associated documentation](cv.md#stratification).

*Last updated: v0.7.6*