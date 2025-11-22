# About cross-validation

Gpredomics currently provides two levels of cross-validation.

## Outer cross-validation

The outer cross-validation follows a K-Fold Cross-Validation scheme. Concretely, when `cv` is `true`, the training
dataset is split into `outer_folds`. The chosen algorithm is then trained `outer_folds` times in parallel, each time
using n-1 folds for training and the remaining fold for validation.

Example with 3 `outer_folds` and 100 samples:

```
Fold A: 33 samples
Fold B: 33 samples
Fold C: 34 samples
-
First launch: 67 samples for training, 33 for validation (AC, B)
Second launch: 67 samples for training, 33 for validation (BC, A)
Third launch: 66 samples for training, 34 for validation (AB, C)
```

Results are reported per fold, showing performance on the k-1 training folds and on the held-out validation fold.

A final population is then built by gathering the individuals from each fold’s Family of Best Models (see the
corresponding chapter) and evaluating that combined population on the full training data and, if provided, on the
Test dataset. Note: if `fit_on_valid` is enabled, individuals from each fold are re-fitted on that fold’s validation
data before computing the Family of Best Models — this can produce models that (in theory) generalize better. However,
this parameter **SHOULD NOT** be enabled unless you have an independent test dataset to evaluate final generalization on,
otherwise you risk data leakage and biased results.

When outer cross-validation is enabled, feature importances are aggregated across the Families of Best Models obtained
from each fold.

## Inner cross-validation

Inner cross-validation serves a different purpose than the outer loop. Also using K-Fold Cross-Validation scheme approach, this
feature splits the training data into `inner_folds` **within a single run** of the algorithm. At every epoch and for
each individual model, the fit is computed `inner_folds` times using the k-1 vs 1 scheme. The fit of each individual is
then computed as:

Mean(train_fit − overfitting_gap * `overfit_penalty`)

where overfitting_gap = |train_fit − valid_fit|.

This mechanism directly penalizes model overfitting. 
For long runs (many epochs) you can also request regular resampling of the inner folds: the folds are re-drawn every 
`resampling_inner_folds_epochs` epochs so that models do not learn the fixed structure of the folds indirectly.

## Stratification

Since version v0.7.4, it has been possible to stratifies folds first by target class y, and then by another specific metadata (hospital, batch, etc.).
To do this, you must add an annotation .tsv file and specify its path using the parameter `sample_annotations`.

| sample |	city | region
| :-- | :-- | :-- 
| sample_a | Paris | Europe
| sample_b | Washington | North America
| sample_c | London | Europe

The stratification of the inner and outer folds is then carried out according to the column indicated in the `stratify_by` parameter:

Example with Double Stratification (`stratify_by=city`):
```
Healthy (Class 0): 45 samples (30 from Paris, 15 from London).
Sick (Class 1): 45 samples (15 from Paris, 30 from London).

Fold A: 30 samples
   - Healthy: 15 (10 Paris, 5 London)
   - Sick:    15 (5 Paris, 10 London)

Fold B: 30 samples
   - Healthy: 15 (10 Paris, 5 London)
   - Sick:    15 (5 Paris, 10 London)

Fold C: 30 samples
   - Healthy: 15 (10 Paris, 5 London)
   - Sick:    15 (5 Paris, 10 London)
```

The algorithm creates folds that respect proportions without discarding data:

Example with Unbalanced Double Stratification:

```
Consider an extreme "Rare Condition" dataset with 7 samples and stratify_by=batch (3 folds):
    Class 0 (Control): 5 samples (4 from Batch A, 1 from Batch B).
    Class 1 (Case): 2 samples (0 from Batch A, 2 from Batch B).


Fold A (3 samples):
   - Class 0: 2 samples (1 Batch A, 1 Batch B)
   - Class 1: 1 sample  (1 Batch B)

Fold B (2 samples):
   - Class 0: 2 samples (2 Batch A, 0 Batch B)
   - Class 1: 0 samples

Fold C (2 samples):
   - Class 0: 1 sample  (1 Batch A, 0 Batch B)
   - Class 1: 1 sample  (1 Batch B)
```

In extreme cases like above, fold sizes may vary slightly (2 vs 3 samples), but the stratification guarantees that the rare "Class 0 / Batch B" sample is isolated for validation exactly once.

*Why are there 0 samples of Class 1 in Fold B?*
Since there are only 2 samples of Class 1 total and we requested 3 folds, it is mathematically impossible to put a sample in every fold (2 items cannot fill 3 boxes).
This is safe however: When validating on Fold B, the model trains on Folds A and C, so it does learn from the Class 1 samples found in those other folds.

*Last updated: v0.7.4*