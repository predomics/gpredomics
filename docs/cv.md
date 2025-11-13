# About cross-validation

Gpredomics currently provides two levels of cross-validation.

## Outer cross-validation

The outer cross-validation follows a Leave-One-Out style scheme. Concretely, when `cv` is `true`, the training
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

Inner cross-validation serves a different purpose than the outer loop. Also using a Leave-One-Out approach, this
feature splits the training data into `inner_folds` **within a single run** of the algorithm. At every epoch and for
each individual model, the fit is computed `inner_folds` times using the k-1 vs 1 scheme. The fit of each individual is
then computed as:

Mean(train_fit − overfitting_gap * `overfit_penalty`)

where overfitting_gap = |train_fit − valid_fit|.

This mechanism directly penalizes model overfitting. 
For long runs (many epochs) you can also request regular resampling of the inner folds: the folds are re-drawn every 
`resampling_inner_folds_epochs` epochs so that models do not learn the fixed structure of the folds indirectly.

*Last updated: v0.7.3*