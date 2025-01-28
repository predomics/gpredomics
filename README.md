# gpredomics

gpredomics is a rewrite of predomics in Rust, which REALLY use GPU since version v0.5 (requires gpu=true in params.yaml, general section). Only a subset of predomics is coded:

- only the Genetic Algorithm is coded, the equivalent of ga2 in Predomics terms,
- the following languages are available bin(ary), ter(nary), ratio, pow2 (a language specific to gpredomics):

  - bin uses only 0 or 1 as coefficients for features (take a feature or ignore it), and makes the sum of them as a score,
  - ter uses 0,1 and -1 as coefficients (so features can now be negative also), and makes the sum of them as a score,
  - ratio is like ter (0,1,-1) but the score is now the ratio of the sum of features associated with 1 divided by the sum of features associated with -1,
  - pow2 is like ter but coefficients can now be powers of 2 (-64, -32, ... -4, -2, -1, 0 , 1, 2, 4, ... 64)

- fit (`general.fit` in param.yaml) can be `auc`, `specificity` or `sensitivity` (a specific care has been taken so as to optimize the computation time of AUC),
(a small test to compare some AUCs with scikit-learn show that values convergent - for some model, a small divergence (less than 0.5%) can appear). The base fit function can be nudged by several useful penalties:

  - k_penalty: a penalty that is deduced from fit multiplied by the number of non nul coefficients,
  - fnr_penalty: a penalty useful when fit is `specificity` to constraint  for some sensitivity as well,
  - fpr_penalty: a penalty useful when fit is `sensitivity` to constraint  for some specificity as well,
  - overfit_penalty: when this penalty is added, a part of the train data is removed, a fold as a test (the number of folds is in `cv.fold_number`), the real test being called then the holdout, and the base fit function is evaluated on test each time, the difference between train and test is deduced from the base fit function multiplied by this coefficient (thus it should be different from 1/cv.fold_number otherwise the penalty does nothing) 

- feature importance is estimated by OOB algorithm using mean decreased AUC,
- several data_type are implemented (`general.data_type` and `general.data_type_minimum`): 

  - `raw` : features are taken as they are (default),
  - `prevalence` : or presence/absence, features above data_type_minimum are 1 others are 0,
  - `log` : features are now equal to their log (`feature.ln()` in Rust). Features below data_type_minimum are set to this value before log transformation.

- lots of features are still not implemented.

At this stage (beta), the program is remains simple, yet it is already versatile and useful. 

## compile

To compile:

- install Rust with [rustup](https://rustup.rs) : `curl –proto '=https' –tlsv1.2 -sSf https://sh.rustup.rs | sh`
- at the root of this repository, compile: `cargo build --release`, which will create the binary in `target/release`


## use

The binary will look for a `param.yaml` file in the current working directory. There is a sample one at the root of this repository. You will of course need some data, in the typical form of a matrix (X) of values with the biological features as lines and the samples in column (in TSV format), and a target vector (y), a two column TSV (which header is ignored) containing a column with the samples (same name as X column names), and a column with the class of each sample, 0 for the first class, 1 for the second class, and 2 when the class is unknown.

You will find two sample sets, in the `samples` folder, one on microbiome and cirrhosis (Qin 2014, a.k.a. PRJEB6337), and one on microbiome and response to treatment in ICI (Immune Checkpoint Inhibitors) (Derosa 2022, a.k.a. PRJNA751792).

## GPU

The supported GPUs are:
- Apple Silicons (Metal),
- All GPU supported by Vulkan.

### Apple Metal
For Apple, Metal is supported out of the box, there is nothing specific to do (appart the `xcode-select --install` which you need anyway for Rust). 

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

## some details about param.yaml

Parameters are set in the `param.yaml` file, a short description of the meaning of the different lines is provided within the file.
There are three sections, general, data and ga.

### general

- seed: gpredomics is fully determinist, re-running a computation with the same seed bear the same results, this should be a number,
- algo: either `random` (not useful, for tests only), `ga` the basic genetic algorithm, `ga+cv` (experimental, uncomplete) the same algorithm but with a simple cross val scheme 
- thread_number: the number of parallel threads used in feature selection and fit evaluation in `ga`, between different `ga` in `ga+cv`.
- gpu: should be true whenever possible (that is almost always)

The following parameter are for the fit function:
- fit: the base parameter, define the base fit function, `auc`, `specificity` or `sensitivity`, the base fit is then modified by different penalty set below,
- k_penalty: the penalty applied per number of feature in the model (be careful with large number of feature, don't set this too high),
- fr_penalty: false rate penalty: add a part of sensitivity when fitting on specificity or symetrically. When set to 1.0 it's roughly a fit on accuracy.
- overfit_penalty: this one trigger the creation of an intermediate test set from the train set (a fold, taken after `cv.fold_number`). The difference of fit function on the test set vs the train set is then deduced from the initial fit function, multiplied by this parameter. This overfit penalty must be carefully balanced, typically fold_number is set to 10. Setting the overfit_penalty to 0.1 is like having included the fold in the train set (it has the weight equal to its size), so it does nothing, better to have 0 in that case (which is less costly). Setting the overfit_penalty to 0.3 or above will give the fold an high importance and is likely to trigger an overfit on the fold, which is likely worse than an overfit on the whole train. Good overfit penalty values should be slightly above 0.1 but not higher than 0.2. This sensitive penalty is likely not the first one to test, keep it to 0 at first.

### data

- The name of the files are pretty obvious: X is the matrix of features, y the target vector.
- Xtest and ytest are optional holdouts (that should not be included in X and y)
- feature_minimal_prevalence_pct is a filter: this is the minimal % of sample non null for the feature in either class for the feature to be kept.
- feature_minimal_feature_value is a filter on the average value of the feature (if either class is above, then the feature is not rejected),
- feature_maximal_pvalue is a filter to remove features which average is not significantly different between the two classes.
- pvalue_method is either studentt (recommanded for features with normal distributions) or wilcoxon (recommanded for sparse/log normal features)

### ga

- population_size: the number of model (a.k.a. Individual in the Rust code) per epoch,
- max_epochs: the target number of epoch (an epoch is a generation in the Genetic Algorithm)
- min_epochs: the minimal number of epoch befor trying to see if AUC are converging
- max_age_best_model: when a best model reaches this age after min_epochs epochs, the ga algorithm is stopped,
- select_elite_pct: the percentage of best N-1 individual selected as parents,
- select_random_pct: the percentage of random N-1 individual selected as parents,
- mutated_children_pct: the percentage of children mutated,
- mutated_features_pct: the percentage of feature mutated in a mutated child,
- mutation_non_null_chance_pct: the likeliness of "positive" mutation, e.g. a mutation that select a new feature in the model.
- feature_importance_permutations: the number of permutations in OOB algorithm to compute feature importance

