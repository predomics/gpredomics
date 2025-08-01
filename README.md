# gpredomics

gpredomics is a rewrite of predomics in Rust, which REALLY use GPU since version v0.5 (requires gpu=true in params.yaml, general section). Only a subset of predomics is coded:

- the Genetic Algorithm is coded, the equivalent of ga2 in Predomics terms,
- the following languages are available bin(ary), ter(nary), ratio, pow2 (a language specific to gpredomics):

  - bin uses only 0 or 1 as coefficients for features (take a feature or ignore it), and makes the sum of them as a score,
  - ter uses 0,1 and -1 as coefficients (so features can now be negative also), and makes the sum of them as a score,
  - ratio is like ter (0,1,-1) but the score is now the ratio of the sum of features associated with 1 divided by the sum of features associated with -1,
  - pow2 is like ter but coefficients can now be powers of 2 (-64, -32, ... -4, -2, -1, 0 , 1, 2, 4, ... 64)

- fit (`general.fit` in param.yaml) can be `auc`, `specificity` or `sensitivity` (a specific care has been taken so as to optimize the computation time of AUC),
(a small test to compare some AUCs with scikit-learn show that values convergent - for some model, a small divergence (less than 0.5%) can appear). The base fit function can be nudged by several useful penalties:

  - k_penalty: a penalty that is deduced from fit multiplied by the number of non nul coefficients,
  - fr_penalty: a penalty used only when fit is specificity or sensitivity, deduce (1 - symetrical metrics) x fr_penalty to fit

- several data_type are implemented (`general.data_type` and `general.epsilon`): 

  - `raw` : features are taken as they are (default),
  - `prevalence` : or presence/absence, features above epsilon are 1 others are 0,
  - `log` : features are now equal to their log (`feature.ln()` in Rust). Features below epsilon are set to this value before log transformation.

- the Beam Algorithm is in beta version. This beta version is not compatible yet with Pow2 language. Two submethods are currently available : 

  - `combinatorial` generates all the possible combinations of features selected at each epoch (an epoch is a population of individuals of size k) which is equivalent of terBeam in Predomics terms. 
  - `extend` adds each of the features selected to each of the best extendable_models at each epoch.

- the Cross-validation, also in beta version, is available since v0.5.8. Feature importance is estimated by OOB algorithm using mean decreased AUC.

- a new algorithm based on Markov chain Monte Carlo - not present in the original version of predomics - is available since v0.5.8. This algorithm explores the search space iteratively and provides a probabilistic prediction based on all the models explored. This algorithm is still in beta version. The MCMC algorithm is run for each number of features between the pre-selected number of features (according to the data parameters) and the `param.mcmc.nmin` number, following the Sequential Backward Selection principle. The MCMC exploration with the highest log-likelihood is retained for prediction. 

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

## some details about param.yaml

Parameters are set in the `param.yaml` file, a short description of the meaning of the different lines is provided within the file.
There are six sections: general, data, cv, ga, beam & gpu.

### general

- seed: gpredomics is fully determinist, re-running a computation with the same seed bear the same results, this should be a number.
- algo: `ga` the genetic algorithm, `beam` the beam algorithm and `mcmc` the Markov chain Monte Carlo based algorithm.
- cv: boolean indicating whether to run cross-validation (depending on the parameters specified in the cv category). Gives an idea of variable importances. Only available for `ga` and `beam` for the moment.
- thread_number: the number of parallel threads used in feature selection and fit evaluation. Note that for cross validation, each fold will be run on one or more threads, depending on avalability.
- gpu: should be true whenever possible (that is almost always).

The following parameter are for model characteristics:
- language: one or more model languages separated by commas (more details above).
- data_type: one or more model data types separated by commas (more details above).
- epsilon: each value equal to or less than this minimum will be converted to 0 (those above will be converted to 1 for data_type prevalence).

The following parameter are for the fit function:
- fit: the base parameter, define the base fit function, `auc`, `specificity` or `sensitivity`, the base fit is then modified by different penalty set below.
- k_penalty: the penalty applied per number of feature in the model (be careful with large number of feature, don't set this too high).
- fr_penalty: false rate penalty: add a part of sensitivity when fitting on specificity or symetrically. When set to 1.0 it's roughly a fit on accuracy.

The following parameter are for logging:
- nb_best_model_to_test: number of models to test and print in the last generation (0 means all models).
- log_base: if specified, logs are saved in a file starting with log_base.
- log_level: possible values are (complete log) trace > debug > info > warning > error (only error).
- display_level: the level of precision in displaying the final results: 2: variable name, 1: variable index; 0: anonymised variables.
- display_colorful: should the terminal results be coloured to make them easier to read?  

Finally, this section also has a parameter that is essential for GpredomicsR to work properly, but which can slow down execution purely from Rust. If you are not using the R version of the programme, it is advisable to leave it to false:

- keep_trace: compute metrics of every models. 

### cv

- outer_folds: number of folds for cross-validation (k-folds strategy).
- cv_best_models_ci_alpha: alpha for the family of best model confidence interval based on the best fit on validation fold. Smaller alpha, larger best_model range.
- n_permutations_oob: number of permutations per feature for OOB importance.
- scaled_importance: should importance be scaled by feature prevalence inside folds?
- importance_aggregation: aggregation method for importances within a fold and between folds : "mean" or "median".     

### data

- The name of the files are pretty obvious: X is the matrix of features, y the target vector.
- Xtest and ytest are optional holdouts (that should not be included in X and y).
- features_maximal_number_per_class: 0: all significant features ; else first X significant features (per class!) sorted according to their pvalue/log_abs_bayes_factor.
- feature_selection_method: possible values are wilcoxon (recommanded for sparse/log normal features), studentt (recommanded for features with normal distributions) and bayesian_fisher. wilcoxon is recommanded in most cases.
- feature_minimal_prevalence_pct is a filter: this is the minimal % of sample non null for the feature in either class for the feature to be kept.
- feature_minimal_feature_value is a filter on the average value of the feature (if either class is above, then the feature is not rejected).
- feature_maximal_pvalue is a filter to remove features which average is not significantly different between the two classes (non-Bayesian methods).
- feature_minimal_log_abs_bayes_factor is a filter to remove features with a fewer log absolute bayes factor will be removed (Bayesian method only)
- classes: the class names, for display in this order:
  - name of the negative class (0),
  - name of the positive class (1),
  - name of the unknown class (2).

### ga

- population_size: the number of model (a.k.a. Individual in the Rust code) per epoch.
- max_epochs: the target number of epoch (an epoch is a generation in the Genetic Algorithm).
- min_epochs: the minimal number of epoch befor trying to see if AUC are converging.
- max_age_best_model: when a best model reaches this age after min_epochs epochs, the ga algorithm is stopped.
- select_elite_pct: the percentage of best N-1 individual selected as parents.
- select_random_pct: the percentage of random N-1 individual selected as parents.
- mutated_children_pct: the percentage of children mutated.
- mutated_features_pct: the percentage of feature mutated in a mutated child.
- mutation_non_null_chance_pct: the likeliness of "positive" mutation, e.g. a mutation that select a new feature in the model.

### beam

- method: combinatorial or incremental, more details above.
- kmin: the number of variables used in the initial population.
- kmax: the maximum number of variables to considere in a single model, the variable count limit for beam algorithm.
- best_models_ci_alpha: alpha for the family of best model confidence interval based on the best fit. Smaller alpha, larger best_model range.
- max_nb_of_models: limits the number of features_to_keep at each epoch according to the number of models made possible by them (truncated according to the significiance).

### mcmc

- n_iter: number of MCMC (Markov Chain Monte Carlo) iterations.
- n_burn: number of MCMC iterations ignored (typically first half of all iterations)
- lambda: bayesian prior parameter for coefficients a, b, c 
- nmin: minimum number of features in a model after Sequential Backward Selection (0 : keep all preselected features - deactivate SBS)
- save_trace_outdir: outdir to save trace of the MCMC model used for the final prediction (for debugging/statistical exploration purposes)

### gpu

Finally, some technical parameters are available for the gpu: 

- fallback_to_cpu: executes the code on the CPU if there is no GPU available.
- memory_policy: Strict -> panic if below limits are not available | Adaptive -> adjusts below limits if not available | Performance -> uses all the available GPU memory regardless of below limits.
- max_total_memory_mb:: limit in mb defining the maximum amount of GPU memory used by all buffers.
- max_buffer_size_mb: limit in mb defining the maximum amount of GPU memory used by one buffer.

We recommend you to keep the default settings. Increasing these values is only useful if the number of models per stage (per GA generation or per k in BEAM) leads to a crash.