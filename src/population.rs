use crate::cv::CV;
use crate::data::Data;
use crate::experiment::ImportanceAggregation;
use crate::experiment::{Importance, ImportanceCollection, ImportanceScope, ImportanceType};
use crate::gpu::GpuAssay;
use crate::individual::Individual;
use crate::individual::{BINARY_LANG, POW2_LANG, RATIO_LANG, TERNARY_LANG};
use crate::param::FitFunction;
use crate::param::Param;
use crate::utils::{
    compute_auc_from_value, compute_metrics_from_classes, compute_roc_and_metrics_from_value,
    compute_threshold_and_metrics_with_precomputed_bootstrap, conf_inter_binomial,
    conf_inter_binomial_method, precompute_bootstrap_indices, PrecomputedBootstrap,
};
use crate::param::FbmCIMethod;
use crate::utils::{mad, mean_and_std, median};
use log::{error, info, warn};
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::prelude::SliceRandom;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::mem;

/// Group of [`Individual`]s
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Population {
    /// Individuals in the population grouped together
    pub individuals: Vec<Individual>,
}

impl Population {
    /// Checks compatibility of this population with data.
    /// Each individual in the population is checked for compatibility with the provided data.
    /// Incompatibility is determined by the individual's own `check_compatibility` method.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to check compatibility against.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether all individuals are compatible with the data.
    ///
    /// # Errors and Warnings
    ///
    /// Logs an error if the data has no features or samples.
    /// Logs a warning if the population is empty.
    ///
    /// Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # let population = Population::new();
    /// # let data = Data::new();
    /// let compatible = population.check_compatibility(&data);
    /// ```
    pub fn check_compatibility(&self, data: &Data) -> bool {
        // Check data is valid
        if data.feature_len == 0 {
            error!("Data has no features (feature_len = 0)");
            return false;
        }
        if data.sample_len == 0 {
            error!("Data has no samples (sample_len = 0)");
            return false;
        }

        // Warn if population is empty
        if self.individuals.is_empty() {
            warn!("Population is empty");
            return true; // Not an error, just a warning
        }

        let mut all_compatible = true;
        let mut incompatible_count = 0;

        // Check each individual using its own check_compatibility method
        for individual in self.individuals.iter() {
            if !individual.check_compatibility(data) {
                incompatible_count += 1;
                all_compatible = false;
            }
        }

        if incompatible_count > 0 {
            error!(
                "Population: {} out of {} individuals are incompatible with data",
                incompatible_count,
                self.individuals.len()
            );
        }

        all_compatible
    }

    /// Generates a display string for the population in a human-readable format.
    ///
    /// # Arguments
    ///
    /// * `data` - The training data used for evaluation.
    /// * `data_to_test` - Optional test data for additional evaluation.
    /// * `param` - Parameters influencing the display format.
    ///
    /// # Returns
    ///
    /// A formatted string representing the population's metrics.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # use gpredomics::param::Param;
    /// # let mut population = Population::new();
    /// # let train_data = Data::new();
    /// # let test_data = Data::new();
    /// # let params = Param::default();
    /// let display_str = population.display(&train_data, Some(&test_data), &params);
    /// ```
    pub fn display(&mut self, data: &Data, data_to_test: Option<&Data>, param: &Param) -> String {
        if param.general.algo == "mcmc" {
            let other_set = if data_to_test.is_some() {
                "Test"
            } else {
                "Train"
            };
            let mut str = format!("Displaying bayesian model with the greatest log evidence. Metrics are shown in the following order: Train/{}.", other_set);

            let (
                train_auc,
                train_best_threshold,
                train_best_acc,
                train_best_sens,
                train_best_spec,
                _,
            ) = self.bayesian_compute_roc_and_metrics(&data);

            str = if let Some(data_to_test) = data_to_test {
                let (test_auc, test_best_acc, test_best_sens, test_best_spec) =
                    self.bayesian_compute_metrics(&data_to_test, train_best_threshold);
                format!("{}\n\nBayesian model {}:{} [n_st = {}] AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}", 
                    str, self.individuals[0].get_language(), self.individuals[0].get_data_type(), self.individuals.len(),
                    train_auc, test_auc, train_best_acc, test_best_acc, train_best_sens, test_best_sens, train_best_spec, test_best_spec)
            } else {
                format!("{}\nBayesian model {}:{} [n_st = {}] AUC {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3}", 
                    str, self.individuals[0].get_language(), self.individuals[0].get_data_type(), self.individuals.len(),
                    train_auc, train_best_acc, train_best_sens, train_best_sens)
            };

            str = format!("{}\n\nThese metrics were calculated by optimizing the probability of decision (threshold={:.3} instead of 0.5)", str, train_best_threshold);
            str = format!("{}\nNote: the number of iterations corresponds to the sum of the number of betas and feature coefficient variations, resulting in a number greater than n_iter-n_burn)", str);
            format!("{}\nTo reproduce these results, relaunch this exact training with data to be predicted and/or export the MCMC trace with save_trace_outdir", str)
        } else {
            let limit = if (self.individuals.len() as u32) < param.general.n_model_to_display
                || param.general.n_model_to_display == 0
            {
                self.individuals.len() as u32
            } else {
                param.general.n_model_to_display
            };

            let mut str = String::new();

            let fbm = self.select_best_population(0.05);

            if !fbm.individuals.is_empty() {
                let train_aucs: Vec<f64> = fbm.individuals.iter().map(|i| i.auc).collect();
                let train_accuracies: Vec<f64> =
                    fbm.individuals.iter().map(|i| i.accuracy).collect();
                let train_sensitivities: Vec<f64> =
                    fbm.individuals.iter().map(|i| i.sensitivity).collect();
                let train_specificities: Vec<f64> =
                    fbm.individuals.iter().map(|i| i.specificity).collect();

                let (train_auc_mean, _) = mean_and_std(&train_aucs);
                let (train_acc_mean, _) = mean_and_std(&train_accuracies);
                let (train_sens_mean, _) = mean_and_std(&train_sensitivities);
                let (train_spec_mean, _) = mean_and_std(&train_specificities);

                let fbm_text = if let Some(data_to_test) = data_to_test {
                    let test_aucs: Vec<f64> = fbm
                        .individuals
                        .iter()
                        .map(|i| i.compute_new_auc(data_to_test))
                        .collect();
                    let mut test_acc_vec = Vec::new();
                    let mut test_sens_vec = Vec::new();
                    let mut test_spec_vec = Vec::new();

                    for individual in &fbm.individuals {
                        let (acc, sens, spec, _, _) = individual.compute_metrics(data_to_test);
                        test_acc_vec.push(acc);
                        test_sens_vec.push(sens);
                        test_spec_vec.push(spec);
                    }

                    let (test_auc_mean, _) = mean_and_std(&test_aucs);
                    let (test_acc_mean, _) = mean_and_std(&test_acc_vec);
                    let (test_sens_mean, _) = mean_and_std(&test_sens_vec);
                    let (test_spec_mean, _) = mean_and_std(&test_spec_vec);

                    format!("\n\x1b[1;33mFBM mean (n={}) - AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}\x1b[0m\n", 
                        fbm.individuals.len(), train_auc_mean, test_auc_mean, train_acc_mean, test_acc_mean,
                        train_sens_mean, test_sens_mean, train_spec_mean, test_spec_mean)
                } else {
                    format!("\n\x1b[1;33mFBM mean (n={}) - AUC {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3}\x1b[0m\n", 
                        fbm.individuals.len(), train_auc_mean, train_acc_mean, train_sens_mean, train_spec_mean)
                };

                str = format!("{}{}", str, fbm_text);
            }

            // Detect CV mode from algo tag containing "cv" or "fold"
            let is_cv_mode = param.tag.contains("cv") || param.tag.contains("fold");

            str = if is_cv_mode {
                format!("{}Displaying {} models. Metrics are shown in the following order: Train fold/Validation fold.", str, limit)
            } else {
                format!(
                    "{}Displaying {} models. Metrics are shown in the following order: Train/Test.",
                    str, limit
                )
            };

            // Compute all metrics if they weren't already computed during fit
            if !param.general.keep_trace {
                self.compute_all_metrics(data, &param.general.fit);
            }

            for i in 0..limit as usize {
                let model_display = self.individuals[i].display(
                    data,
                    data_to_test,
                    &param.general.algo,
                    param.general.threshold_ci_alpha,
                );

                let model_text = format!("\x1b[1;93m#{:?}\x1b[0m", i + 1);
                str = format!("{}\nModel {} {}\n ", str, model_text, model_display);
            }
            str
        }
    }

    //// Generates a new, empty Population with no individuals.
    ///
    /// # Returns
    ///
    /// A new `Population` instance with an empty individuals vector.
    pub fn new() -> Population {
        Population {
            individuals: Vec::new(),
        }
    }

    /// Computes the hash for each individual in the population.
    /// This method iterates over all individuals and invokes their `compute_hash` method.
    ///
    /// Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # let mut population = Population::new();
    /// population.compute_hash();
    /// ```
    pub fn compute_hash(&mut self) {
        for individual in &mut self.individuals {
            individual.compute_hash();
        }
    }

    /// Removes duplicate individuals (clones) from the population based on their hash values.
    ///
    /// # Returns
    ///
    /// The number of clones that were removed from the population.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # let mut population = Population::new();
    /// let num_clones_removed = population.remove_clone();
    /// ```
    pub fn remove_clone(&mut self) -> u32 {
        let mut clone_number: u32 = 0;
        let mut unique_individuals: Vec<Individual> = Vec::new();
        let mut hash_vector: Vec<u64> = Vec::new();

        let individuals = mem::take(&mut self.individuals);
        for individual in individuals.into_iter() {
            if hash_vector.contains(&individual.hash) {
                clone_number += 1;
            } else {
                hash_vector.push(individual.hash);
                unique_individuals.push(individual);
            }
        }
        self.individuals = unique_individuals;

        clone_number
    }

    /// Fits the population to the provided data, applying penalties as specified in the parameters.
    ///
    /// # Arguments
    ///
    /// * `data` - The training data used for fitting.
    /// * `_test_data` - Optional test data for additional evaluation (deprecated)
    /// * `gpu_assay` - Optional GPU assay for accelerated computations
    /// *  `_test_assay` - Optional GPU assay for test data (deprecated)
    /// * `param` - Parameters influencing the fitting process.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # use gpredomics::gpu::GpuAssay;
    /// # use gpredomics::param::Param;
    /// # let mut population = Population::new();
    /// # let train_data = Data::new();
    /// # let mut test_data = Data::new();
    /// # let params = Param::default();
    /// # let gpu_assay: Option<GpuAssay> = None;
    /// # let test_assay: Option<GpuAssay> = None;
    /// population.fit(&train_data, &mut Some(test_data), &gpu_assay, &test_assay, &params);
    /// ```
    pub fn fit(
        &mut self,
        data: &Data,
        _test_data: &mut Option<Data>,
        gpu_assay: &Option<GpuAssay>,
        _test_assay: &Option<GpuAssay>,
        param: &Param,
    ) {
        self.fit_without_penalty(data, _test_data, gpu_assay, _test_assay, param);
        self.penalize(data, param);
    }

    /// Applies penalties to the fitness scores of individuals in the population based on various criteria.
    ///
    /// # Arguments
    ///
    /// * `data` - The training data used for evaluation.
    /// * `param` - Parameters influencing the penalty application.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # use gpredomics::param::Param;
    /// # let mut population = Population::new();
    /// # let train_data = Data::new();
    /// # let params = Param::default();
    /// population.penalize(&train_data, &params);
    /// ```
    pub fn penalize(&mut self, data: &Data, param: &Param) {
        self.individuals.par_iter_mut().for_each(|i| {
            // k
            i.fit = i.fit - i.k as f64 * param.general.k_penalty;
            // Rejection rate
            if let Some(ref mut threshold_ci) = i.threshold_ci {
                i.fit = i.fit - (param.general.threshold_ci_penalty * threshold_ci.rejection_rate);
            };
            // Bias penalty
            if param.general.bias_penalty != 0.0 {
                if i.sensitivity < 0.5 {
                    i.fit = i.fit - (1.0 - i.sensitivity) * param.general.bias_penalty
                };
                if i.specificity < 0.5 {
                    i.fit = i.fit - (1.0 - i.specificity) * param.general.bias_penalty
                };
            };
            // User penalties
            if let Some(feature_annotations) = &data.feature_annotations {
                let lambda = param.general.user_penalties_weight;
                if lambda > 0.0 {
                    let mut acc = 0.0f64;
                    let mut total_weight = 0.0f64;

                    for (feat_idx, coef) in i.features.iter() {
                        let weight = (*coef).abs() as f64;
                        total_weight += weight;

                        if let Some(penalty) = feature_annotations.feature_penalty.get(feat_idx) {
                            if *penalty > 0.0 {
                                acc += weight * penalty;
                            }
                        }
                    }

                    if total_weight > 0.0 {
                        i.fit -= lambda * (acc / total_weight);
                    }
                }
            }
        });
    }

    /// Fits the population to the provided data without applying penalties.
    ///
    /// # Arguments
    ///
    /// * `data` - The training data used for fitting.
    /// * `_test_data` - Optional test data for additional evaluation (deprecated)
    /// * `gpu_assay` - Optional GPU assay for accelerated computations
    /// *  `_test_assay` - Optional GPU assay for test data (deprecated)
    /// * `param` - Parameters influencing the fitting process.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # use gpredomics::gpu::GpuAssay;
    /// # use gpredomics::param::Param;
    /// # let mut population = Population::new();
    /// # let train_data = Data::new();
    /// # let mut test_data = Data::new();
    /// # let params = Param::default();
    /// # let gpu_assay: Option<GpuAssay> = None;
    /// # let test_assay: Option<GpuAssay> = None;
    /// population.fit_without_penalty(&train_data, &mut Some(test_data), &gpu_assay, &test_assay, &params);
    /// ```
    pub fn fit_without_penalty(
        &mut self,
        data: &Data,
        _test_data: &mut Option<Data>,
        gpu_assay: &Option<GpuAssay>,
        _test_assay: &Option<GpuAssay>,
        param: &Param,
    ) {
        // Precompute bootstrap indices if any individual needs threshold CI
        let precomputed_bootstrap: Option<PrecomputedBootstrap> =
            if self.individuals.iter().any(|i| i.threshold_ci.is_some()) {
                let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
                Some(precompute_bootstrap_indices(
                    &data.y,
                    param.general.threshold_ci_n_bootstrap,
                    param.general.threshold_ci_alpha,
                    param.general.threshold_ci_frac_bootstrap,
                    &mut rng,
                ))
            } else {
                None
            };
        let mut all_scores: Vec<f64> = vec![];
        if let Some(assay) = gpu_assay {
            all_scores = assay
                .compute_scores(&self.individuals, param.general.data_type_epsilon as f32)
                .into_iter()
                .map(|x| x as f64)
                .collect();
        }
        self.individuals
            .par_iter_mut()
            .enumerate()
            .for_each(|(n, i)| {
                let scores: Vec<f64>;
                if let Some(_assay) = gpu_assay {
                    scores = all_scores[n * data.sample_len..(n + 1) * data.sample_len].to_vec();
                } else {
                    scores = i.evaluate(data)
                }
                let penalties = match param.general.fit {
                    FitFunction::sensitivity => Some([param.general.fr_penalty, 1.0]),
                    FitFunction::specificity => Some([1.0, param.general.fr_penalty]),
                    _ => None,
                };
                match param.general.fit {
                    FitFunction::auc => {
                        if param.general.keep_trace || param.general.bias_penalty != 0.0 {
                            if let Some(ref mut threshold_ci) = i.threshold_ci {
                                (
                                    i.auc,
                                    [threshold_ci.lower, i.threshold, threshold_ci.upper],
                                    i.accuracy,
                                    i.sensitivity,
                                    i.specificity,
                                    _,
                                    threshold_ci.rejection_rate,
                                ) = compute_threshold_and_metrics_with_precomputed_bootstrap(
                                    &scores,
                                    &data.y,
                                    &param.general.fit,
                                    None,
                                    &precomputed_bootstrap.as_ref().unwrap(),
                                );
                            } else {
                                (
                                    i.auc,
                                    i.threshold,
                                    i.accuracy,
                                    i.sensitivity,
                                    i.specificity,
                                    _,
                                ) = compute_roc_and_metrics_from_value(
                                    &scores,
                                    &data.y,
                                    &param.general.fit,
                                    None,
                                );
                            }
                        } else {
                            // The AUC calculation can be optimised if the user does not wish to calculate the other metrics at the same time.
                            if let Some(ref mut threshold_ci) = i.threshold_ci {
                                (
                                    i.auc,
                                    [threshold_ci.lower, i.threshold, threshold_ci.upper],
                                    _,
                                    _,
                                    _,
                                    _,
                                    threshold_ci.rejection_rate,
                                ) = compute_threshold_and_metrics_with_precomputed_bootstrap(
                                    &scores,
                                    &data.y,
                                    &param.general.fit,
                                    None,
                                    &precomputed_bootstrap.as_ref().unwrap(),
                                );
                            } else {
                                i.auc = compute_auc_from_value(&scores, &data.y);
                            }
                        }
                        i.fit = i.auc;
                    }
                    _ => {
                        if let Some(ref mut threshold_ci) = i.threshold_ci {
                            (
                                i.auc,
                                [threshold_ci.lower, i.threshold, threshold_ci.upper],
                                i.accuracy,
                                i.sensitivity,
                                i.specificity,
                                i.fit,
                                threshold_ci.rejection_rate,
                            ) = compute_threshold_and_metrics_with_precomputed_bootstrap(
                                &scores,
                                &data.y,
                                &param.general.fit,
                                penalties,
                                &precomputed_bootstrap.as_ref().unwrap(),
                            );
                        } else {
                            (
                                i.auc,
                                i.threshold,
                                i.accuracy,
                                i.sensitivity,
                                i.specificity,
                                i.fit,
                            ) = compute_roc_and_metrics_from_value(
                                &scores,
                                &data.y,
                                &param.general.fit,
                                penalties,
                            );
                        }
                        match param.general.fit {
                            FitFunction::mcc => i.metrics.mcc = Some(i.fit),
                            FitFunction::f1_score => i.metrics.f1_score = Some(i.fit),
                            FitFunction::g_mean => i.metrics.g_mean = Some(i.fit),
                            FitFunction::npv => i.metrics.npv = Some(i.fit),
                            FitFunction::ppv => i.metrics.ppv = Some(i.fit),
                            _ => {}
                        }
                    }
                }
            })
    }

    /// Sorts the population based on individuals' fitness scores, handling special cases for NaN and infinite values.
    ///
    /// # Returns
    ///
    /// The sorted `Population` instance.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # let population = Population::new();
    /// let sorted_population = population.sort();
    /// ```
    pub fn sort(mut self) -> Self {
        use std::cmp::Reverse;
        const TIE_EPS: f64 = 1e-6; // tolerance for tie-breaking

        self.individuals.sort_by_key(|ind| {
            // Case: 0=+inf, 1=finite, 2=-inf, 3=NaN (ascending order => +inf first)
            let cat: u8 = if ind.fit.is_nan() {
                3
            } else if ind.fit.is_infinite() {
                if ind.fit.is_sign_positive() {
                    0
                } else {
                    2
                }
            } else {
                1
            };

            // Symmetric quantification of fit by epsilon step
            let qfit: i64 = if cat == 1 {
                let v = (ind.fit / TIE_EPS).round();
                if v > (i64::MAX as f64) {
                    i64::MAX
                } else if v < (i64::MIN as f64) {
                    i64::MIN
                } else {
                    v as i64
                }
            } else {
                0
            };

            // Composite order: category asc, quantified fit desc, hash desc
            (cat, Reverse(qfit), Reverse(ind.hash))
        });

        self
    }

    /// Fits the population using cross-validation folds, applying penalties based on overfitting.
    ///
    /// # Arguments
    ///
    /// * `cv` - The cross-validation structure containing folds.
    /// * `param` - Parameters influencing the fitting process.
    /// * `gpu_assays` - Vector of GPU assays for each fold.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use gpredomics::population::Population;
    /// # use gpredomics::cv::CV;
    /// # use gpredomics::param::Param;
    /// # use gpredomics::gpu::GpuAssay;
    /// # let mut population = Population::new();
    /// # let cv = CV::new();
    /// # let params = Param::default();
    /// # let gpu_assays: Vec<(Option<GpuAssay>, Option<GpuAssay>)> = vec![];
    /// population.fit_on_folds(&cv, &params, &gpu_assays);
    /// ```
    pub fn fit_on_folds(
        &mut self,
        cv: &CV,
        param: &Param,
        gpu_assays: &Vec<(Option<GpuAssay>, Option<GpuAssay>)>,
    ) {
        let num_individuals = self.individuals.len();
        let num_folds = cv.validation_folds.len();

        let mut fold_penalized_fits = vec![vec![0.0; num_folds]; num_individuals];

        for (fold_idx, fold) in cv.validation_folds.iter().enumerate() {
            let (val_gpu_assay, train_gpu_assay) = &gpu_assays[fold_idx];

            // Validation fold - use its GPU assay
            let mut fold_data = fold.clone();
            self.fit_without_penalty(&mut fold_data, &mut None, val_gpu_assay, &None, param);
            let validation_fits: Vec<f64> = self.individuals.iter().map(|ind| ind.fit).collect();

            // Training set - use its GPU assay
            let mut train_data = cv.training_sets[fold_idx].clone();
            self.fit_without_penalty(&mut train_data, &mut None, train_gpu_assay, &None, param);
            let training_fits: Vec<f64> = self.individuals.iter().map(|ind| ind.fit).collect();

            // Penalized scores
            for ind_idx in 0..num_individuals {
                let train_fit = training_fits[ind_idx];
                let val_fit = validation_fits[ind_idx];
                let overfitting_gap = (train_fit - val_fit).abs();

                fold_penalized_fits[ind_idx][fold_idx] =
                    train_fit - overfitting_gap * param.cv.overfit_penalty;
            }
        }

        // Final fit
        for (ind_idx, individual) in self.individuals.iter_mut().enumerate() {
            let fits = &fold_penalized_fits[ind_idx];
            individual.fit = fits.iter().sum::<f64>() / fits.len() as f64;
        }

        // Additional penalties
        // use any data of CV as they all have the information about significance
        self.penalize(&cv.training_sets[0], param);
    }

    /// Selects the best individuals in the population based on a confidence interval for the evaluation metric.
    /// This method returns the Family of Best Models (FBM) by selecting individuals whose fitness exceeds is statistically non-distinguishable from the best individual's fitness.
    ///
    /// # Arguments
    /// * `alpha` - Significance level for the confidence interval.
    ///
    /// # Returns
    ///
    /// A new `Population` instance containing the best individuals.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # let population = Population::new();
    /// let best_population = population.select_best_population(0.05);
    /// ```
    pub fn select_best_population(&self, alpha: f64) -> Population {
        // (Family of best models)
        // This function return a population containing only individuals with a Fit greater than the best individual evaluation metric lower bound
        // it require a sorted population and a fit between 0 et 1.
        let mut best_pop = Population::new();
        let mut eval: Vec<f64> = vec![];

        // Collect evaluation score == ind.fit
        for ind in &self.individuals {
            eval.push(ind.fit)
        }

        // Return empty population if no individuals
        if eval.is_empty() {
            return best_pop;
        }

        // Control the distribution of evaluation metric
        if &eval[0] > &1.0 || &eval[0] < &0.0 {
            warn!("Evaluation metric should be in the [0, 1] interval to compute Family of Best Models! Using AUC instead of fitness...");
            eval = vec![];
            for ind in &self.individuals {
                eval.push(ind.auc)
            }
        }

        let (lower_bound, _, _) = conf_inter_binomial(eval[0], self.individuals.len(), alpha);

        for ind in &self.individuals {
            if ind.fit > lower_bound {
                best_pop.individuals.push(ind.clone());
            }
            // indivduals are theoricaly sorted by fit
            else {
                break;
            }
        }

        best_pop
    }

    /// Select the Family of Best Models using a specified CI method.
    ///
    /// Like [`select_best_population`], but dispatches to the given [`FbmCIMethod`]
    /// instead of always using the Wald interval.
    pub fn select_best_population_with_method(
        &self,
        alpha: f64,
        method: &FbmCIMethod,
    ) -> Population {
        let mut best_pop = Population::new();
        let mut eval: Vec<f64> = self.individuals.iter().map(|i| i.fit).collect();

        if eval.is_empty() {
            return best_pop;
        }

        if eval[0] > 1.0 || eval[0] < 0.0 {
            warn!("Evaluation metric should be in [0, 1] for FBM! Using AUC instead of fitness...");
            eval = self.individuals.iter().map(|i| i.auc).collect();
        }

        let (lower_bound, _, _) =
            conf_inter_binomial_method(eval[0], self.individuals.len(), alpha, method);

        for ind in &self.individuals {
            if ind.fit > lower_bound {
                best_pop.individuals.push(ind.clone());
            } else {
                break;
            }
        }

        best_pop
    }

    /// Populates the population with a set of random individuals.
    ///
    /// This is the unified function that handles both uniform and weighted feature selection.
    /// Use `prior_weight` parameter to enable weighted selection, or pass `None` for uniform selection.
    ///
    /// # Arguments
    ///
    /// * `population_size` - The number of individuals to generate.
    /// * `k_min` - Minimum number of features per individual.
    /// * `k_max` - Maximum number of features per individual.
    /// * `language` - The language type for the individuals.
    /// * `data_type` - The data type for the individuals.
    /// * `epsilon` - Epsilon parameter for the individuals.
    /// * `data` - The data used for feature selection.
    /// * `threshold_ci` - Whether to compute threshold confidence intervals.
    /// * `prior_weight` - Optional reference to a HashMap for weighted feature selection.
    /// * `rng` - Random number generator.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # use gpredomics::individual::BINARY_LANG;
    /// # use rand::SeedableRng;
    /// # use rand_chacha::ChaCha8Rng;
    /// # let mut population = Population::new();
    /// # let data = Data::new();
    /// # let mut rng = ChaCha8Rng::from_entropy();
    /// population.generate(100, 5, 20, BINARY_LANG, 0, 0.1, &data, true, None, &mut rng);
    /// ```
    pub fn generate(
        &mut self,
        population_size: u32,
        k_min: usize,
        k_max: usize,
        language: u8,
        data_type: u8,
        epsilon: f64,
        data: &Data,
        threshold_ci: bool,
        prior_weight: Option<&HashMap<usize, f64>>,
        rng: &mut ChaCha8Rng,
    ) {
        for _ in 0..population_size {
            self.individuals.push(Individual::random_select(
                k_min,
                k_max,
                &data.feature_selection,
                &data.feature_class,
                language,
                data_type,
                epsilon,
                prior_weight,
                threshold_ci,
                rng,
            ))
        }
    }

    /// Adds individuals from another population to the current population.
    ///
    /// # Arguments
    ///
    /// * `population` - The population whose individuals are to be added.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # let mut population = Population::new();
    /// # let other_population = Population::new();
    /// population.add(other_population);
    /// ```
    pub fn add(&mut self, population: Population) {
        self.individuals.extend(population.individuals);
    }

    /// Extracts the top percentage of individuals from the population based on their order.
    ///
    /// # Arguments
    ///
    /// * `pct` - The percentage of individuals to select (0-100).
    ///
    /// # Returns
    ///
    /// A tuple containing the new `Population` with the selected individuals and the number of individuals selected.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # let population = Population::new();
    /// let (top_population, n_selected) = population.select_first_pct(10.0);
    /// ```
    pub fn select_first_pct(&self, pct: f64) -> (Population, usize) {
        let n: usize = (self.individuals.len() as f64 * pct / 100.0) as usize;

        (
            Population {
                individuals: self.individuals.iter().take(n).map(|i| i.clone()).collect(),
            },
            n,
        )
    }

    /// Selects a random subset of individuals from the population above a specified index.
    ///
    /// # Arguments
    ///
    /// * `pct` - The percentage of individuals to select (0-100).
    /// * `n` - The index above which individuals are considered for selection.
    /// * `rng` - Random number generator.
    ///
    /// # Returns
    ///
    /// A new `Population` containing the randomly selected individuals.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use rand::SeedableRng;
    /// # use rand_chacha::ChaCha8Rng;
    /// # let population = Population::new();
    /// # let mut rng = ChaCha8Rng::from_entropy();
    /// let random_population = population.select_random_above_n(10.0, 5, &mut rng);
    /// ```
    pub fn select_random_above_n(&self, pct: f64, n: usize, rng: &mut ChaCha8Rng) -> Population {
        let k = (self.individuals.len() as f64 * pct / 100.0) as usize;

        Population {
            individuals: self.individuals[n..]
                .choose_multiple(rng, k)
                .cloned()
                .collect(),
        }
    }

    /// Computes all evaluation metrics for each individual in the population using the provided data and fitting method.
    ///
    /// # Arguments
    ///
    /// * `data` - The data used for evaluation.
    /// * `method` - The fitting method used for evaluation.
    ///
    /// # Examples
    ///  
    /// ```ignore
    /// population.compute_all_metrics(&data, &FitFunction::auc);
    /// ```
    pub fn compute_all_metrics(&mut self, data: &Data, method: &FitFunction) {
        // Precompute bootstrap indices if any individual needs threshold CI
        let precomputed_bootstrap: Option<PrecomputedBootstrap> =
            if self.individuals.iter().any(|i| i.threshold_ci.is_some()) {
                // Use a fixed seed for reproducibility in metrics computation
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                Some(precompute_bootstrap_indices(
                    &data.y, 1000, 0.05, 0.8, &mut rng,
                ))
            } else {
                None
            };

        self.individuals.par_iter_mut().for_each(|ind| {
            if ind.threshold_ci.is_some() {
                let scores = ind.evaluate(data);
                let threshold_ci = ind.threshold_ci.as_mut().unwrap();
                (
                    ind.auc,
                    [threshold_ci.lower, ind.threshold, threshold_ci.upper],
                    ind.accuracy,
                    ind.sensitivity,
                    ind.specificity,
                    _,
                    threshold_ci.rejection_rate,
                ) = compute_threshold_and_metrics_with_precomputed_bootstrap(
                    &scores,
                    &data.y,
                    method,
                    None,
                    &precomputed_bootstrap.as_ref().unwrap(),
                );
            } else {
                (
                    ind.auc,
                    ind.threshold,
                    ind.accuracy,
                    ind.sensitivity,
                    ind.specificity,
                    _,
                ) = ind.compute_roc_and_metrics(data, method, None);
            }
        });
    }

    /// Retrieves an individual from the population based on its hash value.
    ///
    /// # Arguments
    ///
    /// * `hash` - The hash value of the individual to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the `Individual` if found, or `None` if not found.
    ///
    /// # Panics
    ///
    /// Panics if the hash values of individuals have not been computed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # let population = Population::new();
    /// # let hash_value = 12345u64;
    /// if let Some(individual) = population.get_ind_from_hash(hash_value) {
    ///     // Use the individual
    /// }
    /// ```
    pub fn get_ind_from_hash(&self, hash: u64) -> Option<&Individual> {
        assert!(
            self.individuals[0].hash != 0,
            "Hash should be computed to allow Individual selection"
        );
        self.individuals.par_iter().find_any(|ind| ind.hash == hash)
    }

    /// Computes Mean Decreased Accuracy (MDA) feature importance for the population.
    ///
    /// # Arguments
    ///
    /// * `data` - The data used for evaluation.
    /// * `permutations` - The number of permutations to perform for MDA calculation.
    /// * `main_rng` - The main random number generator for reproducibility.
    /// * `aggregation_method` - The method used for aggregating individual importances.
    /// * `scaled_importance` - Whether to scale the importance values by feature prevalence.
    /// * `cascade` - Whether to include individual-level importances in the final result.
    /// * `population_id` - Optional identifier for the population.
    ///
    /// # Returns
    ///
    /// An `ImportanceCollection` containing the computed feature importances.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # use gpredomics::experiment::ImportanceAggregation;
    /// # use rand::SeedableRng;
    /// # use rand_chacha::ChaCha8Rng;
    /// # let population = Population::new();
    /// # let data = Data::new();
    /// # let mut rng = ChaCha8Rng::from_entropy();
    /// let importance_collection = population.compute_pop_mda_feature_importance(
    ///     &data,
    ///     100,
    ///     &mut rng,
    ///     &ImportanceAggregation::mean,
    ///     true,
    ///     false,
    ///     Some(1),
    /// );
    /// ```
    pub fn compute_pop_mda_feature_importance(
        &self,
        data: &Data,
        permutations: usize,
        main_rng: &mut ChaCha8Rng,
        aggregation_method: &ImportanceAggregation,
        scaled_importance: bool,
        cascade: bool,
        population_id: Option<usize>,
    ) -> ImportanceCollection {
        // Collect all features from all individuals
        let mut all_features: Vec<usize> = self
            .individuals
            .iter()
            .flat_map(|ind| ind.features_index())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        all_features.sort_unstable();

        // Calculate feature prevalences (proportion of individuals possessing each feature)
        let mut feature_prevalences = HashMap::new();
        for &feature_idx in &all_features {
            let count = self
                .individuals
                .iter()
                .filter(|ind| ind.features.contains_key(&feature_idx))
                .count();
            let prevalence = count as f64 / self.individuals.len() as f64;
            feature_prevalences.insert(feature_idx, prevalence);
        }

        // Generate fixed seeds for each feature for reproducibility
        let mut feature_seeds: HashMap<usize, Vec<u64>> = HashMap::new();
        for &feature_idx in &all_features {
            let seeds: Vec<u64> = (0..permutations).map(|_| main_rng.next_u64()).collect();
            feature_seeds.insert(feature_idx, seeds);
        }

        // Sort individuals by hash for reproducibility
        let mut order: Vec<usize> = (0..self.individuals.len()).collect();
        order.sort_by_key(|&i| self.individuals[i].hash);

        let n_individuals = order.len();
        let n_features = all_features.len();
        info!(
            "MDA importance: {} individuals x {} features x {} permutations",
            n_individuals, n_features, permutations
        );

        // Use Individual::compute_mda_feature_importance for each individual
        let completed = AtomicUsize::new(0);
        let individual_results: Vec<_> = order
            .par_iter()
            .map(|&idx| {
                let result = self.individuals[idx].compute_mda_feature_importance(
                    data,
                    permutations,
                    &all_features,
                    &feature_seeds,
                );
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                // Log progress every 10% or at least every 10 individuals
                let step = std::cmp::max(1, n_individuals / 10);
                if done % step == 0 || done == n_individuals {
                    info!(
                        "MDA importance progress: {}/{} individuals ({:.0}%)",
                        done,
                        n_individuals,
                        done as f64 / n_individuals as f64 * 100.0
                    );
                }
                result
            })
            .collect();

        // Aggregate results across individuals for Population-level importances
        let mut feature_importance_values: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut feature_counts: HashMap<usize, usize> = HashMap::new();

        // Extract importance values from individual results for aggregation
        for importance_collection in &individual_results {
            for importance in &importance_collection.importances {
                // Collect values for aggregation
                feature_importance_values
                    .entry(importance.feature_idx)
                    .or_insert_with(Vec::new)
                    .push(importance.importance);

                // Count individuals that actually use this feature (non-zero importance)
                if importance.importance != 0.0 {
                    *feature_counts.entry(importance.feature_idx).or_insert(0) += 1;
                }
            }
        }

        // Calculate population-level aggregated importances
        let mut population_importances = Vec::new();

        for (&feature_idx, values) in &feature_importance_values {
            if values.is_empty() {
                continue;
            }

            // Calculate aggregated value based on method
            let (aggregated_value, dispersion) = match aggregation_method {
                ImportanceAggregation::mean => mean_and_std(&values),
                ImportanceAggregation::median => {
                    let mut buf = values.clone();
                    let med = median(&mut buf);
                    (med, mad(&buf))
                }
            };

            // Get prevalence for this feature
            let prevalence = feature_prevalences
                .get(&feature_idx)
                .copied()
                .unwrap_or(0.0);

            // Apply scaling if requested
            let (final_importance, final_dispersion) = if scaled_importance {
                (aggregated_value * prevalence, dispersion * prevalence)
            } else {
                (aggregated_value, dispersion)
            };

            // Create Population-level Importance object
            let population_importance = Importance {
                importance_type: ImportanceType::MDA,
                feature_idx,
                scope: ImportanceScope::Population {
                    id: population_id.unwrap_or(0),
                },
                aggreg_method: Some(aggregation_method.clone()),
                importance: final_importance,
                is_scaled: scaled_importance,
                dispersion: final_dispersion,
                scope_pct: prevalence,
                direction: None,
            };

            population_importances.push(population_importance);
        }

        population_importances.sort_by_key(|imp| imp.feature_idx);

        // Build final result based on cascade mode
        let mut final_importances = Vec::new();

        // Add individual-level importances if cascade is enabled
        if cascade {
            for importance_collection in &individual_results {
                final_importances.extend(importance_collection.importances.clone());
            }
        }

        final_importances.extend(population_importances);

        ImportanceCollection {
            importances: final_importances,
        }
    }

    /// Performs Bayesian prediction by averaging the predictions of all individuals in the population.
    ///
    /// # Arguments
    ///
    /// * `data` - The data used for prediction.
    ///
    /// # Returns
    ///
    /// A vector of predicted probabilities for each sample.
    ///
    /// # Examples
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # let population = Population::new();
    /// # let data = Data::new();
    /// let predictions = population.bayesian_predict(&data);
    /// ```
    pub fn bayesian_predict(&self, data: &Data) -> Vec<f64> {
        let mut bayesian_prob: Vec<f64> = vec![0.0; data.sample_len];
        for ind in &self.individuals {
            let sample_prob = ind.evaluate(&data);
            for (sample_sum, &val) in bayesian_prob.iter_mut().zip(sample_prob.iter()) {
                *sample_sum += val;
            }
        }

        let pop_size = self.individuals.len() as f64;
        for sample in bayesian_prob.iter_mut() {
            *sample /= pop_size;
        }

        bayesian_prob
    }

    /// Classifies samples based on Bayesian prediction probabilities and a specified threshold.
    ///  
    /// # Arguments
    ///
    /// * `data` - The data used for classification.
    /// * `threshold` - The threshold for classification.
    ///
    pub fn bayesian_class(&self, data: &Data, threshold: f64) -> Vec<u8> {
        let probs = self.bayesian_predict(data);
        probs
            .iter()
            .map(|p| if *p > threshold { 1 } else { 0 })
            .collect()
    }

    /// Computes ROC and evaluation metrics based on Bayesian predictions.
    ///
    /// # Arguments
    ///
    /// * `data` - The data used for evaluation.
    ///
    /// # Returns
    ///
    /// A tuple containing AUC, accuracy, sensitivity, specificity, PPV, and NPV.
    pub fn bayesian_compute_roc_and_metrics(&self, data: &Data) -> (f64, f64, f64, f64, f64, f64) {
        compute_roc_and_metrics_from_value(
            &self.bayesian_predict(data),
            &data.y,
            &FitFunction::auc,
            None,
        )
    }

    /// Computes AUC, accuracy, sensitivity, and specificity based on Bayesian predictions and a specified threshold.
    ///
    /// # Arguments
    ///
    /// * `data` - The data used for evaluation.
    /// * `threshold` - The threshold for classification.
    ///
    /// # Returns
    ///
    /// A tuple containing AUC, accuracy, sensitivity, and specificity.
    pub fn bayesian_compute_metrics(&self, data: &Data, threshold: f64) -> (f64, f64, f64, f64) {
        let (acc, se, sp, _) = compute_metrics_from_classes(
            &self.bayesian_class(data, threshold),
            &data.y,
            [false; 5],
        );
        (
            compute_auc_from_value(&self.bayesian_predict(&data), &data.y),
            acc,
            se,
            sp,
        )
    }

    /// Filters the population based on signed Jaccard dissimilarity between individuals.
    ///
    /// Niches can be considered to group individuals by family and data type before filtering.
    /// This will group individuals into two niches: linear family (Binary, Ternary, Pow2) & Ratio family and filter within each niche.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The dissimilarity threshold (0-100).
    /// * `considere_niche` - Whether to consider family niches during filtering.
    ///
    /// # Returns
    ///
    /// A new `Population` instance containing the filtered individuals.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # let population = Population::new();
    /// let filtered_population = population.filter_by_signed_jaccard_dissimilarity(30.0, true);
    /// ```
    pub fn filter_by_signed_jaccard_dissimilarity(
        &self,
        threshold: f64,
        considere_niche: bool,
    ) -> Population {
        if self.individuals.is_empty() || threshold <= 0.0 {
            return Population {
                individuals: Vec::new(),
            };
        }

        let threshold_normalized = threshold / 100.0;

        let estimated_capacity = (self.individuals.len() as f64 * 0.3) as usize;
        let mut all_filtered = Vec::with_capacity(estimated_capacity);

        let groups: Vec<Vec<usize>> = if considere_niche {
            // Group individuals by family niche: (language family, data type)
            // Linear vs Ratio vs others
            let mut niches: rustc_hash::FxHashMap<(u8, u8), Vec<usize>> =
                rustc_hash::FxHashMap::default();

            for (idx, individual) in self.individuals.iter().enumerate() {
                let family_lang: u8 = match individual.language {
                    BINARY_LANG | TERNARY_LANG | POW2_LANG => 0, // linear family
                    RATIO_LANG => 1,                             // ratio family
                    other => other,                              // MCMC or future languages
                };

                let niche_key = (family_lang, individual.data_type);
                niches.entry(niche_key).or_insert_with(Vec::new).push(idx);
            }

            niches.into_values().collect()
        } else {
            vec![(0..self.individuals.len()).collect()]
        };

        groups
            .par_iter()
            .filter(|group| !group.is_empty())
            .map(|group| {
                let mut sorted_indices = group.clone();
                sorted_indices.par_sort_unstable_by(|&a, &b| {
                    self.individuals[b]
                        .fit
                        .partial_cmp(&self.individuals[a].fit)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut filtered_indices = Vec::with_capacity(sorted_indices.len());
                if !sorted_indices.is_empty() {
                    filtered_indices.push(sorted_indices[0]);

                    for &candidate_idx in sorted_indices.iter().skip(1) {
                        let candidate = &self.individuals[candidate_idx];
                        let mut is_different = true;

                        for &selected_idx in &filtered_indices {
                            let selected = &self.individuals[selected_idx];
                            if candidate.signed_jaccard_dissimilarity_with(selected)
                                < threshold_normalized
                            {
                                is_different = false;
                                break;
                            }
                        }

                        if is_different {
                            filtered_indices.push(candidate_idx);
                        }
                    }
                }

                filtered_indices
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .for_each(|idx| all_filtered.push(self.individuals[idx].clone()));

        Population {
            individuals: all_filtered,
        }
    }

    /// Generates a formatted string displaying the prevalence of features in the population.
    ///
    /// # Arguments
    ///
    /// * `data` - The data containing feature information.
    /// * `nb_features` - The number of top features to display (0 for all).
    ///
    /// # Returns
    ///
    /// A formatted string representing the feature prevalence table.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # let population = Population::new();
    /// # let data = Data::new();
    /// let prevalence_table = population.display_feature_prevalence(&data, 10);
    /// println!("{}", prevalence_table);
    /// ```
    pub fn display_feature_prevalence(&self, data: &Data, nb_features: usize) -> String {
        if self.individuals.is_empty() {
            return "No expert in population".to_string();
        }

        let mut feature_freq: HashMap<usize, usize> = HashMap::new();
        let mut feature_pos_count: HashMap<usize, usize> = HashMap::new();
        let mut feature_neg_count: HashMap<usize, usize> = HashMap::new();
        let total_experts = self.individuals.len();

        for individual in &self.individuals {
            for (&feature_idx, &coef) in individual.features.iter() {
                *feature_freq.entry(feature_idx).or_insert(0) += 1;

                if coef > 0 {
                    *feature_pos_count.entry(feature_idx).or_insert(0) += 1;
                } else if coef < 0 {
                    *feature_neg_count.entry(feature_idx).or_insert(0) += 1;
                }
            }
        }

        let mut freq_vec: Vec<(usize, usize)> = feature_freq.into_iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));

        // Get tag column names from feature annotations
        let tag_names: Vec<String> = if let Some(fa) = &data.feature_annotations {
            fa.tag_column_names.clone()
        } else {
            Vec::new()
        };

        let mut result = String::new();
        result.push_str(&format!(
            "\n\x1b[1m{} FEATURE PREVALENCE IN EXPERT POPULATION {}\x1b[0m\n",
            "~".repeat(25),
            "~".repeat(25)
        ));
        result.push_str(&format!("\n\n{}\n", "─".repeat(80)));

        // Header base
        result.push_str(&format!(
            "{:<29} | {:>12}\t| {:>10}",
            "Feature", "Association", "Prevalence",
        ));
        // Add dynamic tag headers
        for tag in &tag_names {
            result.push_str(&format!(" | {:<20}", tag));
        }
        result.push_str("\n");
        result.push_str(&format!("{}\n", "─".repeat(80 + tag_names.len() * 18)));

        let nb_features_to_show = if nb_features == 0 {
            freq_vec.len()
        } else {
            nb_features
        };

        let features_to_display = std::cmp::min(nb_features_to_show, freq_vec.len());

        for (feature_idx, count) in freq_vec.iter().take(features_to_display) {
            let prevalence_pct = (*count as f64 / total_experts as f64) * 100.0;

            let feature_name = if *feature_idx < data.features.len() {
                &data.features[*feature_idx]
            } else {
                "Unknown"
            };

            let truncated_name = if feature_name.len() > 29 {
                format!("{}...", &feature_name[..26])
            } else {
                feature_name.to_string()
            };

            let pos_count = *feature_pos_count.get(feature_idx).unwrap_or(&0);
            let neg_count = *feature_neg_count.get(feature_idx).unwrap_or(&0);

            let (colored_name, association_info) = if pos_count > 0 && neg_count == 0 {
                // Class 1 -> Blue
                (
                    format!("\x1b[1;96m{}\x1b[0m", truncated_name),
                    format!("\x1b[1;96m+{}\x1b[0m", pos_count),
                )
            } else if neg_count > 0 && pos_count == 0 {
                // Class 0 -> Magenta
                (
                    format!("\x1b[1;95m{}\x1b[0m", truncated_name),
                    format!("\x1b[1;95m-{}\x1b[0m", neg_count),
                )
            } else if pos_count > 0 && neg_count > 0 {
                // Mixed -> White
                (
                    format!("\x1b[0;39m{}\x1b[0m", truncated_name),
                    format!(
                        "\x1b[1;96m+{}\x1b[0m&\x1b[1;95m-{}\x1b[0m",
                        pos_count, neg_count
                    ),
                )
            } else {
                (
                    format!("\x1b[90m{}\x1b[0m", truncated_name),
                    "\x1b[90m~0\x1b[0m".to_string(),
                )
            };
            // Build row base
            result.push_str(&format!(
                "{:<40} | ({}) /{:<4}\t| {:>7.1}%",
                colored_name, association_info, total_experts, prevalence_pct,
            ));

            // Append tag values (if any), pad to tag_names length
            if !tag_names.is_empty() {
                if let Some(fa) = &data.feature_annotations {
                    let tag_vals = fa.feature_tags.get(feature_idx);
                    for i in 0..tag_names.len() {
                        let v = tag_vals
                            .and_then(|vals| vals.get(i))
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        result.push_str(&format!(" | {:<20}", v));
                    }
                } else {
                    for _ in 0..tag_names.len() {
                        result.push_str(&format!(" | {:<20}", ""));
                    }
                }
            }
            result.push_str("\n");
        }

        if freq_vec.len() > nb_features_to_show {
            result.push_str(&format!(
                "... and {} more features\n",
                freq_vec.len() - nb_features_to_show
            ));
        }

        result.push_str(&format!("\n{}\n", "─".repeat(70)));
        result.push_str("Legend: ");
        result.push_str(&format!(
            "\x1b[1;96mBlue\x1b[0m = Always positively associated (+1) | "
        ));
        result.push_str(&format!(
            "\x1b[1;95mMagenta\x1b[0m = Always negatively associated (-1) | "
        ));
        result.push_str("White = Mixed associations\n");
        result.push_str(&format!(
            "Total unique features: {}/{}\n",
            freq_vec.len(),
            data.feature_selection.len()
        ));

        result
    }

    /// Prunes all individuals in the population based on feature importance using parallel processing.
    ///
    /// Applies `Individual::prune_by_importance` to each individual in parallel.
    ///
    /// # Arguments
    ///
    /// * `data` - The data used for importance calculation.
    /// * `n_perm` - The number of permutations for importance calculation.
    /// * `base_rng_seed` - The base random seed for reproducibility.
    /// * `threshold` - Optional threshold for pruning.
    /// * `quantile` - Optional quantile range for pruning.
    /// * `min_k` - Minimum number of features to retain.
    ///
    /// # Returns
    ///
    /// A mutable reference to the `Population` for chaining.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::population::Population;
    /// # use gpredomics::data::Data;
    /// # let mut population = Population::new();
    /// # let data = Data::new();
    /// population.prune_all_by_importance(&data, 100, 42, Some(0.01), None, 5);
    /// ```
    pub fn prune_all_by_importance(
        &mut self,
        data: &Data,
        n_perm: usize,
        base_rng_seed: u64,
        threshold: Option<f64>,
        quantile: Option<(f64, f64)>,
        min_k: usize,
    ) -> &mut Self {
        use rand::RngCore;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rayon::prelude::*;

        // Generate deterministic seeds per individual based on hash
        let mut seed_rng = ChaCha8Rng::seed_from_u64(base_rng_seed);
        let mut individual_seeds: Vec<u64> = Vec::with_capacity(self.individuals.len());

        for ind in &self.individuals {
            individual_seeds.push(base_rng_seed ^ ind.hash ^ seed_rng.next_u64());
        }

        // Parallel pruning with Rayon
        self.individuals
            .par_iter_mut()
            .zip(individual_seeds.par_iter())
            .for_each(|(individual, &seed)| {
                individual.prune_by_importance(data, n_perm, seed, threshold, quantile, min_k);
            });

        self
    }
}

use std::fmt;
impl fmt::Debug for Population {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Population")
            .field("individuals", &self.individuals)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::individual::{AdditionalMetrics, DEFAULT_MINIMUM, RAW_TYPE, TERNARY_LANG};
    use rand::SeedableRng;
    use std::{collections::HashMap, f64::MAX};

    impl Population {
        /// Generates a test population with predefined individuals for testing purposes.
        ///
        /// # Returns
        ///
        /// A `Population` instance containing test individuals.
        pub fn test() -> Population {
            let mut pop = Population {
                individuals: Vec::new(),
            };

            for i in 0..10 {
                let ind = Individual {
                    features: vec![(0, i), (1, -i), (2, i * 2), (3, i % 3)]
                        .into_iter()
                        .collect(),
                    auc: 0.4 + (i as f64 * 0.05),
                    fit: 0.8 - (i as f64 * 0.02),
                    specificity: 0.15 + (i as f64 * 0.01),
                    sensitivity: 0.16 + (i as f64 * 0.01),
                    accuracy: 0.23 + (i as f64 * 0.03),
                    threshold: 42.0 + (i as f64),
                    k: (42 + i) as usize,
                    epoch: (42 + i) as usize,
                    language: (i % 4) as u8,
                    data_type: (i % 3) as u8,
                    hash: i as u64,
                    epsilon: f64::MIN_POSITIVE + (i as f64 * 0.001),
                    parents: None,
                    betas: None,
                    threshold_ci: None,
                    metrics: AdditionalMetrics {
                        mcc: None,
                        f1_score: None,
                        npv: None,
                        ppv: None,
                        g_mean: None,
                    },
                };
                pop.individuals.push(ind);
            }

            pop.individuals.push(pop.individuals[9].clone());

            pop
        }

        /// Generates a test population with a specified number of individuals, each having overlapping features.
        ///
        /// # Arguments
        ///
        /// * `num_individuals` - The number of individuals to generate.
        /// * `features_per_individual` - The number of features each individual should have.
        ///
        /// # Returns
        ///
        /// A `Population` instance containing the generated individuals.
        pub fn test_with_n_overlapping_features(
            num_individuals: usize,
            features_per_individual: usize,
        ) -> Population {
            let mut individuals = Vec::new();
            for i in 0..num_individuals {
                let mut features_map = HashMap::new();
                for feature_idx in 0..features_per_individual {
                    features_map.insert(feature_idx + i, 1i8); // Features shift with i for variety
                }
                individuals.push(Individual {
                    features: features_map,
                    auc: 0.8,
                    fit: 0.7,
                    specificity: 0.75,
                    sensitivity: 0.85,
                    accuracy: 0.80,
                    threshold: 0.5,
                    k: features_per_individual,
                    epoch: 0,
                    language: BINARY_LANG,
                    data_type: RAW_TYPE,
                    hash: i as u64,
                    epsilon: DEFAULT_MINIMUM,
                    parents: None,
                    betas: None,
                    threshold_ci: None,
                    metrics: AdditionalMetrics {
                        mcc: None,
                        f1_score: None,
                        npv: None,
                        ppv: None,
                        g_mean: None,
                    },
                });
            }
            Population { individuals }
        }

        /// Generates a test population with individuals having specified features.
        ///
        /// # Arguments
        ///
        /// * `feature_indices` - A slice of feature indices to be included in the individuals.
        ///
        /// # Returns
        ///
        /// A `Population` instance containing the generated individuals.
        pub fn test_with_these_features(feature_indices: &[usize]) -> Population {
            let mut pop = Population::new();
            for &feature_idx in feature_indices {
                let mut individual = Individual::new();
                individual.features.insert(feature_idx, 1i8);
                individual.k = 1;
                pop.individuals.push(individual);
            }
            pop
        }
    }

    // #[test] outdated - betas are now used to compute hash
    // fn test_hash_reproductibility() {
    //     let mut pop = Population::test();
    //     let test_stdout ="hashes of Individuals of the Population are not the same as generated in the past, indicating a reproducibility problem.";
    //     pop.compute_hash();
    //     assert_eq!(8828241517069783773, pop.individuals[0].hash, "{}", test_stdout);
    //     assert_eq!(10219684453491833036, pop.individuals[1].hash, "{}", test_stdout);
    //     assert_eq!(15701002319609139959, pop.individuals[2].hash, "{}", test_stdout);
    //     assert_eq!(8712745243790315600, pop.individuals[3].hash, "{}", test_stdout);
    //     assert_eq!(5577107020388865403, pop.individuals[4].hash, "{}", test_stdout);
    //     assert_eq!(573762283934476040, pop.individuals[5].hash, "{}", test_stdout);
    //     assert_eq!(8225502949628013706, pop.individuals[6].hash, "{}", test_stdout);
    //     assert_eq!(212344198819259482, pop.individuals[7].hash, "{}", test_stdout);
    //     assert_eq!(16067238433084305925, pop.individuals[8].hash, "{}", test_stdout);
    //     assert_eq!(9859973499533993323, pop.individuals[9].hash, "{}", test_stdout);
    //     assert_eq!(9859973499533993323, pop.individuals[10].hash, "same Individual of the Population should have the same hash");
    // }

    #[test]
    fn test_remove_clone() {
        let mut pop = Population::test();
        assert_eq!(
            1,
            pop.remove_clone(),
            "remove_clone() should return the number of clone eliminated"
        );
        assert_eq!(
            10,
            pop.individuals.len(),
            "remove_clone() should eliminate clones"
        );
        assert_eq!(
            0,
            pop.remove_clone(),
            "remove_clone() should return 0 when there is no clone"
        );
        assert_eq!(
            10,
            pop.individuals.len(),
            "remove_clone() should return the same Population if it does not contain clone"
        );
    }

    // smaller, better ?
    #[test]
    fn test_sort() {
        let mut pop = Population::test();
        let mut previous_fit = MAX as f64;
        pop = pop.sort();
        for i in 1..pop.individuals.len() {
            assert!(pop.individuals[i].fit <= previous_fit);
            previous_fit = pop.individuals[i].fit;
        }
    }

    #[test]
    fn test_generate() {
        let data = &Data::test_disc_data();
        let mut pop = Population::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let expected_features: Vec<(usize, i8)> = vec![
            (1, 1),
            (1, 1),
            (0, -1),
            (1, 1),
            (0, -1),
            (0, -1),
            (0, -1),
            (1, 1),
            (1, 1),
            (0, -1),
        ];

        pop.generate(
            10,
            1,
            3,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            data,
            false,
            None,
            &mut rng,
        );
        assert_eq!(
            pop.individuals.len(),
            10,
            "generated Population should be composed of population_size Individuals"
        );

        for i in 0..pop.individuals.len() {
            assert_eq!(
                pop.individuals[i].language, TERNARY_LANG,
                "each Individual of the generated Population should respect the input language"
            );
            assert_eq!(
                pop.individuals[i].data_type, RAW_TYPE,
                "each Individual of the generated Population should respect the input data_type"
            );
            assert_eq!(
                pop.individuals[i].epsilon, DEFAULT_MINIMUM,
                "each Individual of the generated Population should respect the input epsilon"
            );
            assert!(
                pop.individuals[i].k >= 1,
                "each Individual of the generated Population should have k_min or more k"
            );
            assert!(
                pop.individuals[i].k <= 3,
                "each Individual of the generated Population should have k_max or less k"
            );
            assert_eq!(expected_features.get(i).copied(), pop.individuals[i].features.iter().next().map(|(&k, &v)| (k, v)),
            "the generated Individual' features part of the generated Population are not the same as generated in the past for a same seed, indicating a reproducibility problem");
        }

        let mut empty_pop = Population::new();
        empty_pop.generate(
            0,
            1,
            3,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            data,
            false,
            None,
            &mut rng,
        );
        assert_eq!(
            empty_pop.individuals.len(),
            0,
            "generating an empty population (population_size=0) should return an empty population"
        );
    }

    #[test]
    fn test_generate_weighted() {
        let data = Data::test_disc_data();

        // Create feature annotations with prior weights
        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 1.0);
        prior_weight.insert(1, 1.0);

        let mut pop = Population::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        pop.generate(
            10,
            1,
            3,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &data,
            false,
            Some(&prior_weight),
            &mut rng,
        );

        assert_eq!(
            pop.individuals.len(),
            10,
            "generated Population should be composed of population_size Individuals"
        );

        for i in 0..pop.individuals.len() {
            assert_eq!(
                pop.individuals[i].language, TERNARY_LANG,
                "each Individual of the generated Population should respect the input language"
            );
            assert_eq!(
                pop.individuals[i].data_type, RAW_TYPE,
                "each Individual of the generated Population should respect the input data_type"
            );
            assert_eq!(
                pop.individuals[i].epsilon, DEFAULT_MINIMUM,
                "each Individual of the generated Population should respect the input epsilon"
            );
            assert!(
                pop.individuals[i].k >= 1,
                "each Individual of the generated Population should have k_min or more k"
            );
            assert!(
                pop.individuals[i].k <= 3,
                "each Individual of the generated Population should have k_max or less k"
            );
        }

        let mut empty_pop = Population::new();
        empty_pop.generate(
            0,
            1,
            3,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &data,
            false,
            Some(&prior_weight),
            &mut rng,
        );
        assert_eq!(
            empty_pop.individuals.len(),
            0,
            "generating an empty population (population_size=0) should return an empty population"
        );
    }

    #[test]
    fn test_generate_weighted_with_biased_weights() {
        let data = Data::test_disc_data();

        // Create prior weights favoring feature 0
        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 100.0); // High weight for feature 0
        prior_weight.insert(1, 0.1); // Low weight for feature 1

        let mut pop = Population::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        pop.generate(
            50,
            1,
            1, // Only select 1 feature per individual
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &data,
            false,
            Some(&prior_weight),
            &mut rng,
        );

        // Count how many individuals have feature 0
        let feature_0_count = pop
            .individuals
            .iter()
            .filter(|ind| ind.features.contains_key(&0))
            .count();

        // With heavily biased weights, feature 0 should be selected much more often
        assert!(
            feature_0_count > 40,
            "Feature with higher weight should be selected more often (got {} out of 50)",
            feature_0_count
        );
    }

    #[test]
    fn test_generate_weighted_reproducibility() {
        let data = Data::test_disc_data();

        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 1.0);
        prior_weight.insert(1, 1.0);

        let mut pop1 = Population::new();
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        pop1.generate(
            5,
            1,
            2,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &data,
            false,
            Some(&prior_weight),
            &mut rng1,
        );

        let mut pop2 = Population::new();
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        pop2.generate(
            5,
            1,
            2,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &data,
            false,
            Some(&prior_weight),
            &mut rng2,
        );

        // With same seed, populations should be identical
        assert_eq!(
            pop1.individuals.len(),
            pop2.individuals.len(),
            "Populations with same seed should have same size"
        );

        for i in 0..pop1.individuals.len() {
            assert_eq!(
                pop1.individuals[i].features, pop2.individuals[i].features,
                "Individuals at position {} should have same features with same seed",
                i
            );
        }
    }

    #[test]
    fn test_generate_unified_function() {
        let data = Data::test_disc_data();

        // Test without weights (None) - using TERNARY_LANG to avoid filtering issues
        let mut pop_no_weights = Population::new();
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        pop_no_weights.generate(
            5,
            1,
            2,
            TERNARY_LANG, // Changed to TERNARY_LANG
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &data,
            false,
            None,
            &mut rng1,
        );

        assert_eq!(pop_no_weights.individuals.len(), 5);
        for ind in &pop_no_weights.individuals {
            assert_eq!(ind.language, TERNARY_LANG);
            assert!(ind.k >= 1 && ind.k <= 2);
        }

        // Test with weights (Some)
        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 1.0);
        prior_weight.insert(1, 1.0);

        let mut pop_with_weights = Population::new();
        let mut rng2 = ChaCha8Rng::seed_from_u64(456);
        pop_with_weights.generate(
            5,
            1,
            2,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &data,
            false,
            Some(&prior_weight),
            &mut rng2,
        );

        assert_eq!(pop_with_weights.individuals.len(), 5);
        for ind in &pop_with_weights.individuals {
            assert_eq!(ind.language, TERNARY_LANG);
            assert!(ind.k >= 1 && ind.k <= 2);
        }
    }

    #[test]
    fn test_add() {
        let mut pop = Population::test();
        let mut pop_to_add = Population::new();
        let ind1 = Individual {
            features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(),
            auc: 0.4,
            fit: 0.8,
            specificity: 0.15,
            sensitivity: 0.16,
            accuracy: 0.23,
            threshold: 42.0,
            k: 42,
            epoch: 42,
            language: 0,
            data_type: 0,
            hash: 0,
            epsilon: f64::MIN_POSITIVE,
            parents: None,
            betas: None,
            threshold_ci: None,
            metrics: AdditionalMetrics {
                mcc: None,
                f1_score: None,
                npv: None,
                ppv: None,
                g_mean: None,
            },
        };
        let ind2 = Individual {
            features: vec![(0, -1), (1, 1), (2, 1), (3, 1)].into_iter().collect(),
            auc: 0.2,
            fit: 0.4,
            specificity: 0.6,
            sensitivity: 0.8,
            accuracy: 0.12,
            threshold: 24.0,
            k: 48,
            epoch: 96,
            language: 0,
            data_type: 0,
            hash: 0,
            epsilon: f64::MIN_POSITIVE,
            parents: None,
            betas: None,
            threshold_ci: None,
            metrics: AdditionalMetrics {
                mcc: None,
                f1_score: None,
                npv: None,
                ppv: None,
                g_mean: None,
            },
        };
        let ind_vec = vec![ind1, ind2];
        pop_to_add.individuals = ind_vec.clone();

        pop.add(pop_to_add);
        assert_eq!(pop.individuals.len(), 13);

        for i in 0..1 {
            assert_eq!(
                pop.individuals[11 + i].features,
                ind_vec[i].features,
                "adding an Individual should not change its features"
            );
            assert_eq!(
                pop.individuals[11 + i].auc,
                ind_vec[i].auc,
                "adding an Individual should not change its auc"
            );
            assert_eq!(
                pop.individuals[11 + i].fit,
                ind_vec[i].fit,
                "adding an Individual should not change its fit"
            );
            assert_eq!(
                pop.individuals[11 + i].specificity,
                ind_vec[i].specificity,
                "adding an Individual should not change its specificity"
            );
            assert_eq!(
                pop.individuals[11 + i].sensitivity,
                ind_vec[i].sensitivity,
                "adding an Individual should not change its sensitivity"
            );
            assert_eq!(
                pop.individuals[11 + i].accuracy,
                ind_vec[i].accuracy,
                "adding an Individual should not change its accuracy"
            );
            assert_eq!(
                pop.individuals[11 + i].threshold,
                ind_vec[i].threshold,
                "adding an Individual should not change its threshold"
            );
            assert_eq!(
                pop.individuals[11 + i].language,
                ind_vec[i].language,
                "adding an Individual should not change its language"
            );
            assert_eq!(
                pop.individuals[11 + i].data_type,
                ind_vec[i].data_type,
                "adding an Individual should not change its data_type"
            );
            assert_eq!(
                pop.individuals[11 + i].hash,
                ind_vec[i].hash,
                "adding an Individual should not change its hash"
            );
            assert_eq!(
                pop.individuals[11 + i].epsilon,
                ind_vec[i].epsilon,
                "adding an Individual should not change its epsilon"
            );
        }

        pop.add(Population::new());
        assert_eq!(pop.individuals.len(), 13, "populating a Population with the Individuals of an empty Population should not change it");
    }

    #[test]
    fn test_select_first_pct() {
        let mut pop = Population::test();

        let (selected_pop, n) = pop.select_first_pct(0.0);
        assert_eq!(selected_pop.individuals.len(), 0, "test_select_first_pct() should return the reduced Population and its number of Individuals");
        assert_eq!(
            n, 0,
            "selecting 0% of a Population should return an empty Population"
        );

        let (_selected_pop2, n2) = pop.select_first_pct(100.0);
        assert_eq!(
            n2, 11,
            "selecting 100% of a Population should return the same Population"
        );

        let (_selected_pop3, n3) = pop.select_first_pct(50.0);
        assert_eq!(n3, 5, "selecting 50% of a Population of 11 Individuals should return a Population of 5 Individuals (incomplete individual is not included)");
        pop.remove_clone();
        let (_selected_pop4, n4) = pop.select_first_pct(50.0);
        assert_eq!(n4, 5, "selecting 50% of a Population of 10 Individuals should return a Population of 5 Individuals");

        let (selected_pop5, n5) = pop.select_first_pct(10.0);
        assert_eq!(n5, 1);
        assert_eq!(selected_pop5.individuals[0].accuracy, 0.23,
        "select_first_pct() should be deterministic : selecting 1 Individual (10% of 10 Individuals) should lead to keep only the first Individual of the Population");

        // to change : return n=10 instead of n=100 (currently pop.individuals.len() != n in this case) when pct>100 or panic
        // let (selected_pop5, n5) = pop.select_first_pct(1000.0);
        // println!("{:?}", selected_pop5.individuals.len());
        // assert_eq!(n5, 100, "selecting 1000% of a Population of 10 Individuals should return a Population of 10 Individuals");
    }

    #[test]
    fn test_select_random_above_n() {
        let mut pop = Population::test();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let selected_pop = pop.select_random_above_n(0.0, 0, &mut rng);
        assert_eq!(selected_pop.individuals.len(), 0, "test_select_first_pct() should return the reduced Population and its number of Individuals");

        let selected_pop2 = pop.select_random_above_n(100.0, 0, &mut rng);
        assert_eq!(
            selected_pop2.individuals.len(),
            11,
            "selecting 100% of a Population should return the same Population"
        );

        let selected_pop3 = pop.select_random_above_n(50.0, 0, &mut rng);
        assert_eq!(selected_pop3.individuals.len(), 5, "selecting 50% of a Population of 11 Individuals should return a Population of 5 Individuals (incomplete individual is not included)");
        pop.remove_clone();
        let selected_pop4 = pop.select_random_above_n(50.0, 0, &mut rng);
        assert_eq!(selected_pop4.individuals.len(), 5, "selecting 50% of a Population of 10 Individuals should return a Population of 5 Individuals");

        let selected_pop5 = pop.select_random_above_n(10.0, 0, &mut rng);
        assert_eq!(selected_pop5.individuals.len(), 1);
        assert_eq!(selected_pop5.individuals[0].accuracy, 0.32,
        "the selected Individual is not the same as selected in the past, indicating a reproductibility problem probably linked to the seed interpretation");

        let selected_pop6 = pop.select_random_above_n(100.0, 8, &mut rng);
        assert_eq!(selected_pop6.individuals.len(), 2, "selecting 100% of a 10-Individuals Population since index 8 should return a Population fo 2 Individuals");

        let selected_pop7 = pop.select_random_above_n(100.0, 10, &mut rng);
        assert_eq!(selected_pop7.individuals.len(), 0, "selecting 100% of a 10-Individuals Population since index 10 should return a Population fo 0 Individuals");
    }

    #[test]
    fn test_get_ind_from_hash() {
        let mut pop = Population::test();
        pop.individuals[0].hash = 42;
        assert_eq!(
            pop.get_ind_from_hash(3).unwrap().hash,
            3,
            "Wrong individual selected"
        );
        assert_eq!(
            pop.get_ind_from_hash(3).unwrap().auc,
            0.55,
            "Selected individual has wrong auc"
        );
        assert!(
            pop.get_ind_from_hash(155).is_none(),
            "Unreachable hash should return None"
        );
    }

    #[test]
    #[should_panic(expected = "Hash should be computed to allow Individual selection")]
    fn test_get_ind_from_hash_zero() {
        let pop = Population::test();
        assert_eq!(
            pop.get_ind_from_hash(3).unwrap().hash,
            3,
            "Wrong individual selected"
        );
    }

    use crate::individual::BINARY_LANG;

    fn create_simple_test_data(num_samples: usize, num_features: usize) -> Data {
        let mut X = HashMap::new();
        for sample in 0..num_samples {
            for feature in 0..num_features {
                X.insert((sample, feature), (sample * feature) as f64 * 0.1);
            }
        }
        let y: Vec<u8> = (0..num_samples)
            .map(|i| if i % 2 == 0 { 1 } else { 0 })
            .collect();
        Data {
            X,
            y,
            sample_len: num_samples,
            feature_class: HashMap::new(),
            feature_significance: HashMap::new(),
            features: (0..num_features)
                .map(|i| format!("feature_{}", i))
                .collect(),
            samples: (0..num_samples).map(|i| format!("sample_{}", i)).collect(),
            feature_selection: (0..num_features).collect(),
            feature_len: num_features,
            classes: vec!["class_0".to_string(), "class_1".to_string()],
            feature_annotations: None,
            sample_annotations: None,
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_basic_population_scope() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let pop = Population::test_with_n_overlapping_features(3, 2);
        let data = create_simple_test_data(10, 10);
        let agg_method = ImportanceAggregation::mean;

        let importances = pop.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng,
            &agg_method,
            false,
            false,
            Some(1),
        );

        // All results should be from Population scope
        for imp in &importances.importances {
            assert_eq!(imp.scope, ImportanceScope::Population { id: 1 });
            assert_eq!(imp.importance_type, ImportanceType::MDA);
            assert_eq!(imp.aggreg_method, Some(agg_method.clone()));
            assert!(!imp.is_scaled);
            assert!(
                imp.scope_pct >= 0.0 && imp.scope_pct <= 1.0,
                "Prevalence should be valid"
            );
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_prevalence_calculation() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut pop = Population {
            individuals: Vec::new(),
        };

        // Individual 0: feature [0]
        let mut features1 = HashMap::new();
        features1.insert(0, 1i8);
        pop.individuals.push(Individual {
            features: features1,
            hash: 0,
            k: 1,
            ..Individual::test()
        });

        // Individual 1: feature [0]
        let mut features2 = HashMap::new();
        features2.insert(0, 1i8);
        pop.individuals.push(Individual {
            features: features2,
            hash: 1,
            k: 1,
            ..Individual::test()
        });

        // Individual 2: feature [1]
        let mut features3 = HashMap::new();
        features3.insert(1, 1i8);
        pop.individuals.push(Individual {
            features: features3,
            hash: 2,
            k: 1,
            ..Individual::test()
        });

        let data = create_simple_test_data(15, 3);
        let agg_method = ImportanceAggregation::mean;

        let importances = pop.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng,
            &agg_method,
            false,
            false,
            None,
        );

        for imp in &importances.importances {
            if imp.feature_idx == 0 {
                assert!(
                    (imp.scope_pct - 2.0 / 3.0).abs() < 1e-10,
                    "Feature 0 prevalence should be 2/3"
                );
            } else if imp.feature_idx == 1 {
                assert!(
                    (imp.scope_pct - 1.0 / 3.0).abs() < 1e-10,
                    "Feature 1 prevalence should be 1/3"
                );
            }
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_handles_empty_values() {
        let mut rng = ChaCha8Rng::seed_from_u64(555);
        let mut pop = Population {
            individuals: Vec::new(),
        };

        // Testing `if values.is_empty() { continue; }`
        let mut features = HashMap::new();
        features.insert(999, 1i8); // Unexistent feature
        pop.individuals.push(Individual {
            features,
            hash: 0,
            k: 1,
            ..Individual::test()
        });

        let data = create_simple_test_data(10, 5); // Only 0-4 features
        let agg_method = ImportanceAggregation::mean;

        let importances = pop.compute_pop_mda_feature_importance(
            &data,
            3,
            &mut rng,
            &agg_method,
            false,
            false,
            None,
        );

        for imp in &importances.importances {
            assert!(imp.importance.is_finite());
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_cascade_mode_includes_individuals() {
        let pop = Population::test_with_n_overlapping_features(2, 3);
        let data = create_simple_test_data(8, 8);
        let agg_method = ImportanceAggregation::mean;

        let mut rng1 = ChaCha8Rng::seed_from_u64(789);
        let no_cascade = pop.compute_pop_mda_feature_importance(
            &data,
            3,
            &mut rng1,
            &agg_method,
            false,
            false,
            Some(42),
        );

        let mut rng2 = ChaCha8Rng::seed_from_u64(789);
        let with_cascade = pop.compute_pop_mda_feature_importance(
            &data,
            3,
            &mut rng2,
            &agg_method,
            false,
            true,
            Some(42),
        );

        // Cascade mode should include more importances (individual + population)
        assert!(
            with_cascade.importances.len() > no_cascade.importances.len(),
            "Cascade mode should include individual-level importances"
        );

        // Check Individual scope count
        let individual_scopes = with_cascade
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Individual { .. }))
            .count();
        assert!(
            individual_scopes > 0,
            "Cascade mode should include Individual scope importances"
        );

        // Population scopes should be identical with and without cascade
        let pop_count_without = no_cascade
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Population { .. }))
            .count();
        let pop_count_with = with_cascade
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Population { .. }))
            .count();
        assert_eq!(
            pop_count_without, pop_count_with,
            "Population importances should be same in both modes"
        );
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_empty_population() {
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let pop = Population {
            individuals: Vec::new(),
        };
        let data = create_simple_test_data(10, 10);
        let agg_method = ImportanceAggregation::mean;

        let importances = pop.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng,
            &agg_method,
            false,
            false,
            None,
        );

        assert!(
            importances.importances.is_empty(),
            "Empty population should yield empty importance collection"
        );
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_aggregation_methods() {
        let pop = Population::test_with_n_overlapping_features(5, 2);
        let data = create_simple_test_data(15, 6);

        let mut rng1 = ChaCha8Rng::seed_from_u64(111);
        let mean_result = pop.compute_pop_mda_feature_importance(
            &data,
            4,
            &mut rng1,
            &ImportanceAggregation::mean,
            false,
            false,
            None,
        );

        let mut rng2 = ChaCha8Rng::seed_from_u64(111);
        let median_result = pop.compute_pop_mda_feature_importance(
            &data,
            4,
            &mut rng2,
            &ImportanceAggregation::median,
            false,
            false,
            None,
        );

        assert_eq!(
            mean_result.importances.len(),
            median_result.importances.len()
        );

        for (mean_imp, median_imp) in mean_result
            .importances
            .iter()
            .zip(median_result.importances.iter())
        {
            assert_eq!(mean_imp.aggreg_method, Some(ImportanceAggregation::mean));
            assert_eq!(
                median_imp.aggreg_method,
                Some(ImportanceAggregation::median)
            );
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_reproducibility_hash_sorting() {
        let mut pop = Population {
            individuals: Vec::new(),
        };

        // Individuals avec hashes dans un ordre spécifique
        for i in [3, 1, 2, 0] {
            // Intentionally unordered hashes
            let mut features = HashMap::new();
            features.insert(i, 1i8);
            pop.individuals.push(Individual {
                features,
                hash: i as u64,
                k: 1,
                ..Individual::test()
            });
        }

        let data = create_simple_test_data(10, 6);
        let agg_method = ImportanceAggregation::mean;

        let mut rng1 = ChaCha8Rng::seed_from_u64(222);
        let mut rng2 = ChaCha8Rng::seed_from_u64(222);

        let result1 = pop.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng1,
            &agg_method,
            false,
            false,
            None,
        );
        let result2 = pop.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng2,
            &agg_method,
            false,
            false,
            None,
        );

        // Hash sorting should ensure reproducibility
        assert_eq!(result1.importances.len(), result2.importances.len());

        for (imp1, imp2) in result1.importances.iter().zip(result2.importances.iter()) {
            assert!(
                (imp1.importance - imp2.importance).abs() < 1e-5,
                "Hash-sorted processing should ensure reproducible results"
            );
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_population_id_assignment() {
        let pop = Population::test_with_n_overlapping_features(3, 2);
        let data = create_simple_test_data(12, 5);
        let agg_method = ImportanceAggregation::mean;

        // ✅ Test with specific population ID
        let mut rng1 = ChaCha8Rng::seed_from_u64(333);
        let with_id = pop.compute_pop_mda_feature_importance(
            &data,
            4,
            &mut rng1,
            &agg_method,
            false,
            false,
            Some(99),
        );

        // ✅ Test with None (should default to 0)
        let mut rng2 = ChaCha8Rng::seed_from_u64(333);
        let without_id = pop.compute_pop_mda_feature_importance(
            &data,
            4,
            &mut rng2,
            &agg_method,
            false,
            false,
            None,
        );

        for imp in &with_id.importances {
            assert_eq!(imp.scope, ImportanceScope::Population { id: 99 });
        }

        for imp in &without_id.importances {
            assert_eq!(imp.scope, ImportanceScope::Population { id: 0 });
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_feature_collection_across_individuals() {
        let mut rng = ChaCha8Rng::seed_from_u64(444);
        let mut pop = Population {
            individuals: Vec::new(),
        };

        // Individual 0: features [0, 1]
        let mut features1 = HashMap::new();
        features1.insert(0, 1i8);
        features1.insert(1, 1i8);
        pop.individuals.push(Individual {
            features: features1,
            hash: 0,
            k: 2,
            ..Individual::test()
        });

        // Individual 1: features [2, 3]
        let mut features2 = HashMap::new();
        features2.insert(2, 1i8);
        features2.insert(3, 1i8);
        pop.individuals.push(Individual {
            features: features2,
            hash: 1,
            k: 2,
            ..Individual::test()
        });

        let data = create_simple_test_data(10, 5);
        let agg_method = ImportanceAggregation::mean;

        let importances = pop.compute_pop_mda_feature_importance(
            &data,
            3,
            &mut rng,
            &agg_method,
            false,
            false,
            None,
        );

        // is flat_map collecting all unique features ?
        let feature_indices: HashSet<usize> = importances
            .importances
            .iter()
            .map(|imp| imp.feature_idx)
            .collect();

        let expected_features: HashSet<usize> = [0, 1, 2, 3].iter().cloned().collect();
        assert_eq!(
            feature_indices, expected_features,
            "Should collect all features from all individuals"
        );
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_zero_permutations() {
        let rng = ChaCha8Rng::seed_from_u64(666);
        let pop = Population::test_with_n_overlapping_features(2, 2);
        let data = create_simple_test_data(8, 5);
        let agg_method = ImportanceAggregation::mean;

        // Edge case: zero permutations (should be handled by individual level)
        let result = std::panic::catch_unwind(|| {
            let mut rng_clone = rng.clone();
            pop.compute_pop_mda_feature_importance(
                &data,
                0,
                &mut rng_clone,
                &agg_method,
                false,
                false,
                None,
            )
        });

        // Population level -> Individual level (edge case protection)
        assert!(
            result.is_err(),
            "Zero permutations should cause panic due to individual level division by zero"
        );
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_disjoint_individuals() {
        let mut pop = Population {
            individuals: Vec::new(),
        };

        // Individual avec features complètement en dehors des données
        let mut features = HashMap::new();
        features.insert(100, 1i8);
        features.insert(101, 1i8);
        pop.individuals.push(Individual {
            features,
            hash: 0,
            k: 2,
            ..Individual::test()
        });

        let data = create_simple_test_data(10, 5); // Features 0-4 seulement
        let agg_method = ImportanceAggregation::mean;

        let mut rng = ChaCha8Rng::seed_from_u64(789);
        let importances = pop.compute_pop_mda_feature_importance(
            &data,
            3,
            &mut rng,
            &agg_method,
            false,
            false,
            None,
        );

        // Toutes les importances devraient être 0 car aucune feature match
        for imp in &importances.importances {
            assert_eq!(
                imp.importance, 0.0,
                "Disjoint features should have zero importance"
            );
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_scaling_behavior() {
        let pop1 = Population::test_with_n_overlapping_features(4, 2);
        let data = create_simple_test_data(12, 8);
        let agg_method = ImportanceAggregation::median;

        let mut rng1 = ChaCha8Rng::seed_from_u64(456);
        let mut rng2 = ChaCha8Rng::seed_from_u64(456);

        let unscaled = pop1.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng1,
            &agg_method,
            false,
            false,
            None,
        );
        let scaled = pop1.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng2,
            &agg_method,
            true,
            false,
            None,
        );

        assert_eq!(unscaled.importances.len(), scaled.importances.len());

        for (unscaled_imp, scaled_imp) in unscaled.importances.iter().zip(scaled.importances.iter())
        {
            assert!(!unscaled_imp.is_scaled);
            assert!(scaled_imp.is_scaled);

            assert!(scaled_imp.importance.is_finite());
            assert!(scaled_imp.dispersion.is_finite());

            assert_eq!(unscaled_imp.scope_pct, scaled_imp.scope_pct);
        }
    }

    #[test]
    fn test_compute_pop_mda_feature_importance_dispersion_calculation() {
        let pop = Population::test_with_n_overlapping_features(3, 2);
        let data = create_simple_test_data(12, 6);

        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let mean_result = pop.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng1,
            &ImportanceAggregation::mean,
            false,
            false,
            None,
        );

        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let median_result = pop.compute_pop_mda_feature_importance(
            &data,
            5,
            &mut rng2,
            &ImportanceAggregation::median,
            false,
            false,
            None,
        );

        // Vérifier que dispersion differs between mean (std) vs median (MAD)
        for (mean_imp, median_imp) in mean_result
            .importances
            .iter()
            .zip(median_result.importances.iter())
        {
            assert!(mean_imp.dispersion.is_finite());
            assert!(median_imp.dispersion.is_finite());
            // Mean utilise std, median utilise MAD - peuvent différer
        }
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_with_empty_population() {
        // Test with empty population should return empty
        let pop = Population {
            individuals: vec![],
        };
        let filtered = pop.filter_by_signed_jaccard_dissimilarity(50.0, false);
        assert!(filtered.individuals.is_empty());
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_with_zero_threshold() {
        // Test with threshold <= 0 should return empty
        let ind =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.9, 0, 0);
        let pop = Population {
            individuals: vec![ind],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(0.0, false);
        assert!(filtered.individuals.is_empty());

        let filtered_negative = pop.filter_by_signed_jaccard_dissimilarity(-10.0, false);
        assert!(filtered_negative.individuals.is_empty());
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_with_single_individual() {
        // Test with single individual should always keep it
        let ind =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.9, 0, 0);
        let pop = Population {
            individuals: vec![ind.clone()],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(50.0, false);
        assert_eq!(filtered.individuals.len(), 1);
        assert_eq!(filtered.individuals[0].fit, 0.9);
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_without_niche() {
        // Test filtering without niche consideration
        // Create individuals with different features and fitness values
        let ind1 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.95, 0, 0);
        let ind2 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, -1)], 0.90, 0, 0);
        let ind3 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(2, 1)], 0.85, 0, 0);
        let ind4 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.80, 0, 0); // Same as ind1

        let pop = Population {
            individuals: vec![ind1.clone(), ind2.clone(), ind3.clone(), ind4.clone()],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(50.0, false);

        // ind1: highest fit, always kept
        // ind2: dissimilarity with ind1 = 1.0 (same feature ID, opposite signs) > 0.5, kept
        // ind3: dissimilarity with ind1 = 1.0 (completely different feature IDs) > 0.5, kept
        // ind4: dissimilarity with ind1 = 0.0 (identical features) < 0.5, filtered out

        assert_eq!(filtered.individuals.len(), 3);
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.95));
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.90));
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.85));
        assert!(!filtered.individuals.iter().any(|i| i.fit == 0.80));
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_with_niche() {
        // Test filtering with niche consideration
        // Create individuals with different niches (language, data_type)
        let ind1 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.95, 0, 0); // Niche A
        let ind2 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.90, 0, 0); // Niche A, same features as ind1
        let ind3 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(2, 1)], 0.85, 1, 1); // Niche B
        let ind4 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(2, -1)], 0.80, 1, 1); // Niche B

        let pop = Population {
            individuals: vec![ind1.clone(), ind2.clone(), ind3.clone(), ind4.clone()],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(50.0, true);

        // Niche A (0,0): ind1 kept (highest fit), ind2 filtered out (identical features, dissimilarity = 0.0)
        // Niche B (1,1): ind3 kept (highest fit), ind4 kept (same feature ID, opposite signs, dissimilarity = 1.0 > 0.5)

        assert_eq!(filtered.individuals.len(), 3);
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.95));
        assert!(!filtered.individuals.iter().any(|i| i.fit == 0.90));
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.85));
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.80));
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_with_high_threshold() {
        // Test with threshold = 100% (only completely different individuals kept)
        let ind1 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.95, 0, 0);
        let ind2 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, -1)], 0.90, 0, 0); // Opposite sign
        let ind3 = Individual::test_with_these_given_features_fit_lang_types(
            vec![(1, 1), (2, 1)],
            0.85,
            0,
            0,
        ); // Partial overlap

        let pop = Population {
            individuals: vec![ind1.clone(), ind2.clone(), ind3.clone()],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(100.0, false);

        // ind1: kept (highest fit)
        // ind2: dissimilarity with ind1 = 1.0 (same feature ID, opposite signs) >= 1.0, kept
        // ind3: dissimilarity with ind1 = 1 - 1/2 = 0.5 < 1.0, filtered out
        //   (intersection: {(1,+1)}, union: {(1,+1), (2,+1)})

        assert_eq!(filtered.individuals.len(), 2);
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.95));
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.90));
        assert!(!filtered.individuals.iter().any(|i| i.fit == 0.85));
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_with_low_threshold() {
        // Test with threshold = 1% (very permissive, almost all kept)
        let ind1 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.95, 0, 0);
        let ind2 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.90, 0, 0); // Identical
        let ind3 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, -1)], 0.85, 0, 0); // Different

        let pop = Population {
            individuals: vec![ind1.clone(), ind2.clone(), ind3.clone()],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(1.0, false);

        // ind1: kept (highest fit)
        // ind2: dissimilarity with ind1 = 0.0 (identical features) < 0.01, filtered out
        // ind3: dissimilarity with ind1 = 1.0 (same feature ID, opposite signs) > 0.01, kept

        assert_eq!(filtered.individuals.len(), 2);
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.95));
        assert!(!filtered.individuals.iter().any(|i| i.fit == 0.90));
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.85));
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_with_complex_features() {
        // Test with more complex feature sets
        let ind1 = Individual::test_with_these_given_features_fit_lang_types(
            vec![(1, 1), (2, -1), (3, 1)],
            0.95,
            0,
            0,
        );
        let ind2 = Individual::test_with_these_given_features_fit_lang_types(
            vec![(1, 1), (2, -1)],
            0.90,
            0,
            0,
        );
        let ind3 = Individual::test_with_these_given_features_fit_lang_types(
            vec![(1, -1), (2, 1), (3, -1)],
            0.85,
            0,
            0,
        );

        let pop = Population {
            individuals: vec![ind1.clone(), ind2.clone(), ind3.clone()],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(60.0, false);

        // Manual calculation:
        // ind1 vs ind2: intersection={(1,+1), (2,-1)}, union={(1,+1), (2,-1), (3,+1)}, dissim = 1 - 2/3 ≈ 0.33
        // ind1 vs ind3: intersection={}, union={(1,+1), (1,-1), (2,-1), (2,+1), (3,+1), (3,-1)}, dissim = 1 - 0/6 = 1.0
        // Since 0.33 < 0.6, ind2 filtered out; ind3 kept (1.0 > 0.6)

        assert_eq!(filtered.individuals.len(), 2);
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.95));
        assert!(!filtered.individuals.iter().any(|i| i.fit == 0.90));
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.85));
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_with_empty_features() {
        // Test with individuals having no features
        let ind1 = Individual::test_with_these_given_features_fit_lang_types(vec![], 0.95, 0, 0);
        let ind2 = Individual::test_with_these_given_features_fit_lang_types(vec![], 0.90, 0, 0);
        let ind3 =
            Individual::test_with_these_given_features_fit_lang_types(vec![(1, 1)], 0.85, 0, 0);

        let pop = Population {
            individuals: vec![ind1.clone(), ind2.clone(), ind3.clone()],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(50.0, false);

        // ind1: kept (highest fit)
        // ind2: dissimilarity with ind1 = 0.0 (both empty, intersection=0, union=0) < 0.5, filtered out
        // ind3: dissimilarity with ind1 = 1.0 (empty vs non-empty, intersection=0, union=1) > 0.5, kept

        assert_eq!(filtered.individuals.len(), 2);
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.95));
        assert!(!filtered.individuals.iter().any(|i| i.fit == 0.90));
        assert!(filtered.individuals.iter().any(|i| i.fit == 0.85));
    }

    fn create_test_population() -> Population {
        let mut pop = Population::new();
        for i in 0..5 {
            let mut ind = Individual::new();
            ind.features.insert(0, 1);
            ind.features.insert(1, if i % 2 == 0 { 1 } else { -1 });
            ind.k = 2;
            ind.auc = 0.5 + (i as f64) * 0.05;
            ind.fit = 0.4 + (i as f64) * 0.05;
            ind.accuracy = 0.6;
            ind.sensitivity = 0.7;
            ind.specificity = 0.8;
            ind.threshold = 0.5;
            ind.epoch = 0;
            ind.language = TERNARY_LANG;
            ind.data_type = RAW_TYPE;
            ind.epsilon = 1e-5;
            pop.individuals.push(ind);
        }
        pop
    }

    fn create_small_cv_data() -> (Data, CV) {
        let data = Data::test();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let cv = CV::new(&data, 2, &mut rng);
        (data, cv)
    }

    fn fit_on_folds_test_helper(
        pop: &mut Population,
        cv: &CV,
        param: &Param,
        gpu_assays: &Vec<Option<GpuAssay>>,
    ) {
        // Helper for tests: convert Vec<Option<GpuAssay>> to Vec<(Option<GpuAssay>, Option<GpuAssay>)>
        // For tests, we use None for both validation and training
        let tuple_assays: Vec<(Option<GpuAssay>, Option<GpuAssay>)> =
            gpu_assays.iter().map(|_| (None, None)).collect();
        pop.fit_on_folds(cv, param, &tuple_assays);
    }

    #[test]
    fn test_fit_on_folds_behavior_consistency() {
        let mut pop = create_test_population();
        let (_, cv) = create_small_cv_data();
        let mut param = Param::default();
        param.cv.overfit_penalty = 2.0;
        param.general.k_penalty = 0.1;

        // Store the initial fits
        let initial_fits: Vec<f64> = pop.individuals.iter().map(|ind| ind.fit).collect();

        // Apply fit_on_folds
        fit_on_folds_test_helper(
            &mut pop,
            &cv,
            &param,
            &vec![None; cv.validation_folds.len()],
        );

        // Check that the fits have changed consistently
        let final_fits: Vec<f64> = pop.individuals.iter().map(|ind| ind.fit).collect();

        // The fits must be different from the initial fits
        for (initial, final_fit) in initial_fits.iter().zip(final_fits.iter()) {
            assert_ne!(
                *initial, *final_fit,
                "The fits must be modified by fit_on_folds"
            );
        }

        // The fits must be finite numbers
        for fit in final_fits.iter() {
            assert!(fit.is_finite(), "All fits must be finite");
        }
    }

    #[test]
    fn test_fit_on_folds_empty_population() {
        let mut pop = Population::new();
        let (_, cv) = create_small_cv_data();
        let param = Param::default();

        // Should not panic with an empty population
        fit_on_folds_test_helper(
            &mut pop,
            &cv,
            &param,
            &vec![None; cv.validation_folds.len()],
        );

        assert!(
            pop.individuals.is_empty(),
            "The population must remain empty"
        );
    }

    #[test]
    #[should_panic]
    fn test_fit_on_folds_mismatched_gpu_assays() {
        let mut pop = create_test_population();
        let (_, cv) = create_small_cv_data();
        let param = Param::default();

        // Provide wrong number of GPU assays (fewer than the number of folds)
        let gpu_assays = vec![(None, None)]; // Only 1 instead of cv.validation_folds.len()

        pop.fit_on_folds(&cv, &param, &gpu_assays);
    }

    #[test]
    fn test_fit_on_folds_integration() {
        let (_, cv) = create_controlled_cv_data();
        let mut pop = create_test_population();
        let mut param = Param::default();
        param.general.fit = FitFunction::auc;
        param.cv.overfit_penalty = 1.5;
        param.general.k_penalty = 0.2;

        // Test with different CV parameters
        fit_on_folds_test_helper(
            &mut pop,
            &cv,
            &param,
            &vec![None; cv.validation_folds.len()],
        );

        // Verify that properties are preserved
        assert_eq!(
            pop.individuals.len(),
            5,
            "The number of individuals must be preserved"
        );

        for ind in &pop.individuals {
            assert!(ind.k > 0, "The number of features k must be positive");
            assert!(ind.fit.is_finite(), "The fit must be a finite number");
        }
    }

    #[test]
    fn test_fit_on_folds_structure_preservation() {
        let mut pop = create_test_population();
        let (_, cv) = create_small_cv_data();
        let param = Param::default();

        let initial_count = pop.individuals.len();
        let initial_features: Vec<_> = pop
            .individuals
            .iter()
            .map(|ind| ind.features.clone())
            .collect();

        fit_on_folds_test_helper(
            &mut pop,
            &cv,
            &param,
            &vec![None; cv.validation_folds.len()],
        );

        // Structure preserved
        assert_eq!(
            pop.individuals.len(),
            initial_count,
            "The number of individuals must be preserved"
        );

        for (i, ind) in pop.individuals.iter().enumerate() {
            assert_eq!(
                ind.features, initial_features[i],
                "The features of individual {} must be preserved",
                i
            );
        }
    }

    fn create_controlled_cv_data() -> (Data, CV) {
        // Controlled data for precise mathematical tests
        let mut data = Data::new();
        data.X.insert((0, 0), 0.9);
        data.X.insert((0, 1), 0.1);
        data.X.insert((1, 0), 0.1);
        data.X.insert((1, 1), 0.9);
        data.X.insert((2, 0), 0.8);
        data.X.insert((2, 1), 0.2);
        data.X.insert((3, 0), 0.2);
        data.X.insert((3, 1), 0.8);

        data.y = vec![1, 0, 1, 0];
        data.sample_len = 4;
        data.feature_len = 2;
        data.features = vec!["feature1".to_string(), "feature2".to_string()];
        data.samples = vec![
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
            "sample4".to_string(),
        ];
        data.feature_selection = vec![0, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let cv = CV::new(&data, 2, &mut rng);
        (data, cv)
    }

    #[test]
    fn test_fit_on_folds_step_by_step_mathematical_verification() {
        let (data, cv) = create_controlled_cv_data();

        let mut pop = Population::new();
        let mut ind = Individual::new();
        ind.features.insert(0, 1);
        ind.k = 1;
        pop.individuals.push(ind);

        let mut param = Param::default();
        param.cv.overfit_penalty = 2.0;
        param.general.k_penalty = 0.05;
        param.general.fit = FitFunction::auc;

        pop.fit_without_penalty(&data, &mut None, &None, &None, &param);
        let initial_fit = pop.individuals[0].fit;

        fit_on_folds_test_helper(
            &mut pop,
            &cv,
            &param,
            &vec![None; cv.validation_folds.len()],
        );

        let final_fit = pop.individuals[0].fit;
        let expected_k_penalty = pop.individuals[0].k as f64 * param.general.k_penalty;

        assert!(final_fit < initial_fit - expected_k_penalty * 0.3);

        let fit_difference = initial_fit - final_fit;
        assert!(fit_difference >= expected_k_penalty * 0.8);
    }

    #[test]
    fn test_fit_on_folds_zero_overfit_penalty_mathematical() {
        let mut pop = Population::new();
        let mut ind = Individual::new();
        ind.features.insert(0, 1);
        ind.k = 2;
        ind.auc = 0.7;
        ind.fit = 0.7;
        pop.individuals.push(ind);

        let (_, cv) = create_small_cv_data();
        let mut param = Param::default();
        param.cv.overfit_penalty = 0.0; // No overfitting penalty
        param.general.k_penalty = 0.1; // Only k penalty
        param.general.fit = FitFunction::auc;

        let initial_fit = pop.individuals[0].fit;
        fit_on_folds_test_helper(
            &mut pop,
            &cv,
            &param,
            &vec![None; cv.validation_folds.len()],
        );
        let final_fit = pop.individuals[0].fit;

        let expected_k_penalty = pop.individuals[0].k as f64 * param.general.k_penalty;

        // With overfit_penalty = 0, the main penalty should be k_penalty
        assert_ne!(
            initial_fit, final_fit,
            "The fit must change even without overfit_penalty"
        );
        assert!(
            final_fit <= initial_fit,
            "The final fit should not be higher than initial"
        );

        // Mathematical verification: the reduction should at minimum include k_penalty
        let fit_reduction = initial_fit - final_fit;
        assert!(
            fit_reduction >= expected_k_penalty * 0.5,
            "The fit reduction ({:.6}) should be at least 50% of k_penalty ({:.6})",
            fit_reduction,
            expected_k_penalty
        );
    }

    #[test]
    fn test_fit_on_folds_k_penalty_mathematical_scaling() {
        let k_values = vec![1, 3, 5];
        let k_penalty = 0.1;
        let mut results = Vec::new();

        for k in k_values {
            let mut pop = Population::new();
            let mut ind = Individual::new();
            // Create k features
            for i in 0..k {
                ind.features.insert(i, 1);
            }
            ind.k = k;
            ind.auc = 0.8;
            ind.fit = 0.8;
            pop.individuals.push(ind);

            let (_, cv) = create_small_cv_data();
            let mut param = Param::default();
            param.cv.overfit_penalty = 0.5; // Moderate penalty
            param.general.k_penalty = k_penalty;
            param.general.fit = FitFunction::auc;

            let initial_fit = pop.individuals[0].fit;
            fit_on_folds_test_helper(
                &mut pop,
                &cv,
                &param,
                &vec![None; cv.validation_folds.len()],
            );
            let final_fit = pop.individuals[0].fit;

            let expected_k_penalty = k as f64 * k_penalty;
            let actual_reduction = initial_fit - final_fit;

            results.push((
                k,
                initial_fit,
                final_fit,
                actual_reduction,
                expected_k_penalty,
            ));
        }

        // Verify that penalty increases mathematically with k
        for i in 1..results.len() {
            let (k_prev, _, _, reduction_prev, expected_prev) = results[i - 1];
            let (k_curr, _, _, reduction_curr, expected_curr) = results[i];

            // The actual reduction should follow the increase in k_penalty
            let expected_increase = expected_curr - expected_prev;
            let actual_increase = reduction_curr - reduction_prev;

            assert!(
                reduction_curr > reduction_prev,
                "The penalty for k={} ({:.6}) should be greater than for k={} ({:.6})",
                k_curr,
                reduction_curr,
                k_prev,
                reduction_prev
            );

            // The actual increase should be at least 50% of the expected increase
            assert!(
                actual_increase >= expected_increase * 0.5,
                "The penalty increase ({:.6}) should be at least 50% of expected ({:.6})",
                actual_increase,
                expected_increase
            );
        }
    }

    #[test]
    fn test_fit_on_folds_mathematical_per_fit_function() {
        let fit_functions = vec![
            (FitFunction::auc, "AUC"),
            (FitFunction::sensitivity, "Sensitivity"),
            (FitFunction::specificity, "Specificity"),
            (FitFunction::specificity, "MCC"),
            (FitFunction::npv, "NPV"),
            (FitFunction::ppv, "PPV"),
            (FitFunction::f1_score, "F1-score"),
            (FitFunction::g_mean, "G-mean"),
        ];

        for (fit_func, name) in fit_functions {
            let mut pop = Population::new();
            let mut ind = Individual::new();
            ind.features.insert(0, 1);
            ind.features.insert(1, -1);
            ind.k = 2;
            pop.individuals.push(ind);

            let (data, cv) = create_small_cv_data();
            let mut param = Param::default();
            param.cv.overfit_penalty = 1.0;
            param.general.k_penalty = 0.1;
            param.general.fit = fit_func;

            pop.fit_without_penalty(&data, &mut None, &None, &None, &param);
            let initial_fit = pop.individuals[0].fit;
            let expected_k_penalty = pop.individuals[0].k as f64 * param.general.k_penalty;

            fit_on_folds_test_helper(
                &mut pop,
                &cv,
                &param,
                &vec![None; cv.validation_folds.len()],
            );
            let final_fit = pop.individuals[0].fit;

            // Assertions
            let fit_reduction = initial_fit - final_fit;
            assert!(
                fit_reduction >= expected_k_penalty * 0.3,
                "The reduction ({:.6}) must include at least 30% of k_penalty ({:.6}) for {}",
                fit_reduction,
                expected_k_penalty,
                name
            );
        }
    }

    // -----------------------------------------------------------------
    // Tests for prune_all_by_importance
    // -----------------------------------------------------------------

    #[test]
    fn test_prune_all_by_importance_basic() {
        use crate::individual::{BINARY_LANG, RAW_TYPE};

        let data = Data::specific_test(60, 12);

        let mut population = Population::new();

        for i in 0..5 {
            let mut individual = Individual::new();
            individual.language = BINARY_LANG;
            individual.data_type = RAW_TYPE;
            individual.hash = i as u64;

            for j in 0..8 {
                individual.features.insert((i + j) % 12, 1);
            }
            individual.k = individual.features.len();

            population.individuals.push(individual);
        }

        population.prune_all_by_importance(&data, 15, 54321, Some(0.0), None, 3);

        for ind in &population.individuals {
            assert!(ind.k >= 3, "Individual should have at least min_k features");
            assert!(ind.k <= 8, "Individual should not gain features");
        }
    }

    #[test]
    fn test_prune_all_by_importance_parallel() {
        use crate::individual::{BINARY_LANG, RAW_TYPE, TERNARY_LANG};

        let data = Data::specific_test(80, 15);

        let mut population = Population::new();

        for i in 0..20 {
            let mut individual = Individual::new();
            individual.language = if i % 2 == 0 {
                BINARY_LANG
            } else {
                TERNARY_LANG
            };
            individual.data_type = RAW_TYPE;
            individual.hash = i as u64;

            for j in 0..10 {
                let coef = if individual.language == BINARY_LANG {
                    1
                } else {
                    if j % 2 == 0 {
                        1
                    } else {
                        -1
                    }
                };
                individual.features.insert((i + j) % 15, coef);
            }
            individual.k = individual.features.len();

            population.individuals.push(individual);
        }

        population.prune_all_by_importance(&data, 20, 77777, Some(0.0), None, 4);

        // Verify all individuals were pruned
        for ind in &population.individuals {
            assert!(
                ind.k >= 4,
                "Individual should have at least min_k=4 features"
            );
            assert!(
                ind.k <= 10,
                "Individual should not have more than original features"
            );
        }
    }

    #[test]
    fn test_prune_all_by_importance_determinism() {
        use crate::individual::{BINARY_LANG, RAW_TYPE};

        let data = Data::specific_test(50, 10);

        let mut pop1 = Population::new();
        let mut pop2 = Population::new();

        for i in 0..3 {
            let mut ind = Individual::new();
            ind.language = BINARY_LANG;
            ind.data_type = RAW_TYPE;
            ind.hash = i as u64;

            for j in 0..6 {
                ind.features.insert(j, 1);
            }
            ind.k = 6;

            pop1.individuals.push(ind.clone());
            pop2.individuals.push(ind);
        }

        let seed = 123456;
        pop1.prune_all_by_importance(&data, 10, seed, Some(0.0), None, 2);
        pop2.prune_all_by_importance(&data, 10, seed, Some(0.0), None, 2);

        for (ind1, ind2) in pop1.individuals.iter().zip(pop2.individuals.iter()) {
            assert_eq!(ind1.k, ind2.k, "Feature counts should match");
            assert_eq!(ind1.features, ind2.features, "Features should be identical");
        }
    }

    #[test]
    fn test_prune_all_by_importance_empty_population() {
        let data = Data::specific_test(40, 8);
        let mut population = Population::new();

        // Should not panic
        population.prune_all_by_importance(&data, 10, 999, Some(0.0), None, 1);

        assert!(
            population.individuals.is_empty(),
            "Empty population should remain empty"
        );
    }

    #[test]
    fn test_prune_all_by_importance_with_quantile() {
        use crate::individual::{RAW_TYPE, TERNARY_LANG};

        let data = Data::specific_test(70, 15);

        let mut population = Population::new();

        for i in 0..4 {
            let mut individual = Individual::new();
            individual.language = TERNARY_LANG;
            individual.data_type = RAW_TYPE;
            individual.hash = i as u64;

            for j in 0..10 {
                individual
                    .features
                    .insert(j, if j % 2 == 0 { 1 } else { -1 });
            }
            individual.k = 10;

            population.individuals.push(individual);
        }

        population.prune_all_by_importance(&data, 30, 888, None, Some((0.3, 0.0)), 3);

        for ind in &population.individuals {
            assert!(ind.k >= 3, "Should maintain min_k");
            assert!(ind.k <= 10, "Should not gain features");
        }
    }

    use crate::data::FeatureAnnotations;
    fn make_feature_annotations_with_penalties(penalties: &[(usize, f64)]) -> FeatureAnnotations {
        let mut feature_penalty = HashMap::new();
        for (idx, p) in penalties {
            feature_penalty.insert(*idx, *p);
        }

        FeatureAnnotations {
            tag_column_names: Vec::new(),
            feature_tags: HashMap::new(),
            prior_weight: HashMap::new(),
            feature_penalty,
        }
    }

    #[test]
    fn test_user_penalty_basic_weighted() {
        // Arrange
        let mut param = Param::default();
        param.general.k_penalty = 0.0;
        param.general.threshold_ci_penalty = 0.0;
        param.general.bias_penalty = 0.0;
        param.general.user_penalties_weight = 1.0;

        // Data::test() fournit 2 features (indices 0 et 1)
        let mut data = Data::test();
        let fa = make_feature_annotations_with_penalties(&[(0, 0.5), (1, 0.5)]);
        data.feature_annotations = Some(fa);

        let mut pop = Population::new();
        let mut ind = Individual::new();
        ind.features.insert(0, 1);
        ind.features.insert(1, 1);
        ind.k = ind.features.len();
        ind.fit = 0.0;
        ind.sensitivity = 0.5;
        ind.specificity = 0.5;
        pop.individuals.push(ind);

        // Act
        pop.penalize(&data, &param);

        // Assert
        // Weighted mean = (1*0.5 + 1*0.5) / (1+1) = 0.5
        let fit = pop.individuals[0].fit;
        assert!(
            (fit + 0.5).abs() < 1e-9,
            "Expected fit to be -0.5, got {fit}"
        );
    }

    #[test]
    fn test_user_penalty_weighted_by_coefficients() {
        // Arrange
        let mut param = Param::default();
        param.general.k_penalty = 0.0;
        param.general.threshold_ci_penalty = 0.0;
        param.general.bias_penalty = 0.0;
        param.general.user_penalties_weight = 1.0;

        // On veut au moins 3 features pour tester la pondération
        let mut data = Data::specific_test(10, 3);
        let fa = make_feature_annotations_with_penalties(&[(0, 0.5), (1, 0.1), (2, 0.1)]);
        data.feature_annotations = Some(fa);

        let mut pop = Population::new();
        let mut ind = Individual::new();
        ind.features.insert(0, 64); // très important
        ind.features.insert(1, 1);
        ind.features.insert(2, 1);
        ind.k = ind.features.len();
        ind.fit = 0.0;
        ind.sensitivity = 0.5;
        ind.specificity = 0.5;
        pop.individuals.push(ind);

        // Act
        pop.penalize(&data, &param);

        // Assert
        // Weighted mean = (64*0.5 + 1*0.1 + 1*0.1) / (64+1+1) ≈ 0.488
        let fit = pop.individuals[0].fit;
        let expected = (64.0 * 0.5 + 0.1 + 0.1) / 66.0;
        assert!(
            (fit + expected).abs() < 1e-3,
            "Expected fit ~= -{expected}, got {fit}"
        );
    }

    #[test]
    fn test_user_penalty_zero_lambda_no_effect() {
        // Arrange
        let mut param = Param::default();
        param.general.k_penalty = 0.0;
        param.general.threshold_ci_penalty = 0.0;
        param.general.bias_penalty = 0.0;
        param.general.user_penalties_weight = 0.0; // désactive la pénalité

        let mut data = Data::test();
        let fa = make_feature_annotations_with_penalties(&[(0, 1.0)]);
        data.feature_annotations = Some(fa);

        let mut pop = Population::new();
        let mut ind = Individual::new();
        ind.features.insert(0, 1);
        ind.k = ind.features.len();
        ind.fit = 0.42;
        ind.sensitivity = 0.5;
        ind.specificity = 0.5;
        pop.individuals.push(ind);

        let initial_fit = pop.individuals[0].fit;

        // Act
        pop.penalize(&data, &param);

        // Assert
        let final_fit = pop.individuals[0].fit;
        assert!(
            (final_fit - initial_fit).abs() < 1e-12,
            "Fit should remain unchanged when lambda=0. initial={initial_fit}, final={final_fit}"
        );
    }

    #[test]
    fn test_filter_by_signed_jaccard_dissimilarity_linear_family_shared_niche() {
        use crate::individual::{BINARY_LANG, POW2_LANG, RATIO_LANG, RAW_TYPE, TERNARY_LANG};

        let ind_bin = Individual::test_with_these_given_features_fit_lang_types(
            vec![(1, 1)],
            0.95,
            BINARY_LANG,
            RAW_TYPE,
        );
        let ind_ter = Individual::test_with_these_given_features_fit_lang_types(
            vec![(1, 1)],
            0.90,
            TERNARY_LANG,
            RAW_TYPE,
        );
        let ind_pow2 = Individual::test_with_these_given_features_fit_lang_types(
            vec![(1, 1)],
            0.85,
            POW2_LANG,
            RAW_TYPE,
        );
        let ind_ratio = Individual::test_with_these_given_features_fit_lang_types(
            vec![(1, 1)],
            0.80,
            RATIO_LANG,
            RAW_TYPE,
        );

        let pop = Population {
            individuals: vec![
                ind_bin.clone(),
                ind_ter.clone(),
                ind_pow2.clone(),
                ind_ratio.clone(),
            ],
        };

        let filtered = pop.filter_by_signed_jaccard_dissimilarity(50.0, true);

        assert!(filtered.individuals.iter().any(|i| i.fit == 0.95));
        assert!(!filtered.individuals.iter().any(|i| i.fit == 0.90));
        assert!(!filtered.individuals.iter().any(|i| i.fit == 0.85));

        assert!(filtered.individuals.iter().any(|i| i.fit == 0.80));

        assert_eq!(filtered.individuals.len(), 2);
    }
}
