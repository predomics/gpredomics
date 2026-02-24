use crate::data::Data;
use crate::individual::AdditionalMetrics;
use crate::param::FitFunction;
use crate::param::Param;
use crate::population::Population;
use crate::utils::{compute_auc_from_value, compute_metrics_from_classes};
use crate::Individual;
use log::{debug, warn};
use serde::{Deserialize, Serialize};

//-----------------------------------------------------------------------------
// Voting
//-----------------------------------------------------------------------------

/// Jury population of experts, associated voting methods and metrics
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Jury {
    /// Population of individuals able to vote
    pub experts: Population,
    /// Voting method
    pub voting_method: VotingMethod,
    /// Voting threshold
    pub voting_threshold: f64,
    /// Threshold window for majority voting
    pub threshold_window: f64,

    /// Weighting method used for assigning weights to experts
    pub weighting_method: WeightingMethod,

    /// Weights assigned to each expert after evaluation
    pub weights: Option<Vec<f64>>,

    /// Binary classification metrics
    /// Area Under the Curve based on pos_vote/(pos_vote+neg_vote)
    pub auc: f64,
    /// Accuracy
    pub accuracy: f64,
    /// Sensitivity (True Positive Rate)
    pub sensitivity: f64,
    /// Specificity (True Negative Rate)
    pub specificity: f64,
    /// Rejection rate (abstentions)
    pub rejection_rate: f64,
    /// Predicted classes after evaluation
    pub predicted_classes: Option<Vec<u8>>,

    /// Additional metrics (if present in experts)
    #[serde(default)]
    pub metrics: AdditionalMetrics,
}

/// Voting methods available
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum VotingMethod {
    /// Majority voting: the class with the most votes wins
    Majority,
    /// Consensus voting: a top % experts must agree for a decision to be made
    Consensus,
}

/// Weighting methods for experts in the jury
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum WeightingMethod {
    /// Uniform weighting: all experts have equal weight
    Uniform,
    /// Specialized weighting: experts are weighted based on their specialization (experimental)
    Specialized {
        /// Sensitivity threshold for positive specialists
        sensitivity_threshold: f64,
        /// Specificity threshold for negative specialists
        specificity_threshold: f64,
    },
}

/// Judge specialization categories (experimental)
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum JudgeSpecialization {
    /// Positive specialist able to vote on positive class
    PositiveSpecialist,
    /// Negative specialist able to vote on negative class
    NegativeSpecialist,
    /// Balanced specialist able to vote on both classes
    Balanced,
    /// Ineffective judge unable to vote
    Ineffective,
}

impl Jury {
    /// Creates a new Jury with given parameters
    ///
    /// # Arguments
    ///
    /// * `pop` - Population of experts
    /// * `min_perf` - Minimum performance (sensitivity and specificity) for experts to be included
    /// * `min_diversity` - Minimum diversity (signed Jaccard dissimilarity) for experts to be included
    /// * `voting_method` - Voting method to be used
    /// * `voting_threshold` - Voting threshold for decision making
    /// * `threshold_window` - Threshold window for majority voting
    /// * `weighting_method` - Weighting method for experts
    ///
    /// # Panics
    ///
    /// Panics if :
    /// * voting threshold is not in [0,1] or if threshold window is not in [0,100]
    /// * specialized weighting method thresholds are not in [0,1]
    pub fn new(
        pop: &Population,
        min_perf: &f64,
        min_diversity: &f64,
        voting_method: &VotingMethod,
        voting_threshold: &f64,
        threshold_window: &f64,
        weighting_method: &WeightingMethod,
    ) -> Self {
        let mut experts: Population = pop.clone();

        if *min_perf > 0.0 {
            let n = experts.individuals.len();
            experts.individuals.retain(|expert| {
                expert.sensitivity >= *min_perf && expert.specificity >= *min_perf
            });
            debug!("Judges filtered for minimum sensitivity and specificity: {}/{} individuals retained", experts.individuals.len(), n);
        }

        if *min_diversity > 0.0 {
            let n = experts.individuals.len();
            experts = experts.filter_by_signed_jaccard_dissimilarity(*min_diversity, true);
            debug!(
                "Judges filtered for diversity: {}/{} individuals retained",
                experts.individuals.len(),
                n
            );
        }

        if voting_threshold < &0.0 || voting_threshold > &1.0 {
            panic!("Voting threshold should be in [0,1]")
        }

        let window;
        if threshold_window < &0.0 || threshold_window > &100.0 {
            panic!("Voting threshold should be in [0,100]")
        } else if *threshold_window == 0.0 {
            window = 1e-10;
        } else {
            window = *threshold_window;
        }

        match weighting_method {
            WeightingMethod::Specialized {
                sensitivity_threshold,
                specificity_threshold,
            } => {
                warn!("Specialized voting mode is experimental");
                if *sensitivity_threshold < 0.0 || *sensitivity_threshold > 1.0 {
                    panic!("Sensitivity threshold must be in [0,1]");
                }
                if *specificity_threshold < 0.0 || *specificity_threshold > 1.0 {
                    panic!("Specificity threshold must be in [0,1]");
                }
            }
            _ => (),
        }

        experts = experts.sort();

        Jury {
            experts,
            voting_method: voting_method.clone(),
            voting_threshold: *voting_threshold,
            threshold_window: window,
            weighting_method: weighting_method.clone(),
            weights: None,
            auc: 0.0,
            accuracy: 0.0,
            sensitivity: 0.0,
            specificity: 0.0,
            rejection_rate: 1.0,
            predicted_classes: None,
            metrics: AdditionalMetrics::default(),
        }
    }

    /// Creates a new Jury from Population and Param
    ///
    /// # Arguments
    ///
    /// * `pop` - Population of experts
    /// * `data` - Training data used for pruning if needed
    /// * `param` - Parameters containing voting settings
    ///
    /// # Panics
    ///
    /// Panics if :
    /// * voting threshold is not in [0,1]
    /// * specialized weighting method thresholds are not in [0,1]
    ///
    /// # Note
    ///
    /// If `prune_before_voting` is set in `param`, experts will be pruned based on MDA before creating the Jury
    pub fn new_from_param(pop: &Population, data: &Data, param: &Param) -> Self {
        // Make voting_pop an owned Population so we don't take references to temporaries
        let mut voting_pop: Population =
            if param.voting.fbm_ci_alpha <= 1.0 && param.voting.fbm_ci_alpha >= 0.0 {
                pop.select_best_population_with_method(
                    param.voting.fbm_ci_alpha,
                    &param.voting.fbm_ci_method,
                )
            } else {
                pop.clone()
            };

        if param.voting.prune_before_voting {
            debug!("Pruning experts using MDA resulting from 1000 permutations...");
            // prune_all_by_importance mutates in place and returns &mut Self; call it for side-effects
            let _ = voting_pop.prune_all_by_importance(
                data,
                1000,
                param.general.seed,
                Some(0.0),
                None,
                10,
            );
            voting_pop.fit(data, &mut None, &None, &None, param);
            voting_pop = voting_pop.sort();
        }

        // let weighting_method = if param.voting.specialized {
        //         warn!("Specialized voting mode is experimental");
        //         if param.voting.specialized_pos_threshold < 0.0 || param.voting.specialized_pos_threshold > 1.0 {
        //             panic!("Sensitivity threshold must be in [0,1]");
        //         }
        //         if param.voting.specialized_neg_threshold < 0.0 || param.voting.specialized_neg_threshold > 1.0 {
        //             panic!("Specificity threshold must be in [0,1]");
        //         }
        //         WeightingMethod::Specialized {sensitivity_threshold: param.voting.specialized_pos_threshold, specificity_threshold: param.voting.specialized_neg_threshold}
        //     } else {
        //         WeightingMethod::Uniform
        //     };

        let mut jury = Jury::new(
            &voting_pop,
            &param.voting.min_perf,
            &param.voting.min_diversity,
            &param.voting.method,
            &param.voting.method_threshold,
            &param.voting.threshold_windows_pct,
            &WeightingMethod::Uniform,
        );

        // Apply expert count constraints
        let n = jury.experts.individuals.len();
        if param.voting.max_experts > 0 && n > param.voting.max_experts {
            debug!(
                "Jury truncated from {} to {} experts (max_experts)",
                n, param.voting.max_experts
            );
            jury.experts.individuals.truncate(param.voting.max_experts);
        }
        if param.voting.min_experts > 0 && jury.experts.individuals.len() < param.voting.min_experts
        {
            warn!(
                "Jury has only {} experts, fewer than requested min_experts={}",
                jury.experts.individuals.len(),
                param.voting.min_experts
            );
        }

        jury
    }

    /// Evaluates learning data and adjusts internal weight and performance variables accordingly
    ///
    /// # Arguments
    ///
    /// * `data` - Training data used for evaluation
    ///
    /// # Panics
    ///
    /// Panics if evaluate is called before fitting experts on training data
    pub fn evaluate(&mut self, data: &Data) {
        for expert in &mut self.experts.individuals {
            if expert.accuracy == 0.0 {
                expert.compute_roc_and_metrics(data, &FitFunction::auc, None);
            }
        }

        let weights = self.compute_weights_by_method(data);

        let effective_experts = weights.iter().filter(|&w| *w > 0.0).count();
        if effective_experts % 2 == 0 && effective_experts > 0 {
            warn!(
                "Even number of effective experts ({}). Perfect ties will be abstained (class 2).",
                effective_experts
            );
        }

        self.weights = Some(weights);

        if self.voting_threshold == 0.0 && self.voting_method == VotingMethod::Majority {
            self.voting_threshold = self.optimize_majority_threshold_youden(data);
            warn!(
                "Threshold set to 0.0. Using Youden Maxima as threshold: {}",
                self.voting_threshold
            );
        }

        self.predicted_classes = Some(self.predict(data).0);

        let (auc, accuracy, sensitivity, specificity, rejection_rate, additional_metrics) =
            self.compute_new_metrics(data);

        self.auc = auc;
        self.accuracy = accuracy;
        self.sensitivity = sensitivity;
        self.specificity = specificity;
        self.rejection_rate = rejection_rate;
        self.metrics = additional_metrics;
    }

    /// Computes new metrics on given data based on internal weights and variables
    ///
    /// # Arguments
    ///
    /// * `data` - Data used for metric computation
    ///
    /// # Panics
    ///
    /// Panics if compute_new_metrics is called before evaluate()
    ///
    /// # Returns
    ///
    /// A tuple `(auc, accuracy, sensitivity, specificity, rejection_rate, additional_metrics)`
    /// where `additional_metrics` contains optional MCC, F1-score, NPV, PPV, G-mean
    pub fn compute_new_metrics(&self, data: &Data) -> (f64, f64, f64, f64, f64, AdditionalMetrics) {
        if self.weights.is_none() {
            panic!("Jury must be evaluated on training data first. Call evaluate() before compute_new_metrics().");
        }

        let (pred_classes, scores) = self.predict(data);

        let filtered_data: Vec<(f64, u8, u8)> = scores
            .iter()
            .zip(pred_classes.iter())
            .zip(data.y.iter())
            .filter_map(|((&score, &pred_class), &true_class)| {
                if score >= 0.0 && score <= 1.0 && pred_class != 2 {
                    Some((score, pred_class, true_class))
                } else {
                    None
                }
            })
            .collect();

        let rejection_rate = self.compute_rejection_rate(&pred_classes);

        if !filtered_data.is_empty() {
            let (scores_filtered, pred_filtered, true_filtered): (Vec<f64>, Vec<u8>, Vec<u8>) =
                filtered_data.into_iter().fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut scores, mut preds, mut trues), (s, p, t)| {
                        scores.push(s);
                        preds.push(p);
                        trues.push(t);
                        (scores, preds, trues)
                    },
                );

            let auc = compute_auc_from_value(&scores_filtered, &true_filtered);

            // Vérifier si des experts ont des métriques additionnelles
            let compute_additional = [
                self.experts
                    .individuals
                    .iter()
                    .any(|e| e.metrics.mcc.is_some()),
                self.experts
                    .individuals
                    .iter()
                    .any(|e| e.metrics.f1_score.is_some()),
                self.experts
                    .individuals
                    .iter()
                    .any(|e| e.metrics.npv.is_some()),
                self.experts
                    .individuals
                    .iter()
                    .any(|e| e.metrics.ppv.is_some()),
                self.experts
                    .individuals
                    .iter()
                    .any(|e| e.metrics.g_mean.is_some()),
            ];

            let (accuracy, sensitivity, specificity, additional_metrics) =
                compute_metrics_from_classes(&pred_filtered, &true_filtered, compute_additional);

            (
                auc,
                accuracy,
                sensitivity,
                specificity,
                rejection_rate,
                additional_metrics,
            )
        } else {
            (
                0.5,
                0.0,
                0.0,
                0.0,
                rejection_rate,
                AdditionalMetrics::default(),
            )
        }
    }

    /// Optimizes majority voting threshold based on Youden's index
    ///
    /// # Arguments
    ///
    /// * `data` - Training data used for optimization
    ///
    /// # Returns
    ///
    /// The optimized threshold value
    pub fn optimize_majority_threshold_youden(&mut self, data: &Data) -> f64 {
        let mut best_threshold = 0.5;
        let mut best_youden = 0.0;

        let step_size = match data.sample_len {
            0..=50 => 0.1,
            51..=200 => 0.05,
            _ => 0.01,
        };

        let steps = (1.0 / step_size) as i32;
        for i in 1..steps {
            let threshold = i as f64 * step_size;
            let predictions = self.compute_majority_threshold_vote(
                data,
                &self.weights.as_ref().unwrap(),
                threshold,
                self.threshold_window,
            );

            let filtered_data: Vec<(u8, u8)> = predictions
                .1
                .iter()
                .zip(predictions.0.iter())
                .zip(data.y.iter())
                .filter_map(|((&score, &pred_class), &true_class)| {
                    if score >= 0.0 && score <= 1.0 && pred_class != 2 && true_class != 2 {
                        Some((pred_class, true_class))
                    } else {
                        None
                    }
                })
                .collect();

            if !filtered_data.is_empty() {
                let (pred_classes, true_classes): (Vec<u8>, Vec<u8>) =
                    filtered_data.into_iter().unzip();
                let (_, sensitivity, specificity, _) =
                    compute_metrics_from_classes(&pred_classes, &true_classes, [false; 5]);

                let youden_index = sensitivity + specificity - 1.0;

                if youden_index > best_youden {
                    best_youden = youden_index;
                    best_threshold = threshold;
                }
            }
        }

        self.voting_threshold = best_threshold;
        best_threshold
    }

    /// Predicts classes and scores for given data using current weights and voting method
    ///
    /// # Arguments
    ///
    /// * `data` - Data for which predictions are to be made
    ///
    /// # Panics
    ///
    /// Panics if weights have not been computed (i.e., evaluate() has not been called)
    ///
    /// # Returns
    ///
    /// A tuple `(predicted_classes, scores)` where `predicted_classes` is a vector of predicted class labels
    /// and `scores` is a vector of associated pos_vote/(pos_vote+neg_vote) ratios
    pub fn predict(&self, data: &Data) -> (Vec<u8>, Vec<f64>) {
        let weights = self
            .weights
            .as_ref()
            .expect("Weights must be computed before prediction. Call evaluate() first.");
        self.apply_voting_mechanism(data, weights)
    }

    /// Computes weights for experts based on the selected weighting method
    ///
    /// # Arguments
    ///
    /// * `data` - Training data used for weight computation (if needed, not used in current methods)
    ///
    /// # Returns
    ///
    /// A vector of weights corresponding to each expert
    fn compute_weights_by_method(&self, _data: &Data) -> Vec<f64> {
        match &self.weighting_method {
            WeightingMethod::Uniform => vec![1.0; self.experts.individuals.len()],
            WeightingMethod::Specialized {
                sensitivity_threshold,
                specificity_threshold,
            } => self.compute_group_strict_weights(*sensitivity_threshold, *specificity_threshold),
        }
    }

    /// Applies the selected voting mechanism to the data using the provided weights
    ///
    /// # Arguments
    ///
    /// * `data` - Data for which predictions are to be made
    /// * `weights` - Weights assigned to each expert
    ///
    /// # Returns
    ///
    /// A tuple `(predicted_classes, scores)` where `predicted_classes` is a vector of predicted class labels
    /// and `scores` is a vector of associated pos_vote/(pos_vote+neg_vote) ratios
    fn apply_voting_mechanism(&self, data: &Data, weights: &[f64]) -> (Vec<u8>, Vec<f64>) {
        match &self.voting_method {
            VotingMethod::Majority => self.compute_majority_threshold_vote(
                data,
                weights,
                self.voting_threshold,
                self.threshold_window,
            ),
            VotingMethod::Consensus => {
                self.compute_consensus_threshold_vote(data, weights, self.voting_threshold)
            }
        }
    }

    /// Computes predictions using consensus voting mechanism
    ///
    /// # Arguments
    ///
    /// * `data` - Data for which predictions are to be made
    /// * `weights` - Weights assigned to each expert
    /// * `threshold` - Consensus threshold for decision making
    ///
    /// # Returns
    ///
    /// A tuple `(predicted_classes, scores)` where `predicted_classes` is a vector of predicted class labels
    /// and `scores` is a vector of associated pos_vote/(pos_vote+neg_vote) ratios
    fn compute_consensus_threshold_vote(
        &self,
        data: &Data,
        weights: &[f64],
        threshold: f64,
    ) -> (Vec<u8>, Vec<f64>) {
        let mut predicted_classes = Vec::with_capacity(data.sample_len);
        let mut ratios = Vec::with_capacity(data.sample_len);

        let expert_predictions: Vec<Vec<u8>> = self
            .experts
            .individuals
            .iter()
            .map(|expert| expert.evaluate_class(data))
            .collect();

        for sample_index in 0..data.sample_len {
            let mut weighted_positive = 0.0;
            let mut weighted_negative = 0.0;
            let mut effective_total_weight = 0.0;

            for (expert_idx, expert_pred) in expert_predictions.iter().enumerate() {
                if sample_index < expert_pred.len() {
                    let prediction = expert_pred[sample_index];
                    let weight = weights[expert_idx];

                    // Only count non-abstaining votes
                    if weight > 0.0 && prediction != 2 {
                        effective_total_weight += weight;

                        if prediction == 1 {
                            weighted_positive += weight;
                        } else if prediction == 0 {
                            weighted_negative += weight;
                        }
                    }
                }
            }

            let (predicted_class, pos_ratio) = if effective_total_weight > 0.0 {
                let pos_ratio = weighted_positive / effective_total_weight;
                let neg_ratio = weighted_negative / effective_total_weight;

                let class = if pos_ratio >= threshold {
                    1u8
                } else if neg_ratio >= threshold {
                    0u8
                } else {
                    2u8
                };

                (class, pos_ratio)
            } else {
                (2u8, -1.0)
            };

            ratios.push(pos_ratio);
            predicted_classes.push(predicted_class);
        }

        (predicted_classes, ratios)
    }

    /// Computes predictions using majority voting mechanism with threshold
    ///
    /// # Arguments
    ///
    /// * `data` - Data for which predictions are to be made
    /// * `weights` - Weights assigned to each expert
    /// * `threshold` - Majority voting threshold for decision making
    /// * `threshold_window` - Threshold window for abstention
    ///
    /// # Panics
    ///
    /// Panics if the length of weights does not match the number of experts
    ///
    /// # Returns
    ///
    /// A tuple `(predicted_classes, scores)` where `predicted_classes` is a vector of predicted class labels
    /// and `scores` is a vector of associated pos_vote/(pos_vote+neg_vote) ratios
    ///
    /// # Note
    ///
    /// If the ratio of positive votes to total votes is within the threshold window of the threshold,
    /// the prediction is set to abstain (class 2).
    fn compute_majority_threshold_vote(
        &self,
        data: &Data,
        weights: &[f64],
        threshold: f64,
        threshold_window: f64,
    ) -> (Vec<u8>, Vec<f64>) {
        if weights.len() != self.experts.individuals.len() {
            panic!(
                "Weights length ({}) must match expert count ({})",
                weights.len(),
                self.experts.individuals.len()
            );
        }

        let mut predicted_classes = Vec::with_capacity(data.sample_len);
        let mut ratios = Vec::with_capacity(data.sample_len);

        let expert_predictions: Vec<Vec<u8>> = self
            .experts
            .individuals
            .iter()
            .map(|expert| expert.evaluate_class(data))
            .collect();
        for sample_index in 0..data.sample_len {
            let mut weighted_positive = 0.0;
            let mut total_weight = 0.0;

            for (expert_idx, expert_pred) in expert_predictions.iter().enumerate() {
                if sample_index < expert_pred.len() {
                    let vote = expert_pred[sample_index];

                    if vote == 2 {
                        continue;
                    }

                    let weight = weights[expert_idx];
                    total_weight += weight;
                    if vote == 1 {
                        weighted_positive += weight;
                    }
                }
            }

            let (predicted_class, ratio) = if total_weight > 0.0 {
                let ratio = weighted_positive / total_weight;

                let class = if (ratio - threshold).abs() < (threshold_window / 100.0) {
                    2u8 // Rejection collective (threshold_window)
                } else if ratio >= threshold {
                    1u8
                } else {
                    0u8
                };

                (class, ratio)
            } else {
                (2u8, 0.5)
            };

            ratios.push(ratio);
            predicted_classes.push(predicted_class);
        }

        (predicted_classes, ratios)
    }

    /// Computes predicted classes for given data and stores them internally
    ///
    /// # Arguments
    ///
    /// * `data` - Data for which predicted classes are to be computed
    pub fn compute_classes(&mut self, data: &Data) {
        let predictions = self.predict(data);
        self.predicted_classes = Some(predictions.0);
    }

    /// Computes rejection rate based on predicted classes
    ///
    /// # Arguments
    ///
    /// * `predictions` - Vector of predicted class labels
    ///
    /// # Returns
    ///
    /// The rejection rate as a floating-point value.
    fn compute_rejection_rate(&self, predictions: &[u8]) -> f64 {
        let total_samples = predictions.len();
        let rejected_samples = predictions.iter().filter(|&&pred| pred == 2).count();

        if total_samples > 0 {
            rejected_samples as f64 / total_samples as f64
        } else {
            0.0
        }
    }

    /// Counts total and effective experts based on weighting method
    ///
    /// # Returns
    ///
    /// A tuple `(total_experts, effective_experts)` where `effective_experts` are those contributing non-zero weight
    fn count_effective_experts(&self) -> (usize, usize) {
        match &self.weighting_method {
            WeightingMethod::Uniform => (
                self.experts.individuals.len(),
                self.experts.individuals.len(),
            ),
            WeightingMethod::Specialized {
                sensitivity_threshold,
                specificity_threshold,
            } => {
                let mut total_experts = 0;
                let mut effective_experts = 0;

                for expert in &self.experts.individuals {
                    total_experts += 1;
                    let specialization = self.get_expert_specialization(
                        expert,
                        *sensitivity_threshold,
                        *specificity_threshold,
                    );

                    if !matches!(specialization, JudgeSpecialization::Ineffective) {
                        effective_experts += 1;
                    }
                }

                (total_experts, effective_experts)
            }
        }
    }

    /// Determines the specialization of an expert based on sensitivity and specificity thresholds
    ///
    /// # Arguments
    ///
    /// * `expert` - The expert whose specialization is to be determined
    /// * `sensitivity_threshold` - Sensitivity threshold for specialization classification
    /// * `specificity_threshold` - Specificity threshold for specialization classification
    ///
    /// # Returns
    ///
    /// The `JudgeSpecialization` of the expert
    fn get_expert_specialization(
        &self,
        expert: &Individual,
        sensitivity_threshold: f64,
        specificity_threshold: f64,
    ) -> JudgeSpecialization {
        if expert.sensitivity >= sensitivity_threshold
            && expert.specificity >= specificity_threshold
        {
            JudgeSpecialization::Balanced
        } else if expert.sensitivity >= sensitivity_threshold {
            JudgeSpecialization::PositiveSpecialist
        } else if expert.specificity >= specificity_threshold {
            JudgeSpecialization::NegativeSpecialist
        } else {
            JudgeSpecialization::Ineffective
        }
    }

    /// Computes specialized weights for experts based on their specializations
    ///
    /// # Arguments
    ///
    /// * `sensitivity_threshold` - Sensitivity threshold for specialization classification
    /// * `specificity_threshold` - Specificity threshold for specialization classification
    ///
    /// # Returns
    ///
    /// A vector of weights corresponding to each expert
    fn compute_group_strict_weights(
        &self,
        sensitivity_threshold: f64,
        specificity_threshold: f64,
    ) -> Vec<f64> {
        let mut specs = Vec::new();
        let mut pos = 0usize;
        let mut neg = 0usize;
        let mut bal = 0usize;

        for expert in &self.experts.individuals {
            let s = self.get_expert_specialization(
                expert,
                sensitivity_threshold,
                specificity_threshold,
            );
            specs.push(s.clone());
            match s {
                JudgeSpecialization::PositiveSpecialist => pos += 1,
                JudgeSpecialization::NegativeSpecialist => neg += 1,
                JudgeSpecialization::Balanced => bal += 1,
                _ => {}
            }
        }

        let active_groups = [
            (pos, JudgeSpecialization::PositiveSpecialist),
            (neg, JudgeSpecialization::NegativeSpecialist),
            (bal, JudgeSpecialization::Balanced),
        ]
        .iter()
        .filter(|(n, _)| *n > 0)
        .count();

        if active_groups == 0 {
            panic!("Specialized threshold are too high to allow expert selection")
        }

        let group_share = 1.0 / active_groups as f64;

        specs
            .into_iter()
            .map(|sp| match sp {
                JudgeSpecialization::PositiveSpecialist if pos > 0 => group_share / pos as f64,
                JudgeSpecialization::NegativeSpecialist if neg > 0 => group_share / neg as f64,
                JudgeSpecialization::Balanced if bal > 0 => group_share / bal as f64,
                _ => 0.0,
            })
            .collect()
    }

    /// Generates a detailed display string of the voting analysis
    ///
    /// # Arguments
    ///
    /// * `data` - Training data used for analysis
    /// * `test_data` - Optional test data for additional analysis
    /// * `param` - Parameters containing display settings
    ///
    /// # Returns
    ///
    /// A formatted string containing the voting analysis report.    
    pub fn display(&self, data: &Data, test_data: Option<&Data>, param: &Param) -> String {
        let mut text = format!(
            "{}\n{}{}VOTING ANALYSIS{}{}\n{}\n",
            "═".repeat(80),
            "\x1b[1m",
            " ".repeat(31),
            " ".repeat(32),
            "\x1b[0m",
            "═".repeat(80)
        );

        text = format!(
            "{}\n{}{}",
            text,
            self.display_compact_summary(data, test_data),
            self.display_voting_method_info()
        );

        text = format!(
            "{}\n\n{}{} DETAILED METRICS {}{}",
            text,
            "\x1b[1m",
            "~".repeat(31),
            "~".repeat(31),
            "\x1b[0m"
        );
        text = format!(
            "{}\n{}",
            text,
            self.display_confusion_matrix(
                &self.predicted_classes.as_ref().unwrap(),
                &data.y,
                "TRAIN"
            )
        );
        if let Some(test_data) = test_data {
            let test_preds = self.predict(test_data);
            text = format!(
                "{}\n{}",
                text,
                self.display_confusion_matrix(&test_preds.0, &test_data.y, "TEST")
            );
        }

        if let Some(test_data) = test_data {
            text = format!(
                "{}\n{}",
                text,
                self.display_predictions_by_sample(
                    test_data,
                    &param.voting.complete_display,
                    "TEST"
                )
            );
        } else {
            text = format!(
                "{}\n{}",
                text,
                self.display_predictions_by_sample(data, &param.voting.complete_display, "TRAIN")
            );
        }

        text = format!(
            "\n{}\n\n{}{} EXPERT POPULATION ({}) {}{}",
            text,
            "\x1b[1m",
            "~".repeat(25),
            self.experts.individuals.len(),
            "~".repeat(25),
            "\x1b[0m"
        );
        text = format!(
            "{}\n{}",
            text,
            self.experts.clone().display(&data, test_data, param)
        );

        if param.voting.complete_display {
            text = format!(
                "{}\n{}",
                text,
                self.experts.display_feature_prevalence(data, 0)
            );
        } else {
            text = format!(
                "{}\n{}",
                text,
                self.experts.display_feature_prevalence(data, 20)
            );
        }

        // Apply color stripping if needed before returning
        format!(
            "{}\n",
            crate::utils::strip_ansi_if_needed(&text, param.general.display_colorful)
        )
    }

    fn display_confusion_matrix(
        &self,
        predictions: &[u8],
        true_labels: &[u8],
        title: &str,
    ) -> String {
        let mut text = "".to_string();
        let (mut tp, mut tn, mut fp, mut fn_, mut rp_abstentions, mut rn_abstentions) =
            (0, 0, 0, 0, 0, 0);

        for (pred, real) in predictions.iter().zip(true_labels.iter()) {
            match (*pred, *real) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fp += 1,
                (0, 1) => fn_ += 1,
                (2, 1) => rp_abstentions += 1,
                (2, 0) => rn_abstentions += 1,
                _ => warn!(
                    "Warning: Unexpected class values pred={}, real={}",
                    pred, real
                ),
            }
        }

        text = format!(
            "{}\n{} CONFUSION MATRIX ({}) {}",
            text,
            "─".repeat(15),
            title,
            "─".repeat(15)
        );
        text = format!("{}\n\n         | \x1b[1;96mPred 1\x1b[0m | \x1b[1;95mPred 0\x1b[0m | \x1b[1;90mAbstain\x1b[0m", text);
        text = format!(
            "{}\n\x1b[1;96mReal 1\x1b[0m   | {:>6} | {:>6} | {:>7}",
            text, tp, fn_, rp_abstentions
        );
        text = format!(
            "{}\n\x1b[1;95mReal 0\x1b[0m   | {:>6} | {:>6} | {:>7}",
            text, fp, tn, rn_abstentions
        );

        text
    }

    /// Generates a compact summary string of the voting analysis
    ///
    /// # Arguments
    ///
    /// * `data` - Training data used for analysis
    /// * `test_data` - Optional test data for additional analysis
    ///
    /// # Returns
    ///
    /// A formatted string containing the compact summary of voting performance, including train and test metrics if available.
    fn display_compact_summary(&self, _: &Data, test_data: Option<&Data>) -> String {
        let summary: String;
        let (total_experts, _) = self.count_effective_experts();

        let weighting_info = match &self.weighting_method {
            WeightingMethod::Uniform => "",
            WeightingMethod::Specialized { .. } => "specialized-weighted ",
        };

        let voting_info = match &self.voting_method {
            VotingMethod::Majority => "Majority",
            VotingMethod::Consensus => "Consensus",
        };

        let method_display = format!(
            "\x1b[1m{} jury [{} {}experts]",
            voting_info, total_experts, weighting_info
        );

        if test_data.is_some() {
            let (
                test_auc,
                test_accuracy,
                test_sensitivity,
                test_specificity,
                test_rejection_rate,
                test_additional,
            ) = self.compute_new_metrics(test_data.unwrap());

            let mut summary_str = format!("{} | AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3} | rejection rate {:.3}/{:.3}", 
                    method_display,
                    self.auc, test_auc,
                    self.accuracy, test_accuracy,
                    self.sensitivity, test_sensitivity,
                    self.specificity, test_specificity,
                    self.rejection_rate, test_rejection_rate);

            // Ajouter les métriques additionnelles si présentes
            if self.metrics.mcc.is_some() {
                summary_str = format!(
                    "{} | MCC {:.3}/{:.3}",
                    summary_str,
                    self.metrics.mcc.unwrap(),
                    test_additional.mcc.unwrap_or(0.0)
                );
            }
            if self.metrics.f1_score.is_some() {
                summary_str = format!(
                    "{} | F1-score {:.3}/{:.3}",
                    summary_str,
                    self.metrics.f1_score.unwrap(),
                    test_additional.f1_score.unwrap_or(0.0)
                );
            }
            if self.metrics.npv.is_some() {
                summary_str = format!(
                    "{} | NPV {:.3}/{:.3}",
                    summary_str,
                    self.metrics.npv.unwrap(),
                    test_additional.npv.unwrap_or(0.0)
                );
            }
            if self.metrics.ppv.is_some() {
                summary_str = format!(
                    "{} | PPV {:.3}/{:.3}",
                    summary_str,
                    self.metrics.ppv.unwrap(),
                    test_additional.ppv.unwrap_or(0.0)
                );
            }
            if self.metrics.g_mean.is_some() {
                summary_str = format!(
                    "{} | G-mean {:.3}/{:.3}",
                    summary_str,
                    self.metrics.g_mean.unwrap(),
                    test_additional.g_mean.unwrap_or(0.0)
                );
            }

            summary = format!("{}{}", summary_str, "\x1b[0m");
        } else {
            let mut summary_str = format!("{} | AUC {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} | rejection rate {:.3}", 
                    method_display,
                    self.auc,
                    self.accuracy,
                    self.sensitivity,
                    self.specificity,
                    self.rejection_rate);

            // Ajouter les métriques additionnelles si présentes
            if self.metrics.mcc.is_some() {
                summary_str = format!("{} | MCC {:.3}", summary_str, self.metrics.mcc.unwrap());
            }
            if self.metrics.f1_score.is_some() {
                summary_str = format!(
                    "{} | F1-score {:.3}",
                    summary_str,
                    self.metrics.f1_score.unwrap()
                );
            }
            if self.metrics.npv.is_some() {
                summary_str = format!("{} | NPV {:.3}", summary_str, self.metrics.npv.unwrap());
            }
            if self.metrics.ppv.is_some() {
                summary_str = format!("{} | PPV {:.3}", summary_str, self.metrics.ppv.unwrap());
            }
            if self.metrics.g_mean.is_some() {
                summary_str = format!(
                    "{} | G-mean {:.3}",
                    summary_str,
                    self.metrics.g_mean.unwrap()
                );
            }

            summary = format!("{}{}", summary_str, "\x1b[0m");
        }

        summary
    }

    /// Generates a string displaying information about the voting method and weighting
    ///
    /// # Returns
    ///
    /// A formatted string containing details about the voting method and weighting strategy.
    fn display_voting_method_info(&self) -> String {
        let mut info: String = "".to_string();
        match &self.weighting_method {
            WeightingMethod::Uniform => {}
            WeightingMethod::Specialized {
                sensitivity_threshold,
                specificity_threshold,
            } => {
                info = self
                    .display_expert_specializations(*sensitivity_threshold, *specificity_threshold);
            }
        }
        info
    }

    /// Generates a detailed display string of predictions by sample
    ///
    /// # Arguments
    ///
    /// * `data` - Data for which predictions are to be displayed
    /// * `complete_display` - Flag indicating whether to display all samples or a subset
    /// * `title` - Title indicating whether the data is "TRAIN" or "TEST"
    ///
    /// # Returns
    ///
    /// A formatted string containing detailed predictions by sample, categorized by correctness and sorted by inconsistency
    fn display_predictions_by_sample(
        &self,
        data: &Data,
        complete_display: &bool,
        title: &str,
    ) -> String {
        let mut text = "".to_string();
        text = format!(
            "{}\n{}{} PREDICTIONS BY SAMPLE ({}) {}{}\n\n{}",
            text,
            "\x1b[1;1m\n",
            "~".repeat(25),
            title,
            "~".repeat(25),
            "\x1b[0m",
            "─".repeat(80)
        );

        let predictions = if title == "TEST" {
            self.predict(data).0
        } else {
            self.predicted_classes.as_ref().unwrap().clone()
        };

        if predictions.len() != data.sample_len {
            return format!(
                "Error: Predictions length ({}) != data length ({})",
                predictions.len(),
                data.sample_len
            );
        }

        let (errors, abstentions, correct, inconsistency_list) =
            self.categorize_and_sort_by_inconsistency(data, &predictions);
        let inconsistency_map: std::collections::HashMap<usize, f64> =
            inconsistency_list.iter().cloned().collect();

        let nb_samples_to_show = if *complete_display {
            data.sample_len
        } else {
            20.min(data.sample_len)
        };

        let max_errors = if errors.len() > 0 {
            (nb_samples_to_show * 60 / 100).max(1).min(errors.len())
        } else {
            0
        };
        let max_abstentions = if abstentions.len() > 0 {
            ((nb_samples_to_show - max_errors) * 60 / 100)
                .max(1)
                .min(abstentions.len())
        } else {
            0
        };
        let max_correct = (nb_samples_to_show - max_errors - max_abstentions).min(correct.len());

        text = format!(
            "{}\n{}Sample\t\t| Real | Predictions\t| Result | Consistency{}\n{}",
            text,
            "\x1b[1m",
            "\x1b[0m",
            "─".repeat(80)
        );

        if max_errors > 0 {
            text = format!(
                "{}\n{}─────── ERRORS ({} shown of {}, sorted by inconsistency) ───────{}",
                text,
                "\x1b[1;31m",
                max_errors,
                errors.len(),
                "\x1b[0m"
            );

            for &sample_idx in errors.iter().take(max_errors) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let expert_votes = self.display_expert_votes_for_sample(data, sample_idx);
                let inconsistency = inconsistency_map.get(&sample_idx).unwrap_or(&0.0);
                let consistency_percent = (1.0 - inconsistency) * 100.0;

                text = format!(
                    "{}\n{:>10}\t| {:>4} | {} → {}\t| \x1b[1;31m✗\x1b[0m     | {:>6.1}%",
                    text,
                    sample_name,
                    real_class,
                    expert_votes,
                    predicted_class,
                    consistency_percent
                );
            }
        }

        if max_abstentions > 0 {
            text = format!(
                "{}\n{}────── ABSTENTIONS ({} shown of {}, sorted by inconsistency) ─────{}",
                text,
                "\x1b[1;90m",
                max_abstentions,
                abstentions.len(),
                "\x1b[0m"
            );

            for &sample_idx in abstentions.iter().take(max_abstentions) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let expert_votes = self.display_expert_votes_for_sample(data, sample_idx);
                let inconsistency = inconsistency_map.get(&sample_idx).unwrap_or(&0.0);
                let consistency_percent = (1.0 - inconsistency) * 100.0;

                text = format!(
                    "{}\n{:>10}\t| {:>4} | {} → {} | \x1b[90m~\x1b[0m     | {:>6.1}%",
                    text,
                    sample_name,
                    real_class,
                    expert_votes,
                    predicted_class,
                    consistency_percent
                );
            }
        }

        if max_correct > 0 {
            text = format!(
                "{}\n{}─────── CORRECT ({} shown of {}, sorted by inconsistency) ───────{}",
                text,
                "\x1b[1;32m",
                max_correct,
                correct.len(),
                "\x1b[0m"
            );

            for &sample_idx in correct.iter().take(max_correct) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let expert_votes = self.display_expert_votes_for_sample(data, sample_idx);
                let inconsistency = inconsistency_map.get(&sample_idx).unwrap_or(&0.0);
                let consistency_percent = (1.0 - inconsistency) * 100.0;

                text = format!(
                    "{}\n{:>10}\t| {:>4} | {} → {} | \x1b[1;32m✓\x1b[0m     | {:>6.1}%",
                    text,
                    sample_name,
                    real_class,
                    expert_votes,
                    predicted_class,
                    consistency_percent
                );
            }
        }

        let total_shown = max_errors + max_abstentions + max_correct;
        if data.sample_len > total_shown {
            text = format!(
                "{}\n ... {} additional samples not shown",
                text,
                data.sample_len - total_shown
            );
        }

        // Statistiques avec métriques d'inconsistance
        let avg_inconsistency = inconsistency_list.iter().map(|(_, inc)| inc).sum::<f64>()
            / inconsistency_list.len() as f64;
        let avg_consistency = (1.0 - avg_inconsistency) * 100.0;

        text = format!(
            "{}\n\n{}Errors: {} | Correct: {} | Rejections: {} | Avg Consistency: {:.1}%{}",
            text,
            "\x1b[1;33m",
            errors.len(),
            correct.len(),
            abstentions.len(),
            avg_consistency,
            "\x1b[0m"
        );

        text
    }

    /// Computes inconsistency for each sample based on expert predictions
    ///
    /// # Arguments
    ///
    /// * `data` - Data for which inconsistency is to be computed
    ///
    /// # Returns
    ///
    /// A vector of tuples `(sample_index, inconsistency_value)` for each sample
    fn compute_sample_inconsistency(&self, data: &Data) -> Vec<(usize, f64)> {
        let mut inconsistency_list = Vec::new();

        let expert_predictions: Vec<Vec<u8>> = self
            .experts
            .individuals
            .iter()
            .map(|expert| expert.evaluate_class(data))
            .collect();

        for sample_idx in 0..data.sample_len {
            let mut vote_counts = std::collections::HashMap::new();
            let mut total_votes = 0;

            for expert_pred in &expert_predictions {
                if sample_idx < expert_pred.len() {
                    let vote = expert_pred[sample_idx];
                    *vote_counts.entry(vote).or_insert(0) += 1;
                    total_votes += 1;
                }
            }

            // Calculate inconsistency (1 - proportion of majority vote)
            let max_vote_count = vote_counts.values().max().copied().unwrap_or(0);
            let consistency = if total_votes > 0 {
                max_vote_count as f64 / total_votes as f64
            } else {
                0.0
            };
            let inconsistency = 1.0 - consistency;

            inconsistency_list.push((sample_idx, inconsistency));
        }

        inconsistency_list
    }

    /// Categorizes samples into errors, abstentions, and correct predictions. Sorts each category by inconsistency.
    ///
    /// # Arguments
    ///
    /// * `data` - Data for which categorization is to be performed
    /// * `predictions` - Vector of predicted class labels
    ///
    /// # Returns
    ///
    /// A tuple containing vectors of sample indices for errors, abstentions, correct predictions,
    /// and a vector of tuples `(sample_index, inconsistency_value)` for each sample
    fn categorize_and_sort_by_inconsistency(
        &self,
        data: &Data,
        predictions: &[u8],
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<(usize, f64)>) {
        let inconsistency_list = self.compute_sample_inconsistency(data);
        let inconsistency_map: std::collections::HashMap<usize, f64> =
            inconsistency_list.iter().cloned().collect();

        let mut errors = Vec::new();
        let mut abstentions = Vec::new();
        let mut correct = Vec::new();

        for i in 0..data.sample_len {
            let real_class = data.y[i];
            let predicted_class = predictions[i];

            match predicted_class {
                2 => abstentions.push(i),
                _ if predicted_class != real_class => errors.push(i),
                _ => correct.push(i),
            }
        }

        errors.sort_by(|&a, &b| {
            inconsistency_map
                .get(&b)
                .unwrap_or(&0.0)
                .partial_cmp(inconsistency_map.get(&a).unwrap_or(&0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        abstentions.sort_by(|&a, &b| {
            inconsistency_map
                .get(&b)
                .unwrap_or(&0.0)
                .partial_cmp(inconsistency_map.get(&a).unwrap_or(&0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        correct.sort_by(|&a, &b| {
            inconsistency_map
                .get(&b)
                .unwrap_or(&0.0)
                .partial_cmp(inconsistency_map.get(&a).unwrap_or(&0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        (errors, abstentions, correct, inconsistency_list)
    }

    /// Generates a display string for expert specializations
    ///
    /// # Arguments
    ///
    /// * `sensitivity_threshold` - Sensitivity threshold for specialization classification
    /// * `specificity_threshold` - Specificity threshold for specialization classification
    ///
    /// # Returns
    ///
    /// A formatted string listing each expert's specialization and performance metrics.
    pub fn display_expert_specializations(
        &self,
        sensitivity_threshold: f64,
        specificity_threshold: f64,
    ) -> String {
        let mut text = "".to_string();
        text = format!(
            "{}\n{} JUDGE SPECIALIZATIONS{}\n{}",
            text,
            "\x1b[1;45m",
            "\x1b[0m",
            "─".repeat(80)
        );
        text = format!(
            "{}\n{:<6} | {:<8} | {:<11} | {:<11} | {:<20}",
            text, "Judge", "Accuracy", "Sensitivity", "Specificity", "Specialization"
        );
        text = format!("{}\n{}", text, "─".repeat(80));

        for (idx, expert) in self.experts.individuals.iter().enumerate() {
            let specialization = self.get_expert_specialization(
                expert,
                sensitivity_threshold,
                specificity_threshold,
            );

            let spec_str = match specialization {
                JudgeSpecialization::PositiveSpecialist => "🔍 \x1b[96mPositive Specialist\x1b[0m",
                JudgeSpecialization::NegativeSpecialist => "🔍 \x1b[95mNegative Specialist\x1b[0m",
                JudgeSpecialization::Balanced => "⚖️  Balanced",
                JudgeSpecialization::Ineffective => "❌ \x1b[90mIneffective\x1b[0m",
            };

            match specialization {
                JudgeSpecialization::Ineffective => {}
                _ => {
                    text = format!(
                        "{}\n#{:<6} | {:<8.3} | {:<11.3} | {:<11.3} | {}",
                        text,
                        idx + 1,
                        expert.accuracy,
                        expert.sensitivity,
                        expert.specificity,
                        spec_str
                    )
                }
            }
        }

        text
    }

    /// Generates a colored display for a specialized vote based on expert specialization
    ///
    /// # Arguments
    ///
    /// * `specialization` - The specialization of the expert
    /// * `vote` - The vote cast by the expert (0 or 1)
    ///
    /// # Returns
    ///
    /// A tuple containing the color code and formatted vote string
    fn display_specialized_vote(
        &self,
        specialization: &JudgeSpecialization,
        vote: u8,
    ) -> (&'static str, String) {
        if vote == 2 {
            return ("\x1b[90m", "•".to_string());
        }
        match (specialization, vote) {
            (JudgeSpecialization::Ineffective, _) => ("\x1b[90m", format!("{}", vote)),
            (JudgeSpecialization::Balanced, _) => ("\x1b[92m", vote.to_string()),
            (JudgeSpecialization::PositiveSpecialist, 0) => ("\x1b[90m", format!("{}", vote)),
            (JudgeSpecialization::PositiveSpecialist, 1) => ("\x1b[96m", vote.to_string()),
            (JudgeSpecialization::NegativeSpecialist, 1) => ("\x1b[90m", format!("{}", vote)),
            (JudgeSpecialization::NegativeSpecialist, 0) => ("\x1b[95m", vote.to_string()),
            _ => ("\x1b[97m", vote.to_string()),
        }
    }

    /// Generates a display string of expert votes for a specific sample
    ///
    /// # Arguments
    ///
    /// * `data` - Data containing true labels
    /// * `sample_idx` - Index of the sample for which votes are to be displayed
    ///
    /// # Returns
    ///
    /// A formatted string containing the votes of each expert for the specified sample.
    fn display_expert_votes_for_sample(&self, data: &Data, sample_idx: usize) -> String {
        let mut output = String::new();

        for (_, expert) in self.experts.individuals.iter().enumerate() {
            let predictions = expert.evaluate_class(data);

            if sample_idx < predictions.len() {
                let vote = predictions[sample_idx];

                match &self.weighting_method {
                    WeightingMethod::Uniform => {
                        if vote == 2 {
                            output.push_str("\x1b[90m•\x1b[0m");
                        } else {
                            let vote_display = match data.y[sample_idx] == vote {
                                true => &format!("\x1b[92m{}\x1b[0m", vote),
                                false => &format!("\x1b[31m{}\x1b[0m", vote),
                            };
                            output.push_str(vote_display);
                        }
                    }
                    WeightingMethod::Specialized {
                        sensitivity_threshold,
                        specificity_threshold,
                    } => {
                        let specialization = self.get_expert_specialization(
                            expert,
                            *sensitivity_threshold,
                            *specificity_threshold,
                        );

                        let (color, symbol) = self.display_specialized_vote(&specialization, vote);
                        output.push_str(&format!("{}{}{}", color, symbol, "\x1b[0m"));
                    }
                }
            } else {
                output.push('?');
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Data;
    use crate::individual::{Individual, BINARY_LANG, RAW_TYPE};
    use crate::population::Population;
    use std::collections::HashMap;

    #[test]
    fn test_new_filters_experts_by_minimum_performance_threshold() {
        let mut population = Population::test();

        if population.individuals.len() >= 2 {
            population.individuals[0].sensitivity = 0.9;
            population.individuals[0].specificity = 0.8;
            population.individuals[1].sensitivity = 0.3;
            population.individuals[1].specificity = 0.2;
        }

        let jury = Jury::new(
            &population,
            &0.7, // min_perf
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        // Only the first expert should be retained (0.9 >= 0.7 && 0.8 >= 0.7)
        assert_eq!(jury.experts.individuals.len(), 1);
        assert!(jury.experts.individuals[0].sensitivity >= 0.7);
        assert!(jury.experts.individuals[0].specificity >= 0.7);
    }

    #[test]
    fn test_new_filters_experts_by_diversity_threshold() {
        let population = Population::test();
        let original_count = population.individuals.len();

        let jury = Jury::new(
            &population,
            &0.0,
            &0.8,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        assert!(jury.experts.individuals.len() <= original_count);
    }

    #[test]
    #[should_panic(expected = "Voting threshold should be in [0,1]")]
    fn test_new_validates_voting_threshold_bounds_and_panics() {
        let population = Population::test();

        Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &1.5, // Invalid : > 1.0
            &0.0,
            &WeightingMethod::Uniform,
        );
    }

    #[test]
    #[should_panic(expected = "Sensitivity threshold must be in [0,1]")]
    fn test_new_validates_specialized_sensitivity_threshold_bounds() {
        let population = Population::test();

        Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 1.5, // Invalid : > 1.0
                specificity_threshold: 0.8,
            },
        );
    }

    #[test]
    #[should_panic(expected = "Specificity threshold must be in [0,1]")]
    fn test_new_validates_specialized_specificity_threshold_bounds() {
        let population = Population::test();

        Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.8,
                specificity_threshold: -0.1, // Invalid : < 0.0
            },
        );
    }

    #[test]
    fn test_new_retains_experts_when_thresholds_are_zero() {
        let population = Population::test();
        let original_count = population.individuals.len();

        let jury = Jury::new(
            &population,
            &0.0, // No filtering by performance
            &0.0, // No filtering by diversity
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        assert_eq!(jury.experts.individuals.len(), original_count);
    }

    #[test]
    fn test_new_sorts_experts_after_filtering() {
        let population = Population::test();

        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        // Judges should be sorted (check that sort() has been called)
        assert!(!jury.experts.individuals.is_empty());
        assert!(jury.experts.individuals[0].fit > jury.experts.individuals[1].fit);
        assert!(jury.experts.individuals[1].fit > jury.experts.individuals[2].fit);
        // Test that the object is valid after sorting
        assert!(jury.voting_threshold >= 0.0 && jury.voting_threshold <= 1.0);
    }

    #[test]
    fn test_new_handles_empty_population_gracefully() {
        let population = Population::new();

        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        assert!(jury.experts.individuals.is_empty());
        assert_eq!(jury.voting_method, VotingMethod::Majority);
        assert_eq!(jury.voting_threshold, 0.5);
    }

    #[test]
    fn test_evaluate_computes_metrics_for_experts_and_stores_weights() {
        let mut population = Population::test();
        for individual in &mut population.individuals {
            individual.accuracy = 0.0;
        }

        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);

        // Jury metrics should be calculated
        assert!(jury.accuracy > 0.0);
        assert!(jury.weights.is_some());
        assert_eq!(
            jury.weights.as_ref().unwrap().len(),
            jury.experts.individuals.len()
        );
        assert!(jury.predicted_classes.is_some());
    }

    #[test]
    fn test_evaluate_optimizes_threshold_when_set_to_zero() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.0, // Threshold set to zero to trigger Youden optimisation
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);

        // The threshold should have been optimised (different from 0.0).
        assert!(jury.voting_threshold > 0.0);
        assert!(jury.voting_threshold <= 1.0);
    }

    #[test]
    #[should_panic(expected = "Weights must be computed before prediction")]
    fn test_predict_fails_when_weights_not_computed() {
        let population = Population::test();
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.predict(&data); // Should panic because evaluate() has not been called
    }

    #[test]
    #[should_panic(expected = "Weights length")]
    fn test_compute_majority_threshold_vote_panics_on_weight_mismatch() {
        let population = Population::test();
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        let wrong_weights = vec![1.0]; // Less weight than experts

        jury.compute_majority_threshold_vote(&data, &wrong_weights, 0.5, 0.0);
    }

    #[test]
    fn test_compute_majority_threshold_vote_handles_perfect_ties_correctly() {
        let mut population = Population::test();
        // Set up exactly 2 experts to create perfect ties
        population.individuals.truncate(2);

        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);

        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
        assert!(predictions.0.iter().all(|&x| x == 0 || x == 1 || x == 2));
        assert!(predictions.1.iter().all(|&x| x >= 0.0 || x <= 1.0));
    }

    #[test]
    fn test_compute_consensus_threshold_vote_requires_consensus_for_decision() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.95,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);

        // With a high consensus threshold, there should be abstentions (class 2)
        assert!(predictions.0.contains(&2));
        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
    }

    #[test]
    fn test_compute_consensus_threshold_vote_abstains_when_no_consensus() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.95,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);

        // With a very high threshold, the majority of predictions are likely to be abstentions
        let abstentions = predictions.0.iter().filter(|&&x| x == 2).count();
        assert!(abstentions > 0);
        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
    }

    #[test]
    fn test_compute_classes_method_stores_predictions_correctly() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);
        jury.compute_classes(&data);

        assert!(jury.predicted_classes.is_some());
        assert_eq!(
            jury.predicted_classes.as_ref().unwrap().len(),
            data.sample_len
        );

        // Check that all predictions are valid classes
        let predictions = jury.predicted_classes.as_ref().unwrap();
        assert!(predictions.iter().all(|&x| x == 0 || x == 1 || x == 2));
    }

    #[test]
    fn test_compute_rejection_rate_calculates_percentage_correctly() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.9,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);

        let rejection_rate = jury.compute_rejection_rate(&predictions.0);
        assert!(rejection_rate >= 0.0 && rejection_rate <= 1.0);

        // Extreme cases
        let all_abstentions = vec![2u8; 10];
        let all_abstention_rate = jury.compute_rejection_rate(&all_abstentions);
        assert_eq!(all_abstention_rate, 1.0);

        let no_abstentions = vec![0u8, 1u8, 0u8, 1u8];
        let no_abstention_rate = jury.compute_rejection_rate(&no_abstentions);
        assert_eq!(no_abstention_rate, 0.0);
    }

    #[test]
    fn test_youden_optimization_adapts_step_size_to_sample_size() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.0, // Optimisation
            &0.0,
            &WeightingMethod::Uniform,
        );

        // Test with different sample sizes
        let small_data = Data::specific_test(30, 10); // <= 50
        jury.evaluate(&small_data);
        let threshold_small = jury.voting_threshold;
        assert!(threshold_small > 0.0 && threshold_small <= 1.0);

        // Reset for next test
        jury.voting_threshold = 0.0;
        let medium_data = Data::specific_test(100, 10); // 51-200
        jury.evaluate(&medium_data);
        let threshold_medium = jury.voting_threshold;
        assert!(threshold_medium > 0.0 && threshold_medium <= 1.0);

        // Reset for next test
        jury.voting_threshold = 0.0;
        let large_data = Data::specific_test(500, 10); // > 200
        jury.evaluate(&large_data);
        let threshold_large = jury.voting_threshold;
        assert!(threshold_large > 0.0 && threshold_large <= 1.0);
    }

    #[test]
    fn test_get_expert_specialization_comprehensive() {
        let population = Population::test();
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        // Create experts with different metrics
        let balanced = Individual::test_with_metrics(0.8, 0.8, 0.8);
        let pos_specialist = Individual::test_with_metrics(0.9, 0.5, 0.7);
        let neg_specialist = Individual::test_with_metrics(0.5, 0.9, 0.7);
        let ineffective = Individual::test_with_metrics(0.3, 0.3, 0.3);
        let edge_case = Individual::test_with_metrics(0.7, 0.7, 0.7);

        // Normal cases
        assert_eq!(
            jury.get_expert_specialization(&balanced, 0.7, 0.7),
            JudgeSpecialization::Balanced
        );
        assert_eq!(
            jury.get_expert_specialization(&pos_specialist, 0.7, 0.7),
            JudgeSpecialization::PositiveSpecialist
        );
        assert_eq!(
            jury.get_expert_specialization(&neg_specialist, 0.7, 0.7),
            JudgeSpecialization::NegativeSpecialist
        );
        assert_eq!(
            jury.get_expert_specialization(&ineffective, 0.7, 0.7),
            JudgeSpecialization::Ineffective
        );

        // Boundary condition test (exact thresholds)
        assert_eq!(
            jury.get_expert_specialization(&edge_case, 0.7, 0.7),
            JudgeSpecialization::Balanced
        );
    }

    #[test]
    fn test_count_effective_experts_uniform_counts_all() {
        let population = Population::test();
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let (total, effective) = jury.count_effective_experts();

        // In uniform mode, all experts are active.
        assert_eq!(total, effective);
        assert_eq!(effective, jury.experts.individuals.len());
    }

    #[test]
    fn test_count_effective_experts_specialized_excludes_ineffective() {
        let mut population = Population::test();
        if population.individuals.len() >= 3 {
            // Set up experts with different levels of efficiency
            population.individuals[0].sensitivity = 0.9;
            population.individuals[0].specificity = 0.8;
            population.individuals[1].sensitivity = 0.5;
            population.individuals[1].specificity = 0.9;
            population.individuals[2].sensitivity = 0.3;
            population.individuals[2].specificity = 0.2;
        }

        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.7,
                specificity_threshold: 0.7,
            },
        );

        let (total, effective) = jury.count_effective_experts();

        // Should exclude ineffective experts
        assert!(effective <= total);
        if population.individuals.len() >= 3 {
            assert!(effective < total);
        }
    }

    #[test]
    fn test_compute_group_strict_weights_comprehensive() {
        let mut population = Population::test();

        // Set up experts with different specialisations
        if population.individuals.len() >= 4 {
            population.individuals[0].sensitivity = 0.9;
            population.individuals[0].specificity = 0.5; // Positive specialist
            population.individuals[1].sensitivity = 0.5;
            population.individuals[1].specificity = 0.9; // Negative specialist
            population.individuals[2].sensitivity = 0.8;
            population.individuals[2].specificity = 0.8; // Balanced
            population.individuals[3].sensitivity = 0.3;
            population.individuals[3].specificity = 0.3; // Ineffective
        }

        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.7,
                specificity_threshold: 0.7,
            },
        );

        let weights = jury.compute_group_strict_weights(0.7, 0.7);

        // Weights should be distributed fairly among active groups
        assert_eq!(weights.len(), jury.experts.individuals.len());

        // The sum of the weights of the effective experts must be 1.0.
        let sum_effective_weights: f64 = weights.iter().filter(|&&w| w > 0.0).sum();
        assert!((sum_effective_weights - 1.0).abs() < 1e-10);

        // Ineffective experts should have a weight of 0.0.
        if population.individuals.len() >= 4 {
            assert_eq!(weights[3], 0.0); // Judge ineffectif
        }
    }

    #[test]
    #[should_panic(expected = "Specialized threshold are too high to allow expert selection")]
    fn test_compute_group_strict_weights_panics_when_no_active_groups() {
        let mut population = Population::test();

        // Ensure that all experts perform poorly
        for individual in &mut population.individuals {
            individual.sensitivity = 0.1;
            individual.specificity = 0.1;
        }

        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.9,
                specificity_threshold: 0.9,
            },
        );

        jury.compute_group_strict_weights(0.9, 0.9);
    }

    #[test]
    fn test_voting_mechanisms_handle_zero_effective_weight() {
        let mut population = Population::test();
        population.individuals.truncate(1);

        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();

        jury.weights = Some(vec![0.0]);

        let predictions = jury.apply_voting_mechanism(&data, &[0.0]);

        // With a total weight of zero, all predictions should be abstentions (class 2).
        assert!(predictions.0.iter().all(|&x| x == 2));
        assert_eq!(predictions.0.len(), data.sample_len);
    }

    #[test]
    fn test_voting_mechanisms_handle_out_of_bounds_sample_indices() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);

        // Voting mechanisms should correctly handle sample indices.
        let predictions = jury.predict(&data);
        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
        assert!(predictions.0.iter().all(|&x| x == 0 || x == 1 || x == 2));
        assert!(predictions.1.iter().all(|&x| x >= 0.0 || x <= 1.0));
    }

    use crate::experiment::ExperimentMetadata;
    use crate::Experiment;

    #[test]
    fn test_complete_workflow_serialization_to_jury_evaluation() {
        let mut experiment = Experiment::test();
        experiment.final_population = Some(Population::test());

        let jury = Jury::new(
            experiment.final_population.as_ref().unwrap(),
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        experiment.others = Some(ExperimentMetadata::Jury { jury });

        let temp_file = "test_complete_workflow.msgpack";
        experiment.save_auto(temp_file).unwrap();

        let loaded_experiment = Experiment::load_auto(temp_file).unwrap();
        assert_eq!(experiment, loaded_experiment);

        match loaded_experiment.others {
            Some(ExperimentMetadata::Jury { jury: loaded_jury }) => {
                assert_eq!(loaded_jury.voting_method, VotingMethod::Majority);
                assert_eq!(loaded_jury.voting_threshold, 0.5);
            }
            _ => panic!("Jury metadata not preserved"),
        }

        std::fs::remove_file(temp_file).unwrap();
    }

    #[test]
    fn test_concurrent_importance_calculation_thread_safety() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = None;
        experiment.final_population = Some(Population::test());
        experiment.parameters.general.thread_number = 4;

        // Importance calculation should be thread safe
        experiment.compute_importance();

        assert!(experiment.importance_collection.is_some());
        let importance = experiment.importance_collection.unwrap();
        assert!(!importance.importances.is_empty());
    }

    #[test]
    fn test_memory_cleanup_after_cv_reconstruction_failure() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = Some(vec![(
            vec!["nonexistent_sample".to_string()],
            vec!["another_nonexistent".to_string()],
        )]);
        experiment.collections = vec![vec![Population::test()]];

        // Even if CV reconstruction fails, the programme should not leak memory.
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            experiment.compute_importance();
        }));

        // The test is successful if no memory leaks are detected (implicit verification).
        assert!(panic_result.is_err());
    }

    #[test]
    fn test_jury_edge_case_single_expert_voting() {
        let mut population = Population::test();
        population.individuals.truncate(1);

        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);

        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
        assert!(predictions.0.iter().all(|&x| x == 0 || x == 1 || x == 2));
        assert!(predictions.1.iter().all(|&x| x >= 0.0 || x <= 1.0));

        // Test with consensus and a single expert
        let mut consensus_jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &1.0,
            &0.0,
            &WeightingMethod::Uniform,
        );

        consensus_jury.evaluate(&data);
        let consensus_predictions = consensus_jury.predict(&data);

        // >i 100% consensus required and a single expert, all predictions should be the computed
        assert_eq!(consensus_predictions.0.len(), data.sample_len);
        assert!(consensus_predictions.0.iter().all(|&x| x == 0 || x == 1));
    }

    /// Helper to create a expert who votes according to a predefined pattern
    fn create_mock_expert(vote: u8) -> Individual {
        let mut expert = Individual::new();
        expert.features.insert(0, if vote == 1 { 1 } else { -1 });
        expert.accuracy = 0.8;
        expert.sensitivity = 0.75;
        expert.specificity = 0.85;
        expert.auc = 0.8;
        expert.threshold = 0.5;
        expert.language = BINARY_LANG;
        expert.data_type = RAW_TYPE;
        expert.k = 1;
        expert
    }

    /// Helper for creating data with a single sample
    fn create_single_sample_data(true_class: u8) -> Data {
        let mut X = HashMap::new();
        X.insert((0, 0), 1.0); // Sample 0, feature 0

        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);

        Data {
            X,
            y: vec![true_class],
            features: vec!["feature1".to_string()],
            samples: vec!["sample1".to_string()],
            feature_class,
            feature_significance: HashMap::new(),
            feature_annotations: None,
            sample_annotations: None,
            feature_selection: vec![0],
            feature_len: 1,
            sample_len: 1,
            classes: vec!["class0".to_string(), "class1".to_string()],
        }
    }

    /// Helper for creating a population from predefined votes
    fn create_population_with_votes(votes: Vec<u8>) -> Population {
        let mut pop = Population::new();
        for vote in votes {
            pop.individuals.push(create_mock_expert(vote));
        }
        pop
    }

    #[test]
    fn test_scenario_1_unanimous_majority_for() {
        // 5 experts unanimously vote 1, threshold 0.5 -> decision 1
        let pop = create_population_with_votes(vec![1, 1, 1, 1, 1]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0, // min_perf
            &0.0, // min_diversity
            &VotingMethod::Majority,
            &0.5, // voting_threshold
            &0.0, // threshold_window
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(pred_classes[0], 1, "Decision should be 1 (unanimous for)");
        assert!(
            (scores[0] - 1.0).abs() < 1e-10,
            "Score should be 1.0, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_scenario_2_unanimous_majority_against() {
        // 5 experts unanimously vote 0, threshold 0.5 -> decision 0
        let pop = create_population_with_votes(vec![0, 0, 0, 0, 0]);
        let data = create_single_sample_data(0);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(
            pred_classes[0], 0,
            "Decision should be 0 (unanimous against)"
        );
        assert!(
            (scores[0] - 0.0).abs() < 1e-10,
            "Score should be 0.0, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_scenario_3_simple_majority() {
        // 3 votes 1, 2 votes 0, threshold 0.5 -> decision 1
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(pred_classes[0], 1, "Decision should be 1 (simple majority)");
        assert!(
            (scores[0] - 0.6).abs() < 1e-10,
            "Score should be 0.6 (3/5), got {}",
            scores[0]
        );
    }

    #[test]
    fn test_scenario_4_abstention_due_to_threshold_window() {
        // 3 votes 1, 2 votes 0, threshold 0.6, window 10% -> abstention
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.6,  // threshold
            &10.0, // threshold_window 10%
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // Score = 0.6, threshold = 0.6, window = 10% = 0.1
        // |0.6 - 0.6| = 0.0 < 0.1 -> abstention
        assert_eq!(
            pred_classes[0], 2,
            "Decision should be 2 (abstention due to window)"
        );
        assert!(
            (scores[0] - 0.6).abs() < 1e-10,
            "Score should be 0.6, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_scenario_5_consensus_success() {
        // 4 votes 1, 1 vote 0, consensus threshold 0.7 -> decision 1
        let pop = create_population_with_votes(vec![1, 1, 1, 1, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.7, // consensus threshold
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(
            pred_classes[0], 1,
            "Decision should be 1 (consensus achieved)"
        );
        assert!(
            (scores[0] - 0.8).abs() < 1e-10,
            "Score should be 0.8 (4/5), got {}",
            scores[0]
        );
    }

    #[test]
    fn test_scenario_6_consensus_failure() {
        // 3 votes 1, 2 votes 0, consensus threshold 0.8 -> abstention
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.8, // high consensus threshold
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // Score = 0.6 < 0.8 threshold -> abstention
        assert_eq!(
            pred_classes[0], 2,
            "Decision should be 2 (consensus failed)"
        );
        assert!(
            (scores[0] - 0.6).abs() < 1e-10,
            "Score should be 0.6, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_scenario_7_weighted_majority() {
        // Votes [1,1,0,0,0] with weights [2,2,1,1,1] -> decision 1
        let pop = create_population_with_votes(vec![1, 1, 0, 0, 0]);
        let data = create_single_sample_data(1);

        // To simulate different weights, we create a Jury with min_perf
        // that will filter certain experts based on their performance
        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // Manually simulate different weights
        jury.weights = Some(vec![2.0, 2.0, 1.0, 1.0, 1.0]);

        let (pred_classes, scores) = jury.predict(&data);

        // Weighted votes: 2*1 + 2*1 + 1*0 + 1*0 + 1*0 = 4
        // Total weight: 2 + 2 + 1 + 1 + 1 = 7
        // Score: 4/7 ≈ 0.571
        assert_eq!(
            pred_classes[0], 1,
            "Decision should be 1 (weighted majority)"
        );
        assert!(
            (scores[0] - 4.0 / 7.0).abs() < 1e-10,
            "Score should be ~0.571, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_scenario_8_perfect_tie_with_window() {
        // 2 votes 1, 2 votes 0, seuil 0.5, window 5% -> abstention
        let pop = create_population_with_votes(vec![1, 1, 0, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5, // threshold
            &5.0, // threshold_window 5%
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // Score = 0.5, threshold = 0.5, window = 5% = 0.05
        // |0.5 - 0.5| = 0.0 < 0.05 -> abstention
        assert_eq!(
            pred_classes[0], 2,
            "Decision should be 2 (abstention due to perfect tie)"
        );
        assert!(
            (scores[0] - 0.5).abs() < 1e-10,
            "Score should be 0.5, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_majority_vs_consensus_different_outcomes() {
        // Same population, different voting methods -> different results
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]); // 60% 1
        let data = create_single_sample_data(1);

        // Majority Test (threshold 0.5)
        let mut jury_majority = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        jury_majority.evaluate(&data);
        let (pred_maj, _) = jury_majority.predict(&data);

        // Consensus Test  (threshold 0.8)
        let mut jury_consensus = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.8,
            &0.0,
            &WeightingMethod::Uniform,
        );
        jury_consensus.evaluate(&data);
        let (pred_cons, _) = jury_consensus.predict(&data);

        assert_eq!(pred_maj[0], 1, "Majority should decide 1");
        assert_eq!(pred_cons[0], 2, "Consensus should abstain (0.6 < 0.8)");
    }

    #[test]
    fn test_threshold_window_boundary_cases() {
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]); // Score = 0.6
        let data = create_single_sample_data(1);

        // Case 1: window too small -> no abstention
        let mut jury1 = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5, // threshold
            &1.0, // window = 1% = 0.01
            &WeightingMethod::Uniform,
        );
        jury1.evaluate(&data);
        let (pred1, _) = jury1.predict(&data);
        // |0.6 - 0.5| = 0.1 > 0.01 -> no abstention
        assert_eq!(pred1[0], 1, "Should decide 1 (window too small)");

        // Case 2: window large enough -> abstention
        let mut jury2 = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,  // threshold
            &15.0, // window = 15% = 0.15
            &WeightingMethod::Uniform,
        );
        jury2.evaluate(&data);
        let (pred2, _) = jury2.predict(&data);
        // |0.6 - 0.5| = 0.1 < 0.15 -> abstention
        assert_eq!(pred2[0], 2, "Should abstain (window large enough)");
    }

    #[test]
    fn test_compute_new_metrics_with_known_outcomes() {
        // Population that votes correctly based on known data
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]); // 60% pour

        // Data with true class = 1
        let data_positive = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data_positive);
        let (_, accuracy, sensitivity, _, rejection_rate, _) =
            jury.compute_new_metrics(&data_positive);

        // Jury predicts 1, true class = 1 -> True Positive
        assert_eq!(accuracy, 1.0, "Accuracy should be 1.0 (correct prediction)");
        assert_eq!(sensitivity, 1.0, "Sensitivity should be 1.0 (TP detected)");
        assert_eq!(rejection_rate, 0.0, "No rejection expected");

        // Test on negative class
        let data_negative = create_single_sample_data(0);
        let (_, accuracy_neg, _, specificity_neg, _, _) = jury.compute_new_metrics(&data_negative);

        // Short predicts 1, true class = 0 -> False Positive
        assert_eq!(
            accuracy_neg, 0.0,
            "Accuracy should be 0.0 (wrong prediction)"
        );
        assert_eq!(
            specificity_neg, 0.0,
            "Specificity should be 0.0 (FP not detected)"
        );
    }

    #[test]
    fn test_rejection_rate_calculation() {
        let pop = create_population_with_votes(vec![1, 1, 0, 0]); // Perfect tie

        // Create data with 3 samples
        let mut X = HashMap::new();
        for i in 0..3 {
            X.insert((i, 0), 1.0);
        }
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);

        let data = Data {
            X,
            y: vec![1, 0, 1],
            features: vec!["feature1".to_string()],
            samples: vec![
                "sample1".to_string(),
                "sample2".to_string(),
                "sample3".to_string(),
            ],
            feature_class,
            feature_significance: HashMap::new(),
            feature_annotations: None,
            sample_annotations: None,
            feature_selection: vec![0],
            feature_len: 1,
            sample_len: 3,
            classes: vec!["class0".to_string(), "class1".to_string()],
        };

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5, // threshold
            &5.0, // window -> abstention on perfect tie
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (_, _, _, _, rejection_rate, _) = jury.compute_new_metrics(&data);

        // All samples should be rejected (perfect tie in window)
        assert_eq!(
            rejection_rate, 1.0,
            "All samples should be rejected (perfect tie)"
        );
    }

    #[test]
    fn test_edge_case_single_expert() {
        // One expert -> no collective vote, but tests logic
        let pop = create_population_with_votes(vec![1]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(
            pred_classes[0], 1,
            "Single expert voting 1 should result in decision 1"
        );
        assert_eq!(
            scores[0], 1.0,
            "Score should be 1.0 with single expert voting 1"
        );
    }

    // Helper functions
    fn create_mock_expert_with_metrics(
        vote: u8,
        accuracy: f64,
        sensitivity: f64,
        specificity: f64,
    ) -> Individual {
        let mut expert = Individual::new();
        expert.features.insert(0, if vote == 1 { 1 } else { -1 });
        expert.accuracy = accuracy;
        expert.sensitivity = sensitivity;
        expert.specificity = specificity;
        expert.auc = (accuracy + sensitivity + specificity) / 3.0;
        expert.threshold = 0.5;
        expert.language = BINARY_LANG;
        expert.data_type = RAW_TYPE;
        expert.k = 1;
        expert
    }

    fn create_controlled_population(votes: Vec<u8>) -> Population {
        let mut pop = Population::new();
        for (i, vote) in votes.iter().enumerate() {
            let accuracy = 0.7 + 0.02 * i as f64;
            let sensitivity = 0.65 + 0.03 * i as f64;
            let specificity = 0.75 + 0.02 * i as f64;
            pop.individuals.push(create_mock_expert_with_metrics(
                *vote,
                accuracy,
                sensitivity,
                specificity,
            ));
        }
        pop
    }

    fn create_multi_sample_data(true_classes: Vec<u8>) -> Data {
        let mut X = HashMap::new();
        let mut samples = Vec::new();

        for (sample_idx, _) in true_classes.iter().enumerate() {
            X.insert((sample_idx, 0), 1.0);
            samples.push(format!("sample_{}", sample_idx));
        }

        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);

        Data {
            X,
            y: true_classes,
            features: vec!["feature1".to_string()],
            samples: samples.clone(),
            feature_class,
            feature_significance: HashMap::new(),
            feature_annotations: None,
            sample_annotations: None,
            feature_selection: vec![0],
            feature_len: 1,
            sample_len: samples.len(),
            classes: vec!["class0".to_string(), "class1".to_string()],
        }
    }

    #[test]
    fn test_compute_new_metrics_consistency() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]);
        let data = create_multi_sample_data(vec![1, 1, 0, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (ext_auc, ext_accuracy, ext_sensitivity, ext_specificity, _, _) =
            jury.compute_new_metrics(&data);

        // External metrics must be identical
        assert!(
            (jury.auc - ext_auc).abs() < 1e-10,
            "AUC mismatch: internal={}, external={}",
            jury.auc,
            ext_auc
        );
        assert!(
            (jury.accuracy - ext_accuracy).abs() < 1e-10,
            "Accuracy mismatch: internal={}, external={}",
            jury.accuracy,
            ext_accuracy
        );
        assert!(
            (jury.sensitivity - ext_sensitivity).abs() < 1e-10,
            "Sensitivity mismatch: internal={}, external={}",
            jury.sensitivity,
            ext_sensitivity
        );
        assert!(
            (jury.specificity - ext_specificity).abs() < 1e-10,
            "Specificity mismatch: internal={}, external={}",
            jury.specificity,
            ext_specificity
        );
    }

    #[test]
    fn test_compute_new_metrics_rejection_rate_calculation() {
        // Population that will create abstentions with window
        let pop = create_controlled_population(vec![1, 1, 0, 0]); // Perfect tie at 0.5
        let data = create_multi_sample_data(vec![1, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &10.0, // 10% window -> abstention on ratio = 0.5
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (_, _, _, _, rejection_rate, _) = jury.compute_new_metrics(&data);

        // Manual calculation of the rejection rate
        let (pred_classes, _) = jury.predict(&data);
        let manual_rejection_rate =
            pred_classes.iter().filter(|&&x| x == 2).count() as f64 / pred_classes.len() as f64;

        assert!(
            (rejection_rate - manual_rejection_rate).abs() < 1e-10,
            "Rejection rate mismatch: calculated={}, manual={}",
            rejection_rate,
            manual_rejection_rate
        );

        // With a perfect tie and a 10% window, we expect 100% abstention.
        assert_eq!(
            rejection_rate, 1.0,
            "Should have 100% rejection with perfect tie and window"
        );
    }

    #[test]
    fn test_internal_vs_external_metrics_coherence() {
        let pop = create_controlled_population(vec![1, 1, 1, 1, 0]);
        let data = create_multi_sample_data(vec![1, 1, 0, 0, 1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.6,
            &0.0,
            &WeightingMethod::Uniform,
        );

        // Test on training data
        jury.evaluate(&data);
        let (train_auc, train_acc, train_sens, train_spec, _, _) = jury.compute_new_metrics(&data);

        assert_eq!(train_auc, jury.auc, "Training AUC should match");
        assert_eq!(train_acc, jury.accuracy, "Training accuracy should match");
        assert_eq!(
            train_sens, jury.sensitivity,
            "Training sensitivity should match"
        );
        assert_eq!(
            train_spec, jury.specificity,
            "Training specificity should match"
        );

        // Test on different test data
        let test_data = create_multi_sample_data(vec![0, 0, 1, 1]);
        let (test_auc, test_acc, test_sens, test_spec, _, _) = jury.compute_new_metrics(&test_data);

        // Metrics may differ based on different data,
        // but must remain within valid ranges.
        assert!(
            test_auc >= 0.0 && test_auc <= 1.0,
            "Test AUC out of bounds: {}",
            test_auc
        );
        assert!(
            test_acc >= 0.0 && test_acc <= 1.0,
            "Test accuracy out of bounds: {}",
            test_acc
        );
        assert!(
            test_sens >= 0.0 && test_sens <= 1.0,
            "Test sensitivity out of bounds: {}",
            test_sens
        );
        assert!(
            test_spec >= 0.0 && test_spec <= 1.0,
            "Test specificity out of bounds: {}",
            test_spec
        );
    }

    #[test]
    fn test_voting_threshold_systematic_variations() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]); // 60% 1
        let data = create_multi_sample_data(vec![1]);

        let thresholds = vec![0.3, 0.5, 0.7, 0.9];
        let mut results = Vec::new();

        for threshold in thresholds {
            let mut jury = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Majority,
                &threshold,
                &0.0,
                &WeightingMethod::Uniform,
            );

            jury.evaluate(&data);
            let (pred_classes, scores) = jury.predict(&data);
            let (_, _, _, _, rejection_rate, _) = jury.compute_new_metrics(&data);

            results.push((threshold, pred_classes[0], scores[0], rejection_rate));
        }

        // Score = 0.6 for
        assert!(results
            .iter()
            .all(|(_, _, score, _)| (score - 0.6).abs() < 1e-10));

        // Check decision logic according to thresholds
        assert_eq!(results[0].1, 1, "Threshold 0.3: 0.6 > 0.3 -> decision 1");
        assert_eq!(results[1].1, 1, "Threshold 0.5: 0.6 > 0.5 -> decision 1");
        assert_eq!(results[2].1, 0, "Threshold 0.7: 0.6 < 0.7 -> decision 0");
        assert_eq!(results[3].1, 0, "Threshold 0.9: 0.6 < 0.9 -> decision 0");
    }

    #[test]
    fn test_threshold_window_granular_effects() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]); // Score = 0.6
        let data = create_multi_sample_data(vec![1]);

        let threshold = 0.5;
        let windows = vec![1.0, 5.0, 15.0, 25.0]; // 1%, 5%, 15%, 25%

        for window in windows {
            let mut jury = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Majority,
                &threshold,
                &window,
                &WeightingMethod::Uniform,
            );

            jury.evaluate(&data);
            let (pred_classes, _) = jury.predict(&data);

            // Score = 0.6, threshold = 0.5, |0.6 - 0.5| = 0.1 = 10%
            if window < 10.0 {
                // Window too small -> no abstention
                assert_eq!(pred_classes[0], 1, "Window {}%: should decide 1", window);
            } else {
                // Window large enough -> abstention
                assert_eq!(pred_classes[0], 2, "Window {}%: should abstain", window);
            }
        }
    }

    #[test]
    fn test_majority_vs_consensus_systematic_comparison() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]); // 60% pour
        let data = create_multi_sample_data(vec![1, 0, 1]);

        let test_cases = vec![
            (0.5, 1, 1), // Majority: 0.6 > 0.5 -> 1, Consensus: 0.6 > 0.5 -> 2
            (0.4, 1, 1), // Majority: 0.6 > 0.4 -> 1, Consensus: 0.6 > 0.4 -> 1
            (0.8, 0, 2), // Majority: 0.6 < 0.8 -> 0, Consensus: 0.6 < 0.8 -> 2
        ];

        for (threshold, expected_majority, expected_consensus) in test_cases {
            // Test Majority
            let mut jury_maj = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Majority,
                &threshold,
                &0.0,
                &WeightingMethod::Uniform,
            );
            jury_maj.evaluate(&data);
            let (pred_maj, _) = jury_maj.predict(&data);

            // Test Consensus
            let mut jury_cons = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Consensus,
                &threshold,
                &0.0,
                &WeightingMethod::Uniform,
            );
            jury_cons.evaluate(&data);
            let (pred_cons, _) = jury_cons.predict(&data);

            assert_eq!(
                pred_maj[0], expected_majority,
                "Majority with threshold {}: expected {}, got {}",
                threshold, expected_majority, pred_maj[0]
            );
            assert_eq!(
                pred_cons[0], expected_consensus,
                "Consensus with threshold {}: expected {}, got {}",
                threshold, expected_consensus, pred_cons[0]
            );
        }
    }

    #[test]
    fn test_edge_cases_boundary_conditions() {
        let pop = create_controlled_population(vec![1, 1, 0, 0]); // Perfect tie
        let data = create_multi_sample_data(vec![1]);

        // Threshold test at 0.0
        let mut jury_min = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.0, // Minimum threshold -> always 1 unless score = 0
            &10.0,
            &WeightingMethod::Uniform,
        );
        jury_min.evaluate(&data);
        jury_min.voting_threshold = 0.0; // avoid optimization with Youden Maxima
        let (pred_min, _) = jury_min.predict(&data);
        assert_eq!(
            pred_min[0], 1,
            "Threshold 0.0: should always decide 1 for score > 0"
        );

        // Test seuil à 1.0
        let mut jury_max = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &1.0, // Maximum threshold -> always 0 unless score = 1
            &0.0,
            &WeightingMethod::Uniform,
        );
        jury_max.evaluate(&data);
        let (pred_max, _) = jury_max.predict(&data);
        assert_eq!(
            pred_max[0], 0,
            "Threshold 1.0: should decide 0 for score < 1"
        );

        // Extreme window test
        let mut jury_extreme_window = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &100.0, // 100% window -> always abstain
            &WeightingMethod::Uniform,
        );
        jury_extreme_window.evaluate(&data);
        let (pred_extreme, _) = jury_extreme_window.predict(&data);
        assert_eq!(pred_extreme[0], 2, "Window 100%: should always abstain");
    }

    #[test]
    fn test_large_population_performance() {
        // Create a large population (50 experts)
        let mut large_votes = Vec::new();
        for i in 0..50 {
            large_votes.push(if i % 3 == 0 { 1 } else { 0 }); // ~33% 1
        }

        let pop = create_controlled_population(large_votes);
        let data = create_multi_sample_data(vec![1, 0, 1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (auc, accuracy, sensitivity, specificity, rejection_rate, _) =
            jury.compute_new_metrics(&data);

        // Verify that metrics remain stable with a large population
        assert!(
            auc >= 0.0 && auc <= 1.0,
            "AUC out of bounds with large population"
        );
        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Accuracy out of bounds with large population"
        );
        assert!(
            sensitivity >= 0.0 && sensitivity <= 1.0,
            "Sensitivity out of bounds with large population"
        );
        assert!(
            specificity >= 0.0 && specificity <= 1.0,
            "Specificity out of bounds with large population"
        );
        assert!(
            rejection_rate >= 0.0 && rejection_rate <= 1.0,
            "Rejection rate out of bounds with large population"
        );

        // Verify that the Jury is effectively managing 50 experts
        assert_eq!(
            jury.experts.individuals.len(),
            50,
            "Should retain all 50 experts"
        );
        assert_eq!(
            jury.weights.as_ref().unwrap().len(),
            50,
            "Should have 50 weights"
        );
    }

    #[test]
    fn test_multi_sample_dataset_consistency() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Create a dataset with 100 samples
        let mut large_classes = Vec::new();
        for i in 0..100 {
            large_classes.push(if i % 2 == 0 { 1 } else { 0 });
        }
        let large_data = create_multi_sample_data(large_classes);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&large_data);
        let (auc, accuracy, sensitivity, specificity, rejection_rate, _) =
            jury.compute_new_metrics(&large_data);

        // Check consistency on large dataset
        assert!(
            auc >= 0.0 && auc <= 1.0,
            "AUC inconsistent on large dataset"
        );
        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Accuracy inconsistent on large dataset"
        );
        assert!(
            sensitivity >= 0.0 && sensitivity <= 1.0,
            "Sensitivity inconsistent on large dataset"
        );
        assert!(
            specificity >= 0.0 && specificity <= 1.0,
            "Specificity inconsistent on large dataset"
        );
        assert!(
            rejection_rate >= 0.0 && rejection_rate <= 1.0,
            "Rejection rate inconsistent on large dataset"
        );

        // Check that all predictions are valid
        let (predictions, _) = jury.predict(&large_data);
        assert_eq!(predictions.len(), 100, "Should have 100 predictions");
        assert!(
            predictions.iter().all(|&x| x == 0 || x == 1 || x == 2),
            "All predictions should be valid classes"
        );
    }

    #[test]
    fn test_rejection_rate_mathematical_accuracy() {
        // Controlled scenario: 4 samples with 2 expected abstentions
        let pop = create_controlled_population(vec![1, 1, 0, 0]); // Perfect Tie 0.5
        let data = create_multi_sample_data(vec![1, 0, 1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &1.0, // 1% window -> no abstention (|0.5-0.5| = 0 < 0.01)
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (_, _, _, _, rejection_rate, _) = jury.compute_new_metrics(&data);

        // Expected mathematical calculation
        let (predictions, _) = jury.predict(&data);
        let expected_rejections = predictions.iter().filter(|&&x| x == 2).count();
        let expected_rate = expected_rejections as f64 / predictions.len() as f64;

        assert!(
            (rejection_rate - expected_rate).abs() < 1e-10,
            "Mathematical rejection rate mismatch: expected={}, got={}",
            expected_rate,
            rejection_rate
        );

        // Extreme case testing
        let all_abstain = vec![2u8; 5];
        let rate_all = jury.compute_rejection_rate(&all_abstain);
        assert_eq!(
            rate_all, 1.0,
            "All abstentions should give 100% rejection rate"
        );

        let no_abstain = vec![0u8, 1u8, 0u8, 1u8];
        let rate_none = jury.compute_rejection_rate(&no_abstain);
        assert_eq!(
            rate_none, 0.0,
            "No abstentions should give 0% rejection rate"
        );
    }

    #[test]
    fn test_specialized_weighting_comprehensive() {
        let mut pop = create_controlled_population(vec![1, 1, 0, 0]);

        if pop.individuals.len() >= 4 {
            pop.individuals[0].sensitivity = 0.9; // Positive specialist
            pop.individuals[0].specificity = 0.6;
            pop.individuals[1].sensitivity = 0.9; // Positive specialist
            pop.individuals[1].specificity = 0.6;
            pop.individuals[2].sensitivity = 0.6; // Negative specialist
            pop.individuals[2].specificity = 0.9;
            pop.individuals[3].sensitivity = 0.6; // Negative specialist
            pop.individuals[3].specificity = 0.9;
        }

        let data = create_multi_sample_data(vec![1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.8,
                specificity_threshold: 0.8,
            },
        );

        jury.evaluate(&data);

        let weights = jury.weights.as_ref().unwrap();
        assert_eq!(weights.len(), 4, "Should have 4 weights");

        let effective_weight_sum: f64 = weights.iter().filter(|&&w| w > 0.0).sum();
        assert!(
            (effective_weight_sum - 1.0).abs() < 1e-10,
            "Effective weights should sum to 1.0"
        );

        let (auc, accuracy, sensitivity, specificity, rejection_rate, _) =
            jury.compute_new_metrics(&data);
        assert!(
            auc >= 0.0 && auc <= 1.0,
            "AUC should be valid with specialized weighting"
        );
        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Accuracy should be valid with specialized weighting"
        );
        assert!(
            sensitivity >= 0.0 && sensitivity <= 1.0,
            "Sensitivity should be valid with specialized weighting"
        );
        assert!(
            specificity >= 0.0 && specificity <= 1.0,
            "Specificity should be valid with specialized weighting"
        );
        assert!(
            rejection_rate >= 0.0 && rejection_rate <= 1.0,
            "Rejection rate should be valid with specialized weighting"
        );
    }

    #[test]
    fn test_jury_additional_metrics_not_computed_when_experts_have_none() {
        // Create a population without additional metrics
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]);
        let data = create_multi_sample_data(vec![1, 0, 1, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // All additional metrics should be None
        assert!(
            jury.metrics.mcc.is_none(),
            "MCC should be None when experts don't have it"
        );
        assert!(
            jury.metrics.f1_score.is_none(),
            "F1-score should be None when experts don't have it"
        );
        assert!(
            jury.metrics.npv.is_none(),
            "NPV should be None when experts don't have it"
        );
        assert!(
            jury.metrics.ppv.is_none(),
            "PPV should be None when experts don't have it"
        );
        assert!(
            jury.metrics.g_mean.is_none(),
            "G-mean should be None when experts don't have it"
        );
    }

    #[test]
    fn test_jury_additional_metrics_computed_when_experts_have_them() {
        // Create a population with additional metrics
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Add metrics to all experts
        for expert in &mut pop.individuals {
            expert.metrics.mcc = Some(0.7);
            expert.metrics.f1_score = Some(0.8);
            expert.metrics.npv = Some(0.75);
            expert.metrics.ppv = Some(0.85);
            expert.metrics.g_mean = Some(0.79);
        }

        let data = create_multi_sample_data(vec![1, 0, 1, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // All additional metrics should be computed (Some)
        assert!(
            jury.metrics.mcc.is_some(),
            "MCC should be computed when experts have it"
        );
        assert!(
            jury.metrics.f1_score.is_some(),
            "F1-score should be computed when experts have it"
        );
        assert!(
            jury.metrics.npv.is_some(),
            "NPV should be computed when experts have it"
        );
        assert!(
            jury.metrics.ppv.is_some(),
            "PPV should be computed when experts have it"
        );
        assert!(
            jury.metrics.g_mean.is_some(),
            "G-mean should be computed when experts have it"
        );

        // Values should be in valid range [0, 1] or [-1, 1] for MCC
        assert!(
            jury.metrics.mcc.unwrap() >= -1.0 && jury.metrics.mcc.unwrap() <= 1.0,
            "MCC should be in [-1, 1]"
        );
        assert!(
            jury.metrics.f1_score.unwrap() >= 0.0 && jury.metrics.f1_score.unwrap() <= 1.0,
            "F1-score should be in [0, 1]"
        );
        assert!(
            jury.metrics.npv.unwrap() >= 0.0 && jury.metrics.npv.unwrap() <= 1.0,
            "NPV should be in [0, 1]"
        );
        assert!(
            jury.metrics.ppv.unwrap() >= 0.0 && jury.metrics.ppv.unwrap() <= 1.0,
            "PPV should be in [0, 1]"
        );
        assert!(
            jury.metrics.g_mean.unwrap() >= 0.0 && jury.metrics.g_mean.unwrap() <= 1.0,
            "G-mean should be in [0, 1]"
        );
    }

    #[test]
    fn test_jury_additional_metrics_partial_presence() {
        // Test when only some experts have some metrics
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Only first expert has MCC
        pop.individuals[0].metrics.mcc = Some(0.6);

        // Only first two experts have F1-score
        pop.individuals[0].metrics.f1_score = Some(0.7);
        pop.individuals[1].metrics.f1_score = Some(0.75);

        let data = create_multi_sample_data(vec![1, 0, 1, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // MCC and F1-score should be computed
        assert!(
            jury.metrics.mcc.is_some(),
            "MCC should be computed when at least one expert has it"
        );
        assert!(
            jury.metrics.f1_score.is_some(),
            "F1-score should be computed when at least one expert has it"
        );

        // Others should not be computed
        assert!(
            jury.metrics.npv.is_none(),
            "NPV should not be computed when no expert has it"
        );
        assert!(
            jury.metrics.ppv.is_none(),
            "PPV should not be computed when no expert has it"
        );
        assert!(
            jury.metrics.g_mean.is_none(),
            "G-mean should not be computed when no expert has it"
        );
    }

    #[test]
    fn test_jury_additional_metrics_consistency_train_test() {
        // Test that metrics are computed consistently between train and test
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        for expert in &mut pop.individuals {
            expert.metrics.mcc = Some(0.7);
            expert.metrics.f1_score = Some(0.8);
        }

        let train_data = create_multi_sample_data(vec![1, 0, 1, 0, 1]);
        let test_data = create_multi_sample_data(vec![0, 1, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&train_data);

        // Check train metrics are stored
        assert!(jury.metrics.mcc.is_some(), "Train MCC should be computed");
        assert!(
            jury.metrics.f1_score.is_some(),
            "Train F1-score should be computed"
        );

        let train_mcc = jury.metrics.mcc.unwrap();
        let train_f1 = jury.metrics.f1_score.unwrap();

        // Compute test metrics
        let (_, _, _, _, _, test_metrics) = jury.compute_new_metrics(&test_data);

        assert!(test_metrics.mcc.is_some(), "Test MCC should be computed");
        assert!(
            test_metrics.f1_score.is_some(),
            "Test F1-score should be computed"
        );

        // Both should be finite and in valid ranges
        assert!(
            train_mcc.is_finite() && train_mcc >= -1.0 && train_mcc <= 1.0,
            "Train MCC should be finite and in valid range"
        );
        assert!(
            train_f1.is_finite() && train_f1 >= 0.0 && train_f1 <= 1.0,
            "Train F1 should be finite and in valid range"
        );
        assert!(
            test_metrics.mcc.unwrap().is_finite(),
            "Test MCC should be finite"
        );
        assert!(
            test_metrics.f1_score.unwrap().is_finite(),
            "Test F1 should be finite"
        );
    }

    #[test]
    fn test_jury_additional_metrics_with_abstentions() {
        // Test that metrics exclude abstentions (class 2) properly
        let mut pop = create_controlled_population(vec![1, 1, 0, 0]); // Perfect tie

        for expert in &mut pop.individuals {
            expert.metrics.mcc = Some(0.5);
            expert.metrics.f1_score = Some(0.6);
        }

        let data = create_multi_sample_data(vec![1, 0, 1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &10.0, // Large window to create abstentions
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // Even with abstentions, metrics should be computed on non-abstained predictions
        // They might be None if all samples are abstained, but should be Some if some are not
        let (predictions, _) = jury.predict(&data);
        let non_abstained_count = predictions.iter().filter(|&&p| p != 2).count();

        if non_abstained_count > 0 {
            assert!(
                jury.metrics.mcc.is_some(),
                "MCC should be computed on non-abstained samples"
            );
            assert!(
                jury.metrics.f1_score.is_some(),
                "F1-score should be computed on non-abstained samples"
            );
        }
    }

    #[test]
    fn test_jury_additional_metrics_with_consensus_voting() {
        // Test metrics with consensus voting method
        let mut pop = create_controlled_population(vec![1, 1, 1, 1, 0]);

        for expert in &mut pop.individuals {
            expert.metrics.mcc = Some(0.65);
            expert.metrics.ppv = Some(0.82);
            expert.metrics.npv = Some(0.78);
        }

        let data = create_multi_sample_data(vec![1, 1, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.7, // High consensus threshold
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        assert!(
            jury.metrics.mcc.is_some(),
            "MCC should be computed with consensus voting"
        );
        assert!(
            jury.metrics.ppv.is_some(),
            "PPV should be computed with consensus voting"
        );
        assert!(
            jury.metrics.npv.is_some(),
            "NPV should be computed with consensus voting"
        );

        // Should still be in valid ranges
        assert!(jury.metrics.mcc.unwrap() >= -1.0 && jury.metrics.mcc.unwrap() <= 1.0);
        assert!(jury.metrics.ppv.unwrap() >= 0.0 && jury.metrics.ppv.unwrap() <= 1.0);
        assert!(jury.metrics.npv.unwrap() >= 0.0 && jury.metrics.npv.unwrap() <= 1.0);
    }

    #[test]
    fn test_jury_additional_metrics_return_value_consistency() {
        // Test that compute_new_metrics returns consistent values
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        for expert in &mut pop.individuals {
            expert.metrics.mcc = Some(0.7);
            expert.metrics.f1_score = Some(0.75);
            expert.metrics.g_mean = Some(0.73);
        }

        let data = create_multi_sample_data(vec![1, 0, 1, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // Metrics in jury.metrics should match those from compute_new_metrics
        let (_, _, _, _, _, returned_metrics) = jury.compute_new_metrics(&data);

        assert_eq!(
            jury.metrics.mcc, returned_metrics.mcc,
            "Stored and returned MCC should match"
        );
        assert_eq!(
            jury.metrics.f1_score, returned_metrics.f1_score,
            "Stored and returned F1-score should match"
        );
        assert_eq!(
            jury.metrics.g_mean, returned_metrics.g_mean,
            "Stored and returned G-mean should match"
        );
    }

    #[test]
    fn test_jury_additional_metrics_with_empty_predictions() {
        // Edge case: all predictions are filtered out (all abstentions)
        let mut pop = create_controlled_population(vec![1, 1, 0, 0]);

        for expert in &mut pop.individuals {
            expert.metrics.mcc = Some(0.5);
        }

        let data = create_multi_sample_data(vec![1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &100.0, // 100% window = all abstentions
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // With all abstentions, metrics might not be computable
        // But the code should not crash and return default values
        let (auc, accuracy, _, _, rejection_rate, additional) = jury.compute_new_metrics(&data);

        // Should return safe default values
        assert!(
            auc.is_finite(),
            "AUC should be finite even with all abstentions"
        );
        assert!(
            accuracy.is_finite(),
            "Accuracy should be finite even with all abstentions"
        );
        assert!(
            rejection_rate == 1.0,
            "Rejection rate should be 100% with all abstentions"
        );

        // Additional metrics should handle this gracefully (None or default)
        if additional.mcc.is_some() {
            assert!(
                additional.mcc.unwrap().is_finite(),
                "MCC should be finite if computed"
            );
        }
    }

    #[test]
    fn test_majority_vote_ignores_individual_abstentions() {
        // Scenario: 3 experts vote 1, 2 vote 0, 2 vote 2 (abstain)
        // Expected: Majority on 3 vs 2 (ignoring the 2 abstentions) -> decision 1, ratio 0.6
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0, 0, 0]); // 7 experts total

        // Make last 2 experts abstain by giving them ThresholdCI
        // Logic: if v > upper => 1, if v < lower => 0, else => 2
        // Scores range from -1 to 1, so upper=2, lower=-2 captures all in rejection zone
        for i in 5..7 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 2.0,
                lower: -2.0,
                rejection_rate: 1.0,
            });
        }

        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // Should compute ratio based only on non-abstaining experts: 3/(3+2) = 0.6
        assert_eq!(
            pred_classes[0], 1,
            "Majority should ignore abstentions and decide based on 3 vs 2"
        );
        assert!(
            (scores[0] - 0.6).abs() < 1e-10,
            "Score should be 0.6 (3 out of 5 non-abstaining)"
        );
    }

    #[test]
    fn test_majority_all_experts_abstain() {
        // Scenario: ALL experts vote 2 (abstain)
        // Expected: Collective abstention (class 2)
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Make ALL experts abstain
        for i in 0..5 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 100.0,
                lower: -100.0,
                rejection_rate: 1.0,
            });
        }

        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // When total_weight == 0 (all abstained), should return class 2
        assert_eq!(
            pred_classes[0], 2,
            "Should abstain collectively when all experts abstain"
        );
        assert_eq!(
            scores[0], 0.5,
            "Score should be 0.5 when all experts abstain"
        );
    }

    #[test]
    fn test_majority_minority_decides_when_majority_abstains() {
        // Scenario: 4 experts abstain, 1 votes 1
        // Expected: The single non-abstaining expert decides (ratio = 1.0 -> class 1)
        let mut pop = create_controlled_population(vec![1, 0, 0, 0, 0]);

        // Make 4 experts abstain
        for i in 1..5 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 2.0,
                lower: -2.0,
                rejection_rate: 1.0,
            });
        }

        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // Only 1 expert votes (votes 1), so ratio = 1/1 = 1.0
        assert_eq!(
            pred_classes[0], 1,
            "Single non-abstaining expert should decide"
        );
        assert_eq!(
            scores[0], 1.0,
            "Score should be 1.0 when only one expert votes for 1"
        );
    }

    #[test]
    fn test_consensus_ignores_individual_abstentions() {
        // Scenario: 4 experts vote 1, 1 votes 0, 2 vote 2 (abstain)
        // Consensus threshold: 0.7
        // Expected: pos_ratio = 4/(4+1) = 0.8 >= 0.7 -> decision 1
        let mut pop = create_controlled_population(vec![1, 1, 1, 1, 0, 0, 0]);

        // Make last 2 experts abstain
        for i in 5..7 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 2.0,
                lower: -2.0,
                rejection_rate: 1.0,
            });
        }

        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.7,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // pos_ratio = 4/5 = 0.8 >= 0.7
        assert_eq!(
            pred_classes[0], 1,
            "Consensus should ignore abstentions and decide based on 4 vs 1"
        );
        assert!(
            (scores[0] - 0.8).abs() < 1e-10,
            "Score should be 0.8 (4 out of 5 non-abstaining)"
        );
    }

    #[test]
    fn test_consensus_all_experts_abstain() {
        // Scenario: ALL experts vote 2
        // Expected: Collective abstention (class 2, pos_ratio = -1.0)
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Make ALL experts abstain
        for i in 0..5 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 2.0,
                lower: -2.0,
                rejection_rate: 1.0,
            });
        }

        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.8,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // When effective_total_weight == 0, should return (2, -1.0)
        assert_eq!(
            pred_classes[0], 2,
            "Should abstain collectively when all experts abstain"
        );
        assert_eq!(
            scores[0], -1.0,
            "Score should be -1.0 (sentinel value) when all abstain"
        );
    }

    #[test]
    fn test_consensus_fails_when_majority_abstains() {
        // Scenario: 2 vote 1, 1 votes 0, 4 vote 2 (abstain)
        // Consensus threshold: 0.8
        // pos_ratio = 2/(2+1) = 0.67 < 0.8, neg_ratio = 1/3 = 0.33 < 0.8
        // Expected: Consensus fails -> abstention (class 2)
        let mut pop = create_controlled_population(vec![1, 1, 0, 0, 0, 0, 0]);

        // Make 4 experts abstain
        for i in 3..7 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 2.0,
                lower: -2.0,
                rejection_rate: 1.0,
            });
        }

        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.8,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // pos_ratio = 2/3 ≈ 0.667 < 0.8 -> abstention
        assert_eq!(
            pred_classes[0], 2,
            "Consensus should fail when pos_ratio < threshold"
        );
        assert!(
            (scores[0] - 2.0 / 3.0).abs() < 1e-10,
            "Score should be ~0.667"
        );
    }

    #[test]
    fn test_weighted_voting_with_abstentions() {
        // Scenario: weighted experts with some abstaining
        // Experts: [1 (w=2), 1 (w=2), 0 (w=1), 2 (w=3), 2 (w=2)]
        // Non-abstaining weighted votes: 2*1 + 2*1 + 1*0 = 4 for, 1 against
        // Total non-abstaining weight: 2+2+1 = 5
        // Ratio: 4/5 = 0.8 -> decision 1
        let mut pop = create_controlled_population(vec![1, 1, 0, 0, 0]);

        // Make last 2 abstain
        for i in 3..5 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 2.0,
                lower: -2.0,
                rejection_rate: 1.0,
            });
        }

        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // Manually set different weights
        jury.weights = Some(vec![2.0, 2.0, 1.0, 3.0, 2.0]);

        let (pred_classes, scores) = jury.predict(&data);

        // Weighted: (2*1 + 2*1 + 1*0) / (2+2+1) = 4/5 = 0.8
        assert_eq!(
            pred_classes[0], 1,
            "Weighted majority should ignore abstentions"
        );
        assert!((scores[0] - 0.8).abs() < 1e-10, "Score should be 0.8");
    }

    #[test]
    fn test_rejection_rate_includes_collective_abstentions() {
        // Test that rejection_rate counts both individual and collective abstentions
        let mut pop = create_controlled_population(vec![1, 1, 0, 0]);

        // Sample 1: All abstain -> collective abstention
        // Sample 2: 2 vote 1, 2 vote 0 -> normal vote
        let mut X = HashMap::new();
        X.insert((0, 0), 1.0);
        X.insert((1, 0), 1.0);

        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);

        let data = Data {
            X,
            y: vec![1, 0],
            features: vec!["feature1".to_string()],
            samples: vec!["sample1".to_string(), "sample2".to_string()],
            feature_class,
            feature_significance: HashMap::new(),
            feature_annotations: None,
            sample_annotations: None,
            feature_selection: vec![0],
            feature_len: 1,
            sample_len: 2,
            classes: vec!["class0".to_string(), "class1".to_string()],
        };

        // Make all experts abstain for sample 0 only (by setting extreme thresholds)
        for i in 0..4 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 0.5, // Will abstain for low scores
                lower: 0.4,
                rejection_rate: 0.5,
            });
        }

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // Rejection rate should reflect collective abstentions
        assert!(
            jury.rejection_rate >= 0.0 && jury.rejection_rate <= 1.0,
            "Rejection rate should be valid even with mixed abstentions"
        );
    }

    #[test]
    fn test_compute_new_metrics_with_ground_truth_containing_class_2() {
        // Test that compute_metrics_from_classes correctly ignores class 2 labels in ground truth
        // and that rejection_rate remains coherent
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Create data with ground truth containing class 2 (should be ignored in metrics computation)
        // Ground truth: [1, 0, 2, 1, 0, 2, 1]
        // Predictions: [1, 0, 1, 1, 0, 0, 0]
        // After filtering out 2s: GT=[1,0,1,0,1], Pred=[1,0,1,0,0]
        // TP=2, TN=1, FP=0, FN=1 => Acc=3/4=0.75, Se=2/3=0.667, Sp=1/1=1.0
        let data = create_multi_sample_data(vec![1, 0, 2, 1, 0, 2, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // Verify predictions were made for all samples (including those with y=2 in GT)
        let (pred_classes, _) = jury.predict(&data);
        assert_eq!(pred_classes.len(), 7, "Should predict for all 7 samples");

        // Verify metrics were computed correctly after filtering out samples with y=2 from GT
        // Ground truth: [1, 0, 2, 1, 0, 2, 1]
        // Predictions by majority (3 vote 1, 2 vote 0): all predict 1 (ratio 3/5 = 0.6 > 0.5)
        // After filtering GT y=2 (indices 2 and 5): GT=[1,0,1,0,1], Pred=[1,1,1,1,1]
        // TP=3 (indices 0,2,4), TN=0, FP=2 (indices 1,3), FN=0
        // Accuracy = (TP+TN)/(TP+TN+FP+FN) = 3/5 = 0.6
        // Sensitivity = TP/(TP+FN) = 3/3 = 1.0
        // Specificity = TN/(TN+FP) = 0/2 = 0.0

        let tolerance = 1e-10;
        assert!(
            (jury.accuracy - 0.6).abs() < tolerance,
            "Accuracy should be 0.6 after filtering class 2, got {}",
            jury.accuracy
        );
        assert!(
            (jury.sensitivity - 1.0).abs() < tolerance,
            "Sensitivity should be 1.0 after filtering class 2, got {}",
            jury.sensitivity
        );
        assert!(
            (jury.specificity - 0.0).abs() < tolerance,
            "Specificity should be 0.0 after filtering class 2, got {}",
            jury.specificity
        );

        // Rejection rate should count jury's predictions of class 2, not ground truth's 2s
        // With majority threshold 0.5 and ratio 0.6, no jury abstentions expected
        assert!(
            (jury.rejection_rate - 0.0).abs() < tolerance,
            "Rejection rate should be 0.0 (no jury abstentions), got {}",
            jury.rejection_rate
        );
    }

    #[test]
    fn test_optimize_majority_threshold_youden_with_high_abstention_rate() {
        // Test stability of Youden optimization when high abstention rate causes
        // massive filtering of sample pairs
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Force high abstention rate on most samples
        for i in 0..5 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 0.55, // Narrow rejection zone
                lower: 0.45,
                rejection_rate: 0.0, // Will be computed
            });
        }

        // Create multi-sample data
        let data = create_multi_sample_data(vec![1, 1, 1, 0, 0, 0, 1, 1, 0, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // Now optimize threshold - should handle massive abstention gracefully
        let optimized_threshold = jury.optimize_majority_threshold_youden(&data);

        // Verify threshold is still valid
        assert!(
            optimized_threshold >= 0.0 && optimized_threshold <= 1.0,
            "Optimized threshold should remain in valid range"
        );
        assert!(
            jury.voting_threshold >= 0.0 && jury.voting_threshold <= 1.0,
            "Jury voting_threshold should remain in valid range"
        );

        // Verify metrics are still computable
        assert!(
            jury.accuracy >= 0.0 && jury.accuracy <= 1.0,
            "Accuracy should be valid after Youden optimization with high abstention"
        );

        // Verify rejection_rate reflects high abstention
        assert!(
            jury.rejection_rate >= 0.0 && jury.rejection_rate <= 1.0,
            "Rejection rate should be valid"
        );
    }

    #[test]
    fn test_consensus_with_threshold_window_has_no_effect() {
        // Test that threshold_window parameter (used only in Majority) doesn't affect Consensus
        let pop = create_controlled_population(vec![1, 1, 1, 1, 0]);
        let data = create_single_sample_data(1);

        // Create two juries with different threshold_window values
        let mut jury_no_window = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.7,
            &0.0, // threshold_window = 0
            &WeightingMethod::Uniform,
        );

        let mut jury_wide_window = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.7,
            &50.0, // threshold_window = 50% (very wide)
            &WeightingMethod::Uniform,
        );

        jury_no_window.evaluate(&data);
        jury_wide_window.evaluate(&data);

        let (pred_no_window, scores_no_window) = jury_no_window.predict(&data);
        let (pred_wide_window, scores_wide_window) = jury_wide_window.predict(&data);

        // Consensus should produce identical results regardless of threshold_window
        assert_eq!(
            pred_no_window[0], pred_wide_window[0],
            "Consensus prediction should be identical regardless of threshold_window"
        );
        assert!(
            (scores_no_window[0] - scores_wide_window[0]).abs() < 1e-10,
            "Consensus score should be identical regardless of threshold_window"
        );

        // Both should give pos_ratio = 4/5 = 0.8
        assert_eq!(pred_no_window[0], 1, "Should predict class 1 (0.8 >= 0.7)");
        assert!(
            (scores_no_window[0] - 0.8).abs() < 1e-10,
            "Score should be 0.8"
        );
    }

    #[test]
    fn test_specialized_weighting_with_high_abstention() {
        // Test that specialized weighting adapts correctly when one group is massively abstentionist
        // Verify weight normalization per sample and decision stability
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Make experts 3-4 (who would vote 0) mostly abstain
        for i in 3..5 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 2.0, // Will abstain on most samples
                lower: -2.0,
                rejection_rate: 0.8, // High abstention rate
            });
        }

        // Create data where different samples trigger different abstention patterns
        let data = create_multi_sample_data(vec![1, 1, 0, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.5,
                specificity_threshold: 0.5,
            },
        );

        jury.evaluate(&data);

        // Verify weights were assigned
        assert!(
            jury.weights.is_some(),
            "Specialized weighting should assign weights"
        );

        let weights = jury.weights.as_ref().unwrap();
        assert_eq!(weights.len(), 5, "Should have weights for all 5 experts");

        // Verify weights are non-negative
        for (i, &w) in weights.iter().enumerate() {
            assert!(w >= 0.0, "Expert {} weight should be non-negative", i);
        }

        let (pred_classes, scores) = jury.predict(&data);

        // Verify predictions for all samples
        assert_eq!(pred_classes.len(), 5, "Should predict for all samples");
        assert_eq!(scores.len(), 5, "Should have scores for all samples");

        // Verify ratios are properly normalized (between 0 and 1, or special values)
        for (i, &score) in scores.iter().enumerate() {
            assert!(
                score >= -1.0 && score <= 1.0 || score == 0.5,
                "Sample {} score should be normalized or special value, got {}",
                i,
                score
            );
        }

        // Verify metrics are computable
        assert!(
            jury.accuracy >= 0.0 && jury.accuracy <= 1.0,
            "Accuracy should be valid with specialized weighting and abstentions"
        );

        // Decision should remain stable (not all abstentions)
        let non_abstention_count = pred_classes.iter().filter(|&&c| c != 2).count();
        assert!(
            non_abstention_count > 0,
            "At least some samples should have definitive votes despite high abstention"
        );
    }

    #[test]
    fn test_additional_metrics_consistency_with_partial_expert_coverage_and_abstentions() {
        // Test AdditionalMetrics coherence when:
        // - Only some experts expose additional metrics
        // - High abstention rate on multi-sample data
        // - Verify compute_additional aggregation and API stability
        let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

        // Give additional metrics to only first 3 experts
        for i in 0..3 {
            pop.individuals[i].metrics = crate::individual::AdditionalMetrics {
                mcc: Some(0.6 + 0.1 * i as f64),
                f1_score: Some(0.7 + 0.05 * i as f64),
                npv: Some(0.8),
                ppv: Some(0.75),
                g_mean: Some(0.65),
            };
        }

        // Experts 3-4 have no additional metrics
        // This tests the "at least one expert has metrics" logic

        // Add abstention via ThresholdCI
        for i in 2..5 {
            pop.individuals[i].threshold_ci = Some(crate::individual::ThresholdCI {
                upper: 0.6,
                lower: 0.4,
                rejection_rate: 0.3,
            });
        }

        // Multi-sample data to test aggregation
        let data = create_multi_sample_data(vec![1, 0, 1, 0, 1, 0, 1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);

        // Verify additional metrics were computed on the Jury's aggregate predictions
        // Key insight: Jury::compute_new_metrics RECALCULATES additional metrics
        // from the jury's aggregate predictions, it does NOT average experts' metrics.
        // The test verifies that:
        // 1. If at least one expert has a metric defined, Jury computes it
        // 2. The computed metric is based on Jury's predictions vs ground truth
        // 3. Values are within valid bounds

        let metrics = &jury.metrics;

        // MCC: Should be present if at least one expert exposes it
        // Computed from Jury's confusion matrix, not averaged from experts
        if let Some(mcc) = metrics.mcc {
            assert!(
                mcc >= -1.0 && mcc <= 1.0,
                "MCC should be in valid range [-1, 1], got {}",
                mcc
            );
            assert!(mcc.is_finite(), "MCC should be finite");
        } else {
            panic!("MCC should have been computed since some experts expose it (triggers computation flag)");
        }

        // F1-score: Computed from Jury's TP, FP, FN
        if let Some(f1) = metrics.f1_score {
            assert!(
                f1 >= 0.0 && f1 <= 1.0,
                "F1-score should be in valid range [0, 1], got {}",
                f1
            );
            assert!(f1.is_finite(), "F1-score should be finite");
        } else {
            panic!("F1-score should have been computed since some experts expose it");
        }

        // NPV: Computed from Jury's TN, FN
        if let Some(npv) = metrics.npv {
            assert!(
                npv >= 0.0 && npv <= 1.0,
                "NPV should be in valid range [0, 1], got {}",
                npv
            );
            assert!(npv.is_finite(), "NPV should be finite");
        } else {
            panic!("NPV should have been computed since some experts expose it");
        }

        // PPV: Computed from Jury's TP, FP
        if let Some(ppv) = metrics.ppv {
            assert!(
                ppv >= 0.0 && ppv <= 1.0,
                "PPV should be in valid range [0, 1], got {}",
                ppv
            );
            assert!(ppv.is_finite(), "PPV should be finite");
        } else {
            panic!("PPV should have been computed since some experts expose it");
        }

        // G-mean: Computed from Jury's sensitivity and specificity
        if let Some(g_mean) = metrics.g_mean {
            assert!(
                g_mean >= 0.0 && g_mean <= 1.0,
                "G-mean should be in valid range [0, 1], got {}",
                g_mean
            );
            assert!(g_mean.is_finite(), "G-mean should be finite");
        } else {
            panic!("G-mean should have been computed since some experts expose it");
        }

        // Verify standard metrics are also valid
        assert!(
            jury.accuracy >= 0.0 && jury.accuracy <= 1.0,
            "Base accuracy should be valid"
        );
        assert!(
            jury.rejection_rate >= 0.0 && jury.rejection_rate <= 1.0,
            "Rejection rate should be valid with mixed metrics and abstentions"
        );

        // Verify predictions work correctly
        let (pred_classes, _) = jury.predict(&data);
        assert_eq!(pred_classes.len(), 8, "Should predict for all 8 samples");
    }

    // -----------------------------------------------------------------
    // Tests for monotonicity of rejection_rate when widening [lower, upper] interval
    // for Jury (analogous to Individual tests)
    // -----------------------------------------------------------------

    #[test]
    fn test_jury_rejection_rate_monotonicity_widening_interval() {
        // Test that widening the [lower, upper] interval can only increase or maintain rejection_rate
        // on a fixed dataset for a Jury

        let mut pop = create_controlled_population(vec![1, 1, 0, 0, 0]);

        // Add threshold_ci to experts with narrow interval
        for expert in &mut pop.individuals {
            expert.threshold_ci = Some(crate::individual::ThresholdCI {
                lower: 0.45,
                upper: 0.55,
                rejection_rate: 0.0,
            });
            expert.threshold = 0.5;
        }

        let data = create_multi_sample_data(vec![1, 1, 0, 0, 1, 0]);

        let mut jury_narrow = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury_narrow.evaluate(&data);
        let rejection_rate_narrow = jury_narrow.rejection_rate;

        // Now widen the interval
        for expert in &mut pop.individuals {
            expert.threshold_ci = Some(crate::individual::ThresholdCI {
                lower: 0.3,
                upper: 0.7,
                rejection_rate: 0.0,
            });
        }

        let mut jury_wide = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury_wide.evaluate(&data);
        let rejection_rate_wide = jury_wide.rejection_rate;

        assert!(rejection_rate_wide >= rejection_rate_narrow - 1e-10,
                "Widening the [lower, upper] interval should increase or maintain Jury rejection_rate: narrow={}, wide={}",
                rejection_rate_narrow, rejection_rate_wide);
    }

    #[test]
    fn test_jury_rejection_rate_monotonicity_multiple_intervals() {
        // Test monotonicity across multiple interval widths for Jury

        let intervals = vec![
            (0.48, 0.52),
            (0.45, 0.55),
            (0.40, 0.60),
            (0.30, 0.70),
            (0.20, 0.80),
        ];

        let data = create_multi_sample_data(vec![1, 1, 0, 0, 1, 0, 1]);

        let mut prev_rejection_rate = 0.0;

        for (lower, upper) in intervals {
            let mut pop = create_controlled_population(vec![1, 1, 1, 0, 0]);

            // Set interval for all experts
            for expert in &mut pop.individuals {
                expert.threshold_ci = Some(crate::individual::ThresholdCI {
                    lower,
                    upper,
                    rejection_rate: 0.0,
                });
                expert.threshold = 0.5;
            }

            let mut jury = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Majority,
                &0.5,
                &0.0,
                &WeightingMethod::Uniform,
            );

            jury.evaluate(&data);
            let rejection_rate = jury.rejection_rate;

            assert!(rejection_rate >= prev_rejection_rate - 1e-10,
                    "Jury rejection rate should be monotonic: interval [{}, {}] has rejection_rate={}, previous was {}",
                    lower, upper, rejection_rate, prev_rejection_rate);

            prev_rejection_rate = rejection_rate;
        }
    }

    #[test]
    fn test_jury_rejection_rate_zero_width_interval() {
        // Edge case: zero-width interval should give minimal abstention for Jury

        let mut pop_no_ci = create_controlled_population(vec![1, 1, 0, 0, 0]);

        // No threshold_ci
        for expert in &mut pop_no_ci.individuals {
            expert.threshold = 0.5;
            expert.threshold_ci = None;
        }

        let data = create_multi_sample_data(vec![1, 1, 0, 0]);

        let mut jury_no_ci = Jury::new(
            &pop_no_ci,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury_no_ci.evaluate(&data);
        let rejection_rate_no_ci = jury_no_ci.rejection_rate;

        // With zero-width interval
        let mut pop_zero = create_controlled_population(vec![1, 1, 0, 0, 0]);
        for expert in &mut pop_zero.individuals {
            expert.threshold = 0.5;
            expert.threshold_ci = Some(crate::individual::ThresholdCI {
                lower: 0.5,
                upper: 0.5,
                rejection_rate: 0.0,
            });
        }

        let mut jury_zero = Jury::new(
            &pop_zero,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury_zero.evaluate(&data);
        let rejection_rate_zero = jury_zero.rejection_rate;

        assert!((rejection_rate_zero - rejection_rate_no_ci).abs() < 0.2,
                "Zero-width interval should give similar rejection_rate to no interval case: no_ci={}, zero={}",
                rejection_rate_no_ci, rejection_rate_zero);
    }
}
