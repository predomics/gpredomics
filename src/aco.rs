//! Ant Colony Optimization (ACO) for sparse model discovery.
//!
//! Implements a Max-Min Ant System (MMAS) variant for feature selection
//! and coefficient assignment. Each ant constructs a model by selecting
//! features probabilistically based on pheromone trails and heuristic
//! information (feature significance).
//!
//! Key differences from GA:
//! - Constructive: builds models feature-by-feature (vs recombinative)
//! - Pheromone memory: collective learning about good features
//! - Natural diversity: probabilistic construction avoids premature convergence

use crate::data::Data;
use crate::experiment::{ExperimentMetadata, PheromoneEntry, PheromoneSnapshot};
use crate::ga::{remove_out_of_bounds, remove_stillborn};
use crate::individual::{self, Individual};
use crate::param::Param;
use crate::population::Population;
use crate::{cinfo, individual::BINARY_LANG};
use log::{debug, info, warn};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::cmp::min;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Pheromone matrix: stores pheromone values for (feature, sign) pairs.
/// Uses BTreeMap for deterministic iteration order across runs.
struct PheromoneMatrix {
    /// Pheromone for positive sign: feature_idx -> τ
    positive: BTreeMap<usize, f64>,
    /// Pheromone for negative sign: feature_idx -> τ
    negative: BTreeMap<usize, f64>,
    /// MMAS bounds
    tau_min: f64,
    tau_max: f64,
}

impl PheromoneMatrix {
    fn new(features: &[usize], tau_init: f64, tau_min: f64, tau_max: f64) -> Self {
        let mut positive = BTreeMap::new();
        let mut negative = BTreeMap::new();
        for &f in features {
            positive.insert(f, tau_init);
            negative.insert(f, tau_init);
        }
        PheromoneMatrix {
            positive,
            negative,
            tau_min,
            tau_max,
        }
    }

    /// Get total pheromone for a feature (sum of positive + negative)
    fn total(&self, feature: usize) -> f64 {
        self.positive.get(&feature).unwrap_or(&0.0) + self.negative.get(&feature).unwrap_or(&0.0)
    }

    /// Get pheromone for a (feature, sign) pair
    fn get(&self, feature: usize, sign: i8) -> f64 {
        if sign > 0 {
            *self.positive.get(&feature).unwrap_or(&0.0)
        } else {
            *self.negative.get(&feature).unwrap_or(&0.0)
        }
    }

    /// Evaporate all pheromones by factor (1 - rho)
    fn evaporate(&mut self, rho: f64) {
        let factor = 1.0 - rho;
        for tau in self.positive.values_mut() {
            *tau = (*tau * factor).max(self.tau_min);
        }
        for tau in self.negative.values_mut() {
            *tau = (*tau * factor).max(self.tau_min);
        }
    }

    /// Deposit pheromone for an individual's features
    fn deposit(&mut self, individual: &Individual, amount: f64) {
        for (&feat, &coef) in &individual.features {
            if coef > 0 {
                if let Some(tau) = self.positive.get_mut(&feat) {
                    *tau = (*tau + amount).min(self.tau_max);
                }
            } else if coef < 0 {
                if let Some(tau) = self.negative.get_mut(&feat) {
                    *tau = (*tau + amount).min(self.tau_max);
                }
            }
        }
    }

    /// Snapshot the top-N features by total pheromone, returning (feature_idx, tau_pos, tau_neg).
    fn snapshot_top(&self, n: usize) -> Vec<(usize, f64, f64)> {
        let mut entries: Vec<(usize, f64, f64, f64)> = self
            .positive
            .keys()
            .map(|&f| {
                let tp = *self.positive.get(&f).unwrap_or(&0.0);
                let tn = *self.negative.get(&f).unwrap_or(&0.0);
                (f, tp, tn, tp + tn)
            })
            .collect();
        entries.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        entries
            .into_iter()
            .take(n)
            .map(|(f, tp, tn, _)| (f, tp, tn))
            .collect()
    }

    /// Compute Shannon entropy of the pheromone distribution (normalized to [0,1]).
    fn entropy(&self) -> f64 {
        let totals: Vec<f64> = self.positive.keys().map(|&f| self.total(f)).collect();
        let sum: f64 = totals.iter().sum();
        if sum <= 0.0 {
            return 1.0;
        }
        let n = totals.len() as f64;
        let max_entropy = n.ln();
        if max_entropy <= 0.0 {
            return 0.0;
        }
        let entropy: f64 = totals
            .iter()
            .filter(|&&t| t > 0.0)
            .map(|&t| {
                let p = t / sum;
                -p * p.ln()
            })
            .sum();
        entropy / max_entropy // normalize to [0,1]
    }

    /// Get precomputed selection probabilities for all features (τ^α × η^β).
    /// Returns sorted vec of (feature_idx, probability) for efficient sampling.
    fn selection_probs(
        &self,
        available: &[usize],
        heuristic: &HeuristicInfo,
        alpha: f64,
        beta: f64,
    ) -> Vec<(usize, f64)> {
        available
            .iter()
            .map(|&f| {
                let tau = self.total(f);
                let eta = heuristic.get(f);
                (f, tau.powf(alpha) * eta.powf(beta))
            })
            .collect()
    }
}

/// Heuristic information for each feature (η), derived from feature significance.
/// Uses BTreeMap for deterministic iteration order.
struct HeuristicInfo {
    values: BTreeMap<usize, f64>,
}

impl HeuristicInfo {
    fn new(feature_selection: &[usize], feature_significance: &HashMap<usize, f64>) -> Self {
        let mut values = BTreeMap::new();
        for &f in feature_selection {
            let sig = feature_significance.get(&f).copied().unwrap_or(1.0);
            // For p-value methods: lower p-value = higher desirability
            // For bayesian_fisher: higher score = better
            let eta = if (0.0..=1.0).contains(&sig) {
                1.0 / (sig + 1e-10)
            } else {
                sig.abs().max(1e-10)
            };
            values.insert(f, eta);
        }
        HeuristicInfo { values }
    }

    fn get(&self, feature: usize) -> f64 {
        *self.values.get(&feature).unwrap_or(&1.0)
    }
}

/// Construct a single ant's solution (one Individual).
fn construct_solution(
    probs: &[(usize, f64)],
    pheromone: &PheromoneMatrix,
    feature_class: &HashMap<usize, u8>,
    language: u8,
    data_type: u8,
    epsilon: f64,
    k: usize,
    threshold_ci: bool,
    rng: &mut ChaCha8Rng,
) -> Individual {
    if probs.is_empty() || k == 0 {
        return Individual::new();
    }

    // Working copy of probabilities (we remove selected features)
    let mut available: Vec<(usize, f64)> = probs.to_vec();
    let mut selected_features: BTreeMap<usize, i8> = BTreeMap::new();

    for _ in 0..k {
        if available.is_empty() {
            break;
        }

        let total_prob: f64 = available.iter().map(|(_, p)| p).sum();
        if total_prob <= 0.0 {
            break;
        }

        // Roulette wheel selection
        let mut r = rng.gen::<f64>() * total_prob;
        let mut chosen_idx = 0;
        for (i, (_, p)) in available.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                chosen_idx = i;
                break;
            }
        }

        let (feat, _) = available.swap_remove(chosen_idx); // O(1) removal

        // Choose coefficient sign based on pheromone and feature class
        let sign = match language {
            individual::BINARY_LANG => 1i8,
            individual::TERNARY_LANG | individual::RATIO_LANG => {
                let tau_pos = pheromone.get(feat, 1);
                let tau_neg = pheromone.get(feat, -1);
                let class = feature_class.get(&feat).copied().unwrap_or(2);
                let pos_prob = tau_pos / (tau_pos + tau_neg + 1e-10);
                if class == 1 {
                    if rng.gen::<f64>() < pos_prob {
                        1
                    } else {
                        -1
                    }
                } else {
                    if rng.gen::<f64>() < 1.0 - pos_prob {
                        -1
                    } else {
                        1
                    }
                }
            }
            individual::POW2_LANG => {
                let class = feature_class.get(&feat).copied().unwrap_or(2);
                let base_sign: i8 = if class == 1 { 1 } else { -1 };
                let power = rng.gen_range(0..=3);
                base_sign * (1i8 << power)
            }
            _ => 1,
        };

        selected_features.insert(feat, sign);
    }

    let mut ind = Individual::new();
    ind.features = selected_features;
    ind.k = ind.features.len();
    ind.language = language;
    ind.data_type = data_type;
    ind.epsilon = epsilon;
    if language == individual::RATIO_LANG {
        ind.cls.threshold = 1.0;
    }
    if threshold_ci {
        ind.cls.threshold_ci = Some(individual::ThresholdCI {
            upper: 0.0,
            lower: 0.0,
            rejection_rate: 0.0,
        });
    }
    ind
}

/// Local search: try removing each feature, keep removal if fitness improves.
/// Returns true if any improvement was made.
fn local_search_remove(individual: &mut Individual, data: &Data, param: &Param) -> bool {
    if individual.k <= 1 {
        return false;
    }

    let features: Vec<(usize, i8)> = individual.features.iter().map(|(&k, &v)| (k, v)).collect();
    let baseline_scores = individual.evaluate(data);
    let (baseline_auc, _, _, _, _, _) = crate::utils::compute_roc_and_metrics_from_value(
        &baseline_scores,
        &data.y,
        &crate::param::FitFunction::auc,
        None,
    );

    let mut improved = false;
    for &(feat, coef) in &features {
        // Try removing this feature
        individual.features.remove(&feat);
        individual.k -= 1;

        let scores = individual.evaluate(data);
        let (auc, _, _, _, _, _) = crate::utils::compute_roc_and_metrics_from_value(
            &scores,
            &data.y,
            &crate::param::FitFunction::auc,
            None,
        );

        // Apply k_penalty to compare fairly
        let penalized_baseline = baseline_auc - param.general.k_penalty * (individual.k + 1) as f64;
        let penalized_new = auc - param.general.k_penalty * individual.k as f64;

        if penalized_new >= penalized_baseline {
            // Keep removal — sparser model with equal or better penalized fitness
            improved = true;
        } else {
            // Restore feature
            individual.features.insert(feat, coef);
            individual.k += 1;
        }
    }

    improved
}

/// Display legend for ACO epoch output
fn display_aco_epoch_legend() -> String {
    "Legend: #iteration | best: Language:DataType  0 ▒▒▒▒▒ 1 [k=features, age=iterations_since_improvement]".to_string()
}

/// Main ACO entry point. Returns populations and pheromone metadata.
pub fn aco(
    data: &mut Data,
    _test_data: &mut Option<Data>,
    _initial_pop: &mut Option<Population>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> (Vec<Population>, Option<ExperimentMetadata>) {
    let time = Instant::now();

    // Feature selection (same as GA — avoid data leakage)
    data.select_features(param);
    debug!("ACO: {} features selected", data.feature_selection.len());

    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    // Parse languages and data types
    let languages: Vec<u8> = param
        .general
        .language
        .split(',')
        .map(individual::language)
        .collect();
    let data_types: Vec<u8> = param
        .general
        .data_type
        .split(',')
        .map(individual::data_type)
        .collect();

    // Initialize heuristic info from feature significance
    let heuristic = HeuristicInfo::new(&data.feature_selection, &data.feature_significance);

    // Initialize pheromone matrix
    let tau_init = param.aco.tau_max; // MMAS: start at max
    let mut pheromone = PheromoneMatrix::new(
        &data.feature_selection,
        tau_init,
        param.aco.tau_min,
        param.aco.tau_max,
    );

    // Precompute available features per language (binary filters to class > 0)
    let available_by_lang: BTreeMap<u8, Vec<usize>> = languages
        .iter()
        .map(|&lang| {
            let avail: Vec<usize> = if lang == BINARY_LANG {
                data.feature_selection
                    .iter()
                    .cloned()
                    .filter(|f| data.feature_class.get(f).copied().unwrap_or(0) > 0)
                    .collect()
            } else {
                data.feature_selection.clone()
            };
            (lang, avail)
        })
        .collect();

    let n_ants = param.aco.n_ants;
    let n_combos = languages.len() * data_types.len();
    let ants_per_combo = n_ants / n_combos.max(1);

    info!(
        "ACO: {} ants × {} iterations, α={}, β={}, ρ={}, k=[{},{}]",
        n_ants,
        param.aco.max_iterations,
        param.aco.alpha,
        param.aco.beta,
        param.aco.rho,
        param.aco.k_min,
        param.aco.k_max
    );

    cinfo!(
        param.general.display_colorful,
        "{}",
        display_aco_epoch_legend()
    );

    let threshold_ci = param.general.threshold_ci_n_bootstrap > 0;
    let mut populations: Vec<Population> = Vec::new();
    let mut global_best: Option<Individual> = None;
    let mut best_age: usize = 0;
    let mut pheromone_timeline: Vec<PheromoneSnapshot> = Vec::new();

    for iteration in 0..param.aco.max_iterations {
        if !running.load(Ordering::Relaxed) {
            info!("ACO: stopped by signal at iteration {}", iteration);
            break;
        }

        // Precompute selection probabilities once per iteration (shared across ants)
        let probs_by_lang: BTreeMap<u8, Vec<(usize, f64)>> = available_by_lang
            .iter()
            .map(|(&lang, avail)| {
                (
                    lang,
                    pheromone.selection_probs(avail, &heuristic, param.aco.alpha, param.aco.beta),
                )
            })
            .collect();

        // Generate per-ant seeds from the master RNG for parallel determinism
        let ant_configs: Vec<(u8, u8, u64)> = languages
            .iter()
            .flat_map(|&lang| {
                data_types
                    .iter()
                    .flat_map(move |&dt| (0..ants_per_combo).map(move |_| (lang, dt)))
            })
            .map(|(lang, dt)| (lang, dt, rng.gen::<u64>()))
            .collect();

        // Parallel ant construction
        let individuals: Vec<Individual> = ant_configs
            .par_iter()
            .map(|&(lang, dt, seed)| {
                let mut ant_rng = ChaCha8Rng::seed_from_u64(seed);
                let probs = &probs_by_lang[&lang];
                let effective_k_max = if param.aco.k_max > 0 {
                    min(param.aco.k_max, probs.len())
                } else {
                    probs.len()
                };
                let effective_k_min = if param.aco.k_min > 0 {
                    param.aco.k_min
                } else {
                    1
                };
                let k = if effective_k_min >= effective_k_max {
                    effective_k_max
                } else {
                    ant_rng.gen_range(effective_k_min..=effective_k_max)
                };

                construct_solution(
                    probs,
                    &pheromone,
                    &data.feature_class,
                    lang,
                    dt,
                    param.general.data_type_epsilon,
                    k,
                    threshold_ci,
                    &mut ant_rng,
                )
            })
            .collect();

        let mut pop = Population { individuals };

        // Remove invalid individuals
        remove_stillborn(&mut pop);
        remove_out_of_bounds(&mut pop, param.aco.k_min, param.aco.k_max);

        if pop.individuals.is_empty() {
            warn!(
                "ACO: all ants produced invalid solutions at iteration {}",
                iteration
            );
            continue;
        }

        // Evaluate fitness (already parallelized inside pop.fit)
        pop.fit(data, &mut None, &None, &None, param);
        pop = pop.sort();
        pop.compute_hash();
        pop.remove_clone();

        // Local search on top models (try removing features to reduce k)
        let n_local_search = min(5, pop.individuals.len());
        for i in 0..n_local_search {
            local_search_remove(&mut pop.individuals[i], data, param);
            pop.individuals[i].k = pop.individuals[i].features.len();
        }
        // Re-evaluate and re-sort after local search
        if n_local_search > 0 {
            pop.fit(data, &mut None, &None, &None, param);
            pop = pop.sort();
        }

        // Track best
        let iteration_best = &pop.individuals[0];
        let improved = match &global_best {
            None => true,
            Some(gb) => iteration_best.fit > gb.fit,
        };

        if improved {
            global_best = Some(iteration_best.clone());
            best_age = 0;
        } else {
            best_age += 1;
        }

        // Display progress
        let best = global_best.as_ref().unwrap();
        let bar_len = 50;
        let filled = (best.fit.clamp(0.0, 1.0) * bar_len as f64) as usize;
        let bar: String = (0..bar_len)
            .map(|i| if i < filled { '█' } else { '░' })
            .collect();
        let marker = if improved { "↑" } else { " " };
        cinfo!(
            param.general.display_colorful,
            "#{:<4} {} | best: {}:{} 0 {} 1 [k={}, age={}]",
            iteration,
            marker,
            best.get_language(),
            best.get_data_type(),
            bar,
            best.k,
            best_age
        );

        // Pheromone update (MMAS: only best ant deposits)
        pheromone.evaporate(param.aco.rho);

        // Deposit from iteration best
        let deposit_amount = iteration_best.fit / iteration_best.k.max(1) as f64;
        pheromone.deposit(iteration_best, deposit_amount);

        // Elite deposit from global best
        if let Some(ref gb) = global_best {
            let elite_amount = param.aco.elite_weight * gb.fit / gb.k.max(1) as f64;
            pheromone.deposit(gb, elite_amount);
        }

        // Snapshot pheromone state for timeline visualization (top 30 features)
        pheromone_timeline.push(PheromoneSnapshot {
            iteration,
            top_features: pheromone.snapshot_top(30),
            entropy: pheromone.entropy(),
        });

        populations.push(pop);

        // Early stopping
        if iteration >= param.aco.min_iterations && best_age >= param.aco.max_age_best_model {
            info!(
                "ACO: early stopping at iteration {} (best model age {} >= {})",
                iteration, best_age, param.aco.max_age_best_model
            );
            break;
        }
    }

    let elapsed = time.elapsed();
    info!(
        "ACO computed {} iterations in {:.2?}",
        populations.len(),
        elapsed
    );

    if populations.is_empty() {
        warn!("ACO produced no valid populations!");
        let mut empty = Population::new();
        empty.individuals.push(Individual::new());
        populations.push(empty);
    }

    // Export final pheromone matrix sorted by total pheromone (descending)
    let mut pheromone_entries: Vec<PheromoneEntry> = data
        .feature_selection
        .iter()
        .map(|&f| {
            let tp = *pheromone.positive.get(&f).unwrap_or(&0.0);
            let tn = *pheromone.negative.get(&f).unwrap_or(&0.0);
            PheromoneEntry {
                feature_idx: f,
                tau_positive: tp,
                tau_negative: tn,
                tau_total: tp + tn,
            }
        })
        .collect();
    pheromone_entries.sort_by(|a, b| b.tau_total.partial_cmp(&a.tau_total).unwrap());

    let meta = Some(ExperimentMetadata::ACOPheromone {
        pheromone: pheromone_entries,
        timeline: pheromone_timeline,
    });

    (populations, meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Data;
    use crate::param::Param;

    #[test]
    fn test_aco_basic_run() {
        let mut data = Data::specific_significant_test(50, 50);
        let mut param = Param::default();
        param.general.algo = "aco".to_string();
        param.general.language = "ternary".to_string();
        param.general.data_type = "prevalence".to_string();
        param.data.feature_maximal_adj_pvalue = 1.0;
        param.aco.n_ants = 50;
        param.aco.max_iterations = 10;
        param.aco.min_iterations = 10;

        let running = Arc::new(AtomicBool::new(true));
        let (populations, _meta) = aco(&mut data, &mut None, &mut None, &param, running);

        assert!(!populations.is_empty(), "ACO should produce populations");
        let final_pop = &populations[populations.len() - 1];
        assert!(
            !final_pop.individuals.is_empty(),
            "Final population should have individuals"
        );

        let best = &final_pop.individuals[0];
        assert!(best.k > 0, "Best model should have features");
        assert!(
            best.cls.auc >= 0.0 && best.cls.auc <= 1.0,
            "Best model should have valid AUC"
        );
    }

    #[test]
    fn test_aco_reproducibility() {
        let run_once = || {
            let mut data = Data::specific_significant_test(50, 50);
            let mut param = Param::default();
            param.general.algo = "aco".to_string();
            param.general.language = "ternary".to_string();
            param.general.data_type = "prevalence".to_string();
            param.data.feature_maximal_adj_pvalue = 1.0;
            param.general.seed = 42;
            param.aco.n_ants = 50;
            param.aco.max_iterations = 10;
            param.aco.min_iterations = 10;

            let running = Arc::new(AtomicBool::new(true));
            let (populations, _meta) = aco(&mut data, &mut None, &mut None, &param, running);
            let best = &populations.last().unwrap().individuals[0];
            (best.fit, best.cls.auc, best.k, best.hash)
        };

        let (fit1, auc1, k1, hash1) = run_once();
        let (fit2, auc2, k2, hash2) = run_once();

        assert_eq!(fit1, fit2, "Fitness should be identical across runs");
        assert_eq!(auc1, auc2, "AUC should be identical");
        assert_eq!(k1, k2, "k should be identical");
        assert_eq!(hash1, hash2, "Hash should be identical");
    }

    #[test]
    fn test_pheromone_matrix() {
        let features = vec![0, 1, 2, 3];
        let mut pm = PheromoneMatrix::new(&features, 0.5, 0.01, 1.0);

        assert!((pm.total(0) - 1.0).abs() < 1e-10);
        assert!((pm.get(0, 1) - 0.5).abs() < 1e-10);

        pm.evaporate(0.1);
        assert!((pm.get(0, 1) - 0.45).abs() < 1e-10);

        let mut ind = Individual::new();
        ind.features.insert(0, 1);
        ind.features.insert(1, -1);
        ind.k = 2;
        pm.deposit(&ind, 0.3);

        assert!(pm.get(0, 1) > 0.45);
        assert!(pm.get(1, -1) > 0.45);
    }

    #[test]
    fn test_aco_produces_sparse_models_with_k_penalty() {
        let mut data = Data::specific_significant_test(50, 50);
        let mut param = Param::default();
        param.general.algo = "aco".to_string();
        param.general.language = "ternary".to_string();
        param.general.data_type = "prevalence".to_string();
        param.data.feature_maximal_adj_pvalue = 1.0;
        param.general.k_penalty = 0.01; // Strong k penalty
        param.aco.n_ants = 100;
        param.aco.max_iterations = 20;
        param.aco.min_iterations = 20;
        param.aco.k_max = 20; // Limit model size

        let running = Arc::new(AtomicBool::new(true));
        let (populations, _meta) = aco(&mut data, &mut None, &mut None, &param, running);
        let best = &populations.last().unwrap().individuals[0];

        assert!(
            best.k <= 20,
            "Best model k ({}) should be <= k_max (20)",
            best.k
        );
    }
}
