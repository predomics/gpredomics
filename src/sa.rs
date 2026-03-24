//! Simulated Annealing (SA) for sparse model discovery.
//!
//! Single-solution metaheuristic that explores the search space by proposing
//! small modifications to a current solution and accepting worse solutions
//! with decreasing probability as the temperature cools.
//!
//! Key differences from GA/ACO:
//! - Single solution, not population-based — deep local exploration
//! - Temperature-controlled acceptance of worse solutions escapes local optima
//! - Very simple: only needs propose + accept/reject loop

use crate::data::Data;
use crate::ga::{remove_out_of_bounds, remove_stillborn};
use crate::individual::{self, Individual};
use crate::param::Param;
use crate::population::Population;
use crate::{cinfo, individual::BINARY_LANG};
use log::{debug, info, warn};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::min;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Propose a neighbor solution by modifying the current individual.
///
/// Randomly performs one of:
/// - Add a random feature (if below k_max)
/// - Remove a random feature (if above k_min)
/// - Swap a feature (remove one, add another)
/// - Flip coefficient sign (for ternary/ratio/pow2)
pub fn propose_neighbor(
    current: &Individual,
    feature_selection: &[usize],
    feature_class: &std::collections::HashMap<usize, u8>,
    k_min: usize,
    k_max: usize,
    rng: &mut ChaCha8Rng,
) -> Individual {
    let mut neighbor = current.clone();
    let current_features: Vec<usize> = neighbor.features.keys().copied().collect();
    let available: Vec<usize> = feature_selection
        .iter()
        .filter(|f| !neighbor.features.contains_key(f))
        .copied()
        .collect();

    // Choose move type based on current state
    let move_type = if current_features.is_empty() {
        0 // must add
    } else if available.is_empty() {
        1 // must remove
    } else {
        rng.gen_range(0..4) // 0=add, 1=remove, 2=swap, 3=flip sign
    };

    match move_type {
        0 if !available.is_empty() && (k_max == 0 || neighbor.k < k_max) => {
            // Add a random feature
            let feat = available[rng.gen_range(0..available.len())];
            let sign = if neighbor.language == BINARY_LANG {
                1i8
            } else {
                let class = feature_class.get(&feat).copied().unwrap_or(2);
                if class == 1 {
                    1
                } else {
                    -1
                }
            };
            neighbor.features.insert(feat, sign);
            neighbor.k += 1;
        }
        1 if current_features.len() > k_min => {
            // Remove a random feature
            let feat = current_features[rng.gen_range(0..current_features.len())];
            neighbor.features.remove(&feat);
            neighbor.k -= 1;
        }
        2 if !available.is_empty() && !current_features.is_empty() => {
            // Swap: remove one, add another
            let remove_feat = current_features[rng.gen_range(0..current_features.len())];
            let add_feat = available[rng.gen_range(0..available.len())];
            neighbor.features.remove(&remove_feat);
            let sign = if neighbor.language == BINARY_LANG {
                1i8
            } else {
                let class = feature_class.get(&add_feat).copied().unwrap_or(2);
                if class == 1 {
                    1
                } else {
                    -1
                }
            };
            neighbor.features.insert(add_feat, sign);
            // k stays the same
        }
        3 if !current_features.is_empty() && neighbor.language != BINARY_LANG => {
            // Flip coefficient sign
            let feat = current_features[rng.gen_range(0..current_features.len())];
            if let Some(coef) = neighbor.features.get_mut(&feat) {
                *coef = -*coef;
            }
        }
        _ => {
            // Fallback: try add or remove
            if !available.is_empty() && (k_max == 0 || neighbor.k < k_max) {
                let feat = available[rng.gen_range(0..available.len())];
                neighbor.features.insert(feat, 1);
                neighbor.k += 1;
            } else if current_features.len() > k_min {
                let feat = current_features[rng.gen_range(0..current_features.len())];
                neighbor.features.remove(&feat);
                neighbor.k -= 1;
            }
        }
    }

    neighbor.k = neighbor.features.len();
    neighbor
}

/// Main SA entry point. Returns Vec<Population> for compatibility with GA/ACO.
pub fn sa(
    data: &mut Data,
    _test_data: &mut Option<Data>,
    _initial_pop: &mut Option<Population>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> Vec<Population> {
    let time = Instant::now();

    // Feature selection
    data.select_features(param);
    debug!("SA: {} features selected", data.feature_selection.len());

    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    // Parse first language and data type
    let language = individual::language(param.general.language.split(',').next().unwrap_or("ter"));
    let data_type =
        individual::data_type(param.general.data_type.split(',').next().unwrap_or("prev"));

    if param.general.language.contains(',') {
        warn!(
            "SA uses only one language per run. Using: {}",
            param.general.language.split(',').next().unwrap_or("ter")
        );
    }

    // Generate initial solution
    let k_init = min(
        rng.gen_range(param.sa.k_min..=min(10, param.sa.k_max.max(param.sa.k_min))),
        data.feature_selection.len(),
    );
    let mut current = Individual::random_select_k(
        k_init,
        k_init,
        &data.feature_selection,
        &data.feature_class,
        language,
        data_type,
        param.general.data_type_epsilon,
        false,
        &mut rng,
    );

    // Evaluate initial solution
    let mut pop = Population {
        individuals: vec![current.clone()],
    };
    pop.fit(data, &mut None, &None, &None, param);
    current = pop.individuals.into_iter().next().unwrap();

    let mut best = current.clone();
    let mut temperature = param.sa.initial_temperature;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut populations: Vec<Population> = Vec::new();

    info!(
        "SA: T0={}, cooling={}, min_T={}, max_iter={}, k=[{},{}]",
        param.sa.initial_temperature,
        param.sa.cooling_rate,
        param.sa.min_temperature,
        param.sa.max_iterations,
        param.sa.k_min,
        param.sa.k_max,
    );

    cinfo!(
        param.general.display_colorful,
        "Legend: #iteration T=temperature | best fit [k=features, accepted/rejected]"
    );

    for iteration in 0..param.sa.max_iterations {
        if !running.load(Ordering::Relaxed) {
            info!("SA: stopped by signal at iteration {}", iteration);
            break;
        }

        if temperature < param.sa.min_temperature {
            info!(
                "SA: temperature {:.6} below minimum {:.6}, stopping at iteration {}",
                temperature, param.sa.min_temperature, iteration
            );
            break;
        }

        // Propose neighbor
        let mut neighbor = propose_neighbor(
            &current,
            &data.feature_selection,
            &data.feature_class,
            param.sa.k_min,
            param.sa.k_max,
            &mut rng,
        );

        // Evaluate neighbor
        let mut npop = Population {
            individuals: vec![neighbor.clone()],
        };
        npop.fit(data, &mut None, &None, &None, param);
        neighbor = npop.individuals.into_iter().next().unwrap();

        // Accept/reject (Metropolis criterion)
        let delta = neighbor.fit - current.fit;
        let accept = if delta >= 0.0 {
            true
        } else {
            let prob = (delta / temperature).exp();
            rng.gen::<f64>() < prob
        };

        if accept {
            current = neighbor;
            accepted += 1;
            if current.fit > best.fit {
                best = current.clone();
            }
        } else {
            rejected += 1;
        }

        // Cool down
        temperature *= param.sa.cooling_rate;

        // Snapshot for generation_tracking
        if iteration % param.sa.snapshot_interval == 0 || iteration == param.sa.max_iterations - 1 {
            let bar_len = 50;
            let filled = (best.fit.clamp(0.0, 1.0) * bar_len as f64) as usize;
            let bar: String = (0..bar_len)
                .map(|i| if i < filled { '█' } else { '░' })
                .collect();
            let improved = current.fit >= best.fit - 1e-10;
            let marker = if improved { "↑" } else { " " };

            cinfo!(
                param.general.display_colorful,
                "#{:<6} {} T={:.4} | best: {}:{} 0 {} 1 [k={}, acc/rej={}/{}]",
                iteration,
                marker,
                temperature,
                best.get_language(),
                best.get_data_type(),
                bar,
                best.k,
                accepted,
                rejected
            );

            // Store snapshot as a mini-population
            let mut snap = Population::new();
            snap.individuals.push(best.clone());
            snap.individuals.push(current.clone());
            snap.compute_hash();
            populations.push(snap);
        }
    }

    let elapsed = time.elapsed();
    info!(
        "SA computed {} iterations in {:.2?} (accepted: {}, rejected: {}, ratio: {:.1}%)",
        accepted + rejected,
        elapsed,
        accepted,
        rejected,
        if accepted + rejected > 0 {
            accepted as f64 / (accepted + rejected) as f64 * 100.0
        } else {
            0.0
        }
    );

    // Final population: best solution
    if populations.is_empty() {
        let mut final_pop = Population::new();
        final_pop.individuals.push(best);
        final_pop.compute_hash();
        populations.push(final_pop);
    }

    populations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Data;
    use crate::param::Param;

    #[test]
    fn test_sa_basic_run() {
        let mut data = Data::specific_significant_test(50, 50);
        let mut param = Param::default();
        param.general.algo = "sa".to_string();
        param.general.language = "ternary".to_string();
        param.general.data_type = "prevalence".to_string();
        param.data.feature_maximal_adj_pvalue = 1.0;
        param.sa.max_iterations = 500;
        param.sa.snapshot_interval = 100;

        let running = Arc::new(AtomicBool::new(true));
        let populations = sa(&mut data, &mut None, &mut None, &param, running);

        assert!(!populations.is_empty());
        let best = &populations.last().unwrap().individuals[0];
        assert!(best.k > 0);
        assert!(best.fit > 0.0);
    }

    #[test]
    fn test_sa_reproducibility() {
        let run_once = || {
            let mut data = Data::specific_significant_test(50, 50);
            let mut param = Param::default();
            param.general.algo = "sa".to_string();
            param.general.language = "ternary".to_string();
            param.general.data_type = "prevalence".to_string();
            param.data.feature_maximal_adj_pvalue = 1.0;
            param.general.seed = 42;
            param.sa.max_iterations = 200;
            param.sa.snapshot_interval = 50;

            let running = Arc::new(AtomicBool::new(true));
            let populations = sa(&mut data, &mut None, &mut None, &param, running);
            let best = &populations.last().unwrap().individuals[0];
            (best.fit, best.k, best.hash)
        };

        let (f1, k1, h1) = run_once();
        let (f2, k2, h2) = run_once();
        assert_eq!(f1, f2);
        assert_eq!(k1, k2);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_sa_respects_k_bounds() {
        let mut data = Data::specific_significant_test(50, 50);
        let mut param = Param::default();
        param.general.algo = "sa".to_string();
        param.general.language = "ternary".to_string();
        param.general.data_type = "prevalence".to_string();
        param.data.feature_maximal_adj_pvalue = 1.0;
        param.sa.max_iterations = 500;
        param.sa.k_min = 2;
        param.sa.k_max = 5;
        param.sa.snapshot_interval = 100;

        let running = Arc::new(AtomicBool::new(true));
        let populations = sa(&mut data, &mut None, &mut None, &param, running);
        let best = &populations.last().unwrap().individuals[0];
        assert!(
            best.k >= 2 && best.k <= 5,
            "k={} should be in [2,5]",
            best.k
        );
    }

    #[test]
    fn test_propose_neighbor() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let feature_selection = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut feature_class = std::collections::HashMap::new();
        for i in 0..10 {
            feature_class.insert(i, if i < 5 { 0 } else { 1 });
        }

        let mut ind = Individual::new();
        ind.language = individual::TERNARY_LANG;
        ind.features.insert(2, 1);
        ind.features.insert(5, -1);
        ind.k = 2;

        // Run many proposals and check they're valid
        for _ in 0..100 {
            let neighbor =
                propose_neighbor(&ind, &feature_selection, &feature_class, 1, 5, &mut rng);
            assert!(neighbor.k >= 1 && neighbor.k <= 5, "k={}", neighbor.k);
            assert_eq!(neighbor.k, neighbor.features.len());
        }
    }
}
