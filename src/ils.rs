//! Iterated Local Search (ILS) for sparse model discovery.
//!
//! Simple and effective: repeatedly apply local search to a local optimum,
//! then perturbate to escape, and local search again. The perturbation
//! strength controls the exploration/exploitation balance.
//!
//! Key advantage: minimal hyperparameters, often outperforms complex methods.

use crate::data::Data;
use crate::individual::{self, Individual};
use crate::param::Param;
use crate::population::Population;
use crate::sa::propose_neighbor;
use crate::utils::compute_auc_from_value;
use crate::{cinfo, individual::BINARY_LANG};
use log::{debug, info, warn};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::min;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Local search: greedily improve the individual by trying single-feature modifications.
/// Returns the improved individual and whether any improvement was made.
fn local_search(
    ind: &Individual,
    data: &Data,
    param: &Param,
    feature_selection: &[usize],
    feature_class: &std::collections::HashMap<usize, u8>,
    rng: &mut ChaCha8Rng,
    max_steps: usize,
) -> Individual {
    let mut current = ind.clone();
    let mut current_fit = current.fit;

    for _ in 0..max_steps {
        let mut improved = false;

        // Try each type of single-step move
        for _ in 0..current.k.max(5) * 3 {
            let neighbor = propose_neighbor(
                &current,
                feature_selection,
                feature_class,
                param.ils.k_min,
                param.ils.k_max,
                rng,
            );

            // Quick evaluate
            let scores = neighbor.evaluate(data);
            let auc = compute_auc_from_value(&scores, &data.y);
            let fit = auc - param.general.k_penalty * neighbor.k as f64;

            if fit > current_fit {
                current = neighbor;
                current.cls.auc = auc;
                current.fit = fit;
                current_fit = fit;
                improved = true;
                break;
            }
        }

        if !improved {
            break; // Local optimum reached
        }
    }

    current
}

/// Perturbate: apply multiple random moves to escape local optimum.
fn perturbate(
    ind: &Individual,
    feature_selection: &[usize],
    feature_class: &std::collections::HashMap<usize, u8>,
    perturbation_size: usize,
    k_min: usize,
    k_max: usize,
    rng: &mut ChaCha8Rng,
) -> Individual {
    let mut perturbed = ind.clone();
    for _ in 0..perturbation_size {
        perturbed = propose_neighbor(
            &perturbed,
            feature_selection,
            feature_class,
            k_min,
            k_max,
            rng,
        );
    }
    perturbed
}

/// Main ILS entry point.
pub fn ils(
    data: &mut Data,
    _test_data: &mut Option<Data>,
    _initial_pop: &mut Option<Population>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> Vec<Population> {
    let time = Instant::now();

    data.select_features(param);
    debug!("ILS: {} features selected", data.feature_selection.len());

    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    let language = individual::language(param.general.language.split(',').next().unwrap_or("ter"));
    let data_type =
        individual::data_type(param.general.data_type.split(',').next().unwrap_or("prev"));

    if param.general.language.contains(',') {
        warn!(
            "ILS uses only one language per run. Using: {}",
            param.general.language.split(',').next().unwrap_or("ter")
        );
    }

    // Generate initial solution
    let k_init = min(
        rng.gen_range(param.ils.k_min..=min(10, param.ils.k_max.max(param.ils.k_min))),
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
    let scores = current.evaluate(data);
    current.cls.auc = compute_auc_from_value(&scores, &data.y);
    current.fit = current.cls.auc - param.general.k_penalty * current.k as f64;

    info!(
        "ILS: {} iterations, perturbation_size={}, local_search_steps={}, k=[{},{}]",
        param.ils.max_iterations,
        param.ils.perturbation_size,
        param.ils.local_search_steps,
        param.ils.k_min,
        param.ils.k_max,
    );

    cinfo!(
        param.general.display_colorful,
        "Legend: #iteration | best fit [k=features]"
    );

    // Initial local search
    current = local_search(
        &current,
        data,
        param,
        &data.feature_selection.clone(),
        &data.feature_class.clone(),
        &mut rng,
        param.ils.local_search_steps,
    );

    let mut best = current.clone();
    let mut populations: Vec<Population> = Vec::new();
    let mut no_improve_count = 0usize;

    for iteration in 0..param.ils.max_iterations {
        if !running.load(Ordering::Relaxed) {
            info!("ILS: stopped by signal at iteration {}", iteration);
            break;
        }

        // Perturbate
        let feature_sel = data.feature_selection.clone();
        let feature_cls = data.feature_class.clone();
        let perturbed = perturbate(
            &current,
            &feature_sel,
            &feature_cls,
            param.ils.perturbation_size,
            param.ils.k_min,
            param.ils.k_max,
            &mut rng,
        );

        // Local search from perturbed solution
        let candidate = local_search(
            &perturbed,
            data,
            param,
            &feature_sel,
            &feature_cls,
            &mut rng,
            param.ils.local_search_steps,
        );

        // Acceptance criterion: accept if better than current (or equal with fewer features)
        if candidate.fit > current.fit || (candidate.fit == current.fit && candidate.k < current.k)
        {
            current = candidate;
            no_improve_count = 0;

            if current.fit > best.fit {
                best = current.clone();

                let bar_len = 50;
                let filled = (best.fit.clamp(0.0, 1.0) * bar_len as f64) as usize;
                let bar: String = (0..bar_len)
                    .map(|i| if i < filled { '█' } else { '░' })
                    .collect();
                cinfo!(
                    param.general.display_colorful,
                    "#{:<4} ↑ | best: {}:{} 0 {} 1 [k={}, fit={:.4}]",
                    iteration,
                    best.get_language(),
                    best.get_data_type(),
                    bar,
                    best.k,
                    best.fit
                );
            }
        } else {
            no_improve_count += 1;
        }

        // Snapshot
        if iteration % param.ils.snapshot_interval == 0 {
            let mut snap = Population::new();
            snap.individuals.push(best.clone());
            snap.individuals.push(current.clone());
            snap.compute_hash();
            populations.push(snap);
        }

        // Early stopping
        if no_improve_count >= param.ils.max_no_improve {
            info!(
                "ILS: no improvement for {} iterations, stopping at iteration {}",
                param.ils.max_no_improve, iteration
            );
            break;
        }
    }

    let elapsed = time.elapsed();
    info!(
        "ILS computed {} in {:.2?}. Best: AUC={:.4}, k={}, fit={:.4}",
        param.ils.max_iterations, elapsed, best.cls.auc, best.k, best.fit
    );

    // Final population
    if populations.is_empty() || populations.last().unwrap().individuals[0].hash != best.hash {
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
    fn test_ils_basic_run() {
        let mut data = Data::specific_significant_test(50, 50);
        let mut param = Param::default();
        param.general.algo = "ils".to_string();
        param.general.language = "ternary".to_string();
        param.general.data_type = "prevalence".to_string();
        param.data.feature_maximal_adj_pvalue = 1.0;
        param.ils.max_iterations = 20;

        let running = Arc::new(AtomicBool::new(true));
        let populations = ils(&mut data, &mut None, &mut None, &param, running);

        assert!(!populations.is_empty());
        let best = &populations.last().unwrap().individuals[0];
        assert!(best.k > 0);
    }

    #[test]
    fn test_ils_reproducibility() {
        let run_once = || {
            let mut data = Data::specific_significant_test(50, 50);
            let mut param = Param::default();
            param.general.algo = "ils".to_string();
            param.general.language = "ternary".to_string();
            param.general.data_type = "prevalence".to_string();
            param.data.feature_maximal_adj_pvalue = 1.0;
            param.general.seed = 42;
            param.ils.max_iterations = 20;

            let running = Arc::new(AtomicBool::new(true));
            let populations = ils(&mut data, &mut None, &mut None, &param, running);
            let best = &populations.last().unwrap().individuals[0];
            (best.fit, best.k, best.hash)
        };

        let (f1, k1, h1) = run_once();
        let (f2, k2, h2) = run_once();
        assert_eq!(f1, f2);
        assert_eq!(k1, k2);
        assert_eq!(h1, h2);
    }
}
