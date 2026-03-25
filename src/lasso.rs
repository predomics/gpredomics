//! LASSO / Elastic Net via coordinate descent.
//!
//! Produces the mathematically optimal sparse linear model for a given
//! regularization strength. Runs along a regularization path from high
//! alpha (few features) to low alpha (many features), converting each
//! solution into gpredomics Individuals with BTR coefficients.
//!
//! This provides a baseline: how well can an optimal linear model do
//! compared to gpredomics's discrete-coefficient heuristics?

use crate::cinfo;
use crate::data::Data;
use crate::individual::{self, Individual, BINARY_LANG};
use crate::param::Param;
use crate::population::Population;
use crate::utils::compute_auc_from_value;
use log::{debug, info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Soft-thresholding operator for LASSO
#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

/// Standardize features: subtract mean, divide by std. Returns (means, stds).
fn standardize(x: &mut Vec<Vec<f64>>) -> (Vec<f64>, Vec<f64>) {
    let n = x[0].len() as f64;
    let p = x.len();
    let mut means = vec![0.0; p];
    let mut stds = vec![0.0; p];

    for j in 0..p {
        let mean = x[j].iter().sum::<f64>() / n;
        means[j] = mean;
        let var = x[j].iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        stds[j] = var.sqrt().max(1e-10);
        for i in 0..x[j].len() {
            x[j][i] = (x[j][i] - mean) / stds[j];
        }
    }
    (means, stds)
}

/// Coordinate descent for Elastic Net at a single alpha.
///
/// Minimizes: (1/2n) ||y - Xw||² + alpha * (l1_ratio * ||w||₁ + (1-l1_ratio)/2 * ||w||²)
///
/// Returns coefficient vector w.
fn coordinate_descent(
    x: &[Vec<f64>], // features × samples (column-major)
    y: &[f64],      // targets (n samples)
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tolerance: f64,
    warm_start: Option<&[f64]>,
) -> Vec<f64> {
    let n = y.len() as f64;
    let p = x.len();

    let mut w: Vec<f64> = warm_start
        .map(|ws| ws.to_vec())
        .unwrap_or_else(|| vec![0.0; p]);

    // Precompute X^T X diagonal (||x_j||² / n)
    let x_sq: Vec<f64> = x
        .iter()
        .map(|col| col.iter().map(|v| v * v).sum::<f64>() / n)
        .collect();

    // Compute initial residuals: r = y - Xw
    let mut residuals: Vec<f64> = y.to_vec();
    for j in 0..p {
        if w[j] != 0.0 {
            for i in 0..y.len() {
                residuals[i] -= x[j][i] * w[j];
            }
        }
    }

    let l1_penalty = alpha * l1_ratio;
    let l2_penalty = alpha * (1.0 - l1_ratio);

    for _iter in 0..max_iter {
        let mut max_change = 0.0f64;

        for j in 0..p {
            let old_w = w[j];

            // Compute partial residual z_j = (1/n) * x_j^T * (r + x_j * w_j)
            let mut z_j = 0.0;
            for i in 0..y.len() {
                z_j += x[j][i] * (residuals[i] + x[j][i] * old_w);
            }
            z_j /= n;

            // Update w_j via soft-thresholding
            let new_w = soft_threshold(z_j, l1_penalty) / (x_sq[j] + l2_penalty);

            if (new_w - old_w).abs() > 1e-15 {
                // Update residuals
                let diff = new_w - old_w;
                for i in 0..y.len() {
                    residuals[i] -= x[j][i] * diff;
                }
                w[j] = new_w;
                max_change = max_change.max((new_w - old_w).abs());
            }
        }

        if max_change < tolerance {
            break;
        }
    }

    w
}

/// Convert LASSO coefficients to a BTR Individual.
///
/// Features with non-zero coefficients are included. The sign of the
/// coefficient determines the BTR sign (+1/-1). For binary language,
/// only positive coefficients are kept.
fn coefficients_to_individual(
    w: &[f64],
    feature_selection: &[usize],
    language: u8,
    data_type: u8,
    epsilon: f64,
) -> Individual {
    let mut ind = Individual::new();
    ind.language = language;
    ind.data_type = data_type;
    ind.epsilon = epsilon;

    for (j, &feat_idx) in feature_selection.iter().enumerate() {
        if j >= w.len() {
            break;
        }
        if w[j].abs() < 1e-10 {
            continue;
        }
        let sign: i8 = if language == BINARY_LANG {
            if w[j] > 0.0 {
                1
            } else {
                continue;
            } // Binary: only positive
        } else {
            if w[j] > 0.0 {
                1
            } else {
                -1
            }
        };
        ind.features.insert(feat_idx, sign);
    }
    ind.k = ind.features.len();
    ind
}

/// Main LASSO entry point. Runs coordinate descent along regularization path.
pub fn lasso(
    data: &mut Data,
    _test_data: &mut Option<Data>,
    _initial_pop: &mut Option<Population>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> Vec<Population> {
    let time = Instant::now();

    // Feature selection (same as other algorithms)
    data.select_features(param);
    debug!("LASSO: {} features selected", data.feature_selection.len());

    let language = individual::language(param.general.language.split(',').next().unwrap_or("ter"));
    let data_type =
        individual::data_type(param.general.data_type.split(',').next().unwrap_or("prev"));

    if param.general.language.contains(',') {
        warn!(
            "LASSO uses only one language per run. Using: {}",
            param.general.language.split(',').next().unwrap_or("ter")
        );
    }

    // Build feature matrix X (column-major: features × samples)
    let n = data.sample_len;
    let p = data.feature_selection.len();
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(p);
    for &feat_idx in &data.feature_selection {
        let mut col = Vec::with_capacity(n);
        for sample in 0..n {
            col.push(*data.X.get(&(sample, feat_idx)).unwrap_or(&0.0));
        }
        x.push(col);
    }

    // Target vector (convert u8 to f64)
    let y: Vec<f64> = data.y.iter().map(|&v| v as f64).collect();

    // Standardize features
    let (_means, _stds) = standardize(&mut x);

    // Center y
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();

    // Generate alpha path (log-spaced from alpha_max to alpha_min)
    let n_alphas = param.lasso.n_alphas;
    let log_min = param.lasso.alpha_min.ln();
    let log_max = param.lasso.alpha_max.ln();
    let alphas: Vec<f64> = (0..n_alphas)
        .map(|i| {
            let t = i as f64 / (n_alphas - 1).max(1) as f64;
            (log_max + t * (log_min - log_max)).exp()
        })
        .collect();

    info!(
        "LASSO: {} features × {} samples, {} alphas [{:.4} → {:.4}], l1_ratio={}",
        p,
        n,
        n_alphas,
        alphas[0],
        alphas[n_alphas - 1],
        param.lasso.l1_ratio
    );

    cinfo!(
        param.general.display_colorful,
        "Legend: alpha | k=features | AUC"
    );

    let mut populations: Vec<Population> = Vec::new();
    let mut best_individual: Option<Individual> = None;
    let mut warm_start: Option<Vec<f64>> = None;

    for (i, &alpha) in alphas.iter().enumerate() {
        if !running.load(Ordering::Relaxed) {
            info!("LASSO: stopped by signal at alpha {}", alpha);
            break;
        }

        // Coordinate descent with warm start
        let w = coordinate_descent(
            &x,
            &y_centered,
            alpha,
            param.lasso.l1_ratio,
            param.lasso.max_iter,
            param.lasso.tolerance,
            warm_start.as_deref(),
        );
        warm_start = Some(w.clone());

        // Convert to Individual
        let mut ind = coefficients_to_individual(
            &w,
            &data.feature_selection,
            language,
            data_type,
            param.general.data_type_epsilon,
        );

        if ind.k == 0 {
            continue; // Skip empty models
        }

        // Evaluate fitness using gpredomics standard evaluation
        let scores = ind.evaluate(data);
        ind.cls.auc = compute_auc_from_value(&scores, &data.y);
        ind.fit = ind.cls.auc - param.general.k_penalty * ind.k as f64;
        ind.compute_hash();

        // Track best
        let is_best = best_individual.as_ref().map_or(true, |b| ind.fit > b.fit);

        if is_best || i % 10 == 0 {
            let marker = if is_best { "↑" } else { " " };
            cinfo!(
                param.general.display_colorful,
                "#{:<4} {} α={:.4} | k={:<4} | AUC={:.4} fit={:.4}",
                i,
                marker,
                alpha,
                ind.k,
                ind.cls.auc,
                ind.fit
            );
        }

        if is_best {
            best_individual = Some(ind.clone());
        }

        // Store each alpha step as a mini-population
        let mut pop = Population::new();
        pop.individuals.push(ind);
        if let Some(ref best) = best_individual {
            if pop.individuals[0].hash != best.hash {
                pop.individuals.push(best.clone());
            }
        }
        pop.compute_hash();
        populations.push(pop);
    }

    let elapsed = time.elapsed();
    if let Some(ref best) = best_individual {
        info!(
            "LASSO computed {} alphas in {:.2?}. Best: AUC={:.4}, k={}, fit={:.4}",
            populations.len(),
            elapsed,
            best.cls.auc,
            best.k,
            best.fit
        );
        // Final population with full metrics (sensitivity, specificity, accuracy)
        let mut final_pop = Population::new();
        final_pop.individuals.push(best.clone());
        final_pop.fit(data, &mut None, &None, &None, param);
        final_pop.compute_hash();
        populations.push(final_pop);
    } else {
        warn!("LASSO produced no non-empty models");
        let mut empty = Population::new();
        empty.individuals.push(Individual::new());
        populations.push(empty);
    }

    populations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Data;
    use crate::param::Param;

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_eq!(soft_threshold(1.0, 2.0), 0.0);
        assert_eq!(soft_threshold(0.0, 2.0), 0.0);
    }

    #[test]
    fn test_coordinate_descent_zero_alpha() {
        // With alpha=0, should recover OLS solution
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let y = vec![1.0, 2.0, 3.0];
        let w = coordinate_descent(&x, &y, 0.0, 1.0, 100, 1e-6, None);
        // Should have non-zero coefficients
        assert!(w.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_coordinate_descent_high_alpha() {
        // With very high alpha, all coefficients should be zero
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let y = vec![1.0, 2.0, 3.0];
        let w = coordinate_descent(&x, &y, 100.0, 1.0, 100, 1e-6, None);
        assert!(w.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_lasso_basic_run() {
        let mut data = Data::specific_significant_test(50, 50);
        let mut param = Param::default();
        param.general.algo = "lasso".to_string();
        param.general.language = "ternary".to_string();
        param.general.data_type = "prevalence".to_string();
        param.data.feature_maximal_adj_pvalue = 1.0;
        param.lasso.n_alphas = 20;

        let running = Arc::new(AtomicBool::new(true));
        let populations = lasso(&mut data, &mut None, &mut None, &param, running);

        assert!(!populations.is_empty());
        let best = &populations.last().unwrap().individuals[0];
        assert!(best.k > 0);
        assert!(
            best.cls.auc > 0.0,
            "AUC should be computed: {}",
            best.cls.auc
        );
    }

    #[test]
    fn test_lasso_reproducibility() {
        let run_once = || {
            let mut data = Data::specific_significant_test(50, 50);
            let mut param = Param::default();
            param.general.algo = "lasso".to_string();
            param.general.language = "ternary".to_string();
            param.general.data_type = "prevalence".to_string();
            param.data.feature_maximal_adj_pvalue = 1.0;
            param.lasso.n_alphas = 20;

            let running = Arc::new(AtomicBool::new(true));
            let populations = lasso(&mut data, &mut None, &mut None, &param, running);
            let best = &populations.last().unwrap().individuals[0];
            (best.fit, best.k, best.cls.auc)
        };

        let (f1, k1, a1) = run_once();
        let (f2, k2, a2) = run_once();
        assert_eq!(f1, f2);
        assert_eq!(k1, k2);
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_coefficients_to_individual() {
        let w = vec![0.5, 0.0, -0.3, 0.0, 0.8];
        let fs = vec![10, 20, 30, 40, 50];
        let ind = coefficients_to_individual(&w, &fs, individual::TERNARY_LANG, 0, 1e-5);
        assert_eq!(ind.k, 3); // features 10, 30, 50
        assert_eq!(*ind.features.get(&10).unwrap(), 1);
        assert_eq!(*ind.features.get(&30).unwrap(), -1);
        assert_eq!(*ind.features.get(&50).unwrap(), 1);
    }
}

/// Public wrapper for coordinate descent (used by MCMC pre-screening)
pub fn coordinate_descent_pub(
    x: &[Vec<f64>],
    y: &[f64],
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tolerance: f64,
    warm_start: Option<&[f64]>,
) -> Vec<f64> {
    coordinate_descent(x, y, alpha, l1_ratio, max_iter, tolerance, warm_start)
}
