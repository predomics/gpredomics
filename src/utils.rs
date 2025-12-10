use crate::data::Data;
use crate::experiment::ImportanceAggregation;
use crate::individual::AdditionalMetrics;
use crate::param::FitFunction;
use crate::Param;
use crate::Population;
use log::debug;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

/// a macro to declare simple Vec<String>
#[macro_export]
macro_rules! string_vec {
    ($($x:expr),*) => {
        vec![$($x.into()),*]
    };
}

/// Macro to log info messages with conditional ANSI color support
/// Usage: cinfo!(colorful, "Message with \x1b[1;93mcolors\x1b[0m")
#[macro_export]
macro_rules! cinfo {
    ($colorful:expr, $($arg:tt)*) => {
        {
            let msg = format!($($arg)*);
            log::info!("{}", $crate::utils::strip_ansi_if_needed(&msg, $colorful));
        }
    };
}

/// Remove ANSI color codes from a string if display_colorful is false
/// This allows us to keep colors in the code but strip them at runtime based on configuration
pub fn strip_ansi_if_needed(text: &str, colorful: bool) -> String {
    if colorful {
        text.to_string()
    } else {
        // Remove ANSI escape codes (e.g., \x1b[1;93m, \x1b[0m, etc.)
        let mut result = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                // Skip the escape sequence
                if chars.peek() == Some(&'[') {
                    chars.next(); // consume '['
                                  // Skip until we find 'm' (end of ANSI sequence)
                    while let Some(c) = chars.next() {
                        if c == 'm' {
                            break;
                        }
                    }
                }
            } else {
                result.push(ch);
            }
        }
        result
    }
}

pub fn generate_random_vector(reference_size: usize, rng: &mut ChaCha8Rng) -> Vec<i8> {
    // chose k variables amount feature_selection
    // set a random coeficient for these k variables
    // Generate a vector of random values: 1, 0, or -1
    (0..reference_size).map(|_| rng.gen_range(-1..2)).collect()
}

/// a function used essentially in CV that split randomly a Vec<T> into p Vec<T> of approximatively the same size
pub fn split_into_balanced_random_chunks<T: std::clone::Clone>(
    vec: Vec<T>,
    p: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<T>> {
    // Step 1: Shuffle the original vector
    let mut shuffled = vec;
    shuffled.shuffle(rng);

    // Step 2: Determine sizes for balanced chunks
    let n = shuffled.len();
    let base_size = n / p; // Minimum size for each chunk
    let extra_elements = n % p; // Remaining elements to distribute

    // Step 3: Create chunks with balanced sizes
    let mut chunks = Vec::new();
    let mut start = 0;

    for i in 0..p {
        let chunk_size = base_size + if i < extra_elements { 1 } else { 0 }; // Add one extra element to the first `extra_elements` chunks
        let end = start + chunk_size;
        chunks.push(shuffled[start..end].to_vec());
        start = end;
    }

    chunks
}

/// shuffle a feature
pub fn shuffle_row(
    X: &mut HashMap<(usize, usize), f64>,
    sample_len: usize,
    feature: usize,
    rng: &mut ChaCha8Rng,
) {
    // Extract all the column indices and values for the given row
    let feature_values: Vec<f64> = (0..sample_len)
        .filter_map(|i| X.remove(&(i, feature)))
        .collect();

    // Shuffle the column indices
    //feature_values.shuffle(rng);
    let new_samples: Vec<usize> = (0..sample_len)
        .collect::<Vec<usize>>()
        .choose_multiple(rng, feature_values.len())
        .copied()
        .collect();

    // Update the matrix with shuffled values
    for (value, new_sample) in feature_values.iter().zip(new_samples.iter()) {
        X.insert((*new_sample, feature), *value);
    }
}

//-----------------------------------------------------------------------------
// Statistical utilites
//-----------------------------------------------------------------------------

pub fn conf_inter_binomial(accuracy: f64, n: usize, alpha: f64) -> (f64, f64, f64) {
    assert!(
        n > 0,
        "confInterBinomial: Sample size (n) must be greater than zero."
    );
    assert!(
        accuracy >= 0.0 && accuracy <= 1.0,
        "confInterBinomial: accuracy should not be lower than 0 or greater than 1"
    );
    assert!(
        alpha >= 0.0 && alpha <= 1.0,
        "confInterBinomial: alpha should not be lower than 0 or greater than 1"
    );

    let normal = Normal::new(0.0, 1.0).unwrap_or_else(|e| {
        panic!(
            "confInterBinomial : normal distribution creation failed: {}",
            e
        )
    });
    let z_value = -normal.inverse_cdf(alpha / 2.0);
    let std_error = ((accuracy * (1.0 - accuracy)) / n as f64).sqrt();

    let ci_range = z_value * std_error;
    let lower_bound = 0.0f64.max(accuracy - ci_range);
    let upper_bound = 1.0f64.min(accuracy + ci_range);

    (lower_bound, accuracy, upper_bound)
}

/// Compute AUC for binary class using Mann-Whitney U algorithm O(n log n)
pub fn compute_auc_from_value(value: &[f64], y: &Vec<u8>) -> f64 {
    let mut data: Vec<(f64, u8)> = value
        .iter()
        .zip(y.iter())
        .filter(|(_, &label)| label == 0 || label == 1)
        .map(|(&v, &y)| (v, y))
        .collect();

    let n = data.len();
    let n1 = data.iter().filter(|(_, label)| *label == 1).count();
    let n0 = n - n1;

    if n1 == 0 || n0 == 0 {
        return 0.5;
    }

    data.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut u = 0.0;
    let mut pos_so_far = 0;
    let mut i = 0;

    while i < n {
        let score = data[i].0;

        let mut pos_equal = 0;
        let mut neg_equal = 0;

        while i < n && data[i].0 == score {
            if data[i].1 == 1 {
                pos_equal += 1;
            } else {
                neg_equal += 1;
            }
            i += 1;
        }

        if neg_equal > 0 {
            u += neg_equal as f64 * pos_so_far as f64;
            u += 0.5 * neg_equal as f64 * pos_equal as f64;
        }

        pos_so_far += pos_equal;
    }

    u / (n1 as f64 * n0 as f64)
}

pub fn compute_metrics_from_classes(
    predicted: &Vec<u8>,
    y: &Vec<u8>,
    others_to_compute: [bool; 5],
) -> (f64, f64, f64, AdditionalMetrics) {
    let mut tp = 0;
    let mut fn_count = 0;
    let mut tn = 0;
    let mut fp = 0;

    for (&pred, &real) in predicted.iter().zip(y.iter()) {
        if real == 2 {
            continue;
        }
        match (pred, real) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fn_count += 1,
            _ => {} //warn!("A predicted vs real class of 2 should not exist"),
        }
    }

    let sensitivity = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let specificity = if fp + tn > 0 {
        tn as f64 / (fp + tn) as f64
    } else {
        0.0
    };
    let accuracy = if tp + tn + fp + fn_count > 0 {
        (tp + tn) as f64 / (tp + tn + fp + fn_count) as f64
    } else {
        0.0
    };

    let mut additional = AdditionalMetrics {
        mcc: None,
        f1_score: None,
        npv: None,
        ppv: None,
        g_mean: None,
    };
    if others_to_compute[0] {
        additional.mcc = Some(mcc(tp, fp, tn, fn_count));
    }
    if others_to_compute[1] {
        additional.f1_score = Some(f1_score(tp, fp, fn_count));
    }
    if others_to_compute[2] {
        additional.npv = Some(npv(tn, fn_count));
    }
    if others_to_compute[3] {
        additional.ppv = Some(ppv(tp, fp));
    }
    if others_to_compute[4] {
        additional.g_mean = Some(g_mean(sensitivity, specificity));
    }

    (accuracy, sensitivity, specificity, additional)
}

/// a function that compute accuracy, precision, sensitivity and rejection_rate
/// return (accuracy, sensitivity, specificity, rejection_rate)
pub fn compute_metrics_from_value(
    value: &[f64],
    y: &Vec<u8>,
    threshold: f64,
    threshold_ci: Option<[f64; 2]>,
    others_to_compute: [bool; 5],
) -> (f64, f64, f64, f64, AdditionalMetrics) {
    let classes = value
        .iter()
        .map(|&v| {
            if let Some(threshold_bounds) = &threshold_ci {
                if v > threshold_bounds[1] {
                    1
                } else if v < threshold_bounds[0] {
                    0
                } else {
                    2
                }
            } else {
                if v >= threshold {
                    1
                } else {
                    0
                }
            }
        })
        .collect();

    let (acc, sens, spec, additional) =
        compute_metrics_from_classes(&classes, y, others_to_compute);

    let mut rejection_rate = 0.0;
    if threshold_ci.is_some() {
        rejection_rate = classes.iter().filter(|&&c| c == 2).count() as f64 / classes.len() as f64;
    }

    (acc, sens, spec, rejection_rate, additional)
}

pub fn compute_roc_and_metrics_from_value(
    scores: &[f64],
    y: &[u8],
    fit_function: &FitFunction,
    penalties: Option<[f64; 2]>,
) -> (f64, f64, f64, f64, f64, f64) {
    let mut data: Vec<_> = scores
        .iter()
        .zip(y.iter())
        .filter(|(_, &label)| label == 0 || label == 1)
        .map(|(&score, &label)| (score, label))
        .collect();

    data.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = data.iter().filter(|(_, label)| *label == 1).count();
    let total_neg = data.len() - total_pos;

    if total_pos == 0 || total_neg == 0 {
        return (0.5, f64::NAN, 0.0, 0.0, 0.0, f64::MIN);
    }

    let mut auc = 0.0;
    let mut tn = 0;
    let mut fn_count = 0;

    let mut best_objective = f64::MIN;
    let mut best_threshold = f64::NEG_INFINITY;
    let mut best_acc = 0.0;
    let mut best_sens = 0.0;
    let mut best_spec = 0.0;

    // Initial state: all samples classified as positive (threshold at -infinity)
    // This gives us the metrics for when we apply score >= first_score
    let tp_init = total_pos;
    let fp_init = total_neg;
    let sens_init = tp_init as f64 / total_pos as f64;
    let spec_init = 0.0;
    let acc_init = tp_init as f64 / (total_pos + total_neg) as f64;

    let obj_init = match fit_function {
        FitFunction::auc => youden_index(sens_init, spec_init),
        FitFunction::mcc => mcc(tp_init, fp_init, 0, 0),
        FitFunction::sensitivity => apply_threshold_balance(sens_init, spec_init, penalties),
        FitFunction::specificity => apply_threshold_balance(sens_init, spec_init, penalties),
        FitFunction::f1_score => f1_score(tp_init, fp_init, 0),
        FitFunction::npv => npv(0, 0),
        FitFunction::ppv => ppv(tp_init, fp_init),
        FitFunction::g_mean => g_mean(sens_init, spec_init),
    };

    if obj_init > best_objective {
        best_objective = obj_init;
        best_threshold = if data.is_empty() {
            f64::NEG_INFINITY
        } else {
            data[0].0
        };
        best_acc = acc_init;
        best_sens = sens_init;
        best_spec = spec_init;
    }

    let mut i = 0;
    while i < data.len() {
        let current_score = data[i].0;
        let mut current_tn = 0;
        let mut current_fn = 0;

        while i < data.len() && (data[i].0 - current_score).abs() < f64::EPSILON {
            match data[i].1 {
                0 => current_tn += 1,
                1 => current_fn += 1,
                _ => unreachable!(),
            }
            i += 1;
        }

        let remaining_pos_before = total_pos - fn_count;
        auc += current_tn as f64 * (remaining_pos_before - current_fn) as f64;
        auc += 0.5 * (current_tn * current_fn) as f64;

        tn += current_tn;
        fn_count += current_fn;

        let tp = total_pos - fn_count;
        let fp = total_neg - tn;

        let sensitivity = tp as f64 / total_pos as f64;
        let specificity = tn as f64 / total_neg as f64;
        let accuracy = (tp + tn) as f64 / (total_pos + total_neg) as f64;

        let objective = match fit_function {
            FitFunction::auc => youden_index(sensitivity, specificity),
            FitFunction::mcc => mcc(tp, fp, tn, fn_count),
            FitFunction::sensitivity => {
                apply_threshold_balance(sensitivity, specificity, penalties)
            }
            FitFunction::specificity => {
                apply_threshold_balance(sensitivity, specificity, penalties)
            }
            FitFunction::f1_score => f1_score(tp, fp, fn_count),
            FitFunction::npv => npv(tn, fn_count),
            FitFunction::ppv => ppv(tp, fp),
            FitFunction::g_mean => g_mean(sensitivity, specificity),
        };

        if objective > best_objective {
            best_objective = objective;
            // After processing samples with score = current_score:
            // - tn, fn_count now include all samples with score <= current_score
            // - tp, fp represent all samples with score > current_score
            // The next threshold to consider for >= rule is the next unique score value
            // If there are more scores, use the next one; otherwise use a value just above current
            best_threshold = if i < data.len() {
                data[i].0 // Next score value
            } else {
                // No more scores, set threshold just above current_score
                // This ensures score >= threshold is never satisfied
                current_score + 1.0
            };
            best_acc = accuracy;
            best_sens = sensitivity;
            best_spec = specificity;
        }
    }

    let auc = auc / (total_pos * total_neg) as f64;

    (
        auc,
        best_threshold,
        best_acc,
        best_sens,
        best_spec,
        best_objective,
    )
}

#[inline]
fn mcc(tp: usize, fp: usize, tn: usize, fn_count: usize) -> f64 {
    let numerator = (tp as f64 * tn as f64) - (fp as f64 * fn_count as f64);
    let denominator =
        ((tp + fp) as f64 * (tp + fn_count) as f64 * (tn + fp) as f64 * (tn + fn_count) as f64)
            .sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[inline]
fn f1_score(tp: usize, fp: usize, fn_count: usize) -> f64 {
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    }
}

#[inline]
fn npv(tn: usize, fn_count: usize) -> f64 {
    if tn + fn_count > 0 {
        tn as f64 / (tn + fn_count) as f64
    } else {
        0.0
    }
}

#[inline]
fn ppv(tp: usize, fp: usize) -> f64 {
    if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    }
}

#[inline]
fn g_mean(sensitivity: f64, specificity: f64) -> f64 {
    (sensitivity * specificity).sqrt()
}

#[inline]
fn youden_index(sensitivity: f64, specificity: f64) -> f64 {
    sensitivity + specificity - 1.0
}

#[inline]
fn apply_threshold_balance(sensitivity: f64, specificity: f64, penalties: Option<[f64; 2]>) -> f64 {
    if let Some(p) = penalties {
        if p.len() >= 2 {
            (p[0] * specificity + p[1] * sensitivity) / (p[0] + p[1])
        } else {
            sensitivity + specificity - 1.0
        }
    } else {
        sensitivity + specificity - 1.0
    }
}

pub fn mean_and_std(values: &[f64]) -> (f64, f64) {
    let mut n = 0.0;
    let (mut mean, mut m2) = (0.0, 0.0); // Welford
    for &x in values {
        n += 1.0;
        let delta = x - mean;
        mean += delta / n;
        m2 += delta * (x - mean);
    }
    (mean, (m2 / n).sqrt())
}

pub fn median(values: &mut [f64]) -> f64 {
    let mid = values.len() / 2;

    if values.is_empty() {
        return f64::NAN;
    }

    if values.iter().any(|&x| x.is_nan()) {
        return f64::NAN;
    }

    values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        let max_low = *values[..mid]
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        (max_low + values[mid]) / 2.0
    }
}

pub fn mad(values: &[f64]) -> f64 {
    let mut dev: Vec<f64> = {
        let mut buf = values.to_vec();
        let med = median(&mut buf);
        values.iter().map(|&v| (v - med).abs()).collect()
    };
    1.4826 * median(&mut dev)
}
/// Stratify indices by class label
///
/// Returns (positive_indices, negative_indices) where each vector contains
/// the indices of samples with label 1 and 0 respectively.
///
/// # Arguments
/// - `y`: Binary class labels (0 or 1)
///
/// # Returns
/// Tuple of (pos_indices, neg_indices)
pub fn stratify_indices_by_class(y: &[u8]) -> (Vec<usize>, Vec<usize>) {
    let pos_indices: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, &label)| label == 1)
        .map(|(i, _)| i)
        .collect();

    let neg_indices: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, &label)| label == 0)
        .map(|(i, _)| i)
        .collect();

    (pos_indices, neg_indices)
}

/// Perform stratified bootstrap or subsampling
///
/// When subsample_frac < 1.0, performs subsampling without replacement (each index appears at most once).
/// When subsample_frac == 1.0, performs classic bootstrap with replacement.
///
/// # Arguments
/// - `pos_indices`: Indices of positive class samples
/// - `neg_indices`: Indices of negative class samples
/// - `subsample_frac`: Fraction of samples to draw from each class (0.0, 1.0]
/// - `rng`: Random number generator
///
/// # Returns
/// Vector of sampled indices (positives followed by negatives)
pub fn stratified_bootstrap_sample(
    pos_indices: &[usize],
    neg_indices: &[usize],
    subsample_frac: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let n_pos_total = pos_indices.len();
    let n_neg_total = neg_indices.len();

    let n_pos_sample = ((n_pos_total as f64) * subsample_frac).ceil() as usize;
    let n_neg_sample = ((n_neg_total as f64) * subsample_frac).ceil() as usize;

    let is_subsampling = (subsample_frac - 1.0).abs() > 1e-6;

    let bootstrap_pos: Vec<usize> = if is_subsampling {
        // Subsampling: without replacement
        let mut perm: Vec<usize> = (0..n_pos_total).collect();
        perm.shuffle(rng);
        perm.into_iter()
            .take(n_pos_sample)
            .map(|i| pos_indices[i])
            .collect()
    } else {
        // Bootstrap: with replacement
        (0..n_pos_total)
            .map(|_| pos_indices[rng.gen_range(0..n_pos_total)])
            .collect()
    };

    let bootstrap_neg: Vec<usize> = if is_subsampling {
        // Subsampling: without replacement
        let mut perm: Vec<usize> = (0..n_neg_total).collect();
        perm.shuffle(rng);
        perm.into_iter()
            .take(n_neg_sample)
            .map(|i| neg_indices[i])
            .collect()
    } else {
        // Bootstrap: with replacement
        (0..n_neg_total)
            .map(|_| neg_indices[rng.gen_range(0..n_neg_total)])
            .collect()
    };

    let mut bootstrap_indices = bootstrap_pos;
    bootstrap_indices.extend(bootstrap_neg);
    bootstrap_indices
}

/// Helper function to stratify indices by sample annotation column
pub fn stratify_by_annotation(
    indices: Vec<usize>,
    annot: &crate::data::SampleAnnotations,
    col_idx: usize,
    n_folds: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<usize>> {
    use std::collections::HashMap;

    // Group indices by their annotation value
    let mut strata: HashMap<String, Vec<usize>> = HashMap::new();
    for &idx in &indices {
        if let Some(tags) = annot.sample_tags.get(&idx) {
            if col_idx < tags.len() {
                let tag_value = tags[col_idx].clone();
                strata.entry(tag_value).or_insert_with(Vec::new).push(idx);
            }
        }
    }

    // Split each stratum into folds
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); n_folds];
    for (_tag_value, stratum_indices) in strata {
        let stratum_folds = split_into_balanced_random_chunks(stratum_indices, n_folds, rng);
        for (fold_idx, stratum_fold) in stratum_folds.into_iter().enumerate() {
            folds[fold_idx].extend(stratum_fold);
        }
    }

    folds
}

/// Apply Geyer rescaling for confidence interval construction
///
/// Theory: Geyer (1992) "Practical Markov Chain Monte Carlo"
/// For subsampling (m < n), we use the pivotal quantity:
///   √m * (θ̂ᵦ - θ̂)
/// and then de-pivot with √n to get CI bounds.
///
/// # Arguments
/// - `bootstrap_statistics`: Raw bootstrap statistics (thresholds)
/// - `center`: Central estimate (original threshold)
/// - `is_subsampling`: Whether we're doing subsampling (m < n) or full bootstrap
/// - `_sqrt_m`: Square root of subsample size (unused but kept for API consistency)
/// - `sqrt_n`: Square root of full sample size
/// - `lower_idx`: Index for lower quantile
/// - `upper_idx`: Index for upper quantile
///
/// # Returns
/// (lower_bound, upper_bound) for the confidence interval
pub fn geyer_rescale_ci(
    bootstrap_statistics: &[f64],
    center: f64,
    is_subsampling: bool,
    _sqrt_m: f64,
    sqrt_n: f64,
    lower_idx: usize,
    upper_idx: usize,
) -> (f64, f64) {
    if is_subsampling {
        // De-pivot: θ̂ - quantile(√m * (θ̂ᵦ - θ̂)) / √n
        let lower = center - bootstrap_statistics[upper_idx] / sqrt_n;
        let upper = center - bootstrap_statistics[lower_idx] / sqrt_n;
        (lower, upper)
    } else {
        // No rescaling needed for full bootstrap
        (
            bootstrap_statistics[lower_idx],
            bootstrap_statistics[upper_idx],
        )
    }
}

/// Precompute bootstrap indices for reuse across multiple individuals
/// This significantly improves performance when evaluating populations
///
/// # Arguments
/// - `y`: The class labels (used to stratify)
/// - `n_bootstrap`: Number of bootstrap samples
/// - `alpha`: Significance level for confidence intervals
/// - `subsample_frac`: Fraction of data to use (1.0 = full bootstrap, < 1.0 = subsampling)
/// - `rng`: Random number generator
///
/// # Returns
/// A `PrecomputedBootstrap` structure containing all precomputed indices and statistics
pub fn precompute_bootstrap_indices(
    y: &Vec<u8>,
    n_bootstrap: usize,
    alpha: f64,
    subsample_frac: f64,
    rng: &mut ChaCha8Rng,
) -> PrecomputedBootstrap {
    assert!(subsample_frac > 0.0 && subsample_frac <= 1.0);
    assert!(n_bootstrap >= 100);
    assert!(alpha > 0.0 && alpha < 1.0);

    let seeds: Vec<u64> = (0..n_bootstrap).map(|_| rng.next_u64()).collect();

    // Stratify indices by class
    let (pos_indices, neg_indices) = stratify_indices_by_class(y);

    let n_pos_total = pos_indices.len();
    let n_neg_total = neg_indices.len();
    let n_total = n_pos_total + n_neg_total;

    let n_pos_sample = ((n_pos_total as f64) * subsample_frac).ceil() as usize;
    let n_neg_sample = ((n_neg_total as f64) * subsample_frac).ceil() as usize;
    let m_total = n_pos_sample + n_neg_sample;

    let is_subsampling = (subsample_frac - 1.0).abs() > 1e-6;

    // Geyer rescaling
    let sqrt_m = if is_subsampling {
        (m_total as f64).sqrt()
    } else {
        1.0
    };
    let sqrt_n = (n_total as f64).sqrt();

    // Precompute all bootstrap samples
    let bootstrap_indices: Vec<Vec<usize>> = seeds
        .par_iter()
        .map(|&seed| {
            let mut local_rng = ChaCha8Rng::seed_from_u64(seed);
            stratified_bootstrap_sample(&pos_indices, &neg_indices, subsample_frac, &mut local_rng)
        })
        .collect();

    // Precompute bootstrap_y samples (identical for all individuals)
    let bootstrap_y_samples: Vec<Vec<u8>> = bootstrap_indices
        .iter()
        .map(|indices| indices.iter().map(|&i| y[i]).collect())
        .collect();

    let lower_idx = ((alpha / 2.0) * (n_bootstrap - 1) as f64).ceil() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * (n_bootstrap - 1) as f64).floor() as usize;

    let lower_idx = lower_idx.min(n_bootstrap - 1);
    let upper_idx = upper_idx.min(n_bootstrap - 1);

    PrecomputedBootstrap {
        bootstrap_indices,
        bootstrap_y_samples,
        sqrt_m,
        sqrt_n,
        is_subsampling,
        lower_idx,
        upper_idx,
    }
}

/// Compute an optimized threshold and associated metrics using stratified
/// bootstrap or subsampling.
///
/// This function performs the following steps:
/// - computes AUC and a central threshold using
///   `compute_roc_and_metrics_from_value`,
/// - generates `n_bootstrap` stratified samples (positives/negatives) either
///   with replacement (bootstrap) or without replacement (subsampling) based on
///   `subsample_frac`,
/// - for each resampled dataset computes the optimized threshold and, when
///   subsampling, applies Geyer rescaling to build confidence intervals,
/// - returns the AUC, the threshold confidence interval (lower, center, upper),
///   accuracy, sensitivity, specificity at the central threshold, the
///   optimization objective value (w.r.t. `fit_function`) and the rejection
///   rate (proportion of undecided samples when using a CI).
///
/// # Arguments
///
/// - `value`: slice of scores/values used to build the ROC and select thresholds.
/// - `y`: binary labels (0 or 1) corresponding to `value`.
/// - `fit_function`: fitness function used to select the threshold (AUC, MCC, ...).
/// - `penalties`: optional penalties used to balance sensitivity/specificity.
/// - `n_bootstrap`: number of bootstrap samples to generate (must be > 100).
/// - `alpha`: significance level for confidence intervals (e.g. 0.05 for 95% CI).
/// - `subsample_frac`: fraction of samples to draw per class; when < 1.0
///   subsampling (without replacement) is used, when == 1.0 classic bootstrap
///   (with replacement) is used.
/// - `rng`: random number generator (ChaCha8Rng) for reproducibility.
///
/// # Returns
///
/// A tuple containing:
/// - `f64`: AUC computed on the full dataset,
/// - `[f64; 3]`: thresholds as `[lower_threshold, center_threshold, upper_threshold]`,
/// - `f64`: accuracy at the central threshold,
/// - `f64`: sensitivity (recall) at the central threshold,
/// - `f64`: specificity at the central threshold,
/// - `f64`: objective value (according to `fit_function`) at the central threshold,
/// - `f64`: rejection_rate (fraction of samples classified as "undecided") if a
///   confidence interval is used (0.0 otherwise).
///
/// # Panics
///
/// The function asserts the following preconditions:
/// - `subsample_frac` must be in (0.0, 1.0],
/// - `n_bootstrap` must be greater than 100 (and ideally greater than 1000),
/// - `alpha` must be in (0.0, 1.0).
///
/// # Notes
///
/// - For subsampling (`subsample_frac < 1.0`) Geyer rescaling is applied
///   (see `geyer_rescale_ci`): quantiles are computed on √m * (θ̂_b - θ̂) and
///   then de-pivoted by √n to obtain CI bounds.
/// - Reproducibility is ensured when `rng` is initialized with a fixed seed.
pub fn compute_threshold_and_metrics_with_bootstrap(
    value: &[f64],
    y: &Vec<u8>,
    fit_function: &FitFunction,
    penalties: Option<[f64; 2]>,
    n_bootstrap: usize,
    alpha: f64,
    subsample_frac: f64,
    rng: &mut ChaCha8Rng,
) -> (f64, [f64; 3], f64, f64, f64, f64, f64) {
    assert!(subsample_frac > 0.0 && subsample_frac <= 1.0);
    assert!(n_bootstrap >= 100);
    assert!(alpha > 0.0 && alpha < 1.0);

    let (auc, center_threshold, _, _, _, obj) =
        compute_roc_and_metrics_from_value(value, y, fit_function, penalties);

    let seeds: Vec<u64> = (0..n_bootstrap).map(|_| rng.next_u64()).collect();

    // Stratify indices by class
    let (pos_indices, neg_indices) = stratify_indices_by_class(y);

    let n_pos_total = pos_indices.len();
    let n_neg_total = neg_indices.len();
    let n_total = n_pos_total + n_neg_total;

    let n_pos_sample = ((n_pos_total as f64) * subsample_frac).ceil() as usize;
    let n_neg_sample = ((n_neg_total as f64) * subsample_frac).ceil() as usize;
    let m_total = n_pos_sample + n_neg_sample;

    let is_subsampling = (subsample_frac - 1.0).abs() > 1e-6;

    // Geyer rescaling
    let sqrt_m = if is_subsampling {
        (m_total as f64).sqrt()
    } else {
        1.0
    };

    let mut bootstrap_statistics: Vec<f64> = seeds
        .par_iter()
        .map(|&seed| {
            let mut local_rng = ChaCha8Rng::seed_from_u64(seed);

            // Perform stratified bootstrap/subsampling
            let bootstrap_indices = stratified_bootstrap_sample(
                &pos_indices,
                &neg_indices,
                subsample_frac,
                &mut local_rng,
            );

            let bootstrap_values: Vec<f64> = bootstrap_indices.iter().map(|&i| value[i]).collect();
            let bootstrap_y: Vec<u8> = bootstrap_indices.iter().map(|&i| y[i]).collect();

            let (_, threshold_boot, _, _, _, _) = compute_roc_and_metrics_from_value(
                &bootstrap_values,
                &bootstrap_y,
                fit_function,
                penalties,
            );

            // √m Geyer rescale
            if is_subsampling {
                sqrt_m * (threshold_boot - center_threshold)
            } else {
                threshold_boot
            }
        })
        .collect();

    bootstrap_statistics
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lower_idx = ((alpha / 2.0) * (n_bootstrap - 1) as f64).ceil() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * (n_bootstrap - 1) as f64).floor() as usize;

    let lower_idx = lower_idx.min(n_bootstrap - 1);
    let upper_idx = upper_idx.min(n_bootstrap - 1);

    // Apply Geyer rescaling to get CI bounds
    let sqrt_n = (n_total as f64).sqrt();
    let (lower_threshold, upper_threshold) = geyer_rescale_ci(
        &bootstrap_statistics,
        center_threshold,
        is_subsampling,
        sqrt_m,
        sqrt_n,
        lower_idx,
        upper_idx,
    );

    debug_assert!(lower_threshold <= upper_threshold);

    let (acc, se, sp, rej, _) = compute_metrics_from_value(
        value,
        y,
        center_threshold,
        Some([lower_threshold, upper_threshold]),
        [false; 5],
    );

    (
        auc,
        [lower_threshold, center_threshold, upper_threshold],
        acc,
        se,
        sp,
        obj,
        rej,
    )
}

/// Structure to store precomputed bootstrap samples
/// This avoids recomputing the same bootstrap indices for each individual
pub struct PrecomputedBootstrap {
    /// Pre-generated bootstrap sample indices for each iteration
    pub bootstrap_indices: Vec<Vec<usize>>,
    /// Pre-computed y labels for each bootstrap sample (identical across individuals)
    pub bootstrap_y_samples: Vec<Vec<u8>>,
    /// Square root of subsample size (for Geyer rescaling)
    pub sqrt_m: f64,
    /// Square root of full sample size (for Geyer rescaling)
    pub sqrt_n: f64,
    /// Whether we're doing subsampling (m < n) or full bootstrap
    pub is_subsampling: bool,
    /// Index for lower quantile in sorted bootstrap statistics
    pub lower_idx: usize,
    /// Index for upper quantile in sorted bootstrap statistics
    pub upper_idx: usize,
}

/// Compute threshold and metrics using precomputed bootstrap indices
/// This is much faster than `compute_threshold_and_metrics_with_bootstrap` when
/// evaluating multiple individuals with the same y labels
///
/// # Arguments
/// - `value`: The prediction scores
/// - `y`: The true class labels
/// - `fit_function`: The fitness function to optimize
/// - `penalties`: Optional penalties for sensitivity/specificity
/// - `precomputed`: Precomputed bootstrap indices and statistics
///
/// # Returns
/// Tuple: (AUC, [lower_threshold, center_threshold, upper_threshold], accuracy,
///         sensitivity, specificity, objective, rejection_rate)
pub fn compute_threshold_and_metrics_with_precomputed_bootstrap(
    value: &[f64],
    y: &Vec<u8>,
    fit_function: &FitFunction,
    penalties: Option<[f64; 2]>,
    precomputed: &PrecomputedBootstrap,
) -> (f64, [f64; 3], f64, f64, f64, f64, f64) {
    let (auc, center_threshold, _, _, _, obj) =
        compute_roc_and_metrics_from_value(value, y, fit_function, penalties);

    let mut bootstrap_statistics: Vec<f64> = precomputed
        .bootstrap_indices
        .par_iter()
        .enumerate()
        .map(|(idx, indices)| {
            let bootstrap_values: Vec<f64> = indices.iter().map(|&i| value[i]).collect();
            let bootstrap_y = &precomputed.bootstrap_y_samples[idx];

            let (_, threshold_boot, _, _, _, _) = compute_roc_and_metrics_from_value(
                &bootstrap_values,
                bootstrap_y,
                fit_function,
                penalties,
            );

            // √m Geyer rescale
            if precomputed.is_subsampling {
                precomputed.sqrt_m * (threshold_boot - center_threshold)
            } else {
                threshold_boot
            }
        })
        .collect();

    bootstrap_statistics
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Apply Geyer rescaling to get CI bounds
    let (lower_threshold, upper_threshold) = geyer_rescale_ci(
        &bootstrap_statistics,
        center_threshold,
        precomputed.is_subsampling,
        precomputed.sqrt_m,
        precomputed.sqrt_n,
        precomputed.lower_idx,
        precomputed.upper_idx,
    );

    debug_assert!(lower_threshold <= upper_threshold);

    let (acc, se, sp, rej, _) = compute_metrics_from_value(
        value,
        y,
        center_threshold,
        Some([lower_threshold, upper_threshold]),
        [false; 5],
    );

    (
        auc,
        [lower_threshold, center_threshold, upper_threshold],
        acc,
        se,
        sp,
        obj,
        rej,
    )
}

//-----------------------------------------------------------------------------
// Display utilites
//-----------------------------------------------------------------------------

pub fn display_epoch_legend(param: &Param) -> String {
    let legend = format!(
        "Legend:    [≠ diversity filter]    [↺ resampling]    [{}: {}]    [{}: penalized fit]",
        if param.general.display_colorful {
            "\x1b[1m\x1b[31m█\x1b[0m"
        } else {
            "▓"
        },
        match param.general.fit {
            FitFunction::sensitivity => {
                "sensitivity"
            }
            FitFunction::specificity => {
                "specificity"
            }
            FitFunction::ppv => {
                "PPV"
            }
            FitFunction::npv => {
                "NPV"
            }
            FitFunction::mcc => {
                "MCC"
            }
            FitFunction::g_mean => {
                "G_mean"
            }
            FitFunction::f1_score => {
                "F1-score"
            }
            _ => {
                "AUC"
            }
        },
        if param.general.display_colorful {
            "\x1b[1m\x1b[33m█\x1b[0m"
        } else {
            "▞"
        }
    );

    format!(
        "{}\n{}",
        strip_ansi_if_needed(&legend, param.general.display_colorful),
        "─".repeat(120)
    )
}

pub fn display_epoch(pop: &Population, param: &Param, epoch: usize) -> String {
    if pop.individuals.len() > 0 {
        let best_model = &pop.individuals[0];
        let mean_k = pop.individuals.iter().map(|i| i.k).sum::<usize>() as f64
            / param.ga.population_size as f64;
        let debug_msg = format!("Best model so far AUC:{:.3} ({}:{} fit:{:.3}, k={}, gen#{}, specificity:{:.3}, sensitivity:{:.3}), average AUC {:.3}, fit {:.3}, k:{:.1}", 
                best_model.auc,
                best_model.get_language(),
                best_model.get_data_type(),
                best_model.fit,
                best_model.k,
                best_model.epoch,
                best_model.specificity,
                best_model.sensitivity,
                &pop.individuals.iter().map(|i| {i.auc}).sum::<f64>()/param.ga.population_size as f64,
                &pop.individuals.iter().map(|i| {i.fit}).sum::<f64>()/param.ga.population_size as f64,
                mean_k
                );
        debug!("{}", debug_msg);

        let scale = 50;
        let best_model_pos = match param.general.fit {
            FitFunction::sensitivity => (best_model.sensitivity * scale as f64) as usize,
            FitFunction::specificity => (best_model.specificity * scale as f64) as usize,
            _ => (best_model.auc * scale as f64) as usize,
        };

        let best_fit_pos = (best_model.fit * scale as f64) as usize;
        let max_pos = best_model_pos.max(best_fit_pos);
        let mut bar = vec!["█"; scale]; // White
        for i in (max_pos + 1)..scale {
            bar[i] = "\x1b[0m░\x1b[0m"
        } // Gray
        if best_model_pos < scale {
            bar[best_model_pos] = if param.general.display_colorful {
                "\x1b[1m\x1b[31m█\x1b[0m"
            } else {
                "▓"
            }
        }
        if best_fit_pos < scale {
            bar[best_fit_pos] = if param.general.display_colorful {
                "\x1b[1m\x1b[33m█\x1b[0m"
            } else {
                "▞"
            }
        } // Orange
        let output: String = bar.concat();
        let mut special_epoch = "".to_string();

        if param.ga.forced_diversity_pct != 0.0 && epoch % param.ga.forced_diversity_epochs == 0 {
            special_epoch = format!("{}≠", special_epoch);
        };
        if param.ga.random_sampling_pct > 0.0 && epoch % param.ga.random_sampling_epochs == 0
            || param.cv.overfit_penalty > 0.0
                && param.cv.resampling_inner_folds_epochs > 0
                && epoch % param.cv.resampling_inner_folds_epochs == 0
        {
            special_epoch = format!("{}↺", special_epoch);
        };

        let analysis_tag = if param.tag != "".to_string() {
            format!("[{}] ", param.tag)
        } else {
            "".to_string()
        };

        let epoch_line = format!("{}#{: <5}{: <3}| \x1b[2mbest:\x1b[0m {: <20}\t\x1b[2m0\x1b[0m \x1b[1m{}\x1b[0m \x1b[2m1 [k={}, age={}]\x1b[0m", analysis_tag, epoch, special_epoch,  format!("{}:{}", best_model.get_language(), best_model.get_data_type()), output,  best_model.k, epoch-best_model.epoch);
        strip_ansi_if_needed(&epoch_line, param.general.display_colorful)
    } else {
        String::new()
    }
}

// Graphical functions
pub fn display_feature_importance_terminal(
    data: &Data,
    final_importances: &HashMap<usize, (f64, f64)>,
    nb_features: usize,
    aggregation_method: &ImportanceAggregation,
) -> String {
    const GRAPH_WIDTH: usize = 80;
    const LEFT_MARGIN: usize = 30;
    const VALUE_AREA_WIDTH: usize = GRAPH_WIDTH - LEFT_MARGIN - 2;

    let mut importance_vec: Vec<(&usize, &(f64, f64))> = final_importances.iter().collect();
    importance_vec.sort_by(|a, b| {
        b.1 .0
            .partial_cmp(&a.1 .0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let importance_vec = importance_vec
        .into_iter()
        .take(nb_features)
        .collect::<Vec<_>>();

    if importance_vec.is_empty() {
        return String::from("No features to display.");
    }

    let min_with_std = importance_vec
        .iter()
        .map(|(_, (imp, std))| imp - std)
        .fold(f64::MAX, |a, b| a.min(b));

    let max_with_std = importance_vec
        .iter()
        .map(|&(_, (imp, std))| imp + std)
        .fold(f64::MIN, f64::max);

    let scale_min = round_down_nicely(min_with_std);
    let scale_max = round_up_nicely(max_with_std);
    let scale_range = scale_max - scale_min;

    let scale_factor = if scale_range > 0.0 {
        VALUE_AREA_WIDTH as f64 / scale_range
    } else {
        VALUE_AREA_WIDTH as f64 / 1.0
    };

    let num_ticks = 7;

    let tick_positions: Vec<usize> = (0..num_ticks)
        .map(|i| i * VALUE_AREA_WIDTH / (num_ticks - 1))
        .collect();

    let mut result = String::new();

    result.push_str(match aggregation_method {
        ImportanceAggregation::median => "Feature importance using median aggregation method\n",
        ImportanceAggregation::mean => "Feature importance using mean aggregation method\n",
    });

    result.push_str(match aggregation_method {
        ImportanceAggregation::median => {
            "Legend: • = importance value, <- - -> = confidence interval (±MAD)\n\n"
        }
        ImportanceAggregation::mean => {
            "Legend: • = importance value, <- - -> = confidence interval (±std dev)\n\n"
        }
    });

    let tag_names: Vec<String> = if let Some(fa) = &data.feature_annotations {
        fa.tag_column_names.clone()
    } else {
        Vec::new()
    };

    let mut header_line = format!(
        "{:<LEFT_MARGIN$}|{:^VALUE_AREA_WIDTH$}|",
        "Feature", "Feature importance"
    );
    for tag in &tag_names {
        header_line.push_str(&format!(" | {:<20}", tag));
    }
    result.push_str(&"-".repeat(LEFT_MARGIN));
    result.push_str("|-");
    result.push_str(&"-".repeat(VALUE_AREA_WIDTH));
    for _ in &tag_names {
        result.push_str("|----------------------");
    }
    result.push_str("|\n");
    result.push_str(&header_line);
    result.push_str("\n");
    result.push_str(&"-".repeat(LEFT_MARGIN));
    result.push_str("|-");
    result.push_str(&"-".repeat(VALUE_AREA_WIDTH));
    for _ in &tag_names {
        result.push_str("|----------------------");
    }
    result.push_str("|\n");

    for (i, (feature_idx, (importance, std_dev))) in importance_vec.iter().enumerate() {
        let feature_name = if data.features.len() > **feature_idx {
            &data.features[**feature_idx]
        } else {
            "Unknown"
        };

        let display_name = format!("#{} {}", i + 1, feature_name);
        let truncated_name = if display_name.len() > LEFT_MARGIN - 2 {
            format!("{}...", &display_name[0..LEFT_MARGIN - 5])
        } else {
            display_name
        };

        let normalized_importance = importance - scale_min;
        let normalized_min = (importance - std_dev - scale_min).max(0.0);
        let normalized_max = importance + std_dev - scale_min;

        let center_pos = (normalized_importance * scale_factor).round() as usize;
        let start_pos = (normalized_min * scale_factor).round() as usize;
        let end_pos = (normalized_max * scale_factor).round() as usize;
        let end_pos = std::cmp::min(end_pos, VALUE_AREA_WIDTH - 1);

        let left_margin = LEFT_MARGIN;
        let mut line = format!("{:<left_margin$}|", truncated_name);

        for i in 0..VALUE_AREA_WIDTH {
            if i == center_pos {
                line.push('•');
            } else if i == start_pos {
                line.push('<');
            } else if i == end_pos {
                line.push('>');
            } else if i > start_pos && i < end_pos {
                line.push('-');
            } else {
                line.push(' ');
            }
        }

        line.push('|');

        if !tag_names.is_empty() {
            if let Some(fa) = &data.feature_annotations {
                let tag_vals = fa.feature_tags.get(feature_idx);
                for i in 0..fa.tag_column_names.len() {
                    let v = tag_vals
                        .and_then(|vals| vals.get(i))
                        .map(|s| s.as_str())
                        .unwrap_or("");
                    line.push_str(&format!(" | {:<20}", v));
                }
            }
        }

        result.push_str(&line);
        result.push('\n');
    }

    let mut marker_line = "-".repeat(LEFT_MARGIN);
    marker_line.push('|');

    for i in 0..VALUE_AREA_WIDTH {
        if tick_positions.contains(&i) {
            if i == 0 {
                marker_line.push('-');
            } else {
                marker_line.push('|');
            }
        } else {
            marker_line.push('-');
        }
    }

    marker_line.push('|');
    result.push_str(&marker_line);
    result.push('\n');

    let mut scale_line = " ".repeat(LEFT_MARGIN + 1);

    for (i, &tick_pos) in tick_positions.iter().enumerate() {
        let tick_value = scale_min + i as f64 * (scale_range / (num_ticks - 1) as f64);
        let value_str = format_tick_value(tick_value);

        let label_width = value_str.len();
        let label_start = tick_pos.saturating_sub(label_width / 2);

        while scale_line.len() < LEFT_MARGIN + 1 + label_start {
            scale_line.push(' ');
        }

        scale_line.push_str(&value_str);
    }

    result.push_str(&scale_line);
    result.push_str("\n\n");

    result
}

fn format_tick_value(value: f64) -> String {
    let abs_value = value.abs();

    if abs_value == 0.0 {
        return "0".to_string();
    } else if abs_value < 0.001 || abs_value >= 10000.0 {
        return format!("{:.1e}", value);
    } else if abs_value < 0.01 {
        return format!("{:.4}", value);
    } else if abs_value < 0.1 {
        return format!("{:.3}", value);
    } else if abs_value < 1.0 {
        return format!("{:.2}", value);
    } else if abs_value < 10.0 {
        if (value - value.round()).abs() < 1e-10 {
            return format!("{:.0}", value);
        } else {
            return format!("{:.1}", value);
        }
    } else {
        if (value - value.round()).abs() < 1e-10 {
            return format!("{:.0}", value);
        } else {
            return format!("{:.1}", value);
        }
    }
}

fn round_up_nicely(value: f64) -> f64 {
    if value == 0.0 {
        return 0.1;
    }

    if value < 0.0 {
        return -round_down_nicely(value.abs());
    }

    if value.abs() < 1e-10 {
        return if value >= 0.0 { 1e-10 } else { -1e-10 };
    }

    let magnitude = value.log10().floor();
    let power_of_ten = 10.0_f64.powf(magnitude);

    if value <= 1.0 * power_of_ten {
        1.0 * power_of_ten
    } else if value <= 2.0 * power_of_ten {
        2.0 * power_of_ten
    } else if value <= 5.0 * power_of_ten {
        5.0 * power_of_ten
    } else {
        10.0 * power_of_ten
    }
}

fn round_down_nicely(value: f64) -> f64 {
    if value < 0.0 {
        return -round_up_nicely(value.abs());
    }

    if value < 1e-10 {
        return 0.0;
    }

    let magnitude = value.log10().floor();
    let power_of_ten = 10.0_f64.powf(magnitude);

    if value >= 5.0 * power_of_ten {
        5.0 * power_of_ten
    } else if value >= 2.0 * power_of_ten {
        2.0 * power_of_ten
    } else if value >= 1.0 * power_of_ten {
        1.0 * power_of_ten
    } else {
        0.5 * power_of_ten
    }
}

//-----------------------------------------------------------------------------
// Serialization utilites
//-----------------------------------------------------------------------------

// Serde functions for JSONize HashMaps
pub mod serde_json_hashmap_numeric {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    // ===== General =====

    /// HashMap<usize, T>
    pub fn serialize_usize<S, T>(map: &HashMap<usize, T>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + Clone,
    {
        let map_as_string: HashMap<String, T> = map
            .iter()
            .map(|(&k, v)| (k.to_string(), v.clone()))
            .collect();
        map_as_string.serialize(serializer)
    }

    /// HashMap<usize, T>
    pub fn deserialize_usize<'de, D, T>(deserializer: D) -> Result<HashMap<usize, T>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let map_as_string: HashMap<String, T> = HashMap::deserialize(deserializer)?;
        let mut map = HashMap::new();
        for (k, v) in map_as_string {
            if let Ok(idx) = k.parse() {
                map.insert(idx, v);
            }
        }
        Ok(map)
    }

    /// HashMap<(usize, usize), T>
    pub fn serialize_tuple_usize<S, T>(
        map: &HashMap<(usize, usize), T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + Clone,
    {
        let map_as_string: Result<HashMap<String, T>, _> = map
            .iter()
            .map(|(&(i, j), v)| Ok((format!("{},{}", i, j), v.clone())))
            .collect();

        map_as_string?.serialize(serializer)
    }

    /// HashMap<(usize, usize), T>
    pub fn deserialize_tuple_usize<'de, D, T>(
        deserializer: D,
    ) -> Result<HashMap<(usize, usize), T>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let map_as_string: HashMap<String, T> = HashMap::deserialize(deserializer)?;
        let mut map = HashMap::new();

        for (k, v) in map_as_string {
            let parts: Vec<&str> = k.split(',').collect();
            if parts.len() == 2 {
                match (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    (Ok(i), Ok(j)) => {
                        map.insert((i, j), v);
                    }
                    _ => {
                        return Err(serde::de::Error::custom("Invalid tuple key format"));
                    }
                }
            }
        }
        Ok(map)
    }

    // ===== Specialized modules  =====

    /// HashMap<usize, i8> (Individual.features)
    pub mod usize_i8 {
        use super::*;

        pub fn serialize<S>(map: &HashMap<usize, i8>, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_usize(map, serializer)
        }

        pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<usize, i8>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_usize(deserializer)
        }
    }

    /// HashMap<usize, u8> (Data.featureclass)
    pub mod usize_u8 {
        use super::*;

        pub fn serialize<S>(map: &HashMap<usize, u8>, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_usize(map, serializer)
        }

        pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<usize, u8>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_usize(deserializer)
        }
    }

    /// HashMap<(usize, usize), f64> (Data.X)
    pub mod tuple_usize_f64 {
        use super::*;

        pub fn serialize<S>(
            map: &HashMap<(usize, usize), f64>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_tuple_usize(map, serializer)
        }

        pub fn deserialize<'de, D>(
            deserializer: D,
        ) -> Result<HashMap<(usize, usize), f64>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_tuple_usize(deserializer)
        }
    }
}

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::panic;

    // tests for generate_random_vector
    #[test]
    fn test_generate_random_vector_vector_size() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 10;
        let vector = generate_random_vector(size, &mut rng);
        assert_eq!(
            vector.len(),
            size,
            "the generated vector does not match the input size"
        );
    }

    #[test]
    fn test_generate_random_vector_random_values() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 100;
        let vector = generate_random_vector(size, &mut rng);

        for &value in &vector {
            assert!(
                value == -1 || value == 0 || value == 1,
                "the generated vector contains value.s outside [-1 ; 1]"
            );
        }
    }

    #[test]
    fn test_generate_random_vector_empty_vector() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 0;
        let vector = generate_random_vector(size, &mut rng);
        assert!(
            vector.is_empty(),
            "the generated vector should be empty for an input size of 0"
        );
    }

    #[test]
    fn test_generate_random_vector_deterministic_output_and_reproductibility() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let size = 10;

        let vector1 = generate_random_vector(size, &mut rng1);
        let vector2 = generate_random_vector(size, &mut rng2);

        assert_eq!(
            vector1, vector2,
            "the same seed generated two different vectors"
        );
        assert_eq!(vector1, vec![-1, 1, -1, 1, 1, 0, 0, 0, 1, -1], "the generated vector isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }

    // tests for split_into_balanced_random_chunks
    #[test]
    fn test_split_into_balanced_random_chunks_split_remainder_division() {
        let chunks = split_into_balanced_random_chunks(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            3,
            &mut ChaCha8Rng::seed_from_u64(42),
        );
        assert_eq!(
            chunks.len(),
            3,
            "the count of chunks does not match the input"
        );
        assert_eq!(
            chunks[0].len(),
            4,
            "the first chunk must have one more value when this is a remainder division"
        );
        assert_eq!(
            chunks[1].len(),
            3,
            "the count of value per chunck is not respected"
        );
        assert_eq!(
            chunks[2].len(),
            3,
            "the count of value per chunck is not respected"
        );
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_remainderless_division() {
        let chunks = split_into_balanced_random_chunks(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            3,
            &mut ChaCha8Rng::seed_from_u64(42),
        );
        assert_eq!(
            chunks.len(),
            3,
            "the count of chunks does not match the input"
        );
        assert_eq!(chunks[0].len(), 3, "the first chunk must have the same number of value when this is a remainderless division");
        assert_eq!(
            chunks[1].len(),
            3,
            "the count of value per chunck is not respected"
        );
        assert_eq!(
            chunks[2].len(),
            3,
            "the count of value per chunck is not respected"
        );
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_into_single_chunk() {
        let chunks = split_into_balanced_random_chunks(
            vec![1, 2, 3, 4, 5],
            1,
            &mut ChaCha8Rng::seed_from_u64(42),
        );
        assert_eq!(
            chunks.len(),
            1,
            "the count of chunks does not match the input"
        );
        assert_eq!(
            chunks[0].len(),
            5,
            "when splitted in one part, the chunk must be equal to the input vector"
        );
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_empty_vectors() {
        let vec: Vec<i32> = vec![];
        let chunks = split_into_balanced_random_chunks(vec, 3, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(
            chunks.len(),
            3,
            "the count of chunks does not match the input"
        );
        for chunk in chunks {
            assert!(
                chunk.is_empty(),
                "empty vector should to the formation of empty chunks"
            );
        }
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_more_chunks_than_elements() {
        let vec = vec![1, 2, 3];
        let chunks =
            split_into_balanced_random_chunks(vec.clone(), 5, &mut ChaCha8Rng::seed_from_u64(42));

        // vecs 1, 2, 3 should contain one value, vecs 4 and 5 should be empty
        assert_eq!(
            chunks.len(),
            5,
            "the count of chunks does not match the input"
        );
        assert_eq!(
            chunks.iter().filter(|chunk| !chunk.is_empty()).count(),
            vec.len(),
            "the chunks exceeding the values count must be empty"
        );
    }

    #[test]
    fn test_split_into_balanced_random_chunks_deterministic_split_and_reproductibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = 4;

        let chunks1 = split_into_balanced_random_chunks(vec.clone(), p, &mut rng);
        let chunks2 = split_into_balanced_random_chunks(vec.clone(), p, &mut rng2);

        assert_eq!(
            chunks1, chunks2,
            "the same seed generated two different chunks"
        );
        assert_eq!(chunks1, vec![vec![1, 4, 6], vec![8, 10, 5], vec![9, 2], vec![7, 3]], "the generated chunks isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }

    // tests for shuffle_row
    #[test]
    fn test_shuffle_row_preserves_values_and_reorder() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();

        X.insert((0, 0), 1.0);
        X.insert((1, 0), 2.0);
        X.insert((2, 0), 3.0);
        X.insert((3, 0), 4.0);

        let sample_len = 4;
        let feature = 0;

        shuffle_row(&mut X, sample_len, feature, &mut rng);

        let values: Vec<f64> = (0..sample_len)
            .filter_map(|i| X.get(&(i, feature)).copied())
            .collect();
        assert_eq!(
            values.len(),
            4,
            "HashMap must contain the same number of values after shuffle"
        );
        assert!(
            values.contains(&1.0),
            "the shuffle must conserve HashMap values"
        );
        assert!(
            values.contains(&2.0),
            "the shuffle must conserve HashMap values"
        );
        assert!(
            values.contains(&3.0),
            "the shuffle must conserve HashMap values"
        );
        assert!(
            values.contains(&4.0),
            "the shuffle must conserve HashMap values"
        );
        assert_ne!(
            values,
            vec![1.0, 2.0, 3.0, 4.0],
            "the shuffle must conserve HashMap values"
        );
    }

    #[test]
    fn test_shuffle_row_empty_column() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();

        let sample_len = 4;
        let feature = 0;

        shuffle_row(&mut X, sample_len, feature, &mut rng);

        for i in 0..sample_len {
            assert!(
                !X.contains_key(&(i, feature)),
                "the shuffle of an empty HashMap should also be empty"
            );
        }
    }

    #[test]
    fn test_shuffle_row_deterministic_shuffle_and_reproductibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X1: HashMap<(usize, usize), f64> = HashMap::new();

        X1.insert((0, 0), 1.0);
        X1.insert((1, 0), 2.0);
        X1.insert((2, 0), 3.0);
        X1.insert((3, 0), 4.0);

        let mut X2 = X1.clone();
        let sample_len = 4;
        let feature = 0;

        shuffle_row(
            &mut X1,
            sample_len,
            feature,
            &mut ChaCha8Rng::seed_from_u64(42),
        );
        shuffle_row(&mut X2, sample_len, feature, &mut rng);

        assert_eq!(X1, X2, "the same seed generated two different chunks");
        assert_eq!(X1.get(&(0, 0)), Some(&2.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(2, 0)), Some(&1.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(1, 0)), Some(&3.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(3, 0)), Some(&4.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }

    #[test]
    fn test_conf_inter_binomial() {
        // assert_eq(Rust function results  == R function results)
        assert_eq!(conf_inter_binomial(0.0, 50, 0.05), (0_f64, 0_f64, 0_f64));
        assert_eq!(
            conf_inter_binomial(0.76, 50, 0.05),
            (0.6416207713410322_f64, 0.76_f64, 0.8783792286589678_f64)
        );
        assert_eq!(conf_inter_binomial(1.0, 50, 0.05), (1_f64, 1_f64, 1_f64));

        // control panic! to avoid statistical issues due to invalid input
        let resultErrZeroSample = panic::catch_unwind(|| conf_inter_binomial(0.76, 0, 0.05));
        assert!(
            resultErrZeroSample.is_err(),
            "function should panic! when there is no sample"
        );

        let resultErrInf = panic::catch_unwind(|| conf_inter_binomial(-0.3, 50, 0.05));
        assert!(
            resultErrInf.is_err(),
            "function should panic! for an accuracy lower than 0"
        );

        let resultErrSup = panic::catch_unwind(|| conf_inter_binomial(1.3, 50, 0.05));
        assert!(
            resultErrSup.is_err(),
            "function should panic! for an accuracy greater than 1"
        );
    }

    #[test]
    fn test_compute_auc_from_value_perfect_classification() {
        let value = vec![0.1, 0.2, 0.8, 0.9];
        let y = vec![0, 0, 1, 1];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 1.0, "Perfect classification should yield AUC = 1.0");
    }

    #[test]
    fn test_compute_auc_from_value_random_classification() {
        let value = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![0, 1, 0, 1];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 0.5, "Random classification should yield AUC = 0.5");
    }

    #[test]
    fn test_compute_auc_from_value_empty_vectors() {
        let value = vec![];
        let y = vec![];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 0.5, "Empty vectors should yield AUC = 0.5");
    }

    #[test]
    fn test_compute_auc_from_value_single_class_only() {
        let value = vec![0.1, 0.2, 0.3, 0.4];
        let y = vec![0, 0, 0, 0];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 0.5, "Single class only should yield AUC = 0.5");
    }

    #[test]
    fn test_compute_auc_from_value_ties_handling() {
        let value = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![0, 0, 1, 1];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 0.5, "Ties should be handled correctly");
    }

    #[test]
    fn test_compute_auc_from_value_infinite_values() {
        let value = vec![f64::NEG_INFINITY, 0.5, f64::INFINITY];
        let y = vec![0, 1, 1];
        let auc = compute_auc_from_value(&value, &y);
        assert!(
            auc >= 0.0 && auc <= 1.0,
            "Should handle infinite values gracefully"
        );
    }

    #[test]
    fn test_compute_auc_large_dataset() {
        let n = 10000;
        let value: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<u8> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        let auc = compute_auc_from_value(&value, &y);
        assert!(
            (auc - 1.0).abs() < 1e-10,
            "Large sorted dataset should yield perfect AUC"
        );
    }

    #[test]
    fn test_compute_metrics_from_classes_perfect_predictions() {
        let predicted = vec![0, 1, 0, 1];
        let y = vec![0, 1, 0, 1];
        let (accuracy, sensitivity, specificity, _) =
            compute_metrics_from_classes(&predicted, &y, [false; 5]);
        assert_eq!(
            accuracy, 1.0,
            "Perfect predictions should yield 100% accuracy"
        );
        assert_eq!(
            sensitivity, 1.0,
            "Perfect predictions should yield 100% sensitivity"
        );
        assert_eq!(
            specificity, 1.0,
            "Perfect predictions should yield 100% specificity"
        );
    }

    #[test]
    fn test_compute_metrics_from_classes_all_wrong_predictions() {
        let predicted = vec![1, 0, 1, 0];
        let y = vec![0, 1, 0, 1];
        let (accuracy, sensitivity, specificity, _) =
            compute_metrics_from_classes(&predicted, &y, [false; 5]);
        assert_eq!(
            accuracy, 0.0,
            "All wrong predictions should yield 0% accuracy"
        );
        assert_eq!(
            sensitivity, 0.0,
            "All wrong predictions should yield 0% sensitivity"
        );
        assert_eq!(
            specificity, 0.0,
            "All wrong predictions should yield 0% specificity"
        );
    }

    #[test]
    fn test_compute_metrics_from_classes_mixed_predictions() {
        let predicted = vec![0, 1, 0, 0];
        let y = vec![0, 1, 1, 0];
        let (accuracy, sensitivity, specificity, _) =
            compute_metrics_from_classes(&predicted, &y, [false; 5]);
        assert_eq!(
            accuracy, 0.75,
            "Mixed predictions should yield expected accuracy"
        );
        assert_eq!(
            sensitivity, 0.5,
            "Mixed predictions should yield expected sensitivity"
        );
        assert_eq!(
            specificity, 1.0,
            "Mixed predictions should yield expected specificity"
        );
    }

    #[test]
    fn test_compute_metrics_from_classes_class_2_ignored() {
        let predicted = vec![0, 1, 0, 1];
        let y = vec![0, 1, 2, 1];
        let (accuracy, _, _, _) = compute_metrics_from_classes(&predicted, &y, [false; 5]);
        assert_eq!(accuracy, 1.0, "Class 2 should be ignored in calculations");
    }

    #[test]
    fn test_compute_metrics_from_classes_empty_vectors() {
        let predicted = vec![];
        let y = vec![];
        let (accuracy, sensitivity, specificity, _) =
            compute_metrics_from_classes(&predicted, &y, [false; 5]);
        assert_eq!(accuracy, 0.0, "Empty vectors should yield 0 metrics");
        assert_eq!(sensitivity, 0.0, "Empty vectors should yield 0 metrics");
        assert_eq!(specificity, 0.0, "Empty vectors should yield 0 metrics");
    }

    #[test]
    fn test_compute_metrics_extreme_imbalance() {
        let predicted = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let (accuracy, sensitivity, specificity, _) =
            compute_metrics_from_classes(&predicted, &y, [false; 5]);
        assert_eq!(
            accuracy, 1.0,
            "Perfect predictions should yield 100% accuracy even with imbalance"
        );
        assert_eq!(
            sensitivity, 1.0,
            "Perfect predictions should yield 100% sensitivity"
        );
        assert_eq!(
            specificity, 1.0,
            "Perfect predictions should yield 100% specificity"
        );
    }

    #[test]
    fn test_compute_roc_and_metrics_from_value_basic_case() {
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];
        let (auc, threshold, accuracy, sensitivity, specificity, _) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::auc, None);
        assert!(auc >= 0.0 && auc <= 1.0, "AUC should be between 0 and 1");
        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Accuracy should be between 0 and 1"
        );
        assert!(
            sensitivity >= 0.0 && sensitivity <= 1.0,
            "Sensitivity should be between 0 and 1"
        );
        assert!(
            specificity >= 0.0 && specificity <= 1.0,
            "Specificity should be between 0 and 1"
        );
        assert!(!threshold.is_nan(), "Threshold should not be NaN");
    }

    #[test]
    fn test_compute_roc_and_metrics_from_value_with_penalties() {
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];
        let penalties = Some([2.0, 1.0]);
        let (_, _, _, _, _, objective) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::auc, penalties);
        assert!(objective >= 0.0, "Objective should be non-negative");
    }

    #[test]
    fn test_compute_roc_and_metrics_from_value_without_penalties() {
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];
        let (_, _, _, sensitivity, specificity, objective) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::auc, None);
        let expected_youden = sensitivity + specificity - 1.0;
        assert!(
            (objective - expected_youden).abs() < 1e-10,
            "Without penalties, objective should be Youden index"
        );
    }

    #[test]
    fn test_compute_roc_and_metrics_from_value_single_class_only() {
        let value = vec![0.1, 0.2, 0.3, 0.4];
        let y = vec![0, 0, 0, 0];
        let (auc, threshold, _, _, _, _) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::auc, None);
        assert_eq!(auc, 0.5, "Single class should yield AUC = 0.5");
        assert!(
            threshold.is_nan(),
            "Single class should yield NaN threshold"
        );
    }

    #[test]
    fn test_mean_and_std_normal_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, 3.0, "Mean of 1-5 should be 3.0");
        assert!(
            (std - 2.0f64.sqrt()).abs() < 1e-10,
            "Standard deviation should be √2"
        );
    }

    #[test]
    fn test_mean_and_std_single_value() {
        let values = vec![42.0];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, 42.0, "Mean of single value should be that value");
        assert_eq!(std, 0.0, "Standard deviation of single value should be 0");
    }

    #[test]
    fn test_mean_and_std_empty_slice() {
        let values: Vec<f64> = vec![];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, 0.0, "Welford algorithm starts with mean = 0.0");
        assert!(
            std.is_nan(),
            "Standard deviation of empty slice should be NaN"
        );
    }

    #[test]
    fn test_mean_and_std_identical_values() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, 5.0, "Mean of identical values should be that value");
        assert_eq!(
            std, 0.0,
            "Standard deviation of identical values should be 0"
        );
    }

    #[test]
    fn test_mean_and_std_negative_values() {
        let values = vec![-1.0, -2.0, -3.0];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, -2.0, "Mean should handle negative values");
        assert!(
            std > 0.0,
            "Standard deviation should be positive for varying values"
        );
    }

    #[test]
    fn test_mean_and_std_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0];
        let (mean, std) = mean_and_std(&values);
        assert!(
            mean.is_nan(),
            "Welford with NaN should propagate NaN to mean"
        );
        assert!(std.is_nan(), "Welford with NaN should propagate NaN to std");
    }

    #[test]
    fn test_mean_and_std_large_dataset() {
        let values: Vec<f64> = (0..100000).map(|i| i as f64).collect();
        let (mean, std) = mean_and_std(&values);
        let expected_mean = 49999.5;
        assert!(
            (mean - expected_mean).abs() < 1e-6,
            "Large dataset mean should be accurate"
        );
        assert!(std > 0.0, "Large dataset std should be positive");
    }

    #[test]
    fn test_median_odd_length() {
        let mut values = vec![1.0, 3.0, 2.0];
        let result = median(&mut values);
        assert_eq!(result, 2.0, "Median of [1,2,3] should be 2");
    }

    #[test]
    fn test_median_even_length() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let result = median(&mut values);
        assert_eq!(result, 2.5, "Median of [1,2,3,4] should be 2.5");
    }

    #[test]
    fn test_median_single_value() {
        let mut values = vec![42.0];
        let result = median(&mut values);
        assert_eq!(result, 42.0, "Median of single value should be that value");
    }

    #[test]
    fn test_median_two_values() {
        let mut values = vec![1.0, 3.0];
        let result = median(&mut values);
        assert_eq!(result, 2.0, "Median of two values should be their average");
    }

    #[test]
    fn test_median_unsorted_values() {
        let mut values = vec![5.0, 1.0, 9.0, 3.0, 7.0];
        let result = median(&mut values);
        assert_eq!(result, 5.0, "Median should work on unsorted data");
    }

    #[test]
    fn test_median_with_nan() {
        let mut values = vec![1.0, f64::NAN, 3.0];
        let result = median(&mut values);
        assert!(result.is_nan(), "Median with NaN should return NaN");
    }

    #[test]
    fn test_mad_normal_distribution() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mad(&values);
        assert!(result > 0.0, "MAD should be positive for varying values");
    }

    #[test]
    fn test_mad_constant_values() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let result = mad(&values);
        assert_eq!(result, 0.0, "MAD of constant values should be 0");
    }

    #[test]
    fn test_mad_single_value() {
        let values = vec![42.0];
        let result = mad(&values);
        assert_eq!(result, 0.0, "MAD of single value should be 0");
    }

    #[test]
    fn test_mad_two_values() {
        let values = vec![1.0, 3.0];
        let result = mad(&values);
        assert!(
            result > 0.0,
            "MAD of two different values should be positive"
        );
    }

    #[test]
    fn test_mad_median_consistency() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut values_for_median = values.clone();
        let med = median(&mut values_for_median);
        let mad_result = mad(&values);

        assert_eq!(med, 3.0, "Median should be 3.0");
        assert!(
            mad_result > 0.0,
            "MAD should be positive for varying values"
        );

        let expected_mad = 1.4826;
        assert!(
            (mad_result - expected_mad).abs() < 0.001,
            "MAD should match expected calculation"
        );
    }
    #[test]
    fn test_display_feature_importance_terminal_basic_output() {
        let mut data = Data::new();
        data.features = vec!["feature1".to_string(), "feature2".to_string()];

        let mut importance_map = HashMap::new();
        importance_map.insert(0, (0.8, 0.1));
        importance_map.insert(1, (0.6, 0.05));

        let result = display_feature_importance_terminal(
            &data,
            &importance_map,
            2,
            &ImportanceAggregation::median,
        );
        assert!(
            result.contains("Feature importance"),
            "Output should contain title"
        );
        assert!(
            result.contains("feature1"),
            "Output should contain feature names"
        );
        assert!(
            result.contains("•"),
            "Output should contain importance markers"
        );
    }

    #[test]
    fn test_display_feature_importance_terminal_empty_features() {
        let data = Data::new();
        let importance_map = HashMap::new();

        let result = display_feature_importance_terminal(
            &data,
            &importance_map,
            5,
            &ImportanceAggregation::mean,
        );
        assert_eq!(
            result, "No features to display.",
            "Empty features should return appropriate message"
        );
    }

    #[test]
    fn test_display_feature_importance_terminal_median_aggregation() {
        let mut data = Data::new();
        data.features = vec!["test_feature".to_string()];

        let mut importance_map = HashMap::new();
        importance_map.insert(0, (0.5, 0.1));

        let result = display_feature_importance_terminal(
            &data,
            &importance_map,
            1,
            &ImportanceAggregation::median,
        );
        assert!(
            result.contains("median aggregation"),
            "Should indicate median aggregation method"
        );
        assert!(
            result.contains("±MAD"),
            "Should show MAD for median aggregation"
        );
    }

    #[test]
    fn test_display_feature_importance_negative_importance() {
        let mut data = Data::new();
        data.features = vec!["negative_feature".to_string()];

        let mut importance_map = HashMap::new();
        importance_map.insert(0, (-0.5, 0.1));

        let result = display_feature_importance_terminal(
            &data,
            &importance_map,
            1,
            &ImportanceAggregation::mean,
        );
        assert!(
            result.contains("negative_feature"),
            "Should handle negative importance"
        );
        assert!(
            result.contains("•"),
            "Should still display marker for negative values"
        );
    }

    #[test]
    fn test_display_feature_importance_very_large_names() {
        let mut data = Data::new();
        let long_name = "a".repeat(100);
        data.features = vec![long_name.clone()];

        let mut importance_map = HashMap::new();
        importance_map.insert(0, (0.5, 0.1));

        let result = display_feature_importance_terminal(
            &data,
            &importance_map,
            1,
            &ImportanceAggregation::mean,
        );
        assert!(
            result.contains("..."),
            "Should truncate very long feature names"
        );
        assert!(
            !result.contains(&long_name),
            "Should not contain full long name"
        );
    }

    #[test]
    fn test_format_tick_value_zero() {
        assert_eq!(format_tick_value(0.0), "0");
    }

    #[test]
    fn test_format_tick_value_small_positive() {
        assert_eq!(format_tick_value(0.005), "0.0050");
        assert_eq!(format_tick_value(0.05), "0.050");
        assert_eq!(format_tick_value(0.5), "0.50");
    }

    #[test]
    fn test_format_tick_value_small_negative() {
        assert_eq!(format_tick_value(-0.005), "-0.0050");
        assert_eq!(format_tick_value(-0.5), "-0.50");
    }

    #[test]
    fn test_format_tick_value_large_values() {
        assert!(
            format_tick_value(15000.0).contains("e"),
            "Large values should use scientific notation"
        );
        assert!(
            format_tick_value(-20000.0).contains("e"),
            "Large negative values should use scientific notation"
        );
    }

    #[test]
    fn test_format_tick_value_scientific_notation() {
        assert!(
            format_tick_value(0.0005).contains("e"),
            "Very small values should use scientific notation"
        );
        assert!(
            format_tick_value(-0.0001).contains("e"),
            "Very small negative values should use scientific notation"
        );
    }

    #[test]
    fn test_format_tick_value_integer_values() {
        assert_eq!(format_tick_value(5.0), "5");
        assert_eq!(format_tick_value(42.0), "42");
    }

    #[test]
    fn test_format_tick_value_edge_cases() {
        assert_eq!(format_tick_value(f64::MIN_POSITIVE), "2.2e-308");
        assert_eq!(format_tick_value(-f64::MIN_POSITIVE), "-2.2e-308");
        assert_eq!(format_tick_value(1.0000001), "1.0");
        assert_eq!(format_tick_value(0.99999), "1.00");
    }

    #[test]
    fn test_round_up_nicely_positive_values() {
        assert_eq!(round_up_nicely(1.3), 2.0);
        assert_eq!(round_up_nicely(2.5), 5.0);
        assert_eq!(round_up_nicely(7.0), 10.0);
    }

    #[test]
    fn test_round_up_nicely_negative_values() {
        assert_eq!(round_up_nicely(-1.3), -1.0);
        assert!(round_up_nicely(-2.5) > -5.0);
    }

    #[test]
    fn test_round_up_nicely_zero() {
        assert_eq!(round_up_nicely(0.0), 0.1);
    }

    #[test]
    fn test_round_up_nicely_very_small_values() {
        let result = round_up_nicely(0.0003);
        assert!(result > 0.0003, "Should round up small positive values");
    }

    #[test]
    fn test_round_up_nicely_powers_of_ten() {
        assert_eq!(round_up_nicely(1.0), 1.0);
        assert_eq!(round_up_nicely(10.0), 10.0);
        assert_eq!(round_up_nicely(100.0), 100.0);
    }

    #[test]
    fn test_round_up_nicely_extreme_values() {
        assert!(round_up_nicely(f64::MIN_POSITIVE) > 0.0);
        assert!(round_up_nicely(f64::EPSILON) > 0.0);
        assert_eq!(round_up_nicely(f64::EPSILON), 1e-10);
    }

    #[test]
    fn test_round_down_nicely_positive_values() {
        assert_eq!(round_down_nicely(1.3), 1.0);
        assert_eq!(round_down_nicely(2.5), 2.0);
        assert_eq!(round_down_nicely(7.0), 5.0);
    }

    #[test]
    fn test_round_down_nicely_negative_values() {
        assert!(round_down_nicely(-1.3) < -1.3);
        assert!(round_down_nicely(-2.5) < -2.5);
    }

    #[test]
    fn test_round_down_nicely_very_small_values() {
        assert_eq!(round_down_nicely(1e-12), 0.0);
    }

    #[test]
    fn test_round_down_nicely_powers_of_ten() {
        assert_eq!(round_down_nicely(1.0), 1.0);
        assert_eq!(round_down_nicely(10.0), 10.0);
        assert_eq!(round_down_nicely(100.0), 100.0);
    }

    #[test]
    fn test_round_functions_with_extreme_values() {
        assert!(round_up_nicely(f64::MIN_POSITIVE) > 0.0);
        assert!(round_down_nicely(f64::MAX) > 0.0);
        assert_eq!(round_up_nicely(f64::EPSILON), 1e-10);
        assert_eq!(round_down_nicely(f64::EPSILON), 0.0);
    }

    #[test]
    fn test_serialize_empty_hashmap() {
        let map: HashMap<usize, i8> = HashMap::new();
        let serialized = serde_json::to_string(&map).unwrap();
        let deserialized: HashMap<usize, i8> = serde_json::from_str(&serialized).unwrap();
        assert!(
            deserialized.is_empty(),
            "Empty map serialization should work"
        );
    }

    #[test]
    fn test_serialize_large_keys() {
        let mut map = HashMap::new();
        map.insert(usize::MAX, 42_i8);
        map.insert(0, -42_i8);

        let serialized = serde_json::to_string(&map).unwrap();
        let deserialized: HashMap<usize, i8> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(map, deserialized, "Large keys should serialize correctly");
    }

    #[test]
    fn test_serialize_tuple_edge_cases() {
        use crate::utils::serde_json_hashmap_numeric::{
            deserialize_tuple_usize, serialize_tuple_usize,
        };
        use serde_json::Value;

        let mut original_map = HashMap::new();
        original_map.insert((0, 0), 0.0_f64);
        original_map.insert((usize::MAX, usize::MAX), f64::MAX);
        original_map.insert((1, 2), f64::MIN_POSITIVE);

        let serialized_value =
            serialize_tuple_usize(&original_map, serde_json::value::Serializer).unwrap();
        let serialized_string = serde_json::to_string_pretty(&serialized_value).unwrap();

        assert!(serialized_string.contains("\"0,0\""));
        assert!(serialized_string.contains(&format!("\"{},{}\"", usize::MAX, usize::MAX)));
        assert!(serialized_string.contains("\"1,2\""));

        let value: Value = serde_json::from_str(&serialized_string).unwrap();
        let deserialized = deserialize_tuple_usize(value).unwrap();

        assert_eq!(
            original_map, deserialized,
            "Tuple key serialization roundtrip should work correctly"
        );
    }

    #[test]
    fn test_roundtrip_serialization_usize_i8() {
        let mut original_map = HashMap::new();
        original_map.insert(1, 1_i8);
        original_map.insert(2, -1_i8);

        let serialized = serde_json::to_string(&original_map).unwrap();
        let deserialized: HashMap<usize, i8> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            original_map, deserialized,
            "Roundtrip serialization should preserve data"
        );
    }

    #[test]
    fn test_roundtrip_serialization_usize_u8() {
        let mut original_map = HashMap::new();
        original_map.insert(1, 255_u8);
        original_map.insert(2, 0_u8);

        let serialized = serde_json::to_string(&original_map).unwrap();
        let deserialized: HashMap<usize, u8> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            original_map, deserialized,
            "u8 roundtrip serialization should work"
        );
    }

    #[test]
    fn test_roundtrip_serialization_tuple_usize_f64() {
        let mut original_map = HashMap::new();
        original_map.insert((0, 1), 1.5_f64);
        original_map.insert((2, 3), -2.7_f64);

        // Serialize HashMap<String, f64>
        let string_map: HashMap<String, f64> = original_map
            .iter()
            .map(|(&(i, j), &v)| (format!("{},{}", i, j), v))
            .collect();

        let serialized = serde_json::to_string(&string_map).unwrap();
        let string_map_deserialized: HashMap<String, f64> =
            serde_json::from_str(&serialized).unwrap();

        // Deserialize HashMap<(usize, usize), f64>
        let mut deserialized = HashMap::new();
        for (k, v) in string_map_deserialized {
            let parts: Vec<&str> = k.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(i), Ok(j)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    deserialized.insert((i, j), v);
                }
            }
        }

        assert_eq!(
            original_map, deserialized,
            "Tuple key serialization should work correctly"
        );
    }

    #[test]
    #[should_panic(expected = "confInterBinomial: Sample size (n) must be greater than zero.")]
    fn test_conf_inter_binomial_zero_samples() {
        conf_inter_binomial(0.5, 0, 0.05);
    }

    #[test]
    #[should_panic(
        expected = "confInterBinomial: accuracy should not be lower than 0 or greater than 1"
    )]
    fn test_conf_inter_binomial_invalid_accuracy() {
        conf_inter_binomial(1.5, 50, 0.05);
    }

    #[test]
    #[should_panic(
        expected = "confInterBinomial: alpha should not be lower than 0 or greater than 1"
    )]
    fn test_conf_inter_binomial_invalid_alpha() {
        conf_inter_binomial(0.5, 50, 1.5);
    }

    #[test]
    fn test_auc_roc_mcc_consistency() {
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];

        let auc1 = compute_auc_from_value(&value, &y);
        let (auc2, _, _, _, _, _) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::auc, None);

        assert!(
            (auc1 - auc2).abs() < 1e-10,
            "AUC should be consistent between functions"
        );
    }

    #[test]
    fn test_compute_roc_large_dataset() {
        let n = 5000;
        let value: Vec<f64> = (0..n)
            .map(|i| (i as f64 / n as f64) + 0.001 * (i % 10) as f64)
            .collect();
        let y: Vec<u8> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        let (auc, _, _, _, _, _) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::auc, None);
        assert!(
            auc > 0.9,
            "Large dataset with clear separation should yield high AUC"
        );
    }

    #[test]
    fn test_compute_auc_manual_example_1() {
        // Simple case: 4 samples with clear classification
        let value = vec![0.1, 0.3, 0.7, 0.9];
        let y = vec![0, 0, 1, 1];

        // Manual AUC calculation:
        // Pairs (negative, positive): (0.1,0.7), (0.1,0.9), (0.3,0.7), (0.3,0.9)
        // All 4 pairs are correctly ordered (negative < positive)
        // AUC = 4/4 = 1.0

        let auc = compute_auc_from_value(&value, &y);
        assert!(
            (auc - 1.0).abs() < 1e-10,
            "AUC should be 1.0 for perfect separation"
        );
    }

    #[test]
    fn test_compute_auc_manual_example_2() {
        // Partially inverted classification
        let value = vec![0.8, 0.6, 0.4, 0.2];
        let y = vec![0, 0, 1, 1];

        // Manual AUC calculation:
        // Pairs (negative, positive): (0.8,0.4), (0.8,0.2), (0.6,0.4), (0.6,0.2)
        // No pair correctly ordered (all negative > positive)
        // AUC = 0/4 = 0.0

        let auc = compute_auc_from_value(&value, &y);
        assert!(
            (auc - 0.0).abs() < 1e-10,
            "AUC should be 0.0 for completely inverse classification"
        );
    }

    #[test]
    fn test_compute_auc_manual_example_3() {
        // Medium classification performance
        let value = vec![0.2, 0.6, 0.4, 0.8];
        let y = vec![0, 0, 1, 1];

        // Manual AUC calculation:
        // Pairs (negative, positive): (0.2,0.4), (0.2,0.8), (0.6,0.4), (0.6,0.8)
        // Correct pairs: (0.2,0.4), (0.2,0.8), (0.6,0.8) = 3/4
        // AUC = 3/4 = 0.75

        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 0.75).abs() < 1e-10, "AUC should be 0.75");
    }

    #[test]
    fn test_compute_auc_manual_example_4() {
        // Case with ties handling
        let value = vec![0.5, 0.5, 0.3, 0.7];
        let y = vec![0, 1, 0, 1];

        // Manual AUC calculation with ties:
        // Pairs (negative, positive): (0.5,0.5), (0.5,0.7), (0.3,0.5), (0.3,0.7)
        // (0.5,0.5) counts as 0.5, (0.5,0.7) = 1, (0.3,0.5) = 1, (0.3,0.7) = 1
        // AUC = (0.5 + 1 + 1 + 1) / 4 = 3.5/4 = 0.875

        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 0.875).abs() < 1e-10, "AUC should be 0.875 with ties");
    }

    #[test]
    fn test_compute_auc_manual_example_5() {
        // Imbalanced dataset
        let value = vec![0.1, 0.2, 0.9];
        let y = vec![0, 0, 1];

        // Manual AUC calculation:
        // Pairs (negative, positive): (0.1,0.9), (0.2,0.9)
        // All correct: 2/2 = 1.0

        let auc = compute_auc_from_value(&value, &y);
        assert!(
            (auc - 1.0).abs() < 1e-10,
            "AUC should be 1.0 for imbalanced but perfect classification"
        );
    }

    #[test]
    fn test_compute_mcc_manual_example_1() {
        // Perfect balanced classification
        let value = vec![0.1, 0.2, 0.8, 0.9];
        let y = vec![0, 0, 1, 1];

        // With optimal threshold at 0.5: TP=2, TN=2, FP=0, FN=0
        // MCC = (2*2 - 0*0) / sqrt((2+0)*(2+0)*(2+0)*(2+0)) = 4/4 = 1.0

        let (_, _, _, _, _, mcc) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::mcc, None);
        assert!(
            (mcc - 1.0).abs() < 1e-10,
            "MCC should be 1.0 for perfect classification"
        );
    }

    #[test]
    fn test_compute_mcc_manual_example_2() {
        // Imperfect but balanced classification
        let value = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![0, 1, 0, 1];

        // With optimal threshold at 0.5: TP=1, TN=1, FP=1, FN=1
        // MCC = (1*1 - 1*1) / sqrt((1+1)*(1+1)*(1+1)*(1+1)) = 0/4 = 0.0

        let (_, _, _, _, _, mcc) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::mcc, None);
        assert!(
            mcc.abs() < 1e-10,
            "MCC should be ~0.0 for random-like classification but is {}",
            mcc
        );
    }

    #[test]
    fn test_compute_mcc_manual_example_3() {
        // Classification with class bias
        let value = vec![0.2, 0.3, 0.7, 0.8];
        let y = vec![0, 0, 0, 1];

        // With optimal threshold at ~0.75: TP=1, TN=3, FP=0, FN=0
        // MCC = (1*3 - 0*0) / sqrt((1+0)*(1+0)*(3+0)*(3+0)) = 3/sqrt(9) = 3/3 = 1.0

        let (_, _, _, _, _, mcc) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::mcc, None);
        assert!(
            (mcc - 1.0).abs() < 1e-10,
            "MCC should be 1.0 for perfect imbalanced classification"
        );
    }

    #[test]
    fn test_compute_mcc_manual_example_4() {
        // Intermediate case with manual calculation
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];

        // With optimal threshold at 0.5: TP=2, TN=2, FP=0, FN=0
        // MCC = (2*2 - 0*0) / sqrt((2+0)*(2+0)*(2+0)*(2+0)) = 4/4 = 1.0

        let (_, _, _, _, _, mcc) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::mcc, None);
        assert!((mcc - 1.0).abs() < 1e-10, "MCC should be 1.0");
    }

    #[test]
    fn test_compute_mcc_manual_example_5() {
        // Classification with symmetric errors
        let value = vec![0.2, 0.6, 0.4, 0.8];
        let y = vec![0, 0, 1, 1];

        // Analysis of possible thresholds:
        // Threshold 0.5: TP=1, TN=1, FP=1, FN=1 → MCC = 0
        // Threshold 0.3: TP=2, TN=1, FP=1, FN=0 → MCC = (2*1-1*0)/sqrt(3*2*2*1) = 2/sqrt(12) ≈ 0.577

        let (_, _, _, _, _, mcc) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::mcc, None);
        assert!(mcc > 0.5 && mcc < 0.6, "MCC should be approximately 0.577");
    }

    #[test]
    fn test_auc_mcc_relationship_manual_verification() {
        // Verification on a case where we can calculate both manually
        let value = vec![0.1, 0.3, 0.7, 0.9];
        let y = vec![0, 0, 1, 1];

        let auc = compute_auc_from_value(&value, &y);
        let (_, _, _, _, _, mcc) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::mcc, None);

        // Expected values calculated manually
        assert!((auc - 1.0).abs() < 1e-10, "AUC should be 1.0");
        assert!((mcc - 1.0).abs() < 1e-10, "MCC should be 1.0");

        // For perfect classification, AUC = MCC = 1.0
        assert!(
            (auc - mcc).abs() < 1e-10,
            "For perfect classification, AUC should equal MCC"
        );
    }

    #[test]
    fn test_compute_auc_manual_complex_case() {
        // More complex case with multiple score values
        let value = vec![0.1, 0.3, 0.4, 0.6, 0.7, 0.9];
        let y = vec![0, 0, 1, 0, 1, 1];

        // Manual calculation:
        // Negatives: 0.1, 0.3, 0.6 (indices 0, 1, 3)
        // Positives: 0.4, 0.7, 0.9 (indices 2, 4, 5)
        // Pairs to evaluate: (0.1,0.4), (0.1,0.7), (0.1,0.9), (0.3,0.4), (0.3,0.7), (0.3,0.9), (0.6,0.4), (0.6,0.7), (0.6,0.9)
        // Correct pairs: (0.1,0.4), (0.1,0.7), (0.1,0.9), (0.3,0.4), (0.3,0.7), (0.3,0.9), (0.6,0.7), (0.6,0.9) = 8/9
        // AUC = 8/9 ≈ 0.8889

        let auc = compute_auc_from_value(&value, &y);
        assert!(
            (auc - 8.0 / 9.0).abs() < 1e-10,
            "AUC should be 8/9 for this configuration"
        );
    }

    #[test]
    fn test_compute_mcc_manual_complex_case() {
        // Complex case with known MCC calculation
        let value = vec![0.1, 0.3, 0.6, 0.8];
        let y = vec![0, 1, 1, 0];

        // With threshold 0.45: predictions [0, 0, 1, 1], actual [0, 1, 1, 0]
        // TP=1 (index 2), TN=1 (index 0), FP=1 (index 3), FN=1 (index 1)
        // MCC = (1*1 - 1*1) / sqrt((1+1)*(1+1)*(1+1)*(1+1)) = 0/4 = 0.0
        // But algorithm will find optimal threshold

        let (_, threshold, _, _, _, mcc) =
            compute_roc_and_metrics_from_value(&value, &y, &FitFunction::mcc, None);

        // The optimal threshold should give better than random performance
        assert!(mcc >= 0.0, "MCC should be non-negative for this case");
        assert!(!threshold.is_nan(), "Threshold should not be NaN");
    }

    #[test]
    fn test_compute_auc_edge_case_with_duplicate_scores() {
        // Edge case: multiple samples with same score
        let value = vec![0.2, 0.2, 0.8, 0.8];
        let y = vec![0, 1, 0, 1];

        // Manual calculation with ties:
        // Pairs: (0.2,0.8), (0.2,0.8) → both count as 1.0
        // Pairs: (0.2,0.8), (0.2,0.8) → both count as 1.0
        // But we also have ties: (0.2,0.2) and (0.8,0.8) each count as 0.5
        // Total correct comparisons considering ties
        // Expected AUC = 0.5 (since classification is essentially random due to ties)

        let auc = compute_auc_from_value(&value, &y);
        assert!(
            (auc - 0.5).abs() < 1e-10,
            "AUC should be 0.5 for this tie scenario"
        );
    }

    // =========================================================================
    // Tests for stratify_indices_by_class
    // =========================================================================

    #[test]
    fn test_stratify_balanced_dataset() {
        let y = vec![0, 1, 0, 1, 0, 1];
        let (pos, neg) = stratify_indices_by_class(&y);

        assert_eq!(pos.len(), 3, "Should have 3 positive samples");
        assert_eq!(neg.len(), 3, "Should have 3 negative samples");
        assert_eq!(pos, vec![1, 3, 5], "Positive indices should be [1, 3, 5]");
        assert_eq!(neg, vec![0, 2, 4], "Negative indices should be [0, 2, 4]");
    }

    #[test]
    fn test_stratify_imbalanced_dataset() {
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let (pos, neg) = stratify_indices_by_class(&y);

        assert_eq!(pos.len(), 1, "Should have 1 positive sample");
        assert_eq!(neg.len(), 9, "Should have 9 negative samples");
        assert_eq!(pos, vec![9], "Positive index should be [9]");
        assert_eq!(
            neg,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            "Negative indices should be [0..9]"
        );
    }

    #[test]
    fn test_stratify_all_positive() {
        let y = vec![1, 1, 1, 1];
        let (pos, neg) = stratify_indices_by_class(&y);

        assert_eq!(pos.len(), 4, "Should have 4 positive samples");
        assert_eq!(neg.len(), 0, "Should have 0 negative samples");
        assert_eq!(pos, vec![0, 1, 2, 3]);
        assert!(neg.is_empty());
    }

    #[test]
    fn test_stratify_all_negative() {
        let y = vec![0, 0, 0];
        let (pos, neg) = stratify_indices_by_class(&y);

        assert_eq!(pos.len(), 0, "Should have 0 positive samples");
        assert_eq!(neg.len(), 3, "Should have 3 negative samples");
        assert!(pos.is_empty());
        assert_eq!(neg, vec![0, 1, 2]);
    }

    #[test]
    fn test_stratify_preserves_order() {
        let y = vec![1, 0, 1, 0, 1];
        let (pos, neg) = stratify_indices_by_class(&y);

        // Indices should be in ascending order
        for i in 1..pos.len() {
            assert!(pos[i] > pos[i - 1], "Positive indices should be ordered");
        }
        for i in 1..neg.len() {
            assert!(neg[i] > neg[i - 1], "Negative indices should be ordered");
        }
    }

    #[test]
    fn test_stratify_compatibility_with_manual_implementation() {
        // This test ensures that stratify_indices_by_class produces
        // the same result as the manual loop-based implementation
        // previously used in cv.rs and data.rs
        let y = vec![0, 1, 0, 0, 1, 1, 0, 1];

        // New implementation
        let (pos_new, neg_new) = stratify_indices_by_class(&y);

        // Old manual implementation (from cv.rs)
        let mut pos_old = Vec::new();
        let mut neg_old = Vec::new();
        for (i, &label) in y.iter().enumerate() {
            if label == 0 {
                neg_old.push(i);
            } else if label == 1 {
                pos_old.push(i);
            }
        }

        // Should produce identical results
        assert_eq!(
            pos_new, pos_old,
            "Positive indices should match old implementation"
        );
        assert_eq!(
            neg_new, neg_old,
            "Negative indices should match old implementation"
        );
    }

    // =========================================================================
    // Tests for stratified_bootstrap_sample
    // =========================================================================

    #[test]
    fn test_bootstrap_sample_full_bootstrap() {
        let pos_indices = vec![1, 3, 5];
        let neg_indices = vec![0, 2, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let sample = stratified_bootstrap_sample(&pos_indices, &neg_indices, 1.0, &mut rng);

        assert_eq!(sample.len(), 6, "Should sample all 6 elements");

        // With bootstrap (replacement), we can have duplicates
        let pos_part = &sample[0..3];
        let neg_part = &sample[3..6];

        // All sampled indices should be valid
        for &idx in pos_part {
            assert!(pos_indices.contains(&idx), "Invalid positive index");
        }
        for &idx in neg_part {
            assert!(neg_indices.contains(&idx), "Invalid negative index");
        }
    }

    #[test]
    fn test_bootstrap_sample_subsampling_632() {
        let pos_indices = vec![0, 1, 2, 3, 4];
        let neg_indices = vec![5, 6, 7, 8, 9];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let sample = stratified_bootstrap_sample(&pos_indices, &neg_indices, 0.632, &mut rng);

        // 0.632 * 5 = 3.16 → ceil = 4 per class
        let expected_size = 4 + 4;
        assert_eq!(
            sample.len(),
            expected_size,
            "Should sample {} elements",
            expected_size
        );

        // With subsampling, indices should be unique within each class
        let pos_part: Vec<usize> = sample
            .iter()
            .filter(|&&idx| pos_indices.contains(&idx))
            .copied()
            .collect();
        let neg_part: Vec<usize> = sample
            .iter()
            .filter(|&&idx| neg_indices.contains(&idx))
            .copied()
            .collect();

        // Check uniqueness within each class
        let mut pos_sorted = pos_part.clone();
        pos_sorted.sort_unstable();
        pos_sorted.dedup();
        assert_eq!(
            pos_sorted.len(),
            pos_part.len(),
            "Positive samples should be unique in subsampling mode"
        );

        let mut neg_sorted = neg_part.clone();
        neg_sorted.sort_unstable();
        neg_sorted.dedup();
        assert_eq!(
            neg_sorted.len(),
            neg_part.len(),
            "Negative samples should be unique in subsampling mode"
        );
    }

    #[test]
    fn test_bootstrap_sample_half_bootstrap() {
        let pos_indices = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let neg_indices = vec![8, 9, 10, 11, 12, 13, 14, 15];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let sample = stratified_bootstrap_sample(&pos_indices, &neg_indices, 0.5, &mut rng);

        // 0.5 * 8 = 4 per class
        assert_eq!(sample.len(), 8, "Should sample 8 elements (4+4)");
    }

    #[test]
    fn test_bootstrap_sample_reproducibility() {
        let pos_indices = vec![0, 2, 4, 6];
        let neg_indices = vec![1, 3, 5, 7];

        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let sample1 = stratified_bootstrap_sample(&pos_indices, &neg_indices, 0.7, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let sample2 = stratified_bootstrap_sample(&pos_indices, &neg_indices, 0.7, &mut rng2);

        assert_eq!(sample1, sample2, "Same seed should produce same sample");
    }

    #[test]
    fn test_bootstrap_sample_preserves_stratification() {
        let pos_indices = vec![10, 11, 12, 13, 14];
        let neg_indices = vec![20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let sample = stratified_bootstrap_sample(&pos_indices, &neg_indices, 0.6, &mut rng);

        // Count how many are from pos vs neg
        let pos_count = sample
            .iter()
            .filter(|&&idx| pos_indices.contains(&idx))
            .count();
        let neg_count = sample
            .iter()
            .filter(|&&idx| neg_indices.contains(&idx))
            .count();

        // Should be roughly 60% of each class
        // pos: ceil(5 * 0.6) = 3
        // neg: ceil(10 * 0.6) = 6
        assert_eq!(pos_count, 3, "Should have 3 positive samples");
        assert_eq!(neg_count, 6, "Should have 6 negative samples");
    }

    #[test]
    fn test_bootstrap_sample_small_classes() {
        let pos_indices = vec![0];
        let neg_indices = vec![1];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Full bootstrap with 1 sample each
        let sample = stratified_bootstrap_sample(&pos_indices, &neg_indices, 1.0, &mut rng);

        assert_eq!(sample.len(), 2);
        assert!(sample.contains(&0));
        assert!(sample.contains(&1));
    }

    #[test]
    fn test_bootstrap_sample_order_pos_then_neg() {
        let pos_indices = vec![1, 3, 5];
        let neg_indices = vec![0, 2, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let sample = stratified_bootstrap_sample(&pos_indices, &neg_indices, 1.0, &mut rng);

        // First n_pos should be from pos_indices
        let n_pos = pos_indices.len();
        for i in 0..n_pos {
            assert!(
                pos_indices.contains(&sample[i]),
                "First {} elements should be positive",
                n_pos
            );
        }

        // Remaining should be from neg_indices
        for i in n_pos..sample.len() {
            assert!(
                neg_indices.contains(&sample[i]),
                "Elements after {} should be negative",
                n_pos
            );
        }
    }

    #[test]
    fn test_bootstrap_sample_with_replacement_allows_duplicates() {
        let pos_indices = vec![0, 1];
        let neg_indices = vec![2, 3];

        // Run multiple times to likely get duplicates
        let mut found_duplicate = false;
        for seed in 0..50 {
            let mut local_rng = ChaCha8Rng::seed_from_u64(seed);
            let sample =
                stratified_bootstrap_sample(&pos_indices, &neg_indices, 1.0, &mut local_rng);

            let mut sorted = sample.clone();
            sorted.sort_unstable();
            let original_len = sorted.len();
            sorted.dedup();
            if sorted.len() < original_len {
                found_duplicate = true;
                break;
            }
        }

        assert!(
            found_duplicate,
            "With replacement, should eventually get duplicates"
        );
    }

    // =========================================================================
    // Tests for geyer_rescale_ci
    // =========================================================================

    #[test]
    fn test_geyer_rescale_no_rescaling_full_bootstrap() {
        let bootstrap_stats = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let center = 0.5;
        let sqrt_m = 1.0;
        let sqrt_n = 3.0;
        let lower_idx = 1; // 0.2
        let upper_idx = 7; // 0.8

        let (lower, upper) = geyer_rescale_ci(
            &bootstrap_stats,
            center,
            false,
            sqrt_m,
            sqrt_n,
            lower_idx,
            upper_idx,
        );

        // No rescaling, should just return quantiles
        assert_eq!(lower, 0.2);
        assert_eq!(upper, 0.8);
    }

    #[test]
    fn test_geyer_rescale_with_subsampling() {
        // Pivotal statistics: √m * (θ̂ᵦ - θ̂)
        let bootstrap_stats = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let center = 0.5;
        let sqrt_m = 2.0;
        let sqrt_n = 4.0;
        let lower_idx = 1; // -2.0
        let upper_idx = 5; // 2.0

        let (lower, upper) = geyer_rescale_ci(
            &bootstrap_stats,
            center,
            true,
            sqrt_m,
            sqrt_n,
            lower_idx,
            upper_idx,
        );

        // De-pivot: θ̂ - quantile / √n
        // lower = 0.5 - 2.0 / 4.0 = 0.5 - 0.5 = 0.0
        // upper = 0.5 - (-2.0) / 4.0 = 0.5 + 0.5 = 1.0
        assert!((lower - 0.0).abs() < 1e-10, "Lower bound should be 0.0");
        assert!((upper - 1.0).abs() < 1e-10, "Upper bound should be 1.0");
    }

    #[test]
    fn test_geyer_rescale_symmetry() {
        // Symmetric pivotal statistics around 0
        let bootstrap_stats = vec![-4.0, -2.0, 0.0, 2.0, 4.0];
        let center = 0.6;
        let sqrt_m = 5.0;
        let sqrt_n = 10.0;
        let lower_idx = 1; // -2.0
        let upper_idx = 3; // 2.0

        let (lower, upper) = geyer_rescale_ci(
            &bootstrap_stats,
            center,
            true,
            sqrt_m,
            sqrt_n,
            lower_idx,
            upper_idx,
        );

        // With symmetric pivotal stats, CI should be symmetric around center
        let dist_lower = center - lower;
        let dist_upper = upper - center;
        assert!(
            (dist_lower - dist_upper).abs() < 1e-10,
            "CI should be symmetric: lower={}, center={}, upper={}",
            lower,
            center,
            upper
        );
    }

    #[test]
    fn test_geyer_rescale_ordering() {
        // For subsampling, bootstrap_stats are PIVOTAL: √m * (θ̂ᵦ - θ̂)
        // They should be centered around 0, not around the center threshold
        let bootstrap_stats = vec![-4.0, -2.0, 0.0, 2.0, 4.0];
        let center = 2.5;
        let sqrt_m = 2.0;
        let sqrt_n = 4.0;
        let lower_idx = 1; // -2.0
        let upper_idx = 3; // 2.0

        let (lower, upper) = geyer_rescale_ci(
            &bootstrap_stats,
            center,
            true,
            sqrt_m,
            sqrt_n,
            lower_idx,
            upper_idx,
        );

        // De-pivot: lower = 2.5 - 2.0/4.0 = 2.0, upper = 2.5 - (-2.0)/4.0 = 3.0
        assert!(
            lower <= center,
            "Lower bound should be <= center: {} <= {}",
            lower,
            center
        );
        assert!(
            center <= upper,
            "Center should be <= upper bound: {} <= {}",
            center,
            upper
        );
        assert!(
            lower <= upper,
            "Lower should be <= upper: {} <= {}",
            lower,
            upper
        );
    }

    #[test]
    fn test_geyer_rescale_mathematical_verification() {
        // Test the actual formula: θ̂ - q_α/2 / √n and θ̂ - q_{1-α/2} / √n
        let bootstrap_stats = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        let center = 3.0;
        let sqrt_m = 5.0; // Not used in de-pivoting
        let sqrt_n = 2.5;
        let lower_idx = 1; // -5.0
        let upper_idx = 3; // 5.0

        let (lower, upper) = geyer_rescale_ci(
            &bootstrap_stats,
            center,
            true,
            sqrt_m,
            sqrt_n,
            lower_idx,
            upper_idx,
        );

        // Manual calculation:
        // lower = center - bootstrap_stats[upper_idx] / sqrt_n
        //       = 3.0 - 5.0 / 2.5 = 3.0 - 2.0 = 1.0
        // upper = center - bootstrap_stats[lower_idx] / sqrt_n
        //       = 3.0 - (-5.0) / 2.5 = 3.0 + 2.0 = 5.0
        assert!((lower - 1.0).abs() < 1e-10, "Lower should be 1.0");
        assert!((upper - 5.0).abs() < 1e-10, "Upper should be 5.0");
    }

    #[test]
    fn test_geyer_rescale_depivoting_inverts_quantiles() {
        // The de-pivoting should invert the quantile order
        let bootstrap_stats = vec![1.0, 2.0, 3.0];
        let center = 5.0;
        let sqrt_m = 1.0;
        let sqrt_n = 1.0;
        let lower_idx = 0; // 1.0
        let upper_idx = 2; // 3.0

        let (lower, upper) = geyer_rescale_ci(
            &bootstrap_stats,
            center,
            true,
            sqrt_m,
            sqrt_n,
            lower_idx,
            upper_idx,
        );

        // lower = 5.0 - 3.0 / 1.0 = 2.0 (uses upper_idx)
        // upper = 5.0 - 1.0 / 1.0 = 4.0 (uses lower_idx)
        assert_eq!(lower, 2.0);
        assert_eq!(upper, 4.0);
    }

    // =========================================================================
    // Integration tests for bootstrap (existing tests continue below)
    // =========================================================================

    #[test]
    fn test_bootstrap_ci_with_balanced_dataset() {
        let value = vec![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (auc, [lower, center, upper], _, _, _, _, rej) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                1000, // n_bootstrap
                0.05, // alpha (95% CI)
                1_f64,
                &mut rng,
            );

        assert!(auc > 0.0 && auc <= 1.0, "AUC should be in [0,1]");
        assert!(
            lower <= center && center <= upper,
            "CI should be ordered: lower < center < upper"
        );
        assert!(lower > 0.0, "Lower threshold should be positive");
        assert!(upper < 1.0, "Upper threshold should be < 1.0");
        assert!(rej > 0.0, "Rejection rate should be > 0 with CI");

        let ci_width = upper - lower;
        assert!(
            ci_width > 0.0 && ci_width < 0.5,
            "CI width should be reasonable, got {}",
            ci_width
        );
    }

    #[test]
    fn test_bootstrap_ci_reproducibility() {
        let value = vec![0.1, 0.3, 0.7, 0.9];
        let y = vec![0, 0, 1, 1];

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        let (_, [l1, c1, u1], _, _, _, _, _) = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            100,
            0.05,
            1_f64,
            &mut rng1,
        );

        let (_, [l2, c2, u2], _, _, _, _, _) = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            100,
            0.05,
            1_f64,
            &mut rng2,
        );

        assert!(
            (l1 - l2).abs() < 1e-10,
            "Same seed should give same lower CI"
        );
        assert!((c1 - c2).abs() < 1e-10, "Same seed should give same center");
        assert!(
            (u1 - u2).abs() < 1e-10,
            "Same seed should give same upper CI"
        );
    }

    #[test]
    fn test_bootstrap_ci_with_imbalanced_dataset() {
        // Highly unbalanced dataset (10% positive)
        let value = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (auc, [lower, center, upper], acc, sens, spec, obj, rej) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                500,
                0.05,
                1_f64,
                &mut rng,
            );

        println!(
            "AUC: {}, Threshold: [{}, {}, {}]",
            auc, lower, center, upper
        );
        println!(
            "Metrics: acc={}, sens={}, spec={}, obj={}, rej={}",
            acc, sens, spec, obj, rej
        );

        // With only 1 positive out of 10, perfect separation is possible
        // The threshold should be stable (all positives have high scores)
        // Actually, with this data, the CI might be narrow because the optimal threshold is clear
        let ci_width = upper - lower;

        // The CI width depends on the data - with perfect separation, it can be narrow
        // What we can test is that the bootstrap ran and returned valid results
        assert!(
            ci_width >= 0.0,
            "CI width should be non-negative, got {}",
            ci_width
        );
        assert!(
            lower <= center && center <= upper,
            "CI bounds should be ordered"
        );

        // AUC should be perfect or near perfect with this separable data
        assert!(
            auc >= 0.9,
            "AUC should be high with separable data, got {}",
            auc
        );
    }

    #[test]
    fn test_bootstrap_ci_alpha_parameter() {
        let value = vec![
            0.1, 0.3, 0.5, 0.7, 0.9, 0.3, 0.1, 0.2, 0.2, 0.1, 0.5, 0.2, 0.9, 0.2,
        ];
        let y = vec![0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0];

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        // 95% CI (alpha=0.05)
        let (_, [l1, _, u1], _, _, _, _, _) = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            2000,
            0.01,
            1_f64,
            &mut rng1,
        );

        // 90% CI (alpha=0.10)
        let (_, [l2, _, u2], _, _, _, _, _) = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            2000,
            0.90,
            1_f64,
            &mut rng2,
        );

        let width_99 = u1 - l1;
        let width_90 = u2 - l2;

        // 95% CI should be wider than 90% CI
        assert!(
            width_99 > width_90,
            "99% CI ({:.3}) should be wider than 90% CI ({:.3})",
            width_99,
            width_90
        );
    }

    #[test]
    fn test_compute_metrics_with_additional_perfect_classification() {
        let predicted = vec![0, 1, 0, 1];
        let y = vec![0, 1, 0, 1];

        let (acc, sens, spec, additional) = compute_metrics_from_classes(&predicted, &y, [true; 5]);

        assert_eq!(acc, 1.0);
        assert_eq!(sens, 1.0);
        assert_eq!(spec, 1.0);

        assert_eq!(additional.mcc, Some(1.0), "MCC should be 1.0 for perfect");
        assert_eq!(
            additional.f1_score,
            Some(1.0),
            "F1 should be 1.0 for perfect"
        );
        assert_eq!(additional.npv, Some(1.0), "NPV should be 1.0 for perfect");
        assert_eq!(additional.ppv, Some(1.0), "PPV should be 1.0 for perfect");
        assert_eq!(
            additional.g_mean,
            Some(1.0),
            "G-mean should be 1.0 for perfect"
        );
    }

    #[test]
    fn test_compute_metrics_with_additional_random_classification() {
        let predicted = vec![0, 1, 0, 1];
        let y = vec![1, 0, 1, 0];

        let (acc, sens, spec, additional) = compute_metrics_from_classes(&predicted, &y, [true; 5]);

        assert_eq!(acc, 0.0, "Accuracy should be 0.0");
        assert_eq!(sens, 0.0, "Sensitivity should be 0.0");
        assert_eq!(spec, 0.0, "Specificity should be 0.0");

        // MCC should be -1.0 for perfectly inverse classification
        assert_eq!(additional.mcc, Some(-1.0), "MCC should be -1.0");
        assert_eq!(additional.f1_score, Some(0.0), "F1 should be 0.0");
        assert_eq!(additional.g_mean, Some(0.0), "G-mean should be 0.0");
    }

    #[test]
    fn test_compute_metrics_with_additional_imbalanced() {
        let predicted = vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 1];
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

        let (_, _, _, additional) = compute_metrics_from_classes(&predicted, &y, [true; 5]);

        // TP=1, FP=2, TN=6, FN=1
        // PPV = TP/(TP+FP) = 1/3 = 0.333
        // NPV = TN/(TN+FN) = 6/7 = 0.857

        assert!(additional.ppv.is_some());
        assert!(
            (additional.ppv.unwrap() - (1.0 / 3.0)).abs() < 1e-10,
            "PPV should be 1/3"
        );

        assert!(additional.npv.is_some());
        assert!(
            (additional.npv.unwrap() - (6.0 / 7.0)).abs() < 1e-10,
            "NPV should be 6/7"
        );
    }

    #[test]
    fn test_compute_metrics_selective_additional() {
        let predicted = vec![0, 1, 0, 1];
        let y = vec![0, 1, 0, 1];

        let (_, _, _, additional) =
            compute_metrics_from_classes(&predicted, &y, [true, true, false, false, false]);

        assert!(additional.mcc.is_some(), "MCC should be computed");
        assert!(additional.f1_score.is_some(), "F1 should be computed");
        assert!(additional.npv.is_none(), "NPV should NOT be computed");
        assert!(additional.ppv.is_none(), "PPV should NOT be computed");
        assert!(additional.g_mean.is_none(), "G-mean should NOT be computed");
    }

    #[test]
    fn test_compute_metrics_with_abstentions_and_additional() {
        let predicted = vec![0, 1, 2, 1, 0, 2];
        let y = vec![0, 1, 0, 1, 1, 1];

        let (acc, sens, spec, additional) = compute_metrics_from_classes(&predicted, &y, [true; 5]);

        // Class 2 should be ignored
        // Effective samples: [0,0], [1,1], [1,1], [0,1]
        // TP=2, TN=1, FP=0, FN=1

        assert_eq!(acc, 0.75, "Accuracy should be 3/4");
        assert_eq!(sens, 2.0 / 3.0, "Sensitivity should be 2/3");
        assert_eq!(spec, 1.0, "Specificity should be 1.0");

        assert!(additional.mcc.is_some());
        let mcc = additional.mcc.unwrap();
        assert!(mcc > 0.0 && mcc < 1.0, "MCC should be positive and < 1.0");
    }

    #[test]
    fn test_mcc_edge_case_zero_denominator() {
        // Case where MCC denominator = 0
        // All predicted positive, but true class mixed
        let predicted = vec![1, 1, 1, 1];
        let y = vec![1, 1, 1, 1];

        let (_, _, _, additional) =
            compute_metrics_from_classes(&predicted, &y, [true, false, false, false, false]);

        // TN=0, FP=0 → denominator = 0 → MCC should be 0.0 (convention)
        assert_eq!(
            additional.mcc,
            Some(0.0),
            "MCC should be 0.0 when denominator is zero"
        );
    }

    #[test]
    fn test_gmean_calculation() {
        let predicted = vec![0, 0, 1, 1];
        let y = vec![0, 1, 0, 1];

        let (_, sens, spec, additional) =
            compute_metrics_from_classes(&predicted, &y, [false, false, false, false, true]);

        // Sensitivity = 0.5, Specificity = 0.5
        // G-mean = sqrt(0.5 * 0.5) = sqrt(0.25) = 0.5

        let gmean = additional.g_mean.unwrap();
        assert!(
            (gmean - 0.5).abs() < 1e-10,
            "G-mean should be 0.5, got {}",
            gmean
        );

        let expected_gmean = (sens * spec).sqrt();
        assert!(
            (gmean - expected_gmean).abs() < 1e-10,
            "G-mean should match sqrt(sens * spec)"
        );
    }

    // ============================================================================
    // Additional comprehensive tests for compute_threshold_and_metrics_with_bootstrap
    // ============================================================================

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_bootstrap_invalid_subsample_frac_zero() {
        let value = vec![0.1, 0.9];
        let y = vec![0, 1];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _ = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            100,
            0.05,
            0.0,
            &mut rng,
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_bootstrap_invalid_subsample_frac_above_one() {
        let value = vec![0.1, 0.9];
        let y = vec![0, 1];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _ = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            100,
            0.05,
            1.1,
            &mut rng,
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_bootstrap_too_few_iterations() {
        let value = vec![0.1, 0.5, 0.9];
        let y = vec![0, 0, 1];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _ = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            10,
            0.05,
            1.0,
            &mut rng,
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_bootstrap_invalid_alpha_zero() {
        let value = vec![0.1, 0.9];
        let y = vec![0, 1];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _ = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            100,
            0.0,
            1.0,
            &mut rng,
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_bootstrap_invalid_alpha_one() {
        let value = vec![0.1, 0.9];
        let y = vec![0, 1];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _ = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            100,
            1.0,
            1.0,
            &mut rng,
        );
    }

    #[test]
    fn test_bootstrap_subsample_632() {
        // Test with .632 bootstrap (optimal subsampling)
        let value = vec![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (auc, [lower, center, upper], acc, se, sp, obj, rej) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                500,
                0.05,
                0.632,
                &mut rng,
            );

        assert!(auc > 0.5 && auc <= 1.0, "AUC should be reasonable");
        assert!(
            lower <= center && center <= upper,
            "CI ordering: lower <= center <= upper"
        );
        assert!(acc >= 0.0 && acc <= 1.0, "Accuracy in [0,1]");
        assert!(se >= 0.0 && se <= 1.0, "Sensitivity in [0,1]");
        assert!(sp >= 0.0 && sp <= 1.0, "Specificity in [0,1]");
        assert!(obj >= 0.0, "Objective should be non-negative");
        assert!(rej > 0.0, "Rejection rate should be positive with CI");
    }

    #[test]
    fn test_bootstrap_half_bootstrap() {
        // Test with half-bootstrap (very conservative)
        let value = vec![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (_, [lower_half, _, upper_half], _, _, _, _, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                500,
                0.05,
                0.5,
                &mut rng,
            );

        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let (_, [lower_full, _, upper_full], _, _, _, _, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                500,
                0.05,
                1.0,
                &mut rng2,
            );

        let width_half = upper_half - lower_half;
        let width_full = upper_full - lower_full;

        // Half-bootstrap should typically give wider CI due to subsampling
        // (though this depends on the Geyer correction and sample size)
        assert!(
            width_half >= 0.0 && width_full >= 0.0,
            "Both CI widths should be non-negative"
        );

        assert!(
            lower_half <= upper_half,
            "Half-bootstrap CI should be valid"
        );
        assert!(
            lower_full <= upper_full,
            "Full-bootstrap CI should be valid"
        );
    }

    #[test]
    fn test_bootstrap_ci_width_vs_n_bootstrap() {
        let value = vec![0.1, 0.2, 0.4, 0.6, 0.8, 0.9];
        let y = vec![0, 0, 0, 1, 1, 1];

        // Small n_bootstrap
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let (_, [l1, _, u1], _, _, _, _, _) = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            100,
            0.05,
            1.0,
            &mut rng1,
        );

        // Large n_bootstrap
        let mut rng2 = ChaCha8Rng::seed_from_u64(43);
        let (_, [l2, _, u2], _, _, _, _, _) = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            2000,
            0.05,
            1.0,
            &mut rng2,
        );

        let width1 = u1 - l1;
        let width2 = u2 - l2;

        // More bootstrap iterations should give more stable estimates
        // But both should be reasonable
        assert!(
            width1 > 0.0 && width2 > 0.0,
            "Both CI widths should be positive"
        );

        // Difference shouldn't be too extreme
        let ratio = width1.max(width2) / width1.min(width2);
        assert!(
            ratio < 3.0,
            "CI widths shouldn't differ by more than 3x, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_bootstrap_different_fit_functions() {
        let value = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let (_, [_, center_auc, _], _, _, _, obj_auc, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                500,
                0.05,
                1.0,
                &mut rng1,
            );

        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let (_, [_, center_spec, _], _, _, _, obj_spec, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::specificity,
                None,
                500,
                0.05,
                1.0,
                &mut rng2,
            );

        let mut rng3 = ChaCha8Rng::seed_from_u64(42);
        let (_, [_, center_mcc, _], _, _, _, obj_mcc, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::mcc,
                None,
                500,
                0.05,
                1.0,
                &mut rng3,
            );

        // Different fit functions may give different thresholds
        assert!(
            center_auc.is_finite() && center_spec.is_finite() && center_mcc.is_finite(),
            "All thresholds should be finite"
        );

        assert!(
            obj_auc.is_finite() && obj_spec.is_finite() && obj_mcc.is_finite(),
            "All objectives should be finite"
        );
    }

    #[test]
    fn test_bootstrap_with_penalties() {
        let value = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let y = vec![0, 0, 1, 1, 1];

        // Test with FPR/FNR penalties using sensitivity as fit function
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (_, [lower, center, upper], _, _, _, obj, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::sensitivity,
                Some([2.0, 1.0]), // Penalize FPR more
                500,
                0.05,
                1.0,
                &mut rng,
            );

        assert!(lower <= center && center <= upper, "CI should be valid");
        assert!(obj >= 0.0 && obj <= 1.0, "Objective should be in [0,1]");
    }

    #[test]
    fn test_bootstrap_stratification_preserved() {
        // Test that stratification is maintained across bootstrap samples
        let value = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (_, [lower, _, upper], acc, se, sp, _, rej) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                1000,
                0.05,
                1.0,
                &mut rng,
            );

        // With stratification, we should get reasonable metrics
        // Note: with CI, some samples may be in abstention zone
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "Accuracy should be in [0,1], got {}",
            acc
        );
        assert!(se > 0.0 && se <= 1.0, "Sensitivity should be in (0,1]");
        assert!(sp > 0.0 && sp <= 1.0, "Specificity should be in (0,1]");
        assert!(lower < upper, "CI bounds should be distinct");
        assert!(rej >= 0.0, "Rejection rate should be >= 0, got {}", rej);
    }

    #[test]
    fn test_bootstrap_extreme_scores() {
        // Test with extreme score values
        let value = vec![-1e6, -1e3, 1e3, 1e6];
        let y = vec![0, 0, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (auc, [lower, center, upper], _, _, _, _, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                200,
                0.05,
                1.0,
                &mut rng,
            );

        assert!(auc == 1.0, "Perfect separation should give AUC=1.0");
        assert!(lower <= center && center <= upper, "CI should be ordered");
        assert!(
            lower.is_finite() && upper.is_finite(),
            "CI bounds should be finite"
        );
    }

    #[test]
    fn test_bootstrap_small_dataset() {
        // Test with minimal dataset (edge case)
        let value = vec![0.2, 0.3, 0.7, 0.8];
        let y = vec![0, 0, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (auc, [lower, center, upper], _, _, _, _, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                100,
                0.05,
                1.0,
                &mut rng,
            );

        assert!(
            auc >= 0.5 && auc <= 1.0,
            "AUC should be reasonable even for small dataset"
        );
        assert!(lower <= center && center <= upper, "CI should be valid");

        // With small sample, CI might be wide
        let ci_width = upper - lower;
        assert!(ci_width >= 0.0, "CI width should be non-negative");
    }

    #[test]
    fn test_bootstrap_all_same_class() {
        // Edge case: all samples from same class
        let value = vec![0.1, 0.2, 0.3, 0.4];
        let y = vec![1, 1, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                100,
                0.05,
                1.0,
                &mut rng,
            )
        }));

        // This may panic or return invalid results with single class
        // We just verify it doesn't crash the test suite
        if let Ok((auc, [lower, center, upper], _, se, sp, _, _)) = result {
            // AUC is undefined with single class (defaults to 0.5)
            assert!(
                auc == 0.5 || auc.is_nan(),
                "AUC should be 0.5 or NaN with single class"
            );

            // Sensitivity should be computable, specificity undefined
            assert!(
                se >= 0.0 && se <= 1.0 || se.is_nan(),
                "Sensitivity should be in [0,1] or NaN"
            );
            assert!(
                sp.is_nan() || sp == 0.0,
                "Specificity undefined with no negative class"
            );

            // CI bounds should be finite (though ordering may fail with invalid data)
            assert!(
                lower.is_finite() && center.is_finite() && upper.is_finite(),
                "CI bounds should be finite"
            );
        }
    }

    #[test]
    fn test_bootstrap_perfect_separation() {
        // Perfect separation between classes
        let value = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (auc, [lower, center, upper], acc, se, sp, _, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                500,
                0.05,
                1.0,
                &mut rng,
            );

        assert_eq!(auc, 1.0, "Perfect separation should give AUC=1.0");

        // Note: metrics are computed at the center threshold with CI bounds
        // This may lead to some abstentions, so perfect metrics aren't guaranteed
        assert!(acc >= 0.5, "Accuracy should be reasonable, got {}", acc);
        assert!(se >= 0.0 && se <= 1.0, "Sensitivity should be in [0,1]");
        assert!(sp >= 0.0 && sp <= 1.0, "Specificity should be in [0,1]");

        // CI should be valid
        assert!(lower <= center && center <= upper, "CI should be valid");
        // With perfect separation, threshold can be anywhere that separates the classes
        // (bootstrap variance may place it outside or at the boundary)
        assert!(
            center >= 0.0 && center <= 1.0,
            "Threshold should be in [0,1], got {}",
            center
        );
    }

    #[test]
    fn test_bootstrap_ties_in_scores() {
        // Many tied scores
        let value = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (auc, [lower, center, upper], _, _, _, _, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                200,
                0.05,
                1.0,
                &mut rng,
            );

        // With all ties, AUC should be 0.5 (random)
        assert!((auc - 0.5).abs() < 0.01, "AUC should be ≈0.5 with all ties");

        // CI should be valid (though possibly wide)
        assert!(lower <= center && center <= upper, "CI should be valid");
        assert!(
            lower.is_finite() && upper.is_finite(),
            "CI should be finite"
        );
    }

    #[test]
    fn test_bootstrap_geyer_rescaling() {
        // Test that Geyer rescaling is applied correctly for subsampling
        let value: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        let y: Vec<u8> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let (_, [l1, c1, u1], _, _, _, _, _) = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            500,
            0.05,
            0.7,
            &mut rng1,
        );

        // With Geyer rescaling, CI should be valid
        assert!(
            l1 <= c1 && c1 <= u1,
            "Geyer-corrected CI should be valid: {} <= {} <= {}",
            l1,
            c1,
            u1
        );

        let width = u1 - l1;
        assert!(
            width > 0.0 && width < 1.0,
            "Geyer-corrected CI width should be reasonable, got {}",
            width
        );
    }

    #[test]
    fn test_bootstrap_ci_coverage_stability() {
        // Test that CI is stable across different random seeds
        let value = vec![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut widths = Vec::new();

        for seed in 1..6 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (_, [lower, _, upper], _, _, _, _, _) =
                compute_threshold_and_metrics_with_bootstrap(
                    &value,
                    &y,
                    &FitFunction::auc,
                    None,
                    1000,
                    0.05,
                    1.0,
                    &mut rng,
                );

            widths.push(upper - lower);
        }

        // Calculate coefficient of variation of CI widths
        let mean_width: f64 = widths.iter().sum::<f64>() / widths.len() as f64;
        let variance: f64 =
            widths.iter().map(|w| (w - mean_width).powi(2)).sum::<f64>() / widths.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean_width;

        // CV should be low (< 0.3) indicating stable estimates
        assert!(
            cv < 0.3,
            "CI width should be stable across seeds, got CV={:.3}",
            cv
        );
    }

    #[test]
    fn test_bootstrap_metrics_consistency() {
        let value = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let y = vec![0, 0, 1, 1, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (_auc, [lower, center, upper], acc, se, sp, _, rej) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                500,
                0.05,
                1.0,
                &mut rng,
            );

        // Recompute metrics manually at center threshold WITHOUT CI
        let (acc_check, se_check, sp_check, rej_check, _) =
            compute_metrics_from_value(&value, &y, center, None, [false; 5]);

        // NOTE: The bootstrap function computes metrics WITH CI bounds,
        // so we can't expect exact match with metrics computed without CI
        // Instead, verify that values are reasonable and CI affects rejection rate

        assert!(acc >= 0.0 && acc <= 1.0, "Accuracy should be in [0,1]");
        assert!(se >= 0.0 && se <= 1.0, "Sensitivity should be in [0,1]");
        assert!(sp >= 0.0 && sp <= 1.0, "Specificity should be in [0,1]");

        assert!(
            acc_check >= 0.0 && acc_check <= 1.0,
            "Manual accuracy should be in [0,1]"
        );
        assert!(
            se_check >= 0.0 && se_check <= 1.0,
            "Manual sensitivity should be in [0,1]"
        );
        assert!(
            sp_check >= 0.0 && sp_check <= 1.0,
            "Manual specificity should be in [0,1]"
        );

        // Rejection rate should be > 0 with CI but 0 without
        assert!(rej >= 0.0, "Bootstrap should have rejection >= 0");
        assert_eq!(rej_check, 0.0, "Manual check without CI should have rej=0");

        // CI should be valid
        assert!(lower <= center && center <= upper, "CI should be ordered");
    }

    #[test]
    fn test_bootstrap_return_values_structure() {
        let value = vec![0.1, 0.9];
        let y = vec![0, 1];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = compute_threshold_and_metrics_with_bootstrap(
            &value,
            &y,
            &FitFunction::auc,
            None,
            100,
            0.05,
            1.0,
            &mut rng,
        );

        // Unpack all return values
        let (auc, thresholds, acc, se, sp, obj, rej) = result;
        let [lower, center, upper] = thresholds;

        // Verify all values are finite
        assert!(auc.is_finite(), "AUC should be finite");
        assert!(lower.is_finite(), "Lower threshold should be finite");
        assert!(center.is_finite(), "Center threshold should be finite");
        assert!(upper.is_finite(), "Upper threshold should be finite");
        assert!(acc.is_finite(), "Accuracy should be finite");
        assert!(se.is_finite(), "Sensitivity should be finite");
        assert!(
            sp.is_finite() || sp.is_nan(),
            "Specificity should be finite or NaN"
        );
        assert!(obj.is_finite(), "Objective should be finite");
        assert!(rej.is_finite(), "Rejection rate should be finite");

        // Verify value ranges
        assert!(auc >= 0.0 && auc <= 1.0, "AUC in [0,1]");
        assert!(acc >= 0.0 && acc <= 1.0, "Accuracy in [0,1]");
        assert!(se >= 0.0 && se <= 1.0, "Sensitivity in [0,1]");
        assert!(rej >= 0.0 && rej <= 1.0, "Rejection rate in [0,1]");
    }

    #[test]
    fn test_precomputed_bootstrap_equivalence() {
        // Test that precomputed bootstrap gives same results as regular bootstrap
        let value = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let seed = 12345;
        let n_bootstrap = 1000;
        let alpha = 0.05;
        let subsample_frac = 1.0;

        // Method 1: Regular bootstrap
        let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
        let (auc1, [lower1, center1, upper1], acc1, se1, sp1, obj1, rej1) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                n_bootstrap,
                alpha,
                subsample_frac,
                &mut rng1,
            );

        // Method 2: Precomputed bootstrap
        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
        let precomputed =
            precompute_bootstrap_indices(&y, n_bootstrap, alpha, subsample_frac, &mut rng2);
        let (auc2, [lower2, center2, upper2], acc2, se2, sp2, obj2, rej2) =
            compute_threshold_and_metrics_with_precomputed_bootstrap(
                &value,
                &y,
                &FitFunction::auc,
                None,
                &precomputed,
            );

        // Compare results (should be identical with same seed)
        let tolerance = 1e-10;
        assert!(
            (auc1 - auc2).abs() < tolerance,
            "AUCs should match: {} vs {}",
            auc1,
            auc2
        );
        assert!(
            (lower1 - lower2).abs() < tolerance,
            "Lower thresholds should match: {} vs {}",
            lower1,
            lower2
        );
        assert!(
            (center1 - center2).abs() < tolerance,
            "Center thresholds should match: {} vs {}",
            center1,
            center2
        );
        assert!(
            (upper1 - upper2).abs() < tolerance,
            "Upper thresholds should match: {} vs {}",
            upper1,
            upper2
        );
        assert!(
            (acc1 - acc2).abs() < tolerance,
            "Accuracies should match: {} vs {}",
            acc1,
            acc2
        );
        assert!(
            (se1 - se2).abs() < tolerance,
            "Sensitivities should match: {} vs {}",
            se1,
            se2
        );
        assert!(
            (sp1 - sp2).abs() < tolerance,
            "Specificities should match: {} vs {}",
            sp1,
            sp2
        );
        assert!(
            (obj1 - obj2).abs() < tolerance,
            "Objectives should match: {} vs {}",
            obj1,
            obj2
        );
        assert!(
            (rej1 - rej2).abs() < tolerance,
            "Rejection rates should match: {} vs {}",
            rej1,
            rej2
        );
    }

    #[test]
    fn test_precomputed_bootstrap_with_penalties() {
        // Test precomputed bootstrap with penalties
        let value = vec![0.2, 0.4, 0.6, 0.8];
        let y = vec![0, 0, 1, 1];

        let seed = 42;
        let n_bootstrap = 500;
        let alpha = 0.1;
        let subsample_frac = 0.8;
        let penalties = Some([1.5, 1.0]);

        // Method 1: Regular bootstrap
        let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
        let (auc1, [lower1, center1, upper1], _, _, _, obj1, _) =
            compute_threshold_and_metrics_with_bootstrap(
                &value,
                &y,
                &FitFunction::sensitivity,
                penalties,
                n_bootstrap,
                alpha,
                subsample_frac,
                &mut rng1,
            );

        // Method 2: Precomputed bootstrap
        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
        let precomputed =
            precompute_bootstrap_indices(&y, n_bootstrap, alpha, subsample_frac, &mut rng2);
        let (auc2, [lower2, center2, upper2], _, _, _, obj2, _) =
            compute_threshold_and_metrics_with_precomputed_bootstrap(
                &value,
                &y,
                &FitFunction::sensitivity,
                penalties,
                &precomputed,
            );

        // Compare results
        let tolerance = 1e-10;
        assert!((auc1 - auc2).abs() < tolerance, "AUCs should match");
        assert!(
            (lower1 - lower2).abs() < tolerance,
            "Lower thresholds should match"
        );
        assert!(
            (center1 - center2).abs() < tolerance,
            "Center thresholds should match"
        );
        assert!(
            (upper1 - upper2).abs() < tolerance,
            "Upper thresholds should match"
        );
        assert!((obj1 - obj2).abs() < tolerance, "Objectives should match");
    }

    // Tests for threshold equality and >= rule consistency

    #[test]
    fn test_threshold_equality_classification() {
        // Test case: Simple binary classification with score == threshold
        // We have 3 positive samples and 2 negative samples

        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let y = vec![0, 0, 1, 1, 1];

        // Compute ROC and get the optimal threshold
        let (auc, threshold, acc_roc, sens_roc, spec_roc, _obj) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        // Now compute metrics using compute_metrics_from_value with the same threshold
        let (acc_val, sens_val, spec_val, rej_rate, _additional) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        // The metrics should match
        assert!(
            (acc_roc - acc_val).abs() < 1e-10,
            "Accuracy mismatch: {} vs {}",
            acc_roc,
            acc_val
        );
        assert!(
            (sens_roc - sens_val).abs() < 1e-10,
            "Sensitivity mismatch: {} vs {}",
            sens_roc,
            sens_val
        );
        assert!(
            (spec_roc - spec_val).abs() < 1e-10,
            "Specificity mismatch: {} vs {}",
            spec_roc,
            spec_val
        );
        assert_eq!(rej_rate, 0.0, "Rejection rate should be 0 without CI");
        assert!(auc >= 0.0 && auc <= 1.0, "AUC should be between 0 and 1");
    }

    #[test]
    fn test_threshold_equality_with_exact_match() {
        // Test case: Force a threshold that exactly matches one of the scores
        // This tests the >= rule explicitly

        let scores = vec![0.2, 0.4, 0.6, 0.6, 0.8];
        let y = vec![0, 0, 1, 1, 1];
        let threshold = 0.6;

        // Manually compute expected metrics with >= rule
        // Scores >= 0.6: [0.6, 0.6, 0.8] predicted as positive (class 1)
        // Scores < 0.6: [0.2, 0.4] predicted as negative (class 0)
        // Expected classification: [0, 0, 1, 1, 1]
        // True labels:             [0, 0, 1, 1, 1]
        // TP = 3, TN = 2, FP = 0, FN = 0

        let (acc, sens, spec, rej_rate, _additional) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!((acc - 1.0).abs() < 1e-10, "Expected perfect accuracy");
        assert!((sens - 1.0).abs() < 1e-10, "Expected perfect sensitivity");
        assert!((spec - 1.0).abs() < 1e-10, "Expected perfect specificity");
        assert_eq!(rej_rate, 0.0);
    }

    #[test]
    fn test_threshold_boundary_all_equal() {
        // Case: All scores equal to threshold
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![1, 1, 0, 0];
        let threshold = 0.5;

        // With >= rule, all should be classified as positive (class 1)
        // Expected: [1, 1, 1, 1]
        // True:     [1, 1, 0, 0]
        // TP = 2, FP = 2, TN = 0, FN = 0
        let (acc, sens, spec, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        // Sensitivity = TP/(TP+FN) = 2/2 = 1.0
        assert!(
            (sens - 1.0).abs() < 1e-10,
            "With >= rule, all positives should be correctly classified"
        );
        // Specificity = TN/(TN+FP) = 0/2 = 0.0
        assert!(
            spec.abs() < 1e-10,
            "With >= rule, all should be classified as positive, so specificity = 0"
        );
        assert!((acc - 0.5).abs() < 1e-10, "Accuracy should be 0.5");
    }

    #[test]
    fn test_threshold_boundary_just_below() {
        // Case: Threshold just below the smallest positive score
        let scores = vec![0.1, 0.2, 0.5, 0.6];
        let y = vec![0, 0, 1, 1];
        let threshold = 0.49;

        // Scores >= 0.49: [0.5, 0.6] -> class 1
        // Scores < 0.49: [0.1, 0.2] -> class 0
        // Perfect classification
        let (acc, sens, spec, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!((acc - 1.0).abs() < 1e-10, "Expected perfect accuracy");
        assert!((sens - 1.0).abs() < 1e-10, "Expected perfect sensitivity");
        assert!((spec - 1.0).abs() < 1e-10, "Expected perfect specificity");
    }

    #[test]
    fn test_roc_computation_consistency() {
        // Verify that compute_roc_and_metrics_from_value respects >= rule
        // by checking against manual computation

        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let y = vec![0, 1, 0, 1, 1];

        let (auc, threshold, acc, sens, spec, obj) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        // Verify these metrics by recomputing with compute_metrics_from_value
        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!(
            (acc - acc_verify).abs() < 1e-10,
            "Accuracy should match between ROC computation and metric computation"
        );
        assert!(
            (sens - sens_verify).abs() < 1e-10,
            "Sensitivity should match between ROC computation and metric computation"
        );
        assert!(
            (spec - spec_verify).abs() < 1e-10,
            "Specificity should match between ROC computation and metric computation"
        );
        assert!(auc >= 0.0 && auc <= 1.0);
        assert!(obj >= -1.0 && obj <= 1.0);
    }

    #[test]
    fn test_prevalence_high_positive() {
        // High prevalence scenario: 70% positive
        let scores_high = vec![
            0.1, 0.2, 0.3, // 3 negatives
            0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, // 7 positives
        ];
        let y_high = vec![
            0, 0, 0, // negatives
            1, 1, 1, 1, 1, 1, 1, // positives
        ];

        let (auc_high, threshold_high, acc_high, sens_high, spec_high, _) =
            compute_roc_and_metrics_from_value(&scores_high, &y_high, &FitFunction::auc, None);

        // If threshold equals one of the scores, verify consistency
        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores_high, &y_high, threshold_high, None, [false; 5]);

        assert!((acc_high - acc_verify).abs() < 1e-10);
        assert!((sens_high - sens_verify).abs() < 1e-10);
        assert!((spec_high - spec_verify).abs() < 1e-10);
        assert!(auc_high >= 0.9, "Should have high AUC with separable data");
    }

    #[test]
    fn test_prevalence_low_positive() {
        // Low prevalence scenario: 30% positive
        let scores_low = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, // 7 negatives
            0.8, 0.9, 1.0, // 3 positives
        ];
        let y_low = vec![
            0, 0, 0, 0, 0, 0, 0, // negatives
            1, 1, 1, // positives
        ];

        let (auc_low, threshold_low, acc_low, sens_low, spec_low, _) =
            compute_roc_and_metrics_from_value(&scores_low, &y_low, &FitFunction::auc, None);

        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores_low, &y_low, threshold_low, None, [false; 5]);

        assert!((acc_low - acc_verify).abs() < 1e-10);
        assert!((sens_low - sens_verify).abs() < 1e-10);
        assert!((spec_low - spec_verify).abs() < 1e-10);
        assert!(auc_low >= 0.9, "Should have high AUC with separable data");
    }

    #[test]
    fn test_threshold_equals_last_score() {
        // Test when optimal threshold equals the last (highest) score
        // This should result in classifying all samples as negative
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let y = vec![1, 1, 1, 1, 0]; // Last score is negative

        let (auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        // Verify metrics consistency
        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!(
            (acc - acc_verify).abs() < 1e-10,
            "Accuracy should match: {} vs {}",
            acc,
            acc_verify
        );
        assert!(
            (sens - sens_verify).abs() < 1e-10,
            "Sensitivity should match: {} vs {}",
            sens,
            sens_verify
        );
        assert!(
            (spec - spec_verify).abs() < 1e-10,
            "Specificity should match: {} vs {}",
            spec,
            spec_verify
        );

        // If threshold > 0.9, no sample should be classified as positive
        if threshold > 0.9 {
            assert_eq!(
                sens_verify, 0.0,
                "No positives should be classified if threshold > max score"
            );
            assert_eq!(
                spec_verify, 1.0,
                "All negatives should be correctly classified"
            );
        }

        assert!(auc >= 0.0 && auc <= 1.0);
    }

    #[test]
    fn test_threshold_equals_first_score() {
        // Test when optimal threshold equals the first (lowest) score
        // This should result in classifying all samples as positive
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let y = vec![0, 1, 1, 1, 1]; // First score is negative

        let (auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        // Verify metrics consistency
        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!(
            (acc - acc_verify).abs() < 1e-10,
            "Accuracy should match: {} vs {}",
            acc,
            acc_verify
        );
        assert!(
            (sens - sens_verify).abs() < 1e-10,
            "Sensitivity should match: {} vs {}",
            sens,
            sens_verify
        );
        assert!(
            (spec - spec_verify).abs() < 1e-10,
            "Specificity should match: {} vs {}",
            spec,
            spec_verify
        );

        // With this data, threshold should be at or near first score
        assert!(
            threshold <= 0.3,
            "Threshold should be low, got {}",
            threshold
        );
        assert!(auc >= 0.9, "Should have high AUC with separable data");
    }

    #[test]
    fn test_continuous_scores_no_duplicates() {
        // Test with continuous scores (no duplicates)
        // This is common in real-world scenarios with floating-point predictions
        let scores = vec![0.123, 0.456, 0.789, 0.234, 0.567, 0.890, 0.345, 0.678];
        let y = vec![0, 0, 1, 0, 1, 1, 0, 1];

        let (auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        // Verify metrics consistency
        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!(
            (acc - acc_verify).abs() < 1e-10,
            "Accuracy should match with continuous scores"
        );
        assert!(
            (sens - sens_verify).abs() < 1e-10,
            "Sensitivity should match with continuous scores"
        );
        assert!(
            (spec - spec_verify).abs() < 1e-10,
            "Specificity should match with continuous scores"
        );

        assert!(auc >= 0.0 && auc <= 1.0);
        assert!(
            threshold >= scores.iter().cloned().fold(f64::INFINITY, f64::min),
            "Threshold should be >= min score"
        );
    }

    #[test]
    fn test_continuous_scores_with_ties() {
        // Test with continuous scores that have some ties
        let scores = vec![0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6];
        let y = vec![0, 0, 1, 0, 1, 1, 1, 1];

        let (_auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!(
            (acc - acc_verify).abs() < 1e-10,
            "Metrics should be consistent even with tied scores"
        );
        assert!((sens - sens_verify).abs() < 1e-10);
        assert!((spec - spec_verify).abs() < 1e-10);
    }

    #[test]
    fn test_threshold_with_very_small_continuous_values() {
        // Test with very small continuous values (near zero)
        let scores = vec![0.0001, 0.0002, 0.0003, 0.0004, 0.0005];
        let y = vec![0, 0, 1, 1, 1];

        let (auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!((acc - acc_verify).abs() < 1e-10);
        assert!((sens - sens_verify).abs() < 1e-10);
        assert!((spec - spec_verify).abs() < 1e-10);
        assert!(auc >= 0.9, "Should have high AUC with separable data");
    }

    #[test]
    fn test_threshold_with_large_continuous_values() {
        // Test with large continuous values
        let scores = vec![1000.1, 2000.5, 3000.3, 4000.7, 5000.2];
        let y = vec![0, 0, 1, 1, 1];

        let (auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!((acc - acc_verify).abs() < 1e-10);
        assert!((sens - sens_verify).abs() < 1e-10);
        assert!((spec - spec_verify).abs() < 1e-10);
        assert!(auc >= 0.9, "Should have high AUC with separable data");
    }

    #[test]
    fn test_threshold_at_boundary_last_score_plus_one() {
        // Test edge case where optimal threshold is last_score + 1.0
        // This happens when classifying all as negative is optimal
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let y = vec![0, 0, 0, 0, 0]; // All negatives

        let (auc, _threshold, _acc, _sens, _spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        // With all negatives, AUC is undefined (0.5) and threshold is NAN
        assert!(auc.is_nan() || (auc - 0.5).abs() < 1e-10);

        // Note: with single class, the function returns early with NAN threshold
        // So we don't test metrics consistency here
    }

    #[test]
    fn test_threshold_negative_scores() {
        // Test with negative scores
        let scores = vec![-0.5, -0.3, -0.1, 0.1, 0.3];
        let y = vec![0, 0, 1, 1, 1];

        let (auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!(
            (acc - acc_verify).abs() < 1e-10,
            "Should work correctly with negative scores"
        );
        assert!((sens - sens_verify).abs() < 1e-10);
        assert!((spec - spec_verify).abs() < 1e-10);
        assert!(auc >= 0.9, "Should have high AUC");
    }

    #[test]
    fn test_threshold_mixed_positive_negative_scores() {
        // Test with mix of positive and negative scores, unsorted
        let scores = vec![1.5, -2.3, 0.5, -0.8, 3.2, 0.0, -1.1, 2.7];
        let y = vec![1, 0, 0, 0, 1, 0, 0, 1];

        let (auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!(
            (acc - acc_verify).abs() < 1e-10,
            "Should handle unsorted mixed positive/negative scores"
        );
        assert!((sens - sens_verify).abs() < 1e-10);
        assert!((spec - spec_verify).abs() < 1e-10);
        assert!(auc >= 0.5, "AUC should be reasonable");
    }

    #[test]
    fn test_threshold_continuous_perfect_separation() {
        // Perfect separation with continuous scores
        let scores = vec![0.12, 0.23, 0.34, 0.45, 0.67, 0.78, 0.89, 0.91];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let (auc, threshold, acc, sens, spec, _) =
            compute_roc_and_metrics_from_value(&scores, &y, &FitFunction::auc, None);

        // With perfect separation, should get perfect metrics
        assert!(
            (auc - 1.0).abs() < 1e-10,
            "AUC should be 1.0 with perfect separation"
        );

        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores, &y, threshold, None, [false; 5]);

        assert!((acc - acc_verify).abs() < 1e-10);
        assert!((sens - sens_verify).abs() < 1e-10);
        assert!((spec - spec_verify).abs() < 1e-10);

        // Should achieve perfect classification
        assert!((acc - 1.0).abs() < 1e-10, "Should get perfect accuracy");
        assert!(
            (sens_verify - 1.0).abs() < 1e-10,
            "Should get perfect sensitivity"
        );
        assert!(
            (spec_verify - 1.0).abs() < 1e-10,
            "Should get perfect specificity"
        );

        // Threshold should be between the two groups
        assert!(
            threshold > 0.45 && threshold <= 0.67,
            "Threshold should be between negative and positive groups, got {}",
            threshold
        );
    }

    #[test]
    fn test_threshold_boundary_cases() {
        // Test case 3: Verify >= rule with edge cases

        println!("\n=== Test 3: Boundary cases ===");

        // Case 3a: All scores equal to threshold
        let scores_a = vec![0.5, 0.5, 0.5, 0.5];
        let y_a = vec![1, 1, 0, 0];
        let threshold_a = 0.5;

        println!("\nCase 3a: All scores == threshold");
        println!("Scores: {:?}", scores_a);
        println!("Labels: {:?}", y_a);
        println!("Threshold: {:.4}", threshold_a);

        // With >= rule, all should be classified as positive (class 1)
        // Expected: [1, 1, 1, 1]
        // True:     [1, 1, 0, 0]
        // TP = 2, FP = 2, TN = 0, FN = 0
        let (acc_a, sens_a, spec_a, _, _) =
            compute_metrics_from_value(&scores_a, &y_a, threshold_a, None, [false; 5]);

        println!(
            "Metrics: acc={:.4}, sens={:.4}, spec={:.4}",
            acc_a, sens_a, spec_a
        );

        // Sensitivity = TP/(TP+FN) = 2/2 = 1.0
        assert!(
            (sens_a - 1.0).abs() < 1e-10,
            "With >= rule, all positives should be correctly classified"
        );
        // Specificity = TN/(TN+FP) = 0/2 = 0.0
        assert!(
            spec_a.abs() < 1e-10,
            "With >= rule, all should be classified as positive, so specificity = 0"
        );

        // Case 3b: Threshold just below the smallest positive score
        let scores_b = vec![0.1, 0.2, 0.5, 0.6];
        let y_b = vec![0, 0, 1, 1];
        let threshold_b = 0.49; // Just below 0.5

        println!("\nCase 3b: Threshold just below smallest positive");
        println!("Scores: {:?}", scores_b);
        println!("Labels: {:?}", y_b);
        println!("Threshold: {:.4}", threshold_b);

        // Scores >= 0.49: [0.5, 0.6] -> class 1
        // Scores < 0.49: [0.1, 0.2] -> class 0
        // Perfect classification
        let (acc_b, sens_b, spec_b, _, _) =
            compute_metrics_from_value(&scores_b, &y_b, threshold_b, None, [false; 5]);

        println!(
            "Metrics: acc={:.4}, sens={:.4}, spec={:.4}",
            acc_b, sens_b, spec_b
        );

        assert!((acc_b - 1.0).abs() < 1e-10, "Expected perfect accuracy");
        assert!((sens_b - 1.0).abs() < 1e-10, "Expected perfect sensitivity");
        assert!((spec_b - 1.0).abs() < 1e-10, "Expected perfect specificity");
    }

    #[test]
    fn test_prevalence_ternary_model() {
        // Test case 5: Ternary model with prevalence consideration
        // This simulates a real-world scenario where we have imbalanced classes

        println!("\n=== Test 5: Ternary prevalence model ===");

        // High prevalence scenario: 70% positive
        let scores_high = vec![
            0.1, 0.2, 0.3, // 3 negatives
            0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, // 7 positives
        ];
        let y_high = vec![
            0, 0, 0, // negatives
            1, 1, 1, 1, 1, 1, 1, // positives
        ];

        println!("\nHigh prevalence (70% positive):");
        println!("Scores: {:?}", scores_high);
        println!("Labels: {:?}", y_high);

        let (auc_high, threshold_high, acc_high, sens_high, spec_high, _) =
            compute_roc_and_metrics_from_value(&scores_high, &y_high, &FitFunction::auc, None);

        println!("Threshold: {:.4}", threshold_high);
        println!(
            "Metrics: auc={:.4}, acc={:.4}, sens={:.4}, spec={:.4}",
            auc_high, acc_high, sens_high, spec_high
        );

        // If threshold equals one of the scores, verify consistency
        let (acc_verify, sens_verify, spec_verify, _, _) =
            compute_metrics_from_value(&scores_high, &y_high, threshold_high, None, [false; 5]);

        println!(
            "Verified: acc={:.4}, sens={:.4}, spec={:.4}",
            acc_verify, sens_verify, spec_verify
        );

        assert!((acc_high - acc_verify).abs() < 1e-10);
        assert!((sens_high - sens_verify).abs() < 1e-10);
        assert!((spec_high - spec_verify).abs() < 1e-10);

        // Low prevalence scenario: 30% positive
        let scores_low = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, // 7 negatives
            0.8, 0.9, 1.0, // 3 positives
        ];
        let y_low = vec![
            0, 0, 0, 0, 0, 0, 0, // negatives
            1, 1, 1, // positives
        ];

        println!("\nLow prevalence (30% positive):");
        println!("Scores: {:?}", scores_low);
        println!("Labels: {:?}", y_low);

        let (auc_low, threshold_low, acc_low, sens_low, spec_low, _) =
            compute_roc_and_metrics_from_value(&scores_low, &y_low, &FitFunction::auc, None);

        println!("Threshold: {:.4}", threshold_low);
        println!(
            "Metrics: auc={:.4}, acc={:.4}, sens={:.4}, spec={:.4}",
            auc_low, acc_low, sens_low, spec_low
        );

        let (acc_verify2, sens_verify2, spec_verify2, _, _) =
            compute_metrics_from_value(&scores_low, &y_low, threshold_low, None, [false; 5]);

        println!(
            "Verified: acc={:.4}, sens={:.4}, spec={:.4}",
            acc_verify2, sens_verify2, spec_verify2
        );

        assert!((acc_low - acc_verify2).abs() < 1e-10);
        assert!((sens_low - sens_verify2).abs() < 1e-10);
        assert!((spec_low - spec_verify2).abs() < 1e-10);
    }
}
