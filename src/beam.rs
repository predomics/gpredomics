// BE CAREFUL : the adaptation of the Predomics beam algorithm for Gpredomics is still under development
use crate::cinfo;
use crate::cv::CV;
use crate::data::Data;
use crate::gpu::GpuAssay;
use crate::individual::data_type;
use crate::individual::language;
use crate::individual::AdditionalMetrics;
use crate::individual::Individual;
use crate::individual::ThresholdCI;
use crate::individual::BINARY_LANG;
use crate::individual::POW2_LANG;
use crate::individual::RATIO_LANG;
use crate::individual::RAW_TYPE;
use crate::individual::TERNARY_LANG;
use crate::param::Param;
use crate::population::Population;
use crate::utils::{display_epoch, display_epoch_legend};
use log::{debug, error, info, warn};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

//-----------------------------------------------------------------------------
// Mathematical utilities
//-----------------------------------------------------------------------------

fn generate_combinations(features: &Vec<usize>, k: usize) -> Vec<Vec<usize>> {
    let mut combinations = HashSet::new();
    let mut indices = (0..k).collect::<Vec<_>>();

    if k > features.len() || k == 0 {
        return vec![];
    }

    loop {
        let combination: Vec<usize> = indices.iter().map(|&i| features[i]).collect();
        combinations.insert(combination);

        let mut i = k as isize - 1;
        while i >= 0 && indices[i as usize] == features.len() - k + i as usize {
            i -= 1;
        }

        if i < 0 {
            break;
        }

        indices[i as usize] += 1;
        for j in i + 1..k as isize {
            indices[j as usize] = indices[(j - 1) as usize] + 1;
        }
    }

    combinations.into_iter().collect()
}

fn combine_with_best(best_combinations: Vec<Vec<usize>>, features: &Vec<usize>) -> Vec<Vec<usize>> {
    let single_feature_combinations = generate_combinations(features, 1);

    let all_combinations: Vec<Vec<usize>> = best_combinations
        .into_par_iter()
        .flat_map(|best_combination| {
            single_feature_combinations
                .iter()
                .filter_map(|single_feature| {
                    let feature = single_feature[0];
                    if !best_combination.contains(&feature) {
                        let mut combined_vec = best_combination.clone();
                        combined_vec.push(feature);
                        combined_vec.sort();
                        Some(combined_vec)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let unique_combinations: HashSet<Vec<usize>> = all_combinations.into_iter().collect();
    unique_combinations.into_iter().collect()
}

fn binomial_coefficient(n: u128, k: u128) -> u128 {
    if k > n {
        return 0;
    }
    let k = if k > n / 2 { n - k } else { k };

    let mut result = 1u128;
    for i in 0..k {
        let numerator = n - i;
        let denominator = i + 1;

        if let Some(new_result) = result.checked_mul(numerator) {
            result = new_result / denominator;
        } else {
            error!("Error: Arithmetic overflow in binomial coefficient calculation (there are more than 2e128-1 possible combinations !) \
                \nSuggestions: Reduce k_max, make the preselection criteria for features considerably stricter, or even better, set a max_nb_of_models");
            panic!();
        }
    }
    result
}

fn max_n_for_combinations(k: u128, target: u128) -> u128 {
    let mut n = k;
    loop {
        let combinations = binomial_coefficient(n, k);
        if combinations > target {
            break;
        }
        n += 1;
    }
    n - 1
}

//-----------------------------------------------------------------------------
// Individual & Population utilities
//-----------------------------------------------------------------------------

fn pop_from_combinations(
    features_combination: Vec<Vec<usize>>,
    pattern_ind: Individual,
    expected_k: usize,
    param: &Param,
) -> Population {
    let languages: Vec<u8> = param
        .general
        .language
        .split(",")
        .map(language)
        .filter(|x| *x != POW2_LANG)
        .collect();
    let data_types: Vec<u8> = param.general.data_type.split(",").map(data_type).collect();

    let mut converted_count = 0;
    let mut ind_vec: Vec<Individual> = vec![];
    for combination in &features_combination {
        for language in &languages {
            for data_type in &data_types {
                let mut tmp_ind = pattern_ind.clone();

                tmp_ind.language = *language;
                tmp_ind.data_type = *data_type;
                tmp_ind.features = combination
                    .iter()
                    .filter_map(|&k| pattern_ind.features.get(&k).map(|&coeff| (k, coeff)))
                    .filter(|(_, coeff)| *coeff != 0)
                    .collect();

                // Convert to Binary if all features are positives
                let has_negative = tmp_ind.features.values().any(|&coef| coef < 0);
                if !has_negative && (tmp_ind.language == TERNARY_LANG) {
                    tmp_ind.features.retain(|_, &mut coeff| coeff > 0);
                    if !languages.contains(&BINARY_LANG) {
                        converted_count += 1;
                    };
                    tmp_ind.language = BINARY_LANG;
                } else if !has_negative && (tmp_ind.language == RATIO_LANG) {
                    tmp_ind.features = HashMap::new();
                }

                // Ter & Ratio without any positive features are meaningless
                let has_positive = tmp_ind.features.values().any(|&coef| coef > 0);
                if !has_positive
                    && (tmp_ind.language == TERNARY_LANG || tmp_ind.language == RATIO_LANG)
                {
                    tmp_ind.features = HashMap::new();
                }

                // Remove -1 for Binary Individual
                if language == &BINARY_LANG {
                    tmp_ind.features = tmp_ind
                        .features
                        .into_iter()
                        .filter(|(_, coeff)| *coeff > 0)
                        .collect();
                }

                tmp_ind.k = tmp_ind.features.len();

                // Prevent the generation of model with less than k features
                if tmp_ind.k == expected_k {
                    ind_vec.push(tmp_ind);
                }
            }
        }
    }

    if converted_count > 0 && !languages.contains(&BINARY_LANG) {
        warn!(
            "Converted {} TERNARY models without negatives features to BINARY. \
            Consider adding 'bin' to param.general.language if unintended.",
            converted_count
        )
    }

    let mut pop = Population {
        individuals: ind_vec,
    };
    pop.compute_hash();

    pop
}

// fn count_feature_appearances(population: &Population) -> HashMap<usize, usize> {
//     let mut feature_counts = HashMap::new();

//     for individual in &population.individuals {
//         for feature in individual.features.keys() {
//             *feature_counts.entry(*feature).or_insert(0) += 1;
//         }
//     }

//     feature_counts
// }

// fn get_important_features(feature_counts: HashMap<usize, usize>, nb_features_to_keep: usize) -> Vec<usize> {
//     let mut important_features: Vec<_> = feature_counts.into_iter().collect();
//     important_features.sort_by(|a, b| b.1.cmp(&a.1));
//     important_features.into_iter().take(nb_features_to_keep).map(|(feature, _)| feature).collect()
// }

fn select_features_from_best(best_pop: &Population) -> Vec<usize> {
    let mut unique_features = HashSet::new();
    let mut features: Vec<usize> = vec![];

    for individual in &best_pop.individuals {
        for best_feature in individual.features.keys() {
            unique_features.insert(*best_feature);
        }
    }

    debug!("Features kept: {:?}", unique_features.len());

    features.extend(unique_features.into_iter());

    features
}

pub fn generate_individual(data: &Data, language: u8, data_type: u8, param: &Param) -> Individual {
    let mut features = HashMap::new();
    for &feature_idx in &data.feature_selection {
        if let Some(&feature_class) = data.feature_class.get(&feature_idx) {
            let coefficient = if feature_class == 0 {
                if language != 0 {
                    -1
                } else {
                    0
                }
            } else {
                1
            };

            features.insert(feature_idx, coefficient);
        }
    }

    Individual {
        features: features.clone(),
        auc: 0.0,
        fit: 0.0,
        specificity: 0.0,
        sensitivity: 0.0,
        accuracy: 0.0,
        threshold: 0.0,
        threshold_ci: if param.general.threshold_ci_n_bootstrap > 0 {
            Some(ThresholdCI {
                lower: 0.0,
                upper: 0.0,
                rejection_rate: 0.0,
            })
        } else {
            None
        },
        k: features.len(),
        epoch: 0,
        language: language,
        data_type: data_type,
        hash: 0,
        epsilon: param.general.data_type_epsilon,
        parents: None,
        betas: None,
        metrics: AdditionalMetrics {
            mcc: None,
            f1_score: None,
            npv: None,
            ppv: None,
            g_means: None,
        },
    }
}

//-----------------------------------------------------------------------------
// Beam core functions
//-----------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum BeamMethod {
    #[serde(alias = "combinatorial")]
    LimitedExhaustive,
    #[serde(alias = "incremental")]
    ParallelForward,
}

pub fn beam(
    data: &mut Data,
    _no_overfit_data: &mut Option<Data>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> Vec<Population> {
    let time = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    cinfo!(
        param.general.display_colorful,
        "\x1b[1;96mLaunching Beam algorithm for a feature interval [{}, {}]\x1b[0m",
        param.beam.kmin,
        param.beam.kmax
    );

    data.select_features(param);
    debug!("FEATURES {:?}", data.feature_class);

    // Building the GPU assay and checking the prerequisites
    let gpu_assay = get_gpu_assay(data, param);

    // Initialize first population
    let base_pop = generate_pop(data, param);

    cinfo!(
        param.general.display_colorful,
        "{}",
        display_epoch_legend(param)
    );
    let populations = iterative_growth(&base_pop, &data, &gpu_assay, param, running, &mut rng);

    let elapsed = time.elapsed();
    info!(
        "Beam algorithm ({} mode) computed {:?} generations in {:.2?}",
        format!("{:?}", param.beam.method),
        populations.len(),
        elapsed
    );

    if populations.is_empty() {
        error!("Beam did not produce any results because the criteria for selecting the best features was too restrictive. Please lower best_models_ci_alpha to allow results.")
    }

    populations
}

pub fn generate_pop(data: &Data, param: &Param) -> Population {
    let initial_combinations = generate_combinations(&data.feature_selection, param.beam.kmin);
    let combinations = initial_combinations.clone();

    let pattern_ind = generate_individual(&data, TERNARY_LANG, RAW_TYPE, &param);
    let mut pop = pop_from_combinations(combinations.clone(), pattern_ind, param.beam.kmin, param);

    debug!("{} individuals generated", pop.individuals.len());

    pop = pop.sort();

    pop
}

pub fn iterative_growth(
    base_pop: &Population,
    data: &Data,
    gpu_assay: &Option<GpuAssay>,
    param: &Param,
    running: Arc<AtomicBool>,
    rng: &mut ChaCha8Rng,
) -> Vec<Population> {
    let mut epoch = param.beam.kmin;
    let mut cv: Option<CV> = None;
    let mut populations: Vec<Population> = vec![];

    let mut data_rng = rng.clone();
    //let mut evolution_rng = rng.clone();

    // Prepare epoch associated data
    let epoch_data = if param.cv.overfit_penalty != 0.0 {
        let folds_nb = if param.cv.inner_folds > 1 {
            param.cv.inner_folds
        } else {
            3
        };
        info!("Learning on {:?}-folds.", folds_nb);
        cv = Some(CV::new(&data, folds_nb, &mut data_rng));
        Data::new()
    } else {
        data.clone()
    };

    // Clean data before process
    let mut pop = base_pop.clone();
    pop.compute_hash();
    let _clone_number = pop.remove_clone();

    // Create GPU assays once for CV (will be reused across all epochs)
    let gpu_assays_per_fold = if let Some(ref cv) = cv {
        create_gpu_assays_for_folds_beam(cv, &param)
    } else {
        vec![]
    };

    // Fitting base population on data
    if let Some(ref cv) = cv {
        debug!("Fitting population on folds...");
        pop.fit_on_folds(cv, &param, &gpu_assays_per_fold);
    } else {
        debug!("Fitting population...");
        pop.fit(&epoch_data, &mut None, gpu_assay, &None, param);
    }

    pop = pop.sort();

    let mut last_pop = Population::new();
    //Grow!
    loop {
        epoch += 1;

        // Stop the loop to avoid a panic! if previous generation has not k features
        if epoch != 1 && pop.individuals[0].k < epoch - 2 {
            cinfo!(
                param.general.display_colorful,
                "\x1b[1;91mLimit reached: new models contain all pertinent features\x1b[0m"
            );

            if last_pop.individuals.len() > 0 {
                if param.general.keep_trace {
                    populations.push(pop);
                } else {
                    populations = vec![last_pop]
                }
            } else {
                if !param.general.keep_trace {
                    populations = vec![last_pop]
                }
            }

            break;
        }

        debug!("Generating models with {:?} features...", epoch);
        debug!(
            "[k={:?}] initial population length = {:?}",
            epoch,
            pop.individuals.len()
        );

        // Compute best population
        let best_pop = pop.select_best_population(param.beam.best_models_ci_alpha);
        debug!(
            "Kept {:?} individuals from the family of best models",
            best_pop.individuals.len()
        );

        // Generate new population based on best one - reuse GPU assays
        last_pop = pop;
        pop = grow(
            best_pop,
            data,
            &mut cv,
            param,
            gpu_assay,
            &gpu_assays_per_fold,
            epoch,
        );

        cinfo!(
            param.general.display_colorful,
            "{}",
            display_epoch(&pop, param, epoch)
        );

        // Stop critera
        let mut need_to_break = false;

        // Stop the loop if someone kill the program
        if !running.load(Ordering::Relaxed) {
            need_to_break = true;
        }

        // If the generated population is empty, stop it
        if pop.individuals.len() == 0 {
            need_to_break = true;
        }

        if param.beam.kmax == epoch {
            info!("Limite reached...");
            need_to_break = true;
        }

        if need_to_break {
            if pop.individuals.len() > 0 {
                if param.general.keep_trace {
                    populations.push(pop);
                } else {
                    populations = vec![pop]
                }
            } else {
                if !param.general.keep_trace {
                    populations = vec![last_pop]
                }
            }
            break;
        }

        if param.general.keep_trace {
            populations.push(pop.clone())
        };
    }

    populations
}

pub fn grow(
    best_pop: Population,
    data: &Data,
    cv: &mut Option<CV>,
    param: &Param,
    gpu_assay: &Option<GpuAssay>,
    gpu_assays_per_fold: &Vec<(Option<GpuAssay>, Option<GpuAssay>)>,
    epoch: usize,
) -> Population {
    let pattern_ind = generate_individual(data, TERNARY_LANG, RAW_TYPE, param);

    // Get pertinent features to use it for the new iteration
    let features_to_keep = select_features_from_best(&best_pop);

    let combinations = match param.beam.method {
        BeamMethod::LimitedExhaustive => combine(features_to_keep, epoch, data, param),
        BeamMethod::ParallelForward => increment(best_pop, features_to_keep, param),
    };

    if combinations.is_empty() {
        debug!(
            "No valid combinations for k={}; returning empty population...",
            epoch
        );
        return Population::new();
    }

    debug!(
        "{:?} unique combinations of features generated",
        combinations.len()
    );

    let mut new_pop = pop_from_combinations(combinations, pattern_ind, epoch, param);

    // Fit new population - reuse GPU assays
    if let Some(ref cv) = cv {
        debug!("Fitting children on folds...");
        new_pop.fit_on_folds(cv, &param, gpu_assays_per_fold);
    } else {
        debug!("Fitting children...");
        new_pop.fit(&data, &mut None, gpu_assay, &None, param);
    }

    new_pop = new_pop.sort();
    new_pop
}

// Generate new combinations Mk + 1 feature_to_keep for next step
// Combinations are limited both by kept Mk maximum (param.beam.max_nb_of_models) and features_to_keep (param.beam.best_models_ci_alpha)
// Combinations are currently generated at each epoch in each languages and data_type
pub fn increment(
    best_pop: Population,
    features_to_keep: Vec<usize>,
    param: &Param,
) -> Vec<Vec<usize>> {
    debug!("Selecting best combinations...");
    let potential_combinations = best_pop.individuals.len() * features_to_keep.len();

    let mut reduced_best_pop = Population::new();

    if potential_combinations > param.beam.max_nb_of_models && param.beam.max_nb_of_models != 0 {
        let max_parents = param.beam.max_nb_of_models / features_to_keep.len();
        let min_parents = std::cmp::min(5, best_pop.individuals.len());
        let adjusted_parents = std::cmp::max(max_parents, min_parents);

        debug!("Limiting parent models from {} to {} to respect max_nb_of_models={} (with {} features)",
            best_pop.individuals.len(), adjusted_parents, param.beam.max_nb_of_models, features_to_keep.len());

        reduced_best_pop.individuals = best_pop.individuals
            [..std::cmp::min(adjusted_parents, best_pop.individuals.len())]
            .to_vec();
    } else {
        reduced_best_pop.individuals = best_pop.individuals;
    }

    let best_combinations: Vec<Vec<usize>> = reduced_best_pop
        .individuals
        .clone()
        .par_iter()
        .map(|ind| ind.features.keys().cloned().collect())
        .collect();
    debug!(
        "Computing {} new combinations with these features...",
        best_combinations.len()
    );

    combine_with_best(best_combinations.clone(), &features_to_keep)
}

// Generate all possible combinations between the features_to_keep
// Combinations are limited by features_to_keep (by param.beam.best_models_ci_alpha)
// These features can be limited with param.beam.max_nb_of_models
pub fn combine(
    features_to_keep: Vec<usize>,
    k: usize,
    data: &Data,
    param: &Param,
) -> Vec<Vec<usize>> {
    let mut languages: Vec<u8> = param.general.language.split(",").map(language).collect();
    let mut combinations: Vec<Vec<usize>>;
    let bin_combinations: Vec<Vec<usize>>;

    let mut new_features_to_keep = features_to_keep.clone();
    let mut features_to_keep_neg: Vec<usize> = vec![];
    let mut features_to_keep_pos: Vec<usize> = vec![];
    let mut bin_features_to_keep: Vec<usize> = vec![];

    let possible_nb = binomial_coefficient(features_to_keep.len() as u128, k as u128);

    if possible_nb > param.beam.max_nb_of_models as u128 && param.beam.max_nb_of_models != 0 {
        let max_nb: usize =
            max_n_for_combinations(k as u128, param.beam.max_nb_of_models as u128) as usize;

        // Get features rank
        let iter = data
            .feature_selection
            .iter()
            .copied()
            .filter(|idx| features_to_keep.contains(idx))
            .filter_map(|idx| {
                let class = *data.feature_class.get(&idx)?;
                let value = *data.feature_significance.get(&idx)?;
                Some((idx, class, value))
            });

        let (class_0_features, class_1_features): (Vec<_>, Vec<_>) =
            iter.partition(|&(_, class, _)| class == 0);

        // For Binary languages, pick directly max_nb positive-associated features to avoid different ind_k within an iteration (like 10+10 features Ternary vs 10 features Binary for k=20)
        if languages.contains(&BINARY_LANG) {
            bin_features_to_keep = class_1_features
                .iter()
                .take(max_nb)
                .map(|&(index, _, _)| index)
                .collect();
            if bin_features_to_keep.len() <= k {
                debug!("Limit reached for Binary: skipping Binary...");
                languages.retain(|&x| x != BINARY_LANG);
            }
        }

        // For ternary and ratio, smartly pick the higher feature count leading to stay under max_nb_of_models
        if languages.contains(&TERNARY_LANG) || languages.contains(&RATIO_LANG) {
            let ideal_nb = max_nb / 2;
            let f0_nb = class_0_features.len();
            let f1_nb = class_1_features.len();
            if f0_nb < ideal_nb && f1_nb >= (ideal_nb + (ideal_nb - f0_nb)) {
                if max_nb > ideal_nb * 2 && f1_nb >= (ideal_nb + 1 + (ideal_nb - f0_nb)) {
                    features_to_keep_neg = class_0_features
                        .iter()
                        .take(f0_nb)
                        .map(|&(index, _, _)| index)
                        .collect();
                    features_to_keep_pos = class_1_features
                        .iter()
                        .take(ideal_nb + 1 + (ideal_nb - f0_nb))
                        .map(|&(index, _, _)| index)
                        .collect();
                } else {
                    features_to_keep_neg = class_0_features
                        .iter()
                        .take(f0_nb)
                        .map(|&(index, _, _)| index)
                        .collect();
                    features_to_keep_pos = class_1_features
                        .iter()
                        .take(ideal_nb + (ideal_nb - f0_nb))
                        .map(|&(index, _, _)| index)
                        .collect();
                }
            } else if f1_nb < ideal_nb && f0_nb >= (ideal_nb + (ideal_nb - f1_nb)) {
                if max_nb > ideal_nb * 2 && f0_nb >= (ideal_nb + 1 + (ideal_nb - f1_nb)) {
                    features_to_keep_neg = class_0_features
                        .iter()
                        .take(ideal_nb + (ideal_nb + 1 - f1_nb))
                        .map(|&(index, _, _)| index)
                        .collect();
                    features_to_keep_pos = class_1_features
                        .iter()
                        .take(f1_nb)
                        .map(|&(index, _, _)| index)
                        .collect();
                } else {
                    features_to_keep_neg = class_0_features
                        .iter()
                        .take(ideal_nb + (ideal_nb - f1_nb))
                        .map(|&(index, _, _)| index)
                        .collect();
                    features_to_keep_pos = class_1_features
                        .iter()
                        .take(f1_nb)
                        .map(|&(index, _, _)| index)
                        .collect();
                }
            } else {
                // if ideal_nb can be reached by both or can't be, .take(ideal_nb) (if less take stops before)
                if max_nb > ideal_nb * 2 {
                    // if max_nb is odd, take one additionnal positive-associated feature
                    features_to_keep_neg = class_0_features
                        .iter()
                        .take(ideal_nb)
                        .map(|&(index, _, _)| index)
                        .collect();
                    features_to_keep_pos = class_1_features
                        .iter()
                        .take(ideal_nb + 1)
                        .map(|&(index, _, _)| index)
                        .collect();
                } else {
                    features_to_keep_neg = class_0_features
                        .iter()
                        .take(ideal_nb)
                        .map(|&(index, _, _)| index)
                        .collect();
                    features_to_keep_pos = class_1_features
                        .iter()
                        .take(ideal_nb)
                        .map(|&(index, _, _)| index)
                        .collect();
                }
            }
            new_features_to_keep = features_to_keep_neg.clone();
            new_features_to_keep.extend(features_to_keep_pos.clone());
        }

        if languages.contains(&BINARY_LANG)
            && (languages.contains(&TERNARY_LANG) || languages.contains(&RATIO_LANG))
        {
            warn!("Too many features (leading to {} > {} models). Feature ideal count = {}. Keeping {}+{} features ({} {}-associated for Binary)", possible_nb, param.beam.max_nb_of_models, max_nb, features_to_keep_neg.len(), features_to_keep_pos.len(), bin_features_to_keep.len(), data.classes[1]);
            combinations = generate_combinations(&new_features_to_keep, k);
            bin_combinations = generate_combinations(&bin_features_to_keep, k);
            combinations.extend(bin_combinations);
        } else if languages.contains(&TERNARY_LANG) || languages.contains(&RATIO_LANG) {
            warn!("Too many features (leading to {} > {} models). Feature ideal count = {}. Keeping only {}+{} features.", possible_nb, param.beam.max_nb_of_models, max_nb, features_to_keep_neg.len(), features_to_keep_pos.len());
            combinations = generate_combinations(&new_features_to_keep, k);
        } else {
            warn!("Too many features (leading to {} > {} models). Feature ideal count = {}. Keeping {} {}-associated features for Binary", possible_nb, param.beam.max_nb_of_models, max_nb, bin_features_to_keep.len(), data.classes[1]);
            combinations = generate_combinations(&bin_features_to_keep, k);
        }
    } else {
        combinations = generate_combinations(&new_features_to_keep, k);
    }

    combinations
}

// Function to extract best models among all epochs, not necessarily having k_max features
pub fn keep_n_best_model_within_collection(collection: &Vec<Population>, n: usize) -> Population {
    let mut all_models = Population::new();
    for population in collection {
        for individual in population.individuals.clone() {
            all_models.individuals.push(individual);
        }
    }
    all_models = all_models.sort();

    let best_n_pop;
    if all_models.individuals.len() > n as usize && n as usize != 0 {
        best_n_pop = Population {
            individuals: all_models.individuals.clone()[..n as usize].to_vec(),
        }
    } else {
        best_n_pop = all_models
    }
    best_n_pop
}

fn get_gpu_assay(data: &Data, param: &Param) -> Option<GpuAssay> {
    let languages: Vec<u8> = param
        .general
        .language
        .split(",")
        .map(language)
        .filter(|x| *x != POW2_LANG)
        .collect();
    let data_types: Vec<u8> = param.general.data_type.split(",").map(data_type).collect();
    let lang_data_combinations = languages.len() * data_types.len();

    let gpu_assay = if param.general.gpu {
        let buffer_binding_size = GpuAssay::get_max_buffer_size(&param.gpu) as usize;
        let gpu_max_nb_models =
            buffer_binding_size / (data.sample_len * std::mem::size_of::<f32>());
        if param.beam.max_nb_of_models == 0 {
            warn!("GPU requires a maximum number of models. Setting max_nb_of_models={} to prevent crashes.", 
                gpu_max_nb_models / lang_data_combinations);
            None
        } else if (param.beam.max_nb_of_models as usize) * lang_data_combinations
            > gpu_max_nb_models
        {
            warn!("GPU requires a maximum number of models that you exceed (GPU max_nb_of_models = {}). \
            \nAccording to the input parameters, please fix max_nb_of_models to {} \
            \nIf your configuration supports it and you know what you're doing, consider alternatively increasing the size of the buffers to {:.0} MB (do not forget to adjust the total size accordingly) \
            \nThis Gpredomics session will therefore be launched without a GPU.", gpu_max_nb_models,
            gpu_max_nb_models/lang_data_combinations,
            ((param.beam.max_nb_of_models*lang_data_combinations) as usize * data.sample_len * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)+1.0);
            None
        } else {
            let max_nb = (param.beam.max_nb_of_models as usize) * lang_data_combinations;
            Some(GpuAssay::new(
                &data.X,
                &data.feature_selection,
                data.sample_len,
                max_nb,
                &param.gpu,
            ))
        }
    } else {
        None
    };

    gpu_assay
}

/// Create GPU assays for each fold when inner CV is enabled with GPU for beam algorithm
/// Returns a vector of tuples (validation_assay, training_assay) for each fold
fn create_gpu_assays_for_folds_beam(
    cv: &CV,
    param: &Param,
) -> Vec<(Option<GpuAssay>, Option<GpuAssay>)> {
    if !param.general.gpu {
        return vec![(None, None); cv.validation_folds.len()];
    }

    let languages: Vec<u8> = param
        .general
        .language
        .split(",")
        .map(language)
        .filter(|x| *x != POW2_LANG)
        .collect();
    let data_types: Vec<u8> = param.general.data_type.split(",").map(data_type).collect();
    let lang_data_combinations = languages.len() * data_types.len();

    cv.validation_folds.iter().enumerate().map(|(idx, fold)| {
        let buffer_binding_size = GpuAssay::get_max_buffer_size(&param.gpu) as usize;

        // GPU assay for validation fold
        let gpu_max_nb_models_val = buffer_binding_size / (fold.sample_len * std::mem::size_of::<f32>());
        let val_assay = if param.beam.max_nb_of_models == 0 {
            warn!("GPU cannot be used for validation fold {}: max_nb_of_models is 0", idx);
            None
        } else if (param.beam.max_nb_of_models as usize) * lang_data_combinations > gpu_max_nb_models_val {
            warn!("GPU cannot be used for validation fold {}: requires max {} models but {} requested",
                idx, gpu_max_nb_models_val, param.beam.max_nb_of_models * lang_data_combinations as usize);
            None
        } else {
            let max_nb = (param.beam.max_nb_of_models as usize) * lang_data_combinations;
            Some(GpuAssay::new(&fold.X, &fold.feature_selection, fold.sample_len, max_nb, &param.gpu))
        };

        // GPU assay for training set
        let training_set = &cv.training_sets[idx];
        let gpu_max_nb_models_train = buffer_binding_size / (training_set.sample_len * std::mem::size_of::<f32>());
        let train_assay = if param.beam.max_nb_of_models == 0 {
            warn!("GPU cannot be used for training set {}: max_nb_of_models is 0", idx);
            None
        } else if (param.beam.max_nb_of_models as usize) * lang_data_combinations > gpu_max_nb_models_train {
            warn!("GPU cannot be used for training set {}: requires max {} models but {} requested",
                idx, gpu_max_nb_models_train, param.beam.max_nb_of_models * lang_data_combinations as usize);
            None
        } else {
            let max_nb = (param.beam.max_nb_of_models as usize) * lang_data_combinations;
            Some(GpuAssay::new(&training_set.X, &training_set.feature_selection, training_set.sample_len, max_nb, &param.gpu))
        };

        (val_assay, train_assay)
    }).collect()
}

// still have to write unit-tests to confirm every function behavior in any situation
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_individual() -> Individual {
        Individual {
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
                g_means: None,
            },
        }
    }

    #[test]
    fn test_generate_combinations() {
        let ind: Individual = create_test_individual();
        let combination: Vec<Vec<usize>> =
            generate_combinations(&ind.features.keys().cloned().collect(), 2);
        let truth: Vec<Vec<usize>> = vec![
            vec![0, 3],
            vec![0, 2],
            vec![0, 1],
            vec![3, 2],
            vec![3, 1],
            vec![2, 1],
        ];

        let sorted_combination: Vec<Vec<usize>> = combination
            .into_iter()
            .map(|mut vec| {
                vec.sort();
                vec
            })
            .collect();

        let sorted_truth: Vec<Vec<usize>> = truth
            .into_iter()
            .map(|mut vec| {
                vec.sort();
                vec
            })
            .collect();

        assert_eq!(
            sorted_combination.len(),
            sorted_truth.len(),
            "Combinatorial of 2 out of 4 is 6 but the generated vector contain {:?} combinations",
            sorted_combination.len()
        );
        sorted_combination.iter().for_each(|vec| {
            assert!(
                sorted_truth.contains(vec),
                "{:?} is not a valid combination of length two for 0,1,2,3",
                vec
            );
        });
    }

    #[test]
    fn test_combine_with_best() {
        let ind: Individual = create_test_individual();
        let truth: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3]];

        // Generate sorted combinations to allow comparaison
        let combinations = generate_combinations(&ind.features.keys().cloned().collect(), 2);
        let mut sorted_combination: Vec<Vec<usize>> = combinations
            .into_iter()
            .map(|mut vec| {
                vec.sort();
                vec
            })
            .collect();
        sorted_combination.sort();

        // Combine the first 2 combinations with features of ind and sort it to allow comparaison
        let combined = combine_with_best(
            sorted_combination.into_iter().take(2).collect(),
            &ind.features.keys().cloned().collect(),
        );
        let sorted_combined: Vec<Vec<usize>> = combined
            .into_iter()
            .map(|mut vec| {
                vec.sort();
                vec
            })
            .collect();

        assert_eq!(sorted_combined.len(), truth.len(), "Combining [0, 1] and [0, 2] with 0/1/2/3 should lead to 3 new combinations ({:?} combinations generated)", sorted_combined.len());
        sorted_combined.iter().for_each(|vec| {
            assert!(
                truth.contains(vec),
                "Two best combinations [0, 1] and [0, 2] completed with 0/1/2/3 can't lead to {:?}",
                vec
            );
        });
    }

    fn create_test_individual_for_test_pop_from_combinations() -> Individual {
        Individual {
            features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(),
            auc: 0.4,
            fit: 0.8,
            specificity: 0.15,
            sensitivity: 0.16,
            accuracy: 0.23,
            threshold: 42.0,
            k: 4,
            epoch: 42,
            language: TERNARY_LANG,
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
                g_means: None,
            },
        }
    }

    #[test]
    fn test_pop_from_combinations_ternary_only() {
        let ind = create_test_individual_for_test_pop_from_combinations();

        let combinations = vec![
            vec![0, 1], // (0,1), (1,-1) → valid
            vec![0, 2], // (0,1), (2,1) → valid
            vec![0, 3], // (0,1) only → rejected (k=1)
        ];

        let mut param = Param::default();
        param.general.language = "ter".to_string();
        param.general.data_type = "raw".to_string();

        let generated = pop_from_combinations(combinations, ind, 2, &param);

        // Assert: population not empty
        assert!(
            !generated.individuals.is_empty(),
            "Population should not be empty for TERNARY language"
        );

        // Assert: all individuals have k=2
        for individual in &generated.individuals {
            assert_eq!(
                individual.k, 2,
                "All individuals must have k=2, got k={} with features {:?}",
                individual.k, individual.features
            );
        }

        // Assert: specific combination exists
        let features: Vec<Vec<(usize, i8)>> = generated
            .individuals
            .iter()
            .map(|ind| {
                let mut f: Vec<_> = ind
                    .features
                    .iter()
                    .map(|(feat, coeff)| (*feat, *coeff))
                    .collect();
                f.sort();
                f
            })
            .collect();

        assert!(
            features
                .iter()
                .any(|f| f.len() == 2 && f.contains(&(0, 1)) && f.contains(&(1, -1))),
            "Should have generated [0,1] with features [(0,1), (1,-1)]"
        );
    }

    #[test]
    fn test_pop_from_combinations_binary_only() {
        let ind = create_test_individual_for_test_pop_from_combinations();

        let combinations = vec![
            vec![0, 2], // (0,1), (2,1) → valid for BINARY
            vec![0, 1], // has (1,-1) → invalid for BINARY
        ];

        let mut param = Param::default();
        param.general.language = "bin".to_string();
        param.general.data_type = "raw".to_string();

        let generated = pop_from_combinations(combinations, ind, 2, &param);

        // Assert: at least one individual
        assert!(
            !generated.individuals.is_empty(),
            "Should generate at least one BINARY individual from [0,2]"
        );

        // Assert: all are BINARY
        for individual in &generated.individuals {
            assert_eq!(
                individual.language, BINARY_LANG,
                "All individuals should be BINARY"
            );

            // Assert: only positive coefficients
            for (_, coeff) in &individual.features {
                assert!(
                    *coeff > 0,
                    "BINARY must only have positive coefficients, found {}",
                    coeff
                );
            }

            assert_eq!(individual.k, 2, "BINARY individual must have k=2");
        }

        // Assert: [0,2] was extracted
        let has_0_2 = generated.individuals.iter().any(|ind| {
            ind.features.contains_key(&0)
                && ind.features.contains_key(&2)
                && ind.features.get(&0) == Some(&1)
                && ind.features.get(&2) == Some(&1)
        });
        assert!(has_0_2, "Should have generated [0,2]");
    }

    #[test]
    fn test_pop_from_combinations_mixed_languages() {
        let ind = create_test_individual_for_test_pop_from_combinations();

        let combinations = vec![
            vec![0, 1], // TERNARY only
            vec![0, 2], // Both BINARY and TERNARY
        ];

        let mut param = Param::default();
        param.general.language = "bin,ter".to_string();
        param.general.data_type = "raw".to_string();

        let generated = pop_from_combinations(combinations, ind, 2, &param);

        assert!(
            !generated.individuals.is_empty(),
            "Should generate individuals"
        );

        let mut _ternary_count = 0;
        let mut _binary_count = 0;

        for individual in &generated.individuals {
            match individual.language {
                TERNARY_LANG => _ternary_count += 1,
                BINARY_LANG => _binary_count += 1,
                _ => panic!("Unexpected language"),
            }

            assert_eq!(individual.k, 2, "All must have k=2");

            // Verify constraints per language
            if individual.language == BINARY_LANG {
                for (_, coeff) in &individual.features {
                    assert!(*coeff > 0, "BINARY: only positive");
                }
            }
        }
    }

    #[test]
    fn test_pop_from_combinations_zero_coefficient_filtering() {
        let ind = create_test_individual_for_test_pop_from_combinations();

        let combinations = vec![vec![0, 3]]; // (3,0) will be filtered → k=1

        let mut param = Param::default();
        param.general.language = "ter".to_string();
        param.general.data_type = "raw".to_string();

        let generated = pop_from_combinations(combinations, ind, 2, &param);

        // Should be rejected because k < expected_k
        assert_eq!(
            generated.individuals.len(),
            0,
            "Should reject [0,3]: k=1 < expected_k=2"
        );
    }

    #[test]
    fn test_pop_from_combinations_auto_conversion() {
        let ind = create_test_individual_for_test_pop_from_combinations();

        let combinations = vec![vec![0, 2]]; // No negatives → auto-converts

        let mut param = Param::default();
        param.general.language = "ter".to_string();
        param.general.data_type = "raw".to_string();

        let generated = pop_from_combinations(combinations, ind, 2, &param);

        assert!(
            !generated.individuals.is_empty(),
            "Should generate individual"
        );

        for individual in &generated.individuals {
            assert_eq!(individual.k, 2, "Should have k=2");

            // All coefficients should be positive
            for (_, coeff) in &individual.features {
                assert!(*coeff > 0, "All coefficients should be positive");
            }
        }
    }

    #[test]
    fn test_pop_from_combinations_comprehensive() {
        let ind = create_test_individual_for_test_pop_from_combinations();

        // Scenario 1: TERNARY valid
        {
            let combinations = vec![vec![0, 1]];
            let mut param = Param::default();
            param.general.language = "ter".to_string();
            param.general.data_type = "raw".to_string();

            let pop = pop_from_combinations(combinations, ind.clone(), 2, &param);
            assert_eq!(pop.individuals.len(), 1);
            assert_eq!(pop.individuals[0].k, 2);
        }

        // Scenario 2: BINARY valid
        {
            let combinations = vec![vec![0, 2]];
            let mut param = Param::default();
            param.general.language = "bin".to_string();
            param.general.data_type = "raw".to_string();

            let pop = pop_from_combinations(combinations, ind.clone(), 2, &param);
            assert_eq!(pop.individuals.len(), 1);
            assert_eq!(pop.individuals[0].k, 2);
        }

        // Scenario 3: Rejected
        {
            let combinations = vec![vec![0, 3]];
            let mut param = Param::default();
            param.general.language = "ter".to_string();
            param.general.data_type = "raw".to_string();

            let pop = pop_from_combinations(combinations, ind.clone(), 2, &param);
            assert_eq!(pop.individuals.len(), 0);
        }
    }

    #[test]
    fn test_binomial_coefficient_basic() {
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(10, 3), 120);
        assert_eq!(binomial_coefficient(5, 0), 1);
        assert_eq!(binomial_coefficient(5, 5), 1);
        assert_eq!(binomial_coefficient(1, 0), 1);
        assert_eq!(binomial_coefficient(0, 0), 1);
    }

    #[test]
    fn test_binomial_coefficient_edge_cases() {
        // k > n should return 0
        assert_eq!(binomial_coefficient(3, 5), 0);

        // Large but valid combinations
        assert_eq!(binomial_coefficient(20, 10), 184756);
        assert_eq!(binomial_coefficient(15, 5), 3003);
    }

    #[test]
    #[should_panic]
    fn test_binomial_coefficient_overflow() {
        // Should panic on overflow (too many combinations)
        binomial_coefficient(1000, 500);
    }

    #[test]
    fn test_max_n_for_combinations_basic() {
        // For k=2, target=10
        // C(4,2) = 6, C(5,2) = 10, C(6,2) = 15
        let n = max_n_for_combinations(2, 10);
        assert_eq!(n, 5, "For k=2 and target=10, max n should be 5");

        // Verify boundaries
        assert!(binomial_coefficient(n, 2) <= 10);
        assert!(binomial_coefficient(n + 1, 2) > 10);
    }

    #[test]
    fn test_max_n_for_combinations_various() {
        // k=3, target=35
        // C(6,3) = 20, C(7,3) = 35, C(8,3) = 56
        let n = max_n_for_combinations(3, 35);
        assert_eq!(n, 7);

        // k=1, target=10
        let n = max_n_for_combinations(1, 10);
        assert_eq!(n, 10);

        // k=2, target=1
        let n = max_n_for_combinations(2, 1);
        assert_eq!(n, 2);
    }

    #[test]
    fn test_select_features_from_best() {
        let mut pop = Population::new();

        let mut ind1 = Individual::new();
        ind1.features = vec![(0, 1), (1, -1), (2, 1)].into_iter().collect();
        ind1.auc = 0.9;

        let mut ind2 = Individual::new();
        ind2.features = vec![(1, 1), (3, 1), (5, -1)].into_iter().collect();
        ind2.auc = 0.85;

        let mut ind3 = Individual::new();
        ind3.features = vec![(2, 1), (4, 1)].into_iter().collect();
        ind3.auc = 0.88;

        pop.individuals = vec![ind1, ind2, ind3];

        let features = select_features_from_best(&pop);

        // Should contain all unique features
        assert!(features.contains(&0), "Should contain feature 0");
        assert!(features.contains(&1), "Should contain feature 1");
        assert!(features.contains(&2), "Should contain feature 2");
        assert!(features.contains(&3), "Should contain feature 3");
        assert!(features.contains(&4), "Should contain feature 4");
        assert!(features.contains(&5), "Should contain feature 5");
        assert_eq!(features.len(), 6, "Should have 6 unique features");
    }

    #[test]
    fn test_select_features_from_best_empty() {
        let pop = Population::new();
        let features = select_features_from_best(&pop);
        assert!(
            features.is_empty(),
            "Empty population should return no features"
        );
    }

    #[test]
    fn test_select_features_from_best_single() {
        let mut pop = Population::new();
        let mut ind = Individual::new();
        ind.features = vec![(10, 1), (20, -1)].into_iter().collect();
        pop.individuals = vec![ind];

        let features = select_features_from_best(&pop);
        assert_eq!(features.len(), 2);
        assert!(features.contains(&10));
        assert!(features.contains(&20));
    }

    #[test]
    fn test_keep_n_best_model_within_collection() {
        let mut pop1 = Population::new();
        let mut ind1 = Individual::new();
        ind1.auc = 0.9;
        ind1.fit = 0.9;
        pop1.individuals = vec![ind1];

        let mut pop2 = Population::new();
        let mut ind2 = Individual::new();
        ind2.auc = 0.95;
        ind2.fit = 0.95;
        let mut ind3 = Individual::new();
        ind3.auc = 0.85;
        ind3.fit = 0.85;
        pop2.individuals = vec![ind2, ind3];

        let mut pop3 = Population::new();
        let mut ind4 = Individual::new();
        ind4.auc = 0.92;
        ind4.fit = 0.92;
        pop3.individuals = vec![ind4];

        let collection = vec![pop1, pop2, pop3];
        let best = keep_n_best_model_within_collection(&collection, 2);

        assert_eq!(best.individuals.len(), 2, "Should keep only 2 best");
        assert_eq!(
            best.individuals[0].auc, 0.95,
            "First should have highest AUC"
        );
        assert_eq!(
            best.individuals[1].auc, 0.92,
            "Second should have second highest AUC"
        );
    }

    #[test]
    fn test_keep_n_best_model_within_collection_request_more_than_available() {
        let mut pop1 = Population::new();
        let mut ind1 = Individual::new();
        ind1.auc = 0.9;
        ind1.fit = 0.9;
        pop1.individuals = vec![ind1];

        let collection = vec![pop1];
        let best = keep_n_best_model_within_collection(&collection, 10);

        // Should return all available (1) even though we asked for 10
        assert_eq!(best.individuals.len(), 1);
    }

    #[test]
    fn test_keep_n_best_model_within_collection_empty() {
        let collection: Vec<Population> = vec![];
        let best = keep_n_best_model_within_collection(&collection, 5);
        assert_eq!(best.individuals.len(), 0);
    }

    #[test]
    fn test_get_gpu_assay_disabled() {
        // Create minimal data
        let mut data = Data::new();
        data.samples = vec!["s1".to_string(), "s2".to_string(), "s3".to_string()];
        data.sample_len = 3;
        data.feature_len = 5;

        let mut param = Param::default();
        param.general.gpu = false;

        let gpu_assay = get_gpu_assay(&data, &param);
        assert!(
            gpu_assay.is_none(),
            "GPU assay should be None when disabled"
        );
    }

    #[test]
    #[ignore = "Requires GPU hardware - run with 'cargo test -- --ignored' if GPU is available"]
    fn test_get_gpu_assay_enabled() {
        // Create minimal data with some X values
        let mut data = Data::new();
        data.samples = vec!["s1".to_string(), "s2".to_string()];
        data.sample_len = 2;
        data.feature_len = 2;
        data.feature_selection = vec![0, 1];
        data.X.insert((0, 0), 1.0);
        data.X.insert((0, 1), 0.5);
        data.X.insert((1, 0), 0.3);
        data.X.insert((1, 1), 0.8);

        let mut param = Param::default();
        param.general.gpu = true;
        param.gpu.fallback_to_cpu = true;
        param.beam.max_nb_of_models = 10;

        let gpu_assay = get_gpu_assay(&data, &param);
        assert!(
            gpu_assay.is_some(),
            "GPU assay should be created when enabled with fallback"
        );
    }

    #[test]
    fn test_increment_basic() {
        // Create a population with one individual with k=2
        let mut best_pop = Population::new();
        let mut ind = Individual::new();
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        ind.k = 2;
        best_pop.individuals = vec![ind];

        let features_to_keep = vec![0, 1, 2, 3];
        let param = Param::default();

        let combinations = increment(best_pop, features_to_keep, &param);

        // Should generate combinations of size k+1 = 3
        assert!(!combinations.is_empty(), "Should generate combinations");
        for combo in &combinations {
            assert_eq!(combo.len(), 3, "All combinations should have size k+1=3");
        }

        // Combinations should include features from original + new features
        // [0,1] + one of [2,3] → [0,1,2] or [0,1,3]
        assert!(combinations
            .iter()
            .any(|c| c.contains(&0) && c.contains(&1) && c.contains(&2)));
        assert!(combinations
            .iter()
            .any(|c| c.contains(&0) && c.contains(&1) && c.contains(&3)));
    }

    #[test]
    fn test_combine_basic() {
        let features = vec![0, 1, 2];
        let k = 2;

        // Create minimal data
        let mut data = Data::new();
        data.samples = vec!["s1".to_string()];
        data.sample_len = 1;
        data.feature_len = 3;

        let mut param = Param::default();
        param.beam.max_nb_of_models = 1000;

        let combinations = combine(features.clone(), k, &data, &param);

        // C(3,2) = 3 combinations expected
        assert_eq!(
            combinations.len(),
            3,
            "Should generate C(3,2)=3 combinations"
        );

        // Check all expected combinations exist
        let contains_01 = combinations
            .iter()
            .any(|c| c.len() == 2 && c.contains(&0) && c.contains(&1));
        let contains_02 = combinations
            .iter()
            .any(|c| c.len() == 2 && c.contains(&0) && c.contains(&2));
        let contains_12 = combinations
            .iter()
            .any(|c| c.len() == 2 && c.contains(&1) && c.contains(&2));

        assert!(contains_01, "Should contain [0,1]");
        assert!(contains_02, "Should contain [0,2]");
        assert!(contains_12, "Should contain [1,2]");
    }

    #[test]
    fn test_combine_with_limit() {
        let features: Vec<usize> = (0..20).collect();
        let k = 3;

        // Create minimal data with feature_class and feature_significance
        let mut data = Data::new();
        data.samples = vec!["s1".to_string()];
        data.sample_len = 1;
        data.feature_len = 20;
        data.feature_selection = features.clone();
        data.classes = vec!["class0".to_string(), "class1".to_string()];

        // Add feature classes and significance for the selection logic
        for i in 0..20 {
            data.feature_class.insert(i, if i % 2 == 0 { 0 } else { 1 });
            data.feature_significance.insert(i, (20 - i) as f64); // Higher significance for lower indices
        }

        let mut param = Param::default();
        param.beam.max_nb_of_models = 100; // Limit to 100 models

        let combinations = combine(features.clone(), k, &data, &param);

        // C(20,3) = 1140, but we limited to 100
        // The function should reduce the number of features considered
        assert!(
            combinations.len() <= 1140,
            "Should not exceed total possible combinations"
        );

        // May generate fewer combinations due to filtering, which is OK
        // The important thing is it doesn't crash and respects some limit
        if !combinations.is_empty() {
            // All should have size k
            for combo in &combinations {
                assert_eq!(combo.len(), k, "All combinations should have size k");
            }
        }
    }

    #[test]
    fn test_combine_k_equals_n() {
        let features = vec![0, 1, 2];
        let k = 3;

        let mut data = Data::new();
        data.samples = vec!["s1".to_string()];
        data.sample_len = 1;
        data.feature_len = 3;

        let mut param = Param::default();
        param.beam.max_nb_of_models = 1000;

        let combinations = combine(features.clone(), k, &data, &param);

        // C(3,3) = 1
        assert_eq!(combinations.len(), 1);
        assert_eq!(combinations[0].len(), 3);
    }

    #[test]
    fn test_combine_k_greater_than_n() {
        let features = vec![0, 1];
        let k = 3;

        let mut data = Data::new();
        data.samples = vec!["s1".to_string(), "s2".to_string()];
        data.sample_len = 2;
        data.feature_len = 2;

        let mut param = Param::default();
        param.beam.max_nb_of_models = 1000;

        let combinations = combine(features.clone(), k, &data, &param);

        // C(2,3) = 0, impossible
        assert!(combinations.is_empty(), "Should return empty when k > n");
    }
}
