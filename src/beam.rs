// BE CAREFUL : the adaptation of the Predomics beam algorithm for Gpredomics is still under development
use crate::cv::CV;
use crate::ga;
use crate::gpu::GpuAssay;
use crate::individual::BINARY_LANG;
use crate::individual::RATIO_LANG;
use crate::individual::TERNARY_LANG;
use crate::population::Population;
use crate::individual::language;
use crate::individual::data_type;
use crate::individual::Individual;
use crate::param::FitFunction;
use crate::data::Data;
use crate::param::Param;
use crate::ga::remove_stillborn;
use std::collections::HashMap;
use std::collections::HashSet;
use log::{debug,info,warn,error};
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

// Beam functions
fn generate_combinations(features: &Vec<usize>, k: usize) -> Vec<Vec<usize>> {
    let mut combinations = HashSet::new();
    let mut indices = (0..k).collect::<Vec<_>>();

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

    let all_combinations: Vec<Vec<usize>> = best_combinations.into_par_iter()
        .flat_map(|best_combination| {
            single_feature_combinations.iter()
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

fn beam_pop_from_combinations(features_combination: Vec<Vec<usize>>, ind:Individual) -> Population {
    let ind_vec: Vec<Individual> = features_combination.into_par_iter()
        .map(|combination| {
            let mut tmp_ind = ind.clone();

            tmp_ind.features = combination.iter()
                .filter_map(|&k| ind.features.get(&k).map(|v| (k, v.clone())))
                .collect();

            tmp_ind
        })
        .collect();

    Population { individuals: ind_vec }
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
                if language != 0 { -1 } else { 0 }
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
        k: features.len(),
        epoch: 0,
        language: language,
        data_type: data_type,
        hash: 0,
        epsilon: param.general.data_type_epsilon,
        parents: None,
        betas: None
    }
}

pub fn beam(data: &mut Data, _no_overfit_data: &mut Option<Data>, param: &Param, running: Arc<AtomicBool>) -> Vec<Population> {
    // _no_overfit_data is not currently implemented

    let time = Instant::now();
    info!("\x1b[1;96mLaunching Beam algorithm for a feature interval [{}, {}]\x1b[0m", param.beam.kmin, param.beam.kmax);

    let mut collection: Vec<Population> = vec![];
    let mut pop = Population::new();
    
    info!("Selecting features...");
    data.select_features(param);
    info!("{} features selected.",data.feature_selection.len());
    debug!("FEATURES {:?}",data.feature_class);

    // Building lang_and_type_pop population 
    let mut languages: Vec<u8> = param.general.language.split(",").map(language).collect();
    let data_types: Vec<u8> = param.general.data_type.split(",").map(data_type).collect();
    let mut lang_and_type_pop = Population::new();
    for language in &languages {
        // Ignore pow2 language as beam algorithm has not (yet?) the ability to explore coefficient
        if *language != 2 as u8 {
            for data_type in &data_types {
                let ind = generate_individual(&data, *language, *data_type, &param);
                lang_and_type_pop.individuals.push(ind);
            }
        }
    }

    // Cross-validation initialization
    let mut cv: Option<CV> = None;
    if param.cv.overfit_penalty != 0.0 {
        let folds_nb = if param.cv.inner_folds > 1 { param.cv.inner_folds } else { 3 };
        info!("Learning on {:?}-folds.", folds_nb);
        let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
        cv = Some(CV::new(&data, folds_nb, &mut rng));
    }

    // Building the GPU assay and checking the prerequisites
    let gpu_assay = if param.general.gpu  {
        if param.cv.overfit_penalty == 0.0 {
            let buffer_binding_size = GpuAssay::get_max_buffer_size(&param.gpu) as usize;
            let gpu_max_nb_models = buffer_binding_size / (data.sample_len * std::mem::size_of::<f32>());
            if param.beam.max_nb_of_models == 0 {
                warn!("GPU requires a maximum number of models. Setting max_nb_of_models={} to prevent crashes.", 
                    gpu_max_nb_models / lang_and_type_pop.individuals.len());
                None
            } else if (param.beam.max_nb_of_models as usize) * lang_and_type_pop.individuals.len() > gpu_max_nb_models {
                warn!("GPU requires a maximum number of models that you exceed (GPU max_nb_of_models = {}). \
                \nAccording to the input parameters, please fix max_nb_of_models to {} \
                \nIf your configuration supports it and you know what you're doing, consider alternatively increasing the size of the buffers to {:.0} MB (do not forget to adjust the total size accordingly) \
                \nThis Gpredomics session will therefore be launched without a GPU.", gpu_max_nb_models,
                gpu_max_nb_models/lang_and_type_pop.individuals.len(),
                ((param.beam.max_nb_of_models*lang_and_type_pop.individuals.len()) as usize * data.sample_len * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)+1.0);
                None
            } else {
                let max_nb = (param.beam.max_nb_of_models as usize) * lang_and_type_pop.individuals.len();
                Some(GpuAssay::new(&data.X, &data.feature_selection, data.sample_len, max_nb, &param.gpu))
            }
        } else {
            warn!("Beam algorithm cannot be started with GPU if overfit_penalty>0.0.");
            None
        }
    } else { None };

    // Starting beam algorithm for initial k
    let initial_combinations = generate_combinations(&data.feature_selection, param.beam.kmin);
    let mut combinations = initial_combinations.clone();
    for ind in &lang_and_type_pop.individuals{
        pop.individuals.extend(beam_pop_from_combinations(combinations.clone(), ind.clone()).individuals)
    }

    if param.beam.kmin == 1 && (!languages.contains(&BINARY_LANG)) {
        warn!("Generating a population of non-binary Individuals with only one feature only results in stillborns. To prevent a panic, all Individuals will be retained for this iteration.")
    } else {
        let n_unvalid = remove_stillborn(&mut pop) as usize;
        if n_unvalid>0 { debug!("{} stillborns removed", n_unvalid) }
    }

    // Fitting first Population composed of all k_start combinations
    let test_assay: Option<GpuAssay> = None;
    if let Some(ref cv) = cv {
        debug!("Fitting population on folds...");
        pop.fit_on_folds(cv, &param, &vec![None; param.cv.inner_folds]);
        if param.general.keep_trace { pop.compute_all_metrics(&data, &param.general.fit); }
    }  else {
        debug!("Fitting population...");
        ga::fit_fn(&mut pop, data, &mut None, &gpu_assay, &test_assay, param);
    }

    pop = pop.sort();

    let pool = ThreadPoolBuilder::new()
        .num_threads(param.general.thread_number)
        .build()
        .expect("Failed to build thread pool");

    pool.install(|| {
        for ind_k in param.beam.kmin+1..param.beam.kmax {
            if param.general.keep_trace {pop.compute_hash()};

            debug!("Generating models with {:?} features...", ind_k);
            debug!("[k={:?}] initial population length = {:?}", ind_k, pop.individuals.len());

            // Dynamically select the best_models where features_to_keep are picked
            debug!("Selecting models...");

            if param.ga.forced_diversity_pct > 0.0 {
                pop = pop.filter_by_signed_jaccard_dissimilarity(param.ga.forced_diversity_pct, true)
            }
            let best_pop = pop.select_best_population(param.beam.best_models_ci_alpha);

            debug!("Kept {:?} individuals from the family of best models of k={:?}", best_pop.individuals.len(), ind_k-1);

            // pertinent features to use for next combinations
            let mut features_to_keep = select_features_from_best(&best_pop);

            // Stop the loop to avoid a panic! if there is not enough features_to_keep
            if (features_to_keep.len() <= ind_k ) & (ind_k != 1) {
                info!("\x1b[1;91mLimit reached: new models contain all pertinent features\x1b[0m");
                break;
            }

            let (mut class_0_features, mut class_1_features) = data.evaluate_features(param);
            class_0_features = class_0_features.iter().filter(|&&(index, _, _)| features_to_keep.contains(&index)).cloned().collect();
            class_1_features = class_1_features.iter().filter(|&&(index, _, _)| features_to_keep.contains(&index)).cloned().collect();
            let mut bin_combinations: Option<Vec<Vec<usize>>> = None; let mut possibilities=0;
            if param.beam.method == "combinatorial" {
                // Generate all possible combinations between the features_to_keep
                // Combinations are limited by features_to_keep (by param.beam.best_models_ci_alpha)
                // These features can be limited with param.beam.max_nb_of_models
                let mut features_to_keep_neg: Vec<usize>=vec![]; let mut features_to_keep_pos: Vec<usize>=vec![]; let mut bin_features_to_keep: Vec<usize>=vec![]; 
                possibilities = binomial_coefficient(features_to_keep.len() as u128, ind_k as u128);
                if possibilities > param.beam.max_nb_of_models as u128 && param.beam.max_nb_of_models != 0 {
                    let max_nb:usize = max_n_for_combinations(ind_k as u128, param.beam.max_nb_of_models as u128) as usize;
                    features_to_keep = vec![];
                    // For Binary languages, pick directly max_nb positive-associated features to avoid different ind_k within an iteration (like 10+10 features Ternary vs 10 features Binary for k=20)
                    if languages.contains(&BINARY_LANG) {
                        bin_features_to_keep = class_1_features.iter().take(max_nb).map(|&(index, _, _)| index).collect();
                        if bin_features_to_keep.len() <= ind_k  {
                            warn!("Limit reached for Binary: new Binary models contain all pertinent features. Binary removed for next iteration.");
                            languages.retain(|&x| x != BINARY_LANG);
                        }
                    } 
                    if languages.contains(&TERNARY_LANG) || languages.contains(&RATIO_LANG) {
                        // ideal_nb conducts to rounding down
                        // Always pick the higher feature count leading to stay under max_nb_of_models
                        let ideal_nb = max_nb/2; let f0_nb = class_0_features.len() ; let f1_nb = class_1_features.len();
                        if f0_nb < ideal_nb && f1_nb >= (ideal_nb + (ideal_nb - f0_nb)) {
                            if max_nb > ideal_nb*2 && f1_nb >= (ideal_nb + 1 + (ideal_nb - f0_nb )){
                                features_to_keep_neg = class_0_features.iter().take(f0_nb).map(|&(index, _, _)| index).collect();
                                features_to_keep_pos = class_1_features.iter().take(ideal_nb + 1 + (ideal_nb - f0_nb)).map(|&(index, _, _)| index).collect();
                            } else {
                                features_to_keep_neg = class_0_features.iter().take(f0_nb).map(|&(index, _, _)| index).collect();
                                features_to_keep_pos = class_1_features.iter().take(ideal_nb + (ideal_nb - f0_nb)).map(|&(index, _, _)| index).collect();
                            }
                        } else if f1_nb < ideal_nb && f0_nb >= (ideal_nb + (ideal_nb - f1_nb)) {
                            if max_nb > ideal_nb*2 && f0_nb >= (ideal_nb + 1 + (ideal_nb - f1_nb)) {
                                features_to_keep_neg = class_0_features.iter().take(ideal_nb + (ideal_nb + 1 - f1_nb)).map(|&(index, _, _)| index).collect();
                                features_to_keep_pos = class_1_features.iter().take(f1_nb).map(|&(index, _, _)| index).collect();
                            } else {
                                features_to_keep_neg = class_0_features.iter().take(ideal_nb + (ideal_nb - f1_nb)).map(|&(index, _, _)| index).collect();
                                features_to_keep_pos = class_1_features.iter().take(f1_nb).map(|&(index, _, _)| index).collect();
                            }
                        } else {
                            // if ideal_nb can be reached by both or can't be, .take(ideal_nb) (if less take stops before)
                            if max_nb > ideal_nb*2 {
                                // if max_nb is odd, take one additionnal positive-associated feature 
                                features_to_keep_neg = class_0_features.iter().take(ideal_nb).map(|&(index, _, _)| index).collect();
                                features_to_keep_pos = class_1_features.iter().take(ideal_nb+1).map(|&(index, _, _)| index).collect();
                            } else {
                                features_to_keep_neg = class_0_features.iter().take(ideal_nb).map(|&(index, _, _)| index).collect();
                                features_to_keep_pos = class_1_features.iter().take(ideal_nb).map(|&(index, _, _)| index).collect();
                            }
                        }
                        features_to_keep = features_to_keep_neg.clone();
                        features_to_keep.extend(features_to_keep_pos.clone()); 
                        }
                    if languages.contains(&BINARY_LANG) && (languages.contains(&TERNARY_LANG) || languages.contains(&RATIO_LANG)) {
                        warn!("Too many features (leading to {} > {} models). Feature ideal count = {}. Keeping {}+{} features ({} {}-associated for Binary)", possibilities, param.beam.max_nb_of_models, max_nb, features_to_keep_neg.len(), features_to_keep_pos.len(), bin_features_to_keep.len(), data.classes[1]);
                        combinations = generate_combinations(&features_to_keep, ind_k);
                        bin_combinations = Some(generate_combinations(&bin_features_to_keep, ind_k));
                    } else if languages.contains(&TERNARY_LANG) || languages.contains(&RATIO_LANG) { 
                        warn!("Too many features (leading to {} > {} models). Feature ideal count = {}. Keeping only {}+{} features.", possibilities, param.beam.max_nb_of_models, max_nb, features_to_keep_neg.len(), features_to_keep_pos.len());
                        combinations = generate_combinations(&features_to_keep, ind_k);
                    } else {
                        warn!("Too many features (leading to {} > {} models). Feature ideal count = {}. Keeping {} {}-associated features for Binary", possibilities, param.beam.max_nb_of_models, max_nb, bin_features_to_keep.len(), data.classes[1]); 
                        bin_combinations = Some(generate_combinations(&bin_features_to_keep, ind_k));
                    }
                } else {
                    combinations = generate_combinations(&features_to_keep, ind_k);
                }
            } else if param.beam.method == "incremental" {
                // Generate new combinations Mk + 1 feature_to_keep for next step
                // Combinations are limited both by kept Mk maximum (param.beam.max_nb_of_models) and features_to_keep (param.beam.best_models_ci_alpha)
                // Combinations are currently generated at each epoch in each languages and data_type
                debug!("Selecting best combinations...");
                let potential_combinations = best_pop.individuals.len() * features_to_keep.len();
                let mut reduced_best_pop = Population::new();
                if potential_combinations > param.beam.max_nb_of_models && param.beam.max_nb_of_models != 0 {
                    let max_parents = param.beam.max_nb_of_models / features_to_keep.len();
                    let min_parents = std::cmp::min(5, best_pop.individuals.len());
                    let adjusted_parents = std::cmp::max(max_parents, min_parents);
                    
                    debug!("Limiting parent models from {} to {} to respect max_nb_of_models={} (with {} features)",
                        best_pop.individuals.len(), adjusted_parents, param.beam.max_nb_of_models, features_to_keep.len());
                    
                    reduced_best_pop.individuals = best_pop.individuals[..std::cmp::min(adjusted_parents, best_pop.individuals.len())].to_vec();
                } else {
                    reduced_best_pop.individuals = best_pop.individuals;
                }

                let best_combinations: Vec<Vec<usize>> = reduced_best_pop.individuals.clone().par_iter().map(|ind| ind.features.keys().cloned().collect()).collect();
                debug!("Computing new combinations with these features...");
                combinations = combine_with_best(best_combinations.clone(), &features_to_keep);
            } else {
                error!("Unknown beam method");
            }
            debug!("{:?} unique combinations generated (~* {} language.data_type)", combinations.len(), lang_and_type_pop.individuals.len());

            // Compute AUC for generated Population and sort it
            pop = Population::new();

            for ind in &lang_and_type_pop.individuals{
                if ind.language == BINARY_LANG && param.beam.method == "combinatorial" && possibilities > param.beam.max_nb_of_models as u128 && param.beam.max_nb_of_models != 0 {
                    if (!bin_combinations.is_none()) && languages.contains(&BINARY_LANG){
                        pop.individuals.extend(beam_pop_from_combinations(bin_combinations.clone().unwrap(), ind.clone()).individuals);
                    } else {
                        pop.individuals.extend(beam_pop_from_combinations(combinations.clone(), ind.clone()).individuals);
                    }}
                else {
                    pop.individuals.extend(beam_pop_from_combinations(combinations.clone(), ind.clone()).individuals);
                }
            }
            
            if class_0_features.is_empty() && (languages.contains(&TERNARY_LANG) || languages.contains(&RATIO_LANG)) {
                warn!("All selected features are associated with class {}. Keeping stillborns to generate next iteration.", data.classes[1]);
            } else {
                let n_unvalid = remove_stillborn(&mut pop) as usize;
                if n_unvalid>0 { warn!("{} stillborns removed", n_unvalid) }
            }

            if let Some(ref cv) = cv {
                debug!("Fitting population on folds...");
                pop.fit_on_folds(cv, &param, &vec![None; param.cv.inner_folds]);
                if param.general.keep_trace { pop.compute_all_metrics(&data, &param.general.fit); }
            }  else {
                debug!("Fitting population...");
                ga::fit_fn(&mut pop, data, &mut None, &gpu_assay, &test_assay, param);
            }

            debug!("Sorting population...");
            pop = pop.sort();

            let best_model = &pop.individuals[0];
            let mean_k = pop.individuals.iter().map(|i| {i.k}).sum::<usize>() as f64/param.ga.population_size as f64;
            debug!("Best model so far AUC:{:.3} ({}:{} fit:{:.3}, k={}, gen#{}, specificity:{:.3}, sensitivity:{:.3}), average AUC {:.3}, fit {:.3}, k:{:.1}", 
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

            let scale = 50;
            let best_model_pos = match param.general.fit {
                FitFunction::sensitivity => {
                    ((best_model.sensitivity - 0.5) / 0.5 * scale as f64) as usize
                },
                FitFunction::specificity => {
                    ((best_model.specificity - 0.5) / 0.5 * scale as f64) as usize
                },
                _ => {
                    ((best_model.auc - 0.5) / 0.5 * scale as f64) as usize
                }
            };

            let best_fit_pos = ((best_model.fit - 0.5) / 0.5 * scale as f64) as usize;

            let max_pos = best_model_pos.max(best_fit_pos);
            let mut bar = vec!["█"; scale];
            for i in (max_pos + 1)..scale {
                bar[i] = "\x1b[0m▒\x1b[0m";
            }
            if best_model_pos < scale {
                bar[best_model_pos] = "\x1b[1m\x1b[31m█\x1b[0m"; 
            }

            if best_fit_pos < scale {
                bar[best_fit_pos] = "\x1b[1m\x1b[33m█\x1b[0m"; 
            }
            let output: String = bar.concat();
            let special_epoch = "".to_string();
            info!("k={: <5}{: <3}| \x1b[2mbest:\x1b[0m {: <20}\t\x1b[2m0.5\x1b[0m \x1b[1m{}\x1b[0m \x1b[2m1\x1b[0m", ind_k, special_epoch,  format!("{}:{}", best_model.get_language(), best_model.get_data_type()), output);

            let mut sorted_pop = Population::new();
            sorted_pop.individuals = pop.individuals.clone();

            if sorted_pop.individuals.len() > 0 {
                collection.push(sorted_pop);
            }

            // Stop the loop if someone kill the program
            if !running.load(Ordering::Relaxed) {
                break;
            }
        }});

        

        let elapsed = time.elapsed();
        info!("Beam algorithm ({} mode) computed {:?} generations in {:.2?}", param.beam.method, collection.len(), elapsed);

        if collection.is_empty() {
            error!("Beam did not produce any results because the criteria for selecting the best features was too restrictive. Please lower best_models_ci_alpha to allow results.")
        }
        
        collection
}

// Function to extract best models among all epochs, not necessarily having k_max features
pub fn keep_n_best_model_within_collection(collection:&Vec<Population>, n:usize) -> Population {
    let mut all_models = Population::new();
    for population in collection {
        for individual in population.individuals.clone() {
            all_models.individuals.push(individual);
        }
    }
    all_models = all_models.sort();

    let best_n_pop;
    if all_models.individuals.len() > n as usize && n as usize != 0 {
        best_n_pop = Population { individuals : all_models.individuals.clone()[..n as usize].to_vec() }
    } else {
        best_n_pop = all_models 
    }
    best_n_pop
}

// still have to write unit-tests to confirm every function behavior in any situation
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_individual() -> Individual {
        Individual  {features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(), auc: 0.4, fit: 0.8,
        specificity: 0.15, sensitivity:0.16, accuracy: 0.23, threshold: 42.0, k: 42, epoch:42,  language: 0, data_type: 0, hash: 0,
        epsilon: f64::MIN_POSITIVE, parents: None, betas: None}
    }

    #[test]
    fn test_generate_combinations() {
        let ind: Individual = create_test_individual();
        let combination: Vec<Vec<usize>> = generate_combinations(&ind.features.keys().cloned().collect(), 2);
        let truth: Vec<Vec<usize>> = vec![
            vec![0, 3],
            vec![0, 2],
            vec![0, 1],
            vec![3, 2],
            vec![3, 1],
            vec![2, 1],
        ];

        let sorted_combination: Vec<Vec<usize>> = combination.into_iter().map(|mut vec| {
            vec.sort();
            vec
        }).collect();

        let sorted_truth: Vec<Vec<usize>> = truth.into_iter().map(|mut vec| {
            vec.sort();
            vec
        }).collect();

        assert_eq!(sorted_combination.len(), sorted_truth.len(), "Combinatorial of 2 out of 4 is 6 but the generated vector contain {:?} combinations", sorted_combination.len());
        sorted_combination.iter().for_each(|vec| {
            assert!(sorted_truth.contains(vec), "{:?} is not a valid combination of length two for 0,1,2,3", vec);
        });
    }

    #[test]
    fn test_combine_with_best() {
        let ind: Individual = create_test_individual();
        let truth: Vec<Vec<usize>> = vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3]];

        // Generate sorted combinations to allow comparaison
        let combinations = generate_combinations(&ind.features.keys().cloned().collect(), 2);
        let mut sorted_combination: Vec<Vec<usize>> = combinations.into_iter().map(|mut vec| {
            vec.sort();
            vec
        }).collect();
        sorted_combination.sort();

        // Combine the first 2 combinations with features of ind and sort it to allow comparaison
        let combined = combine_with_best(sorted_combination.into_iter().take(2).collect(), &ind.features.keys().cloned().collect());
        let sorted_combined: Vec<Vec<usize>> = combined.into_iter().map(|mut vec| {
            vec.sort();
            vec
        }).collect();

        assert_eq!(sorted_combined.len(), truth.len(), "Combining [0, 1] and [0, 2] with 0/1/2/3 should lead to 3 new combinations ({:?} combinations generated)", sorted_combined.len());
        sorted_combined.iter().for_each(|vec| {
            assert!(truth.contains(vec), "Two best combinations [0, 1] and [0, 2] completed with 0/1/2/3 can't lead to {:?}", vec);
        });
    }

    #[test]
    fn test_beam_pop_from_combinations() {
        let ind = create_test_individual();

        // Generate sorted combinations to allow comparaison
        let combinations = generate_combinations(&ind.features.keys().cloned().collect(), 2);
        let mut sorted_combination: Vec<Vec<usize>> = combinations.into_iter().map(|mut vec| {
            vec.sort();
            vec
        }).collect();
        sorted_combination.sort();

        let combined = combine_with_best(sorted_combination.into_iter().take(2).collect(), &ind.features.keys().cloned().collect());
        let mut sorted_combined: Vec<Vec<usize>> = combined.into_iter().map(|mut vec| {
            vec.sort();
            vec
        }).collect();
        sorted_combined.sort();

        let generated = beam_pop_from_combinations(sorted_combined, ind);

        let features = vec![
        vec![(0, 1), (1, -1), (2, 1)],
        vec![(0, 1), (1, -1), (3, 0)],
        vec![(0, 1), (2, 1), (3, 0)],
        ];

        for ind in generated.individuals {
            let mut ind_features: Vec<_> = ind.features.into_iter().collect();
            ind_features.sort();
            assert!(features.contains(&ind_features), "Wrong generated population : {:?} does not contain {:?}", features, &ind_features);
        }
    }
}