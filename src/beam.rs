// BE CAREFUL : the adaptation of the Predomics beam algorithm for Gpredomics is still under development
// IMPORTANT NOTE : this algorithm currently use the last statrs version (v0.18), where Gpredomics 0.5 use the v0.16
use crate::individual::TERNARY_LANG;
use crate::population::Population;
use crate::individual::data_type;
use crate::individual::Individual;
use crate::data::Data;
use crate::param::Param;
use std::collections::HashMap;
use std::collections::HashSet;
use log::{debug,info,warn,error};
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use statrs::stats_tests::fisher::fishers_exact;
use statrs::stats_tests::Alternative;
use std::sync::Arc;

// Beam functions
fn generate_combinations(features: &Vec<usize>, k: usize) -> Vec<Vec<usize>> {
    let mut combinations = Vec::new();
    let mut indices = (0..k).collect::<Vec<_>>();

    loop {
        let combination: Vec<usize> = indices.iter().map(|&i| features[i]).collect();
        combinations.push(combination);

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

    combinations
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

fn beam_pop_from_combinations(features_combination: Vec<Vec<usize>>, ind:Individual) -> Population {
    let ind_vec: Vec<Individual> = features_combination.into_par_iter()
        .map(|combination| {
            let mut tmp_ind = ind.clone();
            tmp_ind.language = TERNARY_LANG;

            tmp_ind.features = combination.iter()
                .filter_map(|&k| ind.features.get(&k).map(|v| (k, v.clone())))
                .collect();

            tmp_ind
        })
        .collect();

    Population { individuals: ind_vec }
}

fn count_feature_appearances(population: &Population) -> HashMap<usize, usize> {
    let mut feature_counts = HashMap::new();

    for individual in &population.individuals {
        for feature in individual.features.keys() {
            *feature_counts.entry(*feature).or_insert(0) += 1;
        }
    }

    feature_counts
}

fn get_important_features(feature_counts: HashMap<usize, usize>, nb_features_to_keep: usize) -> Vec<usize> {
    let mut important_features: Vec<_> = feature_counts.into_iter().collect();
    important_features.sort_by(|a, b| b.1.cmp(&a.1));
    important_features.into_iter().take(nb_features_to_keep).map(|(feature, _)| feature).collect()
}

fn select_features_from_best(sorted_population: &Population, nbBest: usize, nbVeryBest: usize, threshold: f64) -> Vec<usize> {
    let mut unique_features = HashSet::new();
    let mut features: Vec<usize> = vec![];
    let mut nB = 0;
    let mut nVB;

    let mut very_best_models_pop = Population::new();
    very_best_models_pop.individuals = sorted_population.individuals.clone();
    if very_best_models_pop.individuals.len() >= nbVeryBest {
        very_best_models_pop.individuals = very_best_models_pop.individuals[..nbVeryBest].to_vec();
        nVB = nbVeryBest;
    } else {
        (very_best_models_pop, nVB) = very_best_models_pop.select_first_pct(1.0);
    }

    for individual in &very_best_models_pop.individuals {
        for very_best_feature in individual.features.keys() {
            unique_features.insert(*very_best_feature);
        }
    }

    let mut best_models_pop = Population::new();
    best_models_pop.individuals = sorted_population.individuals.clone();
    if best_models_pop.individuals.len() >= nbBest {
        best_models_pop.individuals = best_models_pop.individuals[..nbBest].to_vec();
        nB = nbBest;
    } else {
        (best_models_pop, nB) = best_models_pop.select_first_pct(10.0);
    }

    let features_appearances = count_feature_appearances(&best_models_pop);
    for (feature, count) in features_appearances {
        if very_best_models_pop.individuals[0].features.len() != 1 && (count as f64 / nB as f64) >= threshold {
            unique_features.insert(feature);
        // Keep all best features when k=1
        } else if very_best_models_pop.individuals[0].features.len() == 1 {
            unique_features.insert(feature);
        }
    }

    features.extend(unique_features.into_iter());

    info!(
        "{:?} features present in at least {:?}% of the {:?} best models or present in the {:?} (very) best models kept",
        features.len(),
        threshold * 100.0,
        nB,
        nVB
    );

    features
}

pub fn generate_individual(data: &Data, significant_features: &Vec<usize>, data_type:u8) -> Individual {
    let mut feature_counts = HashMap::new();

    for (&(sample_idx, feature_idx), &value) in &data.X {
        if !significant_features.contains(&feature_idx) {
            continue;
        }

        let response = match data.y.get(sample_idx) {
            Some(0) => 0,
            Some(1) => 1,
            _ => continue,
        };

        let entry = feature_counts.entry(feature_idx).or_insert([0.0, 0.0]);
        entry[response] += value;
    }

    let mut features = HashMap::new();
    let mut count_neg_1 = 0;
    let mut count_0 = 0;
    let mut count_1 = 0;

    for (&feature_idx, &counts) in &feature_counts {
        if counts[0] > counts[1] {
            features.insert(feature_idx, -1);
            count_neg_1 += 1;
        } else if counts[1] > counts[0] {
            features.insert(feature_idx, 1);
            count_1 += 1;
        } else {
            features.insert(feature_idx, 0);
            count_0 += 1;
        }
    }

    info!(
        "\x1b[1;92mGenerating a standard Individual based on the average representativeness of features within samples:\nClass 0 associated: {}\nClass 1 associated: {}\nBalanced representation: {} \x1b[0m",
        count_neg_1, count_1, count_0
    );

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
        language: TERNARY_LANG,
        data_type: data_type,
        hash: 0,
        data_type_minimum: 0.0,
    }
}

pub fn contingency_table(data: &Data, feature_index: usize, min:f64) -> [u64; 4] {
    let mut table = [0u64; 4];

    for (&(sample_index, feat_index), &value) in &data.X {
        if feat_index == feature_index {
            let class = data.y[sample_index] as usize;
            if class == 0 || class == 1 {
                if value >= min {
                    table[class * 2] += 1;
                } else {
                    table[class * 2 + 1] += 1;
                }
            }
        }
    }

    table
}

pub fn fisher_exact_test(data: &Data, feature_index: usize, min: f64) -> f64 {
    let table = contingency_table(data, feature_index, min);
    let p_value = fishers_exact(&table, Alternative::TwoSided).unwrap();

    p_value
}


pub fn significant_features(data: &Data, p_value_threshold: f64, min: f64) -> Vec<usize> {
    let significant_features: Vec<usize> = (0..data.feature_len)
        .into_par_iter()
        .filter(|&feature_index| {
            let p_value = fisher_exact_test(data, feature_index, min);
            p_value < p_value_threshold
        })
        .collect();

    println!(
        "\x1b[1;38;5;214mFisher Exact Test : kept {} on {} features (p-value < {})\x1b[0m",
        significant_features.len(),
        data.feature_len,
        p_value_threshold
    );

    significant_features
}

pub fn run_beam(param: &Param, running: Arc<AtomicBool>) -> Vec<Population> {
    // Load data
    let mut data = Data::new();
    let mut data_test = Data::new();
    data.load_data(&param.data.X.to_string(), &param.data.y.to_string());
    data_test.load_data(&param.data.Xtest.to_string(), &param.data.ytest.to_string());

    let pool = ThreadPoolBuilder::new()
        .num_threads(param.general.thread_number)
        .build()
        .expect("Failed to build thread pool");
    
    pool.install(|| {
            
        info!("\x1b[1;96mLaunching Beam algorithm for a feature interval [{}, {}] with maximum {} models per epoch\x1b[0m", param.beam.kmin, param.beam.kmax, param.beam.max_nb_of_models);

        let mut collection: Vec<Population> = vec![];

        // Feature preselection
        data.select_features(&param);
        let mut features_index = data.feature_selection.clone();

        let initial_combinations = generate_combinations(&features_index, param.beam.kmin);
        let mut combinations = initial_combinations.clone();
        let mut n: usize;

        let ind = generate_individual(&data, &features_index, data_type(&param.general.data_type));
        let mut pop = beam_pop_from_combinations(combinations, ind.clone());

        // Fitting first Population composed of all k_start combinations
        pop.auc_fit(&data, 0.0, 12);
        pop = pop.sort();

        for ind_k in param.beam.kmin..param.beam.kmax {
            info!("Generating models with {:?} features...", ind_k);

            // Select maxNbOfModels
            debug!("Selecting models...");
            let mut selected_pop = Population::new();
            if pop.individuals.len() > param.beam.max_nb_of_models {
                selected_pop.individuals = pop.individuals.clone()[..param.beam.max_nb_of_models].to_vec();
            } else {
                selected_pop.individuals = pop.individuals.clone();
            }

            // Generate new combinations [k, k+1] for next step
            let mut features_to_keep = select_features_from_best(&selected_pop, param.beam.nb_best_models, param.beam.nb_very_best_models, param.beam.features_importance_minimal_pct);
            debug!("Selecting best combinations...");
            let best_combinations: Vec<Vec<usize>> = selected_pop.individuals.clone().par_iter().map(|ind| ind.features.keys().cloned().collect()).collect();
            debug!("Computing new combinations with these features...");
            combinations = combine_with_best(best_combinations.clone(), &features_to_keep);
            debug!("Generating Population...");
            
            // Compute AUC for generated Population and sort it
            pop = beam_pop_from_combinations(combinations.clone(), ind.clone());
            debug!("Fitting AUC and sort...");
            pop.auc_fit(&data, 0.0, 16);
            pop = pop.sort();
            let mut sorted_pop = Population::new();
            sorted_pop.individuals = pop.individuals.clone();
            collection.push(sorted_pop);

            // Stop the loop to avoid a panic! due to not enough models in a few epochs
            if (features_to_keep.len() <= ind_k + 2) & (ind_k != 1) {
                info!("\x1b[1;91mLimit reached : new model contains all pertinent features\x1b[0m");
                break;
            }

            // Stop the loop if someone kill the program
            if !running.load(Ordering::Relaxed) {
                break;
            }
        }

        // Print final 20 best models
        let mut final_pop = Population::new();
        final_pop.individuals = collection.last_mut().unwrap().individuals.clone();
        println!("\x1b[1;93mTop model rankings for k={:?}\x1b[0m", final_pop.individuals[0].features.len());
        let mut limit = 20;
        if final_pop.individuals.len() < 20 {
            limit = final_pop.individuals.len()
        }
        for i in 0..limit {
            let individual = &mut pop.individuals[i];
            let auc_train = individual.auc;
            let auc_test = individual.compute_auc(&data_test);
            let (threshold, acc_train, se_train, sp_train) = individual.compute_threshold_and_metrics(&data);
            let (_, acc_test, se_test, sp_test) = individual.compute_threshold_and_metrics(&data_test);
            println!("\x1b[1;93m#{}\x1b[0m: k={:?} || threshold {:.10} : AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3} \n {:?}",
            i + 1, individual.features.len(), threshold, auc_train, auc_test, acc_train, acc_test, se_train, se_test, sp_train, sp_test, individual);
        }

        println!("\x1b[1;93mTop model rankings for [{}, {}] interval\x1b[0m", param.beam.kmin, final_pop.individuals[0].features.len());
        let mut top_ten_pop = keep_n_best_model_within_collection(&collection, 20);
        for i in 0..20 {
            let individual = &mut top_ten_pop.individuals[i];
            let auc_train = individual.auc;
            let auc_test = individual.compute_auc(&data_test);
            let (threshold, acc_train, se_train, sp_train) = individual.compute_threshold_and_metrics(&data);
            let (_, acc_test, se_test, sp_test) = individual.compute_threshold_and_metrics(&data_test);
            println!("\x1b[1;93m#{}\x1b[0m: k={:?} || threshold {:.10} : AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3} \n {:?}",
            i + 1, individual.features.len(), threshold, auc_train, auc_test, acc_train, acc_test, se_train, se_test, sp_train, sp_test, individual);
        }

        collection
    })
}

// Function to extract best models among all epochs, not necessarily having k_max features
fn keep_n_best_model_within_collection(collection:&Vec<Population>, n:usize) -> Population {
    let mut all_models = Population::new();
    for population in collection {
        for individual in population.individuals.clone() {
            all_models.individuals.push(individual);
        }
    }
    all_models = all_models.sort();

    Population { individuals : all_models.individuals.clone()[..n].to_vec() }
}

// still have to write unit-tests to confirm every function behavior in any situation
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_individual() -> Individual {
        Individual  {features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(), auc: 0.4, fit: 0.8,
        specificity: 0.15, sensitivity:0.16, accuracy: 0.23, threshold: 42.0, k: 42, epoch:42,  language: 0, data_type: 0, hash: 0,
        data_type_minimum: f64::MIN_POSITIVE}
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

        let mut sorted_truth: Vec<Vec<usize>> = truth.into_iter().map(|mut vec| {
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
        let mut combinations = generate_combinations(&ind.features.keys().cloned().collect(), 2);
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

        let mut combined = combine_with_best(sorted_combination.into_iter().take(2).collect(), &ind.features.keys().cloned().collect());
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
