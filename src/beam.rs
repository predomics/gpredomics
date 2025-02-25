// BE CAREFUL : the adaptation of the Predomics beam algorithm for Gpredomics is still under development 
// IMPORTANT NOTE : this algorithm currently use the last statrs version (v0.18) where Gpredomics 0.5 use the v0.16
use crate::individual::TERNARY_LANG;
use crate::population::Population;
use crate::individual::Individual;
use crate::data::Data;
use crate::ga;
use std::collections::HashMap;
use std::collections::HashSet;
use log::{debug,info,warn,error};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use statrs::stats_tests::fisher::fishers_exact;
use statrs::stats_tests::Alternative;

// Beam functions
fn generate_combinations(features: &Vec<usize>, k: usize) -> Vec<Vec<usize>> {
    fn combine(
        features: &Vec<usize>,
        k: usize,
        start: usize,
        tmp_combination: &mut Vec<usize>,
        combinations: &mut Vec<Vec<usize>>
    ) {
        if tmp_combination.len() == k {
            combinations.push(tmp_combination.clone());
            return;
        }

        for i in start..features.len() {
            tmp_combination.push(features[i]);
            combine(features, k, i + 1, tmp_combination, combinations);
            tmp_combination.pop();
        }
    }

    let mut combinations = Vec::new();
    let mut tmp_combination = Vec::new();
    combine(features, k, 0, &mut tmp_combination, &mut combinations);
    combinations
}

fn combine_with_best(best_combinations: Vec<Vec<usize>>, features: &Vec<usize>) -> Vec<Vec<usize>> {
    let single_feature_combinations = generate_combinations(features, 1);

    let mut all_combinations: Vec<Vec<usize>> = best_combinations.into_par_iter()
        .flat_map(|best_combination| {
            single_feature_combinations.iter()
                .filter_map(|single_feature| {
                    let mut combined_vec = best_combination.clone();
                    let feature = single_feature[0];
                    if !combined_vec.contains(&feature) {
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

fn beam_pop_from_combinations(features_combination:Vec<Vec<usize>>, ind:Individual) -> Population {
    let mut ind_vec: Vec<Individual> = Vec::new();

    for combination in features_combination {
        let mut tmp_ind = ind.clone();
        tmp_ind.language = TERNARY_LANG;

        tmp_ind.features = combination.iter()
            .filter_map(|&k| ind.features.get(&k).map(|v| (k, v.clone())))
            .collect();

        ind_vec.push(tmp_ind);
    }

    let pop = Population { individuals: ind_vec };

    pop
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

pub fn generate_individual(data: &Data, significant_features: &Vec<usize>) -> Individual {
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

    println!(
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
        data_type: 0, 
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


fn beam(data: Data, k_start:usize, k_max:usize, nb_of_models:usize, data_test: Data) -> Vec<Population> {
   println!("\x1b[1;96mLaunching Beam algorithm for a feature interval [{}, {}] with maximum {} models per epoch\x1b[0m", k_start, k_max, nb_of_models);
   
   // Initialisation
   let mut rng = ChaCha8Rng::seed_from_u64(42);
   let mut collection: Vec<Population> = vec![];
   //let mut features_index = &(0..=data.feature_len).collect();
   
   // Feature preselection
   let mut features_index = &significant_features(&data, 0.25, 0.0002);

   let initial_combinations = generate_combinations(features_index, k_start);
   let mut combinations = initial_combinations.clone();
   let mut n:usize;

   let ind = generate_individual(&data,&features_index);
   let mut pop = beam_pop_from_combinations(combinations, ind.clone());

   // Fitting first Population composed of all k_start combinations
   pop.auc_fit(&data, 0.0, 12);
   pop = pop.sort();

   for ind_k in k_start..k_max {
        println!("Generating models with {:?} features...", ind_k);

        // Select nb_of_models
        println!("Selecting models...");
        let mut selected_pop = Population::new();
        if pop.individuals.len() > nb_of_models {
            selected_pop.individuals = pop.individuals.clone()[..nb_of_models].to_vec();
        } else {
            selected_pop.individuals = pop.individuals.clone();
        }

        // Keep only 10% of nb_of_models (best in Predomics) - select feature before or after this step ?
        (selected_pop, n) = selected_pop.select_first_pct(10.0);
        println!("{:?} models selected from a Population of {:?} models", selected_pop.individuals.len(), pop.individuals.len());

        // Search for important features
        let mut features_importance = count_feature_appearances(&selected_pop);
        let mut features_to_keep = get_important_features(features_importance, features_index.len());
        // Fisher based on the prevelance of the feature in the n best model ?  

        // Generate new combinations [k, k+1] for next step
        println!("Computing new best combinations...");
        let best_combinations: Vec<Vec<usize>> = selected_pop.individuals.clone().par_iter().map(|ind| { ind.features.keys().cloned().collect()}).collect();
        combinations = combine_with_best(best_combinations.clone(), &features_to_keep);

        // Compute AUC for generated Population and sort it
        pop = beam_pop_from_combinations(combinations.clone(), ind.clone());
        pop.auc_fit(&data, 0.0, 12);
        pop = pop.sort();
        let mut sorted_pop = Population::new();
        sorted_pop.individuals = pop.individuals.clone();
        collection.push(sorted_pop);

        // Stop the loop to avoid a panic! due to not enough models in a few epochs
        if combinations.len() < (nb_of_models as f64 * 0.01) as usize {
            println!("\x1b[1;91mLimit reached : models with {} features results in the selection of fewer models than the number required per epoch. Please consider increasing the number of models selected per epoch.\x1b[0m", (ind_k+1));
            break;
        }
    }

    // Print final 20 best models
    let mut final_pop = Population::new();
    final_pop.individuals = collection.last_mut().unwrap().individuals.clone();
    println!("\x1b[1;93mTop model rankings for k={:?}\x1b[0m", final_pop.individuals[1].features.len());
    for i in 0..10 {
                let auc_on_train = pop.individuals[i].auc;
                let auc_on_test = pop.individuals[i].compute_auc(&data_test);
                println!("\x1b[1;93m#{}\x1b[0m: k={:?} | AUC train:{:?} test:{:?}\n{:?}", i+1, pop.individuals[i].features.len(), auc_on_train, auc_on_test, pop.individuals[i]);
            }

    println!("\x1b[1;93mTop model rankings for [{}, {}] interval\x1b[0m", k_start, final_pop.individuals[1].features.len());
    let mut top_ten_pop = keep_n_best_model_within_collection(&collection, 10);
    for i in 0..10 {
                let auc_on_train = top_ten_pop.individuals[i].auc;
                let auc_on_test = top_ten_pop.individuals[i].compute_auc(&data_test);
                println!("\x1b[1;93m#{}\x1b[0m: k={:?} | AUC train:{:?} test:{:?}\n{:?}", i+1, top_ten_pop.individuals[i].features.len(), auc_on_train, auc_on_test, top_ten_pop.individuals[i]);
            }

    collection
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

// tests - only used to help development and debug at this step
// still have to write unit-tests to confirm every function behavior in any situation
// the function test_beam can be used to test the current beam algorithm : it be launch with : cargo test test_beam -- --nocapture
// (do not forget to provide paths before launching the test)
#[cfg(test)]
mod tests {
    use super::*;
    /*
    fn create_test_individual() -> Individual {
        Individual  {features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(), auc: 0.4, fit: 0.8, 
        specificity: 0.15, sensitivity:0.16, accuracy: 0.23, threshold: 42.0, k: 42, epoch:42,  language: 0, data_type: 0, hash: 0, 
        data_type_minimum: f64::MIN_POSITIVE}
    }

    #[test]
    fn test_generate_combinations() {
        let ind = create_test_individual();
        println!("{:?}", generate_combinations(&ind.features.keys().cloned().collect(), 1));
    }

    #[test]
    fn test_combine_with_best() {
        let ind = create_test_individual();
        let combinations = generate_combinations(&ind.features.keys().cloned().collect(), 3);
        println!("{:?}", combinations);
        println!("{:?}", combine_with_best(combinations.into_iter().take(2).collect(), &ind.features.keys().cloned().collect()));
    }

    #[test]
    fn test_beam_pop_from_combinations() {
        let ind = create_test_individual();
        let combinations = generate_combinations(&ind.features.keys().cloned().collect(), 1);
        println!("{:?}", combinations);
        println!("{:?}", combine_with_best(combinations.clone().into_iter().take(2).collect(), &ind.features.keys().cloned().collect()));
        println!("{:?}", beam_pop_from_combinations(combine_with_best(combinations.into_iter().take(2).collect(), &ind.features.keys().cloned().collect()), ind));
    }*/

    #[test]
    fn test_beam() {
        // Can be launch with : cargo test test_beam -- --nocapture
        let mut data = Data::new();
        let mut data_test = Data::new();
        let _err = data.load_data(Xtrain, ytrain);
        let _err2 = data_test.load_data(Xtest, ytest);
        let pool = ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();
        pool.install(|| {beam(data, 2, 200, 10000, data_test)});
    }
}