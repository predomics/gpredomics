// BE CAREFUL : the adaptation of the Predomics beam algorithm for Gpredomics is still under development
use crate::cv::CV;
use crate::population::Population;
use crate::individual::language;
use crate::individual::data_type;
use crate::individual::Individual;
use crate::data::Data;
use crate::param::Param;
use std::collections::HashMap;
use std::collections::HashSet;
use log::{debug,info};
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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

// fn factorial(n: u64) -> u64 {
//     if n == 0 {
//         1
//     } else {
//         (1..=n).product()
//     }
// }

// fn binomial_coefficient(n: u64, k: u64) -> u64 {
//     if k > n {
//         0
//     } else {
//         factorial(n) / (factorial(k) * factorial(n - k))
//     }
// }

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

fn count_feature_appearances(population: &Population) -> HashMap<usize, usize> {
    let mut feature_counts = HashMap::new();

    for individual in &population.individuals {
        for feature in individual.features.keys() {
            *feature_counts.entry(*feature).or_insert(0) += 1;
        }
    }

    feature_counts
}

// fn get_important_features(feature_counts: HashMap<usize, usize>, nb_features_to_keep: usize) -> Vec<usize> {
//     let mut important_features: Vec<_> = feature_counts.into_iter().collect();
//     important_features.sort_by(|a, b| b.1.cmp(&a.1));
//     important_features.into_iter().take(nb_features_to_keep).map(|(feature, _)| feature).collect()
// }

fn select_features_from_best(best_pop: &Population, very_best_pop: &Population, best_pct:f64) -> Vec<usize> {
    let mut unique_features = HashSet::new();
    let mut features: Vec<usize> = vec![];
    let threshold = best_pop.individuals.len() as f64 * (best_pct / 100.0);

    for individual in &very_best_pop.individuals {
        for very_best_feature in individual.features.keys() {
            unique_features.insert(*very_best_feature);
        }
    }

    debug!("Very best features : {:?}", unique_features.len());
    let features_appearances = count_feature_appearances(&best_pop);
    for (feature, count) in features_appearances {
        if best_pop.individuals[0].features.len() != 1 && (count as f64) >= threshold {
            unique_features.insert(feature);
        // Keep all best features when k=1
        } else if best_pop.individuals[0].features.len() == 1 {
            unique_features.insert(feature);
        }
    }
    debug!("Very best features + best features : {:?}", unique_features.len());

    features.extend(unique_features.into_iter());

    if best_pop.individuals[0].features.len() != 1 {
        info!("{:?} features present in at least {:.2}% of the {:?} best models or present in the {:?} (very) best models kept",
        features.len(), best_pct, best_pop.individuals.len(), very_best_pop.individuals.len())
    } else {
        info!("Single-feature models : all features kept.")
    };

    features
}

pub fn generate_individual(data: &Data, significant_features: &Vec<usize>, language:u8, data_type:u8, param: &Param) -> Individual {
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

    for (&feature_idx, &counts) in &feature_counts {
        if counts[0] > counts[1] && language != 0 {
            features.insert(feature_idx, -1);
        } else if counts[1] > counts[0] || language == 0 {
            features.insert(feature_idx, 1);
        } else {
            features.insert(feature_idx, 0);
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
        data_type_minimum: param.general.data_type_epsilon,
    }
}

pub fn run_beam(param: &Param, running: Arc<AtomicBool>) -> Vec<Population> {
    // Load data
    let mut data = Data::new();
    let mut data_test = Data::new();
    let _ = data.load_data(&param.data.X.to_string(), &param.data.y.to_string());
    let _ = data_test.load_data(&param.data.Xtest.to_string(), &param.data.ytest.to_string());

    // Cross-validation initialization
    let mut cv: Option<CV> = None;
    if param.cv.fold_number > 1 {
        info!("Learning on {:?}-folds.", param.cv.fold_number);
        let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
        cv = Some(CV::new(&data, param.cv.fold_number, &mut rng));
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(param.general.thread_number)
        .build()
        .expect("Failed to build thread pool");

   

    info!("\x1b[1;96mLaunching Beam algorithm for a feature interval [{}, {}]\x1b[0m", param.beam.kmin, param.beam.kmax);

    let mut collection: Vec<Population> = vec![];

    // Feature preselection
    data.select_features(&param);
    let features_index = data.feature_selection.clone();

    let languages: Vec<u8> = param.general.language.split(",").map(language).collect();
    let data_types: Vec<u8> = param.general.data_type.split(",").map(data_type).collect();

    let initial_combinations = generate_combinations(&features_index, param.beam.kmin);
    let mut combinations = initial_combinations.clone();

    let mut pop = Population::new();

    let mut lang_and_type_pop = Population::new();
    for language in &languages {
        // Ignore pow2 language as beam algorithm has not (yet?) the ability to explore coefficient
        if *language != 2 as u8 {
            for data_type in &data_types {
                let ind = generate_individual(&data, &features_index, *language, *data_type, &param);
                lang_and_type_pop.individuals.push(ind);
            }
        }
    }

    for ind in &lang_and_type_pop.individuals{
        pop.individuals.extend(beam_pop_from_combinations(combinations.clone(), ind.clone()).individuals)
    }

    // Fitting first Population composed of all k_start combinations
    pop.auc_fit(&data, param.general.k_penalty, param.general.thread_number);
    pop = pop.sort();

    pool.install(|| {

        for ind_k in param.beam.kmin..param.beam.kmax {
            info!("Generating models with {:?} features...", ind_k);

            // Select maxNbOfModels
            debug!("Selecting models...");
            let best_pop = pop.select_best_population(param.beam.best_models_ci_alpha);
            let (very_best_pop, _) = best_pop.select_first_pct(param.beam.very_best_models_pct);

            debug!("Kept {:?} individuals for k={:?} from the family of best models", best_pop.individuals.len(), ind_k);
            debug!("Kept {:?} individuals for k={:?} from the family of very best models", very_best_pop.individuals.len(), ind_k);

            let features_to_keep = select_features_from_best(&best_pop, &very_best_pop, param.beam.features_importance_minimal_pct);

            // Stop the loop to avoid a panic! due to not enough models in a few epochs
            if (features_to_keep.len() <= ind_k ) & (ind_k != 1) {
                info!("\x1b[1;91mLimit reached : new model contains all pertinent features\x1b[0m");
                break;
            }

            if param.beam.method == "classic" {
                // Generate all possible combinations between the features_to_keep
                combinations = generate_combinations(&features_to_keep, ind_k);
            } else {
                // Generate new combinations Mk + 1 feature_to_keep for next step
                debug!("Selecting best combinations...");
                let best_combinations: Vec<Vec<usize>> = best_pop.individuals.clone().par_iter().map(|ind| ind.features.keys().cloned().collect()).collect();
                debug!("Computing new combinations with these features...");
                combinations = combine_with_best(best_combinations.clone(), &features_to_keep);
            }
            debug!("{:?} unique combinations generated ", combinations.len());


            // Compute AUC for generated Population and sort it
            pop = Population::new();

            for ind in &lang_and_type_pop.individuals{
                pop.individuals.extend(beam_pop_from_combinations(combinations.clone(), ind.clone()).individuals)
            }

            if let Some(ref cv) = cv {
                debug!("Computing penalized AUC (with cross-validation)...");
                pop.fit_on_folds(cv, &param);
            }  else {
                debug!("Computing penalized AUC...");
                pop.auc_fit(&data, param.general.k_penalty, param.general.thread_number);
            }

            debug!("Sorting population...");
            pop = pop.sort();
            let mut sorted_pop = Population::new();
            sorted_pop.individuals = pop.individuals.clone();
            collection.push(sorted_pop);

            debug!("Best fit : {:?}", pop.individuals[0].fit);

            // Stop the loop if someone kill the program
            if !running.load(Ordering::Relaxed) {
                break;
            }
        }

        // Print final 20 best models
        let mut final_pop = Population::new();
        final_pop.individuals = collection.last_mut().unwrap().individuals.clone();
        info!("\x1b[1;93mTop model rankings for k={:?}\x1b[0m", final_pop.individuals[0].features.len());
        info!("{}", final_pop.display(&data, Some(&data_test), param));

        info!("\x1b[1;93mTop model rankings for [{}, {}] interval\x1b[0m", param.beam.kmin, final_pop.individuals[0].features.len());
        let mut top_ten_pop = keep_n_best_model_within_collection(&collection, param.general.nb_best_model_to_test as usize);
        info!("{}", top_ten_pop.display(&data, Some(&data_test), param));

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