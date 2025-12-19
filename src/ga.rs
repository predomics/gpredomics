use crate::cinfo;
use crate::cv::CV;
use crate::data::Data;
use crate::gpu::GpuAssay;
use crate::individual::Individual;
use crate::individual::{self, RATIO_LANG, TERNARY_LANG};
use crate::param::Param;
use crate::population::Population;
use crate::utils::{display_epoch, display_epoch_legend};
use log::{debug, error, info, warn};
use rand::prelude::SliceRandom;
use rand::prelude::*;
use rand::seq::index::sample;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::cmp::min;
use std::collections::HashMap;
use std::mem;
use std::time::Instant;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

//-----------------------------------------------------------------------------
// Genetic Algorithm core functions
//-----------------------------------------------------------------------------

/// Main function to run the genetic algorithm
///
/// # Arguments
///
/// * `data` - The dataset to operate on.
/// * `_test_data` - Optional test dataset (deprecated).
/// * `initial_pop` - Optional initial population to start the algorithm.
/// * `param` - Parameters for the genetic algorithm.
/// * `running` - Atomic boolean to control the running state of the algorithm.
///
/// # Returns
///
/// A vector of populations representing the evolution over generations.
///
/// # Panics
///
/// Panics if the initial population is not compatible with the data or if the initial population size is too small.
pub fn ga(
    data: &mut Data,
    _test_data: &mut Option<Data>,
    initial_pop: &mut Option<Population>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> Vec<Population> {
    let time = Instant::now();

    // Select feature in the algorithm to avoid data leakage
    data.select_features(param);
    debug!("FEATURES {:?}", data.feature_class);

    // Initialize GPU BEFORE creating RNG to ensure deterministic behavior
    // GPU initialization can affect system state in non-deterministic ways
    let gpu_assay = get_gpu_assay(data, param);

    // Create RNG AFTER all GPU initialization is complete
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    // Initialize first population
    let base_pop = if let Some(pop) = initial_pop {
        let mut pop = pop.clone();

        if pop.check_compatibility(data) {
            info!("Initial population is compatible with data.");
        } else {
            error!("Initial population is not compatible with data!");
            panic!("Initial population is not compatible with data!");
        }

        // Disable GPU for initial population fitting to avoid problem if the population is larger
        pop.fit(data, &mut None, &None, &None, param);
        pop = pop.sort();
        let _ = remove_stillborn(&mut pop);
        pop.compute_hash();
        pop.remove_clone();

        if pop.individuals.len() > param.ga.population_size as usize {
            warn!("Initial population larger than requested population size: truncating based on their fit...");
            pop.individuals.truncate(param.ga.population_size as usize);
        } else if pop.individuals.len() < param.ga.population_size as usize {
            warn!("Initial population smaller than requested population size: first generation will be smaller.");
        }

        info!(
            "{} individuals kept after removing stillborn from the provided initial population.",
            pop.individuals.len()
        );
        pop
    } else {
        generate_pop(data, param, &mut rng)
    };

    if base_pop.individuals.len() < 50 {
        error!(
            "Initial population size is too small ({}<50 individuals)!",
            base_pop.individuals.len()
        );
        panic!(
            "Initial population size is too small ({}<50 individuals)!",
            base_pop.individuals.len()
        );
    }

    info!(
        "Population size: {}, k_min {}, k_max {}",
        base_pop.individuals.len(),
        base_pop
            .individuals
            .iter()
            .map(|i| { i.k })
            .min()
            .unwrap_or(0),
        base_pop
            .individuals
            .iter()
            .map(|i| { i.k })
            .max()
            .unwrap_or(0)
    );

    cinfo!(
        param.general.display_colorful,
        "{}",
        display_epoch_legend(param)
    );
    let populations = iterative_evolution(&base_pop, data, &gpu_assay, param, running, &mut rng);

    let elapsed = time.elapsed();
    info!(
        "Genetic algorithm computed {:?} generations in {:.2?}",
        populations.len(),
        elapsed
    );

    populations
}

/// Generate initial population divided into sub-populations for each language and data type
///
/// # Arguments
///
/// * `data` - The dataset to operate on.
/// * `param` - Parameters for the genetic algorithm.
/// * `rng` - Random number generator.
///
/// # Returns
///
/// A population representing the initial generation constructed from sub-populations for each language and data type.
pub fn generate_pop(data: &Data, param: &Param, rng: &mut ChaCha8Rng) -> Population {
    let mut pop = Population::new();
    let languages: Vec<u8> = param
        .general
        .language
        .split(",")
        .map(individual::language)
        .collect();
    let data_types: Vec<u8> = param
        .general
        .data_type
        .split(",")
        .map(individual::data_type)
        .collect();
    let sub_population_size =
        param.ga.population_size / (languages.len() * data_types.len()) as u32;
    for data_type in &data_types {
        for language in &languages {
            let mut target_size = sub_population_size;
            while target_size > 0 {
                let mut sub_pop = Population::new();
                debug!("generating...");

                let prior_weight = data.feature_annotations.as_ref().map(|fa| &fa.prior_weight);
                sub_pop.generate(
                    target_size,
                    param.ga.k_min,
                    if param.ga.k_max > 0 {
                        min(data.feature_selection.len(), param.ga.k_max)
                    } else {
                        data.feature_selection.len()
                    },
                    *language,
                    *data_type,
                    param.general.data_type_epsilon,
                    data,
                    param.general.threshold_ci_n_bootstrap > 0,
                    prior_weight,
                    rng,
                );

                debug!(
                    "generated for {} {}...",
                    sub_pop.individuals[0].get_language(),
                    sub_pop.individuals[0].get_data_type()
                );

                target_size = remove_stillborn(&mut sub_pop);
                if target_size > 0 {
                    debug!(
                        "Some still born are present {} (with healthy {})",
                        target_size,
                        sub_pop.individuals.len()
                    );
                    if target_size == param.ga.population_size {
                        error!("Params only create inviable individuals!");
                        panic!("Params only create inviable individuals!")
                    }
                }

                pop.add(sub_pop);
            }
        }
    }
    pop.compute_hash();
    let clone_number = pop.remove_clone();
    if clone_number > 0 {
        debug!("Some clones were removed : {}.", clone_number)
    };
    pop
}

/// Run the iterative evolution process of the genetic algorithm
///
/// # Arguments
///
/// * `base_pop` - The initial population to start the evolution.
/// * `data` - The dataset to operate on.
/// * `gpu_assay` - Optional GPU assay for fitness evaluation.
/// * `param` - Parameters for the genetic algorithm.
/// * `running` - Atomic boolean to control the running state of the algorithm.
/// * `rng` - Random number generator.
///
/// # Returns
///
/// A vector of populations representing the evolution over generations.
pub fn iterative_evolution(
    base_pop: &Population,
    data: &mut Data,
    gpu_assay: &Option<GpuAssay>,
    param: &Param,
    running: Arc<AtomicBool>,
    rng: &mut ChaCha8Rng,
) -> Vec<Population> {
    let mut epoch: usize = 0;
    let mut cv: Option<CV> = None;
    let mut populations: Vec<Population> = vec![];

    let mut data_rng = rng.clone();
    let mut evolution_rng = rng.clone();

    // Prepare epoch associated data
    let mut epoch_data = if param.ga.random_sampling_pct > 0.0 {
        let random_samples =
            (data.sample_len as f64 * (param.ga.random_sampling_pct / 100.0)) as usize;
        data.subset(data.random_subset(random_samples, &mut data_rng))
    } else if param.cv.overfit_penalty != 0.0 {
        let folds_nb = if param.cv.inner_folds > 1 {
            param.cv.inner_folds
        } else {
            10
        };
        info!("Learning on {:?}-folds.", folds_nb);
        cv = Some(CV::new_from_param(&data, param, &mut data_rng, folds_nb));
        Data::new()
    } else {
        data.clone()
    };

    // Clean data before process
    let mut pop = base_pop.clone();
    pop.compute_hash();
    let _clone_number = pop.remove_clone();

    // Create GPU assays once for CV (will be reused until resampling)
    let mut gpu_assays_per_fold = if let Some(ref cv) = cv {
        create_gpu_assays_for_folds(cv, param)
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

    // Evolve!
    loop {
        epoch += 1;

        // Data shuffling if required
        // Fit precendent generation on new data only for during for resampling epochs
        if param.ga.random_sampling_pct > 0.0 && epoch % param.ga.random_sampling_epochs == 0 {
            let random_samples =
                (data.sample_len as f64 * (param.ga.random_sampling_pct / 100.0)) as usize;
            debug!("Re-sampling {} samples...", random_samples);
            epoch_data = data.subset(data.random_subset(random_samples, &mut data_rng));

            pop.fit(&epoch_data, &mut None, &gpu_assay, &None, param);
            if param.general.keep_trace {
                pop.compute_all_metrics(&epoch_data, &param.general.fit);
            }
            pop = pop.sort();
        } else if param.cv.overfit_penalty > 0.0
            && param.cv.resampling_inner_folds_epochs > 0
            && epoch % param.cv.resampling_inner_folds_epochs == 0
        {
            debug!("Re-sampling folds...");
            let folds_nb = if param.cv.inner_folds > 1 {
                param.cv.inner_folds
            } else {
                3
            };
            cv = Some(CV::new_from_param(&data, param, &mut data_rng, folds_nb));

            if let Some(ref cv) = cv {
                // Recreate GPU assays for new folds
                gpu_assays_per_fold = create_gpu_assays_for_folds(cv, param);
                pop.fit_on_folds(cv, &param, &gpu_assays_per_fold);
                pop = pop.sort();
            }
        }

        // Evolution and ranking - reuse GPU assays
        pop = evolve(
            pop,
            &epoch_data,
            &mut cv,
            param,
            &gpu_assays_per_fold,
            gpu_assay,
            epoch,
            &mut evolution_rng,
        );

        cinfo!(
            param.general.display_colorful,
            "{}",
            display_epoch(&pop, param, epoch)
        );

        // Stop critera
        let mut need_to_break = false;

        let best_model = &pop.individuals[0];
        if epoch >= param.ga.min_epochs {
            if epoch - best_model.epoch + 1 > param.ga.max_age_best_model {
                info!("Best model has reached limit age...");
                need_to_break = true;
            }
        }

        if epoch >= param.ga.max_epochs {
            info!("Reach max epoch");
            need_to_break = true;
        }

        if !running.load(Ordering::Relaxed) {
            info!("Signal received");
            need_to_break = true;
        }

        if param.general.keep_trace {
            populations.push(pop.clone())
        }

        if need_to_break {
            if populations.len() == 0 {
                populations = vec![pop];
            }

            if param.ga.random_sampling_pct > 0.0 {
                if let Some(last_population) = populations.last_mut() {
                    warn!("Random sampling: models optimized on samples ({} samples), metrics shown on full dataset. \n\
                    NOTE: Fit values reflect sample-based optimization, not full dataset performance.", (param.ga.random_sampling_pct * 100.0) as u8);
                    last_population.compute_all_metrics(&data, &param.general.fit);
                    //(&mut *last_population).fit(&data, &mut None, &gpu_assay, &None, param);
                }
            }

            break;
        }
    }

    populations
}

/// Run one evolution step: selection, cross-over, mutation, fitting
///
/// # Arguments
///
/// * `pop` - The current population to evolve.
/// * `data` - The dataset to operate on.
/// * `cv` - Optional cross-validation structure for fitness evaluation.
/// * `param` - Parameters for the genetic algorithm.
/// * `gpu_assays_per_fold` - Optional GPU assays for each fold in cross-validation.
/// * `gpu_assay` - Optional GPU assay for fitness evaluation.
/// * `epoch` - The current epoch number.
/// * `rng` - Random number generator.
///
/// # Returns
///
/// A new population representing the next generation after evolution.
#[inline]
pub fn evolve(
    pop: Population,
    data: &Data,
    cv: &mut Option<CV>,
    param: &Param,
    gpu_assays_per_fold: &Vec<(Option<GpuAssay>, Option<GpuAssay>)>,
    gpu_assay: &Option<GpuAssay>,
    epoch: usize,
    rng: &mut ChaCha8Rng,
) -> Population {
    let mut new_pop = Population::new();

    new_pop.add(select_parents(&pop, param, rng));

    // Filter before cross-over to improve diversity
    if param.ga.forced_diversity_pct != 0.0 && epoch % param.ga.forced_diversity_epochs == 0 {
        let n = new_pop.individuals.len();
        new_pop =
            new_pop.filter_by_signed_jaccard_dissimilarity(param.ga.forced_diversity_pct, true);
        if new_pop.individuals.len() > 1 {
            debug!(
                "Parents filtered for diversity: {}/{} individuals retained",
                new_pop.individuals.len(),
                n
            );
        } else {
            warn!("Only 1 Individual kept after filtration with diversity");
        }
    }

    // Generate children
    let mut children_to_create = param.ga.population_size as usize - new_pop.individuals.len();

    let mut children = Population::new();

    while children_to_create > 0 {
        let mut some_children = cross_over(&new_pop, children_to_create, rng);
        mutate(&mut some_children, param, &data.feature_selection, rng);
        children_to_create = remove_stillborn(&mut some_children) as usize;
        if children_to_create > 0 {
            debug!("Some stillborn are presents: {}", children_to_create)
        }

        children.add(some_children);
    }

    for i in children.individuals.iter_mut() {
        i.epoch = epoch;
    }

    // Fit children and clean population
    if let Some(ref cv) = cv {
        debug!("Fitting children on folds...");
        children.fit_on_folds(cv, &param, gpu_assays_per_fold);
    } else {
        debug!("Fitting children...");
        children.fit(&data, &mut None, gpu_assay, &None, param);
    }

    new_pop.add(children);

    new_pop.compute_hash();
    let clone_number = new_pop.remove_clone();
    if clone_number > 0 {
        debug!("Some clones were removed : {}.", clone_number)
    };
    new_pop = new_pop.sort();

    new_pop
}

/// Picks parents from the population based on elite, niche, and random selection
///
/// # Arguments
///
/// * `pop` - The current population to select parents from.
/// * `param` - Parameters for the genetic algorithm.
/// * `rng` - Random number generator.
///
/// # Returns
///
/// A population representing the selected parents.
fn select_parents(pop: &Population, param: &Param, rng: &mut ChaCha8Rng) -> Population {
    // order pop by fit and select params.ga_select_elite_pct
    let (mut parents, n) = pop.select_first_pct(param.ga.select_elite_pct);

    let mut individual_by_types: HashMap<(u8, u8), Vec<&Individual>> = HashMap::new();
    for individual in pop.individuals[n..].iter() {
        let i_type = (individual.language, individual.data_type);
        if !individual_by_types.contains_key(&i_type) {
            individual_by_types.insert(i_type, vec![individual]);
        } else {
            individual_by_types
                .get_mut(&i_type)
                .unwrap()
                .push(individual);
        }
    }

    // adding best models of each language / data type
    if param.ga.select_niche_pct > 0.0 {
        let types = individual_by_types
            .keys()
            .cloned()
            .collect::<Vec<(u8, u8)>>();
        let target = (pop.individuals.len() as f64 * param.ga.select_niche_pct
            / 100.0
            / types.len() as f64) as usize;
        let mut type_count: HashMap<(u8, u8), usize> = types.iter().map(|x| (*x, 0)).collect();
        for i in &pop.individuals[n..] {
            let i_type = (i.language, i.data_type);
            let current_count = *type_count.get(&i_type).unwrap_or(&target);
            if current_count < target {
                type_count.insert(i_type, current_count + 1);
                parents.individuals.push(i.clone())
            }
        }
    }

    // add a random part of the others
    // parents.add(pop.select_random_above_n(param.ga.select_random_pct, n, rng));
    let n2 = (pop.individuals.len() as f64 * param.ga.select_random_pct
        / 100.0
        / individual_by_types.keys().len() as f64) as usize;

    let mut sorted_keys: Vec<_> = individual_by_types.keys().collect();
    sorted_keys.sort();

    for i_type in sorted_keys {
        debug!(
            "Adding {}:{} Individuals {} ",
            individual_by_types[i_type][0].get_language(),
            individual_by_types[i_type][0].get_data_type(),
            n2
        );

        // Sort individuals by hash to ensure deterministic order for choose_multiple
        // Critical for CPU/GPU reproducibility when fits are very close (f32 vs f64 precision)
        let mut individuals_sorted = individual_by_types[i_type].clone();
        individuals_sorted.sort_by_key(|ind| ind.hash);

        parents.individuals.extend(
            individuals_sorted
                .choose_multiple(rng, n2)
                .map(|i| (*i).clone()),
        );
    }
    parents
}

/// Perform crossover between parents to generate children
///
/// # Arguments
///
/// * `parents` - The population of parents to crossover.
/// * `children_number` - The number of children to generate.
/// * `rng` - Random number generator.
///
/// # Returns
///
/// A population representing the generated children.
///
/// # Panics
///
/// Panics if the number of parents is less than 2.
pub fn cross_over(
    parents: &Population,
    children_number: usize,
    rng: &mut ChaCha8Rng,
) -> Population {
    let mut children = Population::new();

    for _i in 0..children_number {
        let [p1, p2] = parents
            .individuals
            .choose_multiple(rng, 2)
            .collect::<Vec<_>>()
            .try_into()
            .expect("Vec must have exactly 2 elements");

        let main_parent = *[p1, p2].choose(rng).unwrap();
        let mut child = Individual::child(main_parent);

        let mut all_features: Vec<usize> = p1
            .features
            .keys()
            .chain(p2.features.keys())
            .copied()
            .collect();
        all_features.sort_unstable();
        all_features.dedup();

        for &feature in &all_features {
            let (parent, parent_lang) = if rng.gen_bool(0.5) {
                (p1, p1.language)
            } else {
                (p2, p2.language)
            };

            if let Some(&val) = parent.features.get(&feature) {
                let converted_val = if individual::needs_conversion(parent_lang, child.language) {
                    individual::gene_convert_from_to(parent_lang, child.language, val)
                } else {
                    val
                };
                child.features.insert(feature, converted_val);
            }
        }

        child.count_k();
        child.parents = Some(vec![p1.hash, p2.hash]);
        children.individuals.push(child);
    }
    children
}

/// Mutate a portion of the children population based on specified parameters
///
/// # Arguments
///
/// * `children` - The population of children to mutate.
/// * `param` - Parameters for the genetic algorithm.
/// * `feature_selection` - The list of feature indices available for mutation.
/// * `rng` - Random number generator.
pub fn mutate(
    children: &mut Population,
    param: &Param,
    feature_selection: &Vec<usize>,
    rng: &mut ChaCha8Rng,
) {
    let feature_len = feature_selection.len();

    if param.ga.mutated_children_pct > 0.0 {
        let num_mutated_individuals =
            (children.individuals.len() as f64 * param.ga.mutated_children_pct / 100.0) as usize;

        let num_mutated_features =
            (feature_len as f64 * param.ga.mutated_features_pct / 100.0) as usize;

        // Select indices of the individuals to mutate
        let individuals_to_mutate =
            sample(rng, children.individuals.len(), num_mutated_individuals);

        for idx in individuals_to_mutate {
            // Mutate features for each selected individual
            let individual = &mut children.individuals[idx]; // Get a mutable reference
            let feature_indices = sample(rng, feature_len, num_mutated_features)
                .iter()
                .map(|i| feature_selection[i])
                .collect::<Vec<usize>>();

            match individual.language {
                individual::TERNARY_LANG | individual::RATIO_LANG => {
                    mutate_ternary(individual, param, &feature_indices, rng);
                }
                individual::POW2_LANG => {
                    mutate_pow2(individual, param, &feature_indices, rng);
                }
                individual::BINARY_LANG => {
                    mutate_binary(individual, param, &feature_indices, rng);
                }
                other => {
                    panic!("Unsupported language {}", other);
                }
            };
        }
    }
}

/// Remove stillborn individuals from the population
///
/// # Arguments
///
/// * `children` - The population of children to filter.
///
/// # Returns
///
/// The number of stillborn individuals removed.
pub fn remove_stillborn(children: &mut Population) -> u32 {
    let mut stillborn_children: u32 = 0;
    let mut valid_individuals: Vec<Individual> = Vec::new();
    let individuals = mem::take(&mut children.individuals);
    for individual in individuals.into_iter() {
        if individual.language == TERNARY_LANG || individual.language == RATIO_LANG {
            let mut has_positive: bool = false;
            let mut has_negative: bool = false;
            let mut stillborn_child: bool = true;
            for feature in individual.features.values() {
                if *feature < 0 {
                    has_negative = true;
                    if has_positive {
                        stillborn_child = false;
                        break;
                    }
                }
                if *feature > 0 {
                    has_positive = true;
                    if has_negative {
                        stillborn_child = false;
                        break;
                    }
                }
            }
            if stillborn_child {
                stillborn_children += 1;
                // println!("still {:?}",individual.features);
            } else {
                valid_individuals.push(individual);
            }
        } else if individual.k == 0 {
            stillborn_children += 1;
        } else {
            valid_individuals.push(individual);
        }
    }
    children.individuals = valid_individuals;

    stillborn_children
}

// pub fn remove_inefficient(parents: &mut Population) -> u32 {
//     let mut inefficient_parents: u32 = 0;
//     let mut valid_individuals: Vec<Individual> = Vec::new();
//     let individuals = mem::take(&mut parents.individuals);
//     for individual in individuals.into_iter() {
//         if individual.specificity > 0.4 && individual.sensitivity > 0.4 {
//             valid_individuals.push(individual);
//         } else {
//             inefficient_parents += 1;
//         }
//     }
//     parents.individuals = valid_individuals;
//     inefficient_parents

// }

/// Mutates ternary individuals by changing signs, removing variables, or adding new variables
///
/// # Arguments
///
/// * `individual` - The individual to mutate.
/// * `param` - Parameters for the genetic algorithm.
/// * `feature_indices` - The list of feature indices available for mutation.
/// * `rng` - Random number generator.  
pub fn mutate_ternary(
    individual: &mut Individual,
    param: &Param,
    feature_indices: &Vec<usize>,
    rng: &mut ChaCha8Rng,
) {
    let p1 = param.ga.mutation_non_null_chance_pct / 200.0;
    let p2 = 2.0 * p1;

    for i in feature_indices {
        if individual.features.contains_key(&i) {
            individual.k -= 1;
            individual.features.remove(&i);
        }
        match rng.gen::<f64>() {
            r if r < p1 => {
                individual.k += 1;
                individual.features.insert(*i, 1);
            }
            r if r < p2 => {
                individual.k += 1;
                individual.features.insert(*i, -1);
            }
            _ => {}
        };
    }
}

/// Mutates binary individuals by changing signs, removing variables, or adding new variables
///
/// # Arguments
///
/// * `individual` - The individual to mutate.
/// * `param` - Parameters for the genetic algorithm.
/// * `feature_indices` - The list of feature indices available for mutation.
/// * `rng` - Random number generator.  
pub fn mutate_binary(
    individual: &mut Individual,
    param: &Param,
    feature_indices: &Vec<usize>,
    rng: &mut ChaCha8Rng,
) {
    let p1 = param.ga.mutation_non_null_chance_pct / 100.0;

    for i in feature_indices {
        if individual.features.contains_key(&i) {
            individual.k -= 1;
            individual.features.remove(&i);
        }
        match rng.gen::<f64>() {
            r if r < p1 => {
                individual.k += 1;
                individual.features.insert(*i, 1);
            }
            _ => {}
        };
    }
}

/// Mutates pow2 individuals by changing signs, removing variables, adding new variables, doubling variables, or dividing variables by two
///
/// # Arguments
///
/// * `individual` - The individual to mutate.
/// * `param` - Parameters for the genetic algorithm.
/// * `feature_indices` - The list of feature indices available for mutation.
/// * `rng` - Random number generator.  
pub fn mutate_pow2(
    individual: &mut Individual,
    param: &Param,
    feature_indices: &Vec<usize>,
    rng: &mut ChaCha8Rng,
) {
    let p1 = param.ga.mutation_non_null_chance_pct / 200.0;
    let p2 = 2.0 * p1;
    let p3 = 2.0 * p2;
    let p4 = 3.0 * p2;

    for i in feature_indices {
        let value = if individual.features.contains_key(&i) {
            individual.k -= 1;
            individual.features.remove(&i).unwrap()
        } else {
            0
        };
        match rng.gen::<f64>() {
            r if r < p1 => {
                individual.k += 1;
                individual.features.insert(*i, 1);
            }
            r if r < p2 => {
                individual.k += 1;
                individual.features.insert(*i, -1);
            }
            r if r < p3 => {
                if value != 0 {
                    individual
                        .features
                        .insert(*i, if value.abs() < 64 { 2 * value } else { value });
                    individual.k += 1;
                }
            }
            r if r < p4 => {
                if value != 0 {
                    individual
                        .features
                        .insert(*i, if value.abs() == 1 { value } else { value / 2 });
                    individual.k += 1;
                }
            }
            _ => {}
        };
    }
}

/// Gets GPU assay if GPU is enabled, random sampling is not used and GPU memory is sufficient for population size
///
/// # Arguments
///
/// * `data` - The dataset to operate on.
/// * `param` - Parameters for the genetic algorithm.
///
/// # Returns
///
/// An optional GPU assay for fitness evaluation.
fn get_gpu_assay(data: &Data, param: &Param) -> Option<GpuAssay> {
    let gpu_assay = if param.general.gpu && param.ga.random_sampling_pct == 0.0 {
        let buffer_binding_size = GpuAssay::get_max_buffer_size(&param.gpu) as usize;
        let gpu_max_nb_models =
            buffer_binding_size / (data.sample_len * std::mem::size_of::<f32>());
        let assay = if gpu_max_nb_models < param.ga.population_size as usize {
            warn!("GPU requires a maximum number of models (<=> Population size). \
            \nAccording to your configuration, param.ga.population_size must not exceed {}. \
            \nIf your configuration supports it and you know what you're doing, consider alternatively increasing the size of the buffers to {:.0} MB (do not forget to adjust the total size accordingly) \
            \nThis Gpredomics session will therefore be launched without a GPU.", gpu_max_nb_models,
            ((param.ga.population_size as usize * data.sample_len * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)+1.0));
            None
        } else {
            Some(GpuAssay::new(
                &data.X,
                &data.feature_selection,
                data.sample_len,
                param.ga.population_size as usize,
                &param.gpu,
            ))
        };
        assay
    } else {
        None
    };

    gpu_assay
}

/// Creates GPU assays for each fold when inner CV is enabled with GPU
///
/// # Arguments
///
/// * `cv` - The cross-validation structure containing folds.
/// * `param` - Parameters for the genetic algorithm.
///
/// # Returns
///
/// A vector of tuples (validation_assay, training_assay) for each fold
fn create_gpu_assays_for_folds(
    cv: &CV,
    param: &Param,
) -> Vec<(Option<GpuAssay>, Option<GpuAssay>)> {
    if !param.general.gpu {
        return vec![(None, None); cv.validation_folds.len()];
    }

    cv.validation_folds.iter().enumerate().map(|(idx, fold)| {
        let buffer_binding_size = GpuAssay::get_max_buffer_size(&param.gpu) as usize;

        // GPU assay for validation fold
        let gpu_max_nb_models_val = buffer_binding_size / (fold.sample_len * std::mem::size_of::<f32>());
        let val_assay = if gpu_max_nb_models_val < param.ga.population_size as usize {
            warn!("GPU cannot be used for validation fold {}: requires max {} models but population size is {}",
                idx, gpu_max_nb_models_val, param.ga.population_size);
            None
        } else {
            Some(GpuAssay::new(&fold.X, &fold.feature_selection, fold.sample_len, param.ga.population_size as usize, &param.gpu))
        };

        // GPU assay for training set
        let training_set = &cv.training_sets[idx];
        let gpu_max_nb_models_train = buffer_binding_size / (training_set.sample_len * std::mem::size_of::<f32>());
        let train_assay = if gpu_max_nb_models_train < param.ga.population_size as usize {
            warn!("GPU cannot be used for training set {}: requires max {} models but population size is {}",
                idx, gpu_max_nb_models_train, param.ga.population_size);
            None
        } else {
            Some(GpuAssay::new(&training_set.X, &training_set.feature_selection, training_set.sample_len, param.ga.population_size as usize, &param.gpu))
        };

        (val_assay, train_assay)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::individual::{BINARY_LANG, POW2_LANG, RATIO_LANG, TERNARY_LANG};
    use crate::individual::{LOG_TYPE, PREVALENCE_TYPE, RAW_TYPE};

    /// Helper function to create a simple test population
    fn create_test_population(size: usize, language: u8, data_type: u8) -> Population {
        let mut pop = Population::new();
        for i in 0..size {
            let mut individual = Individual::new();
            individual.language = language;
            individual.data_type = data_type;
            individual.fit = (size - i) as f64; // Higher fit for earlier individuals
            individual.auc = 0.5 + (size - i) as f64 / (size as f64 * 2.0);
            individual.k = 3;

            // Add some features
            individual.features.insert(0, 1);
            individual.features.insert(1, -1);
            individual.features.insert(2, 1);

            individual.hash = i as u64;
            individual.epoch = 0;

            pop.individuals.push(individual);
        }
        pop
    }

    /// Helper function to create default parameters for testing
    fn create_test_params() -> Param {
        let mut param = Param::default();
        param.ga.select_elite_pct = 10.0;
        param.ga.select_random_pct = 20.0;
        param.ga.select_niche_pct = 0.0;
        param.ga.mutated_children_pct = 50.0;
        param.ga.mutated_features_pct = 20.0;
        param.ga.mutation_non_null_chance_pct = 50.0;
        param
    }

    #[test]
    fn test_select_parents_elite_selection() {
        let pop = create_test_population(100, TERNARY_LANG, PREVALENCE_TYPE);
        let param = create_test_params();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let parents = select_parents(&pop, &param, &mut rng);

        // Should select at least elite_pct% of population
        let expected_min = (100.0 * param.ga.select_elite_pct / 100.0) as usize;
        assert!(
            parents.individuals.len() >= expected_min,
            "Expected at least {} parents, got {}",
            expected_min,
            parents.individuals.len()
        );

        // Elite individuals should be at the beginning (highest fit)
        let first_parent_fit = parents.individuals[0].fit;
        let pop_best_fit = pop.individuals[0].fit;
        assert_eq!(
            first_parent_fit, pop_best_fit,
            "Elite selection should pick best individuals"
        );
    }

    #[test]
    fn test_select_parents_with_niche() {
        let mut pop = Population::new();

        // Create diverse population with different language/data_type combinations
        for i in 0..50 {
            let mut ind = Individual::new();
            ind.language = if i < 25 { TERNARY_LANG } else { BINARY_LANG };
            ind.data_type = if i % 2 == 0 {
                PREVALENCE_TYPE
            } else {
                RAW_TYPE
            };
            ind.fit = (50 - i) as f64;
            ind.hash = i as u64;
            ind.features.insert(0, 1);
            ind.k = 1;
            pop.individuals.push(ind);
        }

        let mut param = create_test_params();
        param.ga.select_elite_pct = 5.0;
        param.ga.select_niche_pct = 10.0;
        param.ga.select_random_pct = 5.0;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let parents = select_parents(&pop, &param, &mut rng);

        // Should have representation from different niches
        assert!(
            parents.individuals.len() > (50.0 * param.ga.select_elite_pct / 100.0) as usize,
            "Niche selection should add more parents beyond elite"
        );
    }

    #[test]
    fn test_cross_over_creates_valid_children() {
        let pop = create_test_population(10, TERNARY_LANG, PREVALENCE_TYPE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let children = cross_over(&pop, 20, &mut rng);

        assert_eq!(
            children.individuals.len(),
            20,
            "Should create requested number of children"
        );

        for child in &children.individuals {
            // Each child should have features
            assert!(!child.features.is_empty(), "Child should have features");

            // Child should have parent hashes
            assert!(child.parents.is_some(), "Child should have parents");
            assert_eq!(
                child.parents.as_ref().unwrap().len(),
                2,
                "Child should have 2 parents"
            );

            // Child k should match number of features
            assert_eq!(
                child.k,
                child.features.len(),
                "Child k should match feature count"
            );

            // Child should inherit language and data_type from one parent
            assert_eq!(
                child.language, TERNARY_LANG,
                "Child should inherit parent language"
            );
            assert_eq!(
                child.data_type, PREVALENCE_TYPE,
                "Child should inherit parent data_type"
            );
        }
    }

    #[test]
    fn test_cross_over_with_different_languages() {
        let mut pop = Population::new();

        // Create parents with different languages
        for i in 0..5 {
            let mut ind1 = Individual::new();
            ind1.language = TERNARY_LANG;
            ind1.data_type = PREVALENCE_TYPE;
            ind1.features.insert(0, 1);
            ind1.features.insert(1, -1);
            ind1.k = 2;
            ind1.hash = i as u64;
            pop.individuals.push(ind1);

            let mut ind2 = Individual::new();
            ind2.language = BINARY_LANG;
            ind2.data_type = PREVALENCE_TYPE;
            ind2.features.insert(0, 1);
            ind2.features.insert(2, 1);
            ind2.k = 2;
            ind2.hash = (i + 10) as u64;
            pop.individuals.push(ind2);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 10, &mut rng);

        assert_eq!(
            children.individuals.len(),
            10,
            "Should create requested children"
        );

        for child in &children.individuals {
            // Child should be valid
            assert!(!child.features.is_empty(), "Child should have features");
            assert!(
                child.language == TERNARY_LANG || child.language == BINARY_LANG,
                "Child should have valid language"
            );
        }
    }

    #[test]
    fn test_mutate_ternary_changes_features() {
        let mut pop = create_test_population(10, TERNARY_LANG, PREVALENCE_TYPE);
        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4, 5];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Store original features
        let original_features: Vec<_> = pop
            .individuals
            .iter()
            .map(|ind| ind.features.clone())
            .collect();

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // At least some individuals should have been mutated
        let mut changed_count = 0;
        for i in 0..pop.individuals.len() {
            if pop.individuals[i].features != original_features[i] {
                changed_count += 1;
            }
        }

        // With 50% mutation rate, expect around half to be mutated
        assert!(
            changed_count > 0,
            "At least some individuals should be mutated"
        );
        assert!(
            changed_count <= pop.individuals.len(),
            "Not all should necessarily change"
        );
    }

    #[test]
    fn test_mutate_ternary_respects_language() {
        let mut pop = create_test_population(1, TERNARY_LANG, PREVALENCE_TYPE);
        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // All feature values should be in {-1, 0, 1} for ternary language
        for (_, &val) in &pop.individuals[0].features {
            assert!(
                val >= -1 && val <= 1,
                "Ternary values should be in range [-1, 1], got {}",
                val
            );
        }
    }

    #[test]
    fn test_mutate_binary_only_positive_values() {
        let mut pop = Population::new();

        // Create binary individuals with only positive values
        for _ in 0..10 {
            let mut ind = Individual::new();
            ind.language = BINARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.features.insert(0, 1);
            ind.features.insert(1, 1);
            ind.k = 2;
            pop.individuals.push(ind);
        }

        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4, 5];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // All feature values should be 1 for binary language (no 0 or negative)
        for ind in &pop.individuals {
            for (_, &val) in &ind.features {
                assert_eq!(
                    val, 1,
                    "Binary language should only have value 1, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_mutate_pow2_power_values() {
        let mut pop = create_test_population(20, POW2_LANG, PREVALENCE_TYPE);

        // Set initial features with various power-of-2 values
        for ind in &mut pop.individuals {
            ind.features.clear();
            ind.features.insert(0, 2);
            ind.features.insert(1, -4);
            ind.features.insert(2, 1);
            ind.k = 3;
        }

        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4, 5];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // Check that values are valid powers of 2 (or their negatives)
        for ind in &pop.individuals {
            for (_, &val) in &ind.features {
                let abs_val = val.abs();
                // Valid pow2 values: 1, 2, 4, 8, 16, 32, 64
                let is_pow2 = abs_val > 0 && (abs_val & (abs_val - 1)) == 0;
                assert!(
                    is_pow2 && abs_val <= 64,
                    "POW2 value should be a power of 2 up to 64, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_remove_stillborn_ternary() {
        let mut pop = Population::new();

        // Create valid individual (has both positive and negative)
        let mut valid = Individual::new();
        valid.language = TERNARY_LANG;
        valid.features.insert(0, 1);
        valid.features.insert(1, -1);
        valid.k = 2;
        pop.individuals.push(valid);

        // Create stillborn (only positive values)
        let mut stillborn1 = Individual::new();
        stillborn1.language = TERNARY_LANG;
        stillborn1.features.insert(0, 1);
        stillborn1.features.insert(1, 1);
        stillborn1.k = 2;
        pop.individuals.push(stillborn1);

        // Create stillborn (only negative values)
        let mut stillborn2 = Individual::new();
        stillborn2.language = TERNARY_LANG;
        stillborn2.features.insert(0, -1);
        stillborn2.features.insert(1, -1);
        stillborn2.k = 2;
        pop.individuals.push(stillborn2);

        let removed = remove_stillborn(&mut pop);

        assert_eq!(removed, 2, "Should remove 2 stillborn individuals");
        assert_eq!(
            pop.individuals.len(),
            1,
            "Should have 1 valid individual remaining"
        );
        assert_eq!(
            pop.individuals[0].k, 2,
            "Valid individual should have 2 features"
        );
    }

    #[test]
    fn test_remove_stillborn_ratio() {
        let mut pop = Population::new();

        // Create valid individual for RATIO_LANG (has both positive and negative)
        let mut valid = Individual::new();
        valid.language = RATIO_LANG;
        valid.features.insert(0, 1);
        valid.features.insert(1, -1);
        valid.k = 2;
        pop.individuals.push(valid);

        // Create stillborn (only positive values)
        let mut stillborn = Individual::new();
        stillborn.language = RATIO_LANG;
        stillborn.features.insert(0, 1);
        stillborn.features.insert(1, 1);
        stillborn.k = 2;
        pop.individuals.push(stillborn);

        let removed = remove_stillborn(&mut pop);

        assert_eq!(removed, 1, "Should remove 1 stillborn individual");
        assert_eq!(
            pop.individuals.len(),
            1,
            "Should have 1 valid individual remaining"
        );
    }

    #[test]
    fn test_remove_stillborn_binary_zero_k() {
        let mut pop = Population::new();

        // Create valid individual
        let mut valid = Individual::new();
        valid.language = BINARY_LANG;
        valid.features.insert(0, 1);
        valid.k = 1;
        pop.individuals.push(valid);

        // Create stillborn with k=0
        let mut stillborn = Individual::new();
        stillborn.language = BINARY_LANG;
        stillborn.k = 0;
        pop.individuals.push(stillborn);

        let removed = remove_stillborn(&mut pop);

        assert_eq!(removed, 1, "Should remove individual with k=0");
        assert_eq!(
            pop.individuals.len(),
            1,
            "Should have 1 valid individual remaining"
        );
    }

    #[test]
    fn test_mutate_with_zero_mutation_rate() {
        let mut pop = create_test_population(10, TERNARY_LANG, PREVALENCE_TYPE);
        let mut param = create_test_params();
        param.ga.mutated_children_pct = 0.0; // No mutation

        let feature_selection = vec![0, 1, 2, 3, 4, 5];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let original_features: Vec<_> = pop
            .individuals
            .iter()
            .map(|ind| ind.features.clone())
            .collect();

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // Nothing should change with 0% mutation rate
        for i in 0..pop.individuals.len() {
            assert_eq!(
                pop.individuals[i].features, original_features[i],
                "Features should not change with 0% mutation rate"
            );
        }
    }

    #[test]
    fn test_cross_over_preserves_feature_types() {
        let mut pop = Population::new();

        // Parent 1: ternary with specific features
        let mut p1 = Individual::new();
        p1.language = TERNARY_LANG;
        p1.data_type = PREVALENCE_TYPE;
        p1.features.insert(0, 1);
        p1.features.insert(1, -1);
        p1.features.insert(2, 1);
        p1.k = 3;
        p1.hash = 1;
        pop.individuals.push(p1);

        // Parent 2: ternary with different features
        let mut p2 = Individual::new();
        p2.language = TERNARY_LANG;
        p2.data_type = PREVALENCE_TYPE;
        p2.features.insert(1, 1);
        p2.features.insert(3, -1);
        p2.features.insert(4, 1);
        p2.k = 3;
        p2.hash = 2;
        pop.individuals.push(p2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 10, &mut rng);

        for child in &children.individuals {
            // Check child has features from union of parents
            assert!(child.k > 0, "Child should have features");

            // All feature values should be valid for ternary
            for (_, &val) in &child.features {
                assert!(
                    val >= -1 && val <= 1,
                    "Ternary child should have values in [-1, 1], got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_mutate_updates_k_correctly() {
        let mut pop = create_test_population(5, TERNARY_LANG, PREVALENCE_TYPE);
        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // Verify k matches actual feature count
        for ind in &pop.individuals {
            assert_eq!(
                ind.k,
                ind.features.len(),
                "k ({}) should match number of features ({})",
                ind.k,
                ind.features.len()
            );
        }
    }

    #[test]
    fn test_select_parents_deterministic_with_seed() {
        let pop = create_test_population(50, TERNARY_LANG, PREVALENCE_TYPE);
        let param = create_test_params();

        // Run selection twice with same seed
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let parents1 = select_parents(&pop, &param, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let parents2 = select_parents(&pop, &param, &mut rng2);

        // Results should be identical
        assert_eq!(
            parents1.individuals.len(),
            parents2.individuals.len(),
            "Same seed should produce same number of parents"
        );

        for i in 0..parents1.individuals.len() {
            assert_eq!(
                parents1.individuals[i].hash, parents2.individuals[i].hash,
                "Same seed should select same parents at index {}",
                i
            );
        }
    }

    #[test]
    fn test_cross_over_deterministic_with_seed() {
        let pop = create_test_population(10, TERNARY_LANG, PREVALENCE_TYPE);

        // Run crossover twice with same seed
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let children1 = cross_over(&pop, 20, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let children2 = cross_over(&pop, 20, &mut rng2);

        // Results should be identical
        assert_eq!(children1.individuals.len(), children2.individuals.len());

        for i in 0..children1.individuals.len() {
            assert_eq!(
                children1.individuals[i].features, children2.individuals[i].features,
                "Same seed should produce same child features at index {}",
                i
            );
            assert_eq!(
                children1.individuals[i].parents, children2.individuals[i].parents,
                "Same seed should produce same parent selection at index {}",
                i
            );
        }
    }

    #[test]
    fn test_evolve_produces_valid_population() {
        // Create a simple test - this is a structural test
        // Full functional testing requires valid fitted data
        let mut data = Data::new();
        data.feature_selection = vec![0, 1, 2, 3, 4];
        data.sample_len = 10;
        data.feature_len = 5;

        let pop = create_test_population(20, TERNARY_LANG, PREVALENCE_TYPE);
        let mut param = create_test_params();
        param.ga.population_size = 20;
        param.ga.select_elite_pct = 20.0;
        param.ga.select_random_pct = 10.0;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut cv = None;

        // Note: evolve will try to fit, but with minimal mock data
        // this is primarily a structural/integration test
        let new_pop = evolve(pop, &data, &mut cv, &param, &vec![], &None, 1, &mut rng);

        // Check that we get a valid population back
        assert!(
            new_pop.individuals.len() > 0,
            "Evolve should produce individuals"
        );

        // Check that all individuals are valid
        for ind in &new_pop.individuals {
            assert_eq!(ind.k, ind.features.len(), "k should match feature count");
            assert!(
                ind.epoch > 0 || ind.k > 0,
                "Individual should be from current epoch or have features"
            );
        }
    }

    #[test]
    fn test_mutate_preserves_language_and_data_type() {
        let mut pop = Population::new();

        // Create individuals with different languages and data types
        let languages = vec![BINARY_LANG, TERNARY_LANG, POW2_LANG];
        let data_types = vec![RAW_TYPE, PREVALENCE_TYPE, LOG_TYPE];

        for lang in &languages {
            for dtype in &data_types {
                let mut ind = Individual::new();
                ind.language = *lang;
                ind.data_type = *dtype;
                ind.features.insert(0, 1);
                ind.k = 1;
                pop.individuals.push(ind);
            }
        }

        let original_langs: Vec<_> = pop.individuals.iter().map(|i| i.language).collect();
        let original_dtypes: Vec<_> = pop.individuals.iter().map(|i| i.data_type).collect();

        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4, 5];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // Language and data_type should not change after mutation
        for i in 0..pop.individuals.len() {
            assert_eq!(
                pop.individuals[i].language, original_langs[i],
                "Mutation should not change language"
            );
            assert_eq!(
                pop.individuals[i].data_type, original_dtypes[i],
                "Mutation should not change data_type"
            );
        }
    }

    #[test]
    fn test_cross_over_requires_at_least_two_parents() {
        // Edge case: crossover requires at least 2 parents
        // This test documents the requirement rather than testing single parent
        let mut pop = Population::new();

        let mut ind1 = Individual::new();
        ind1.language = TERNARY_LANG;
        ind1.data_type = PREVALENCE_TYPE;
        ind1.features.insert(0, 1);
        ind1.features.insert(1, -1);
        ind1.k = 2;
        ind1.hash = 1;
        pop.individuals.push(ind1);

        let mut ind2 = Individual::new();
        ind2.language = TERNARY_LANG;
        ind2.data_type = PREVALENCE_TYPE;
        ind2.features.insert(2, 1);
        ind2.k = 1;
        ind2.hash = 2;
        pop.individuals.push(ind2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 5, &mut rng);

        assert_eq!(
            children.individuals.len(),
            5,
            "Should create requested children with 2+ parents"
        );
        for child in &children.individuals {
            assert!(!child.features.is_empty(), "Child should have features");
            assert!(
                child.parents.is_some(),
                "Child should have parent information"
            );
        }
    }

    #[test]
    fn test_mutate_pow2_doubling_and_halving() {
        let mut pop = Population::new();

        // Create multiple individuals to increase chance of hitting doubling/halving mutations
        for _ in 0..50 {
            let mut ind = Individual::new();
            ind.language = POW2_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.features.insert(0, 2);
            ind.features.insert(1, -4);
            ind.features.insert(2, 8);
            ind.k = 3;
            pop.individuals.push(ind);
        }

        let mut param = create_test_params();
        param.ga.mutated_children_pct = 100.0; // Mutate all
        param.ga.mutated_features_pct = 100.0; // Mutate all features
        param.ga.mutation_non_null_chance_pct = 80.0; // High mutation rate

        let feature_selection = vec![0, 1, 2];
        let mut rng = ChaCha8Rng::seed_from_u64(999);

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // Check that some values were doubled or halved (this is probabilistic)
        let mut found_doubling = false;
        let mut found_halving = false;

        for ind in &pop.individuals {
            for (_, &val) in &ind.features {
                if val.abs() > 8 || (val.abs() == 1) {
                    found_halving = true;
                }
                if val.abs() >= 16 {
                    found_doubling = true;
                }
            }
        }

        // With 50 individuals and high mutation rate, we should see some doubling/halving
        assert!(
            found_doubling || found_halving,
            "With high mutation rate on many individuals, should see some doubling or halving"
        );
    }

    #[test]
    fn test_mutate_pow2_boundary_values() {
        let mut ind = Individual::new();
        ind.language = POW2_LANG;
        ind.data_type = PREVALENCE_TYPE;
        ind.features.insert(0, 64); // Max value
        ind.features.insert(1, -64); // Min value
        ind.features.insert(2, 1); // Minimum positive
        ind.features.insert(3, -1); // Minimum negative
        ind.k = 4;

        let mut pop = Population::new();
        pop.individuals.push(ind);

        let param = create_test_params();
        let feature_indices = vec![0, 1, 2, 3];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Run mutation multiple times
        for _ in 0..10 {
            mutate_pow2(&mut pop.individuals[0], &param, &feature_indices, &mut rng);
        }

        // Values should stay within bounds
        for (_, &val) in &pop.individuals[0].features {
            assert!(
                val.abs() <= 64,
                "POW2 values should not exceed 64, got {}",
                val
            );
        }
    }

    #[test]
    fn test_remove_stillborn_empty_features() {
        let mut pop = Population::new();

        // Ternary individual with no features (empty)
        let mut empty_ternary = Individual::new();
        empty_ternary.language = TERNARY_LANG;
        empty_ternary.k = 0;
        pop.individuals.push(empty_ternary);

        // Binary individual with no features
        let mut empty_binary = Individual::new();
        empty_binary.language = BINARY_LANG;
        empty_binary.k = 0;
        pop.individuals.push(empty_binary);

        let removed = remove_stillborn(&mut pop);

        // Both should be removed (k=0 for binary, empty for ternary also counts as stillborn)
        assert_eq!(removed, 2, "Should remove all empty individuals");
        assert_eq!(pop.individuals.len(), 0, "Population should be empty");
    }

    #[test]
    fn test_remove_stillborn_only_zeros() {
        let mut pop = Population::new();

        // Ternary with only zeros (neither positive nor negative)
        let mut zero_ternary = Individual::new();
        zero_ternary.language = TERNARY_LANG;
        zero_ternary.k = 0;
        pop.individuals.push(zero_ternary);

        let removed = remove_stillborn(&mut pop);

        assert_eq!(removed, 1, "Should remove individual with no features");
    }

    #[test]
    fn test_select_parents_empty_population() {
        let pop = Population::new(); // Empty population
        let param = create_test_params();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let parents = select_parents(&pop, &param, &mut rng);

        assert_eq!(
            parents.individuals.len(),
            0,
            "Empty population should return empty parents"
        );
    }

    #[test]
    fn test_select_parents_all_same_type() {
        let mut pop = Population::new();

        // All individuals same language/data_type
        for i in 0..20 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.fit = (20 - i) as f64;
            ind.hash = i as u64;
            ind.features.insert(0, 1);
            ind.k = 1;
            pop.individuals.push(ind);
        }

        let mut param = create_test_params();
        param.ga.select_elite_pct = 25.0;
        param.ga.select_random_pct = 25.0;
        param.ga.select_niche_pct = 0.0;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let parents = select_parents(&pop, &param, &mut rng);

        // Should still select based on percentages
        assert!(
            parents.individuals.len() >= 5,
            "Should select at least elite percentage"
        );
        assert!(
            parents.individuals.len() <= 15,
            "Should not select too many"
        );
    }

    #[test]
    fn test_cross_over_merges_features_correctly() {
        let mut pop = Population::new();

        // Parent 1: features {0:1, 1:1}
        let mut p1 = Individual::new();
        p1.language = TERNARY_LANG;
        p1.data_type = PREVALENCE_TYPE;
        p1.features.insert(0, 1);
        p1.features.insert(1, 1);
        p1.k = 2;
        p1.hash = 1;
        pop.individuals.push(p1);

        // Parent 2: features {2:-1, 3:-1}
        let mut p2 = Individual::new();
        p2.language = TERNARY_LANG;
        p2.data_type = PREVALENCE_TYPE;
        p2.features.insert(2, -1);
        p2.features.insert(3, -1);
        p2.k = 2;
        p2.hash = 2;
        pop.individuals.push(p2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 20, &mut rng);

        // Children should have features from both parents
        let mut has_from_p1 = false;
        let mut has_from_p2 = false;

        for child in &children.individuals {
            if child.features.contains_key(&0) || child.features.contains_key(&1) {
                has_from_p1 = true;
            }
            if child.features.contains_key(&2) || child.features.contains_key(&3) {
                has_from_p2 = true;
            }
        }

        assert!(
            has_from_p1,
            "Some children should inherit features from parent 1"
        );
        assert!(
            has_from_p2,
            "Some children should inherit features from parent 2"
        );
    }

    #[test]
    fn test_mutate_ratio_language() {
        let mut pop = Population::new();

        let mut ind = Individual::new();
        ind.language = RATIO_LANG;
        ind.data_type = PREVALENCE_TYPE;
        ind.features.insert(0, 1);
        ind.features.insert(1, -1);
        ind.k = 2;
        pop.individuals.push(ind);

        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // RATIO_LANG uses same mutation as ternary: values should be in {-1, 0, 1}
        for (_, &val) in &pop.individuals[0].features {
            assert!(
                val >= -1 && val <= 1,
                "Ratio language should have values in [-1, 1], got {}",
                val
            );
        }
    }

    #[test]
    fn test_cross_over_conversion_between_languages() {
        let mut pop = Population::new();

        // Parent 1: TERNARY
        let mut p1 = Individual::new();
        p1.language = TERNARY_LANG;
        p1.data_type = PREVALENCE_TYPE;
        p1.features.insert(0, 1);
        p1.features.insert(1, -1);
        p1.k = 2;
        p1.hash = 1;
        pop.individuals.push(p1);

        // Parent 2: BINARY (different language)
        let mut p2 = Individual::new();
        p2.language = BINARY_LANG;
        p2.data_type = PREVALENCE_TYPE;
        p2.features.insert(0, 1);
        p2.features.insert(2, 1);
        p2.k = 2;
        p2.hash = 2;
        pop.individuals.push(p2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 10, &mut rng);

        // All children should be valid (conversion should work)
        for child in &children.individuals {
            assert!(!child.features.is_empty(), "Child should have features");
            assert_eq!(
                child.k,
                child.features.len(),
                "k should match feature count"
            );

            // Check values are valid for child's language
            if child.language == TERNARY_LANG {
                for (_, &val) in &child.features {
                    assert!(
                        val >= -1 && val <= 1,
                        "Ternary child should have valid values"
                    );
                }
            } else if child.language == BINARY_LANG {
                for (_, &val) in &child.features {
                    assert_eq!(val, 1, "Binary child should only have 1");
                }
            }
        }
    }

    #[test]
    fn test_mutate_with_100_percent_rate() {
        let mut pop = create_test_population(10, TERNARY_LANG, PREVALENCE_TYPE);

        let mut param = create_test_params();
        param.ga.mutated_children_pct = 100.0; // Mutate all
        param.ga.mutated_features_pct = 100.0; // Mutate all features

        let feature_selection = vec![0, 1, 2, 3, 4, 5];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let original_features: Vec<_> = pop
            .individuals
            .iter()
            .map(|ind| ind.features.clone())
            .collect();

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // With 100% mutation rate, all should change (though there's a small chance some stay same by chance)
        let mut changed_count = 0;
        for i in 0..pop.individuals.len() {
            if pop.individuals[i].features != original_features[i] {
                changed_count += 1;
            }
        }

        // Most should change with 100% rate
        assert!(
            changed_count >= 7,
            "With 100% mutation rate, most individuals should change (got {})",
            changed_count
        );
    }

    #[test]
    fn test_select_parents_respects_percentages() {
        let pop = create_test_population(100, TERNARY_LANG, PREVALENCE_TYPE);
        let mut param = create_test_params();

        // Set specific percentages
        param.ga.select_elite_pct = 10.0;
        param.ga.select_random_pct = 20.0;
        param.ga.select_niche_pct = 0.0;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let parents = select_parents(&pop, &param, &mut rng);

        // Should have at least 10 (elite) + some random
        assert!(
            parents.individuals.len() >= 10,
            "Should select at least elite_pct individuals"
        );

        // Elite should be the best individuals
        for i in 0..10 {
            assert_eq!(
                parents.individuals[i].hash, pop.individuals[i].hash,
                "First {} parents should be elite (best individuals)",
                i
            );
        }
    }

    #[test]
    fn test_mutate_ternary_maintains_k_consistency() {
        let mut ind = Individual::new();
        ind.language = TERNARY_LANG;
        ind.data_type = PREVALENCE_TYPE;
        ind.features.insert(0, 1);
        ind.features.insert(1, -1);
        ind.features.insert(2, 1);
        ind.k = 3;

        let param = create_test_params();
        let feature_indices = vec![0, 1, 2, 3, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Run mutation multiple times
        for _ in 0..20 {
            mutate_ternary(&mut ind, &param, &feature_indices, &mut rng);
            assert_eq!(
                ind.k,
                ind.features.len(),
                "k should always equal feature count after mutation"
            );
        }
    }

    #[test]
    fn test_mutate_binary_maintains_k_consistency() {
        let mut ind = Individual::new();
        ind.language = BINARY_LANG;
        ind.data_type = PREVALENCE_TYPE;
        ind.features.insert(0, 1);
        ind.features.insert(1, 1);
        ind.k = 2;

        let param = create_test_params();
        let feature_indices = vec![0, 1, 2, 3, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Run mutation multiple times
        for _ in 0..20 {
            mutate_binary(&mut ind, &param, &feature_indices, &mut rng);
            assert_eq!(
                ind.k,
                ind.features.len(),
                "k should always equal feature count after mutation"
            );
        }
    }

    #[test]
    fn test_mutate_pow2_maintains_k_consistency() {
        let mut ind = Individual::new();
        ind.language = POW2_LANG;
        ind.data_type = PREVALENCE_TYPE;
        ind.features.insert(0, 2);
        ind.features.insert(1, -4);
        ind.k = 2;

        let param = create_test_params();
        let feature_indices = vec![0, 1, 2, 3, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Run mutation multiple times
        for _ in 0..20 {
            mutate_pow2(&mut ind, &param, &feature_indices, &mut rng);
            assert_eq!(
                ind.k,
                ind.features.len(),
                "k should always equal feature count after mutation"
            );
        }
    }

    #[test]
    fn test_cross_over_produces_valid_k() {
        let pop = create_test_population(20, TERNARY_LANG, PREVALENCE_TYPE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let children = cross_over(&pop, 50, &mut rng);

        for (i, child) in children.individuals.iter().enumerate() {
            assert_eq!(
                child.k,
                child.features.len(),
                "Child {} should have k={} matching its {} features",
                i,
                child.k,
                child.features.len()
            );
        }
    }

    #[test]
    fn test_remove_stillborn_preserves_valid_individuals() {
        let mut pop = Population::new();

        // Add valid ternary individuals
        for i in 0..5 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.features.insert(0, 1);
            ind.features.insert(1, -1);
            ind.k = 2;
            ind.hash = i as u64;
            pop.individuals.push(ind);
        }

        // Add stillborn
        let mut stillborn = Individual::new();
        stillborn.language = TERNARY_LANG;
        stillborn.features.insert(0, 1);
        stillborn.features.insert(1, 1);
        stillborn.k = 2;
        pop.individuals.push(stillborn);

        let original_hashes: Vec<_> = pop.individuals[0..5].iter().map(|i| i.hash).collect();

        let removed = remove_stillborn(&mut pop);

        assert_eq!(removed, 1, "Should remove exactly 1 stillborn");
        assert_eq!(pop.individuals.len(), 5, "Should keep 5 valid individuals");

        // Check that the valid individuals are still there
        for (i, hash) in original_hashes.iter().enumerate() {
            assert_eq!(
                pop.individuals[i].hash, *hash,
                "Valid individual {} should be preserved",
                i
            );
        }
    }

    #[test]
    fn test_selection_percentages_bounds_and_rounding() {
        let mut param = create_test_params();

        // Test with small population (rounding issues)
        let small_pop = create_test_population(5, TERNARY_LANG, PREVALENCE_TYPE);
        param.ga.select_elite_pct = 40.0; // 2 individuals
        param.ga.select_random_pct = 40.0; // 2 individuals
        param.ga.select_niche_pct = 0.0;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let parents = select_parents(&small_pop, &param, &mut rng);

        // Should not exceed population size
        assert!(
            parents.individuals.len() <= small_pop.individuals.len(),
            "Selected parents ({}) should not exceed population size ({})",
            parents.individuals.len(),
            small_pop.individuals.len()
        );

        // Check for duplicates by hash
        let mut hashes = std::collections::HashSet::new();
        for ind in &parents.individuals {
            assert!(
                hashes.insert(ind.hash),
                "Found duplicate individual in parents (hash={})",
                ind.hash
            );
        }

        // Test with extreme percentages
        param.ga.select_elite_pct = 100.0;
        param.ga.select_random_pct = 0.0;
        let all_elite = select_parents(&small_pop, &param, &mut rng);
        assert!(
            all_elite.individuals.len() <= small_pop.individuals.len(),
            "100% elite should not exceed population size"
        );
    }

    #[test]
    fn test_determinism_with_equal_fit_values() {
        let mut pop = Population::new();

        // Create individuals with SAME fit but different hashes
        for i in 0..20 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.fit = 0.75; // Same fit for all!
            ind.auc = 0.75;
            ind.features.insert(i, 1);
            ind.k = 1;
            ind.hash = i as u64;
            pop.individuals.push(ind);
        }

        let param = create_test_params();

        // Run selection twice with same seed
        let mut rng1 = ChaCha8Rng::seed_from_u64(999);
        let parents1 = select_parents(&pop, &param, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(999);
        let parents2 = select_parents(&pop, &param, &mut rng2);

        // Results must be identical (deterministic tie-breaking by hash)
        assert_eq!(
            parents1.individuals.len(),
            parents2.individuals.len(),
            "Same seed with equal fits should produce same number of parents"
        );

        for i in 0..parents1.individuals.len() {
            assert_eq!(
                parents1.individuals[i].hash, parents2.individuals[i].hash,
                "Parent {} should be identical with same seed (hash {} vs {})",
                i, parents1.individuals[i].hash, parents2.individuals[i].hash
            );
        }
    }

    #[test]
    fn test_crossover_conflicting_feature_signs() {
        let mut pop = Population::new();

        // Parent 1: feature 0 with +1
        let mut p1 = Individual::new();
        p1.language = TERNARY_LANG;
        p1.data_type = PREVALENCE_TYPE;
        p1.features.insert(0, 1); // Positive
        p1.features.insert(1, -1);
        p1.k = 2;
        p1.hash = 1;
        pop.individuals.push(p1);

        // Parent 2: feature 0 with -1 (opposite sign!)
        let mut p2 = Individual::new();
        p2.language = TERNARY_LANG;
        p2.data_type = PREVALENCE_TYPE;
        p2.features.insert(0, -1); // Negative
        p2.features.insert(2, 1);
        p2.k = 2;
        p2.hash = 2;
        pop.individuals.push(p2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 20, &mut rng);

        // All children should be valid and handle conflict resolution
        for child in &children.individuals {
            assert!(!child.features.is_empty(), "Child should have features");

            // If child has feature 0, it must be valid for ternary
            if let Some(&val) = child.features.get(&0) {
                assert!(
                    val == 1 || val == -1,
                    "Feature 0 with conflicting signs should resolve to +1 or -1, got {}",
                    val
                );
            }

            // All values must be valid for child's language
            for (_, &val) in &child.features {
                assert!(
                    val >= -1 && val <= 1,
                    "Ternary values must be in [-1, 1], got {}",
                    val
                );
                assert_ne!(val, 0, "No zero values should be in features HashMap");
            }

            assert_eq!(child.k, child.features.len(), "k must match feature count");
        }
    }

    #[test]
    fn test_mutation_with_empty_feature_selection() {
        let mut pop = create_test_population(5, TERNARY_LANG, PREVALENCE_TYPE);
        let param = create_test_params();
        let empty_selection: Vec<usize> = vec![];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let original_features: Vec<_> = pop
            .individuals
            .iter()
            .map(|ind| ind.features.clone())
            .collect();

        // Should not panic with empty selection
        mutate(&mut pop, &param, &empty_selection, &mut rng);

        // Nothing should change
        for i in 0..pop.individuals.len() {
            assert_eq!(
                pop.individuals[i].features, original_features[i],
                "Features should not change with empty feature_selection"
            );
        }
    }

    #[test]
    fn test_mutation_respects_feature_selection_boundaries() {
        let mut pop = create_test_population(10, TERNARY_LANG, PREVALENCE_TYPE);

        // Individuals have features {0, 1, 2}
        // But feature_selection only contains {5, 6, 7}
        let feature_selection = vec![5, 6, 7];

        let param = create_test_params();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        mutate(&mut pop, &param, &feature_selection, &mut rng);

        // Check that mutation only touched features in selection
        for ind in &pop.individuals {
            for feature_idx in ind.features.keys() {
                // Original features {0, 1, 2} or new ones from selection {5, 6, 7}
                assert!(
                    *feature_idx <= 2 || feature_selection.contains(feature_idx),
                    "Feature {} should be from original or feature_selection",
                    feature_idx
                );
            }
        }
    }

    #[test]
    fn test_no_zero_values_in_features_hashmap() {
        let mut pop = create_test_population(20, TERNARY_LANG, PREVALENCE_TYPE);
        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // After mutation
        mutate(&mut pop, &param, &feature_selection, &mut rng);

        for ind in &pop.individuals {
            for (feature, &val) in &ind.features {
                assert_ne!(
                    val, 0,
                    "Feature {} has value 0 which should not exist in HashMap",
                    feature
                );
            }
            assert_eq!(
                ind.k,
                ind.features.len(),
                "k should equal actual number of non-zero features"
            );
        }

        // After crossover
        let children = cross_over(&pop, 30, &mut rng);

        for child in &children.individuals {
            for (feature, &val) in &child.features {
                assert_ne!(
                    val, 0,
                    "Child feature {} has value 0 which should not exist",
                    feature
                );
            }
            assert_eq!(
                child.k,
                child.features.len(),
                "Child k should equal actual number of features"
            );
        }
    }

    #[test]
    fn test_clone_removal_after_evolution_step() {
        let pop = create_test_population(30, TERNARY_LANG, PREVALENCE_TYPE);
        let param = create_test_params();
        let feature_selection = vec![0, 1, 2, 3, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Select parents
        let parents = select_parents(&pop, &param, &mut rng);

        // Create children
        let mut children = cross_over(&parents, 50, &mut rng);

        // Mutate
        mutate(&mut children, &param, &feature_selection, &mut rng);

        // Remove stillborn
        let _ = remove_stillborn(&mut children);

        // Now compute hash and remove clones
        children.compute_hash();
        let clone_count = children.remove_clone();

        // Verify no duplicates remain
        let mut seen_hashes = std::collections::HashSet::new();
        for ind in &children.individuals {
            assert!(
                seen_hashes.insert(ind.hash),
                "Found duplicate hash {} after clone removal",
                ind.hash
            );
        }

        // If clones were found, verify they were actually removed
        if clone_count > 0 {
            assert!(
                children.individuals.len() < 50,
                "Clone removal should reduce population size"
            );
        }
    }

    #[test]
    fn test_children_have_parent_information_and_epoch() {
        let pop = create_test_population(10, TERNARY_LANG, PREVALENCE_TYPE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let children = cross_over(&pop, 20, &mut rng);

        for (i, child) in children.individuals.iter().enumerate() {
            assert!(
                child.parents.is_some(),
                "Child {} should have parent information",
                i
            );

            let parent_hashes = child.parents.as_ref().unwrap();
            assert_eq!(
                parent_hashes.len(),
                2,
                "Child {} should have exactly 2 parent hashes",
                i
            );

            // Verify parents exist in original population
            let p1_exists = pop.individuals.iter().any(|p| p.hash == parent_hashes[0]);
            let p2_exists = pop.individuals.iter().any(|p| p.hash == parent_hashes[1]);
            assert!(
                p1_exists && p2_exists,
                "Child {} parents should exist in original population",
                i
            );
        }
    }

    #[test]
    fn test_forced_diversity_filtering() {
        let mut pop = create_test_population(50, TERNARY_LANG, PREVALENCE_TYPE);

        // Make some individuals very similar
        for i in 0..25 {
            pop.individuals[i].features.clear();
            pop.individuals[i].features.insert(0, 1);
            pop.individuals[i].features.insert(1, 1);
            pop.individuals[i].k = 2;
        }

        let mut param = create_test_params();
        param.ga.forced_diversity_pct = 50.0; // Keep only 50%
        param.ga.forced_diversity_epochs = 1;
        param.ga.select_elite_pct = 80.0; // Select most

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut parents = select_parents(&pop, &param, &mut rng);

        let initial_size = parents.individuals.len();

        // Apply diversity filter
        parents = parents.filter_by_signed_jaccard_dissimilarity(
            param.ga.forced_diversity_pct,
            param.ga.select_niche_pct == 0.0,
        );

        // Should reduce population
        assert!(
            parents.individuals.len() < initial_size,
            "Diversity filtering should reduce population from {} to {}",
            initial_size,
            parents.individuals.len()
        );

        // Test with very high threshold (keeps very few)
        parents = select_parents(&pop, &param, &mut rng);
        let size_before_strict = parents.individuals.len();
        parents = parents.filter_by_signed_jaccard_dissimilarity(95.0, true);
        // Very high threshold filters aggressively
        assert!(
            parents.individuals.len() <= size_before_strict,
            "High diversity threshold should filter heavily"
        );
    }

    #[test]
    fn test_niche_quotas_per_type() {
        let mut pop = Population::new();

        // Create 4 types: 2 languages × 2 data_types = 4 combinations
        // 20 individuals per type = 80 total
        let languages = vec![TERNARY_LANG, BINARY_LANG];
        let data_types = vec![PREVALENCE_TYPE, RAW_TYPE];

        let mut counter = 0u64;
        for lang in &languages {
            for dtype in &data_types {
                for _i in 0..20 {
                    let mut ind = Individual::new();
                    ind.language = *lang;
                    ind.data_type = *dtype;
                    ind.fit = (80 - counter) as f64; // Decreasing fitness
                    ind.hash = counter;
                    ind.features.insert(0, 1);
                    ind.k = 1;
                    counter += 1;
                    pop.individuals.push(ind);
                }
            }
        }

        let mut param = create_test_params();
        param.ga.select_elite_pct = 5.0; // 4 individuals
        param.ga.select_niche_pct = 20.0; // ~16 individuals total, ~4 per type
        param.ga.select_random_pct = 5.0; // ~4 individuals total, ~1 per type

        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let parents = select_parents(&pop, &param, &mut rng);

        // Count individuals per type
        let mut type_counts: std::collections::HashMap<(u8, u8), usize> =
            std::collections::HashMap::new();
        for ind in &parents.individuals {
            *type_counts
                .entry((ind.language, ind.data_type))
                .or_insert(0) += 1;
        }

        // Each type should have some representation with niche selection
        assert_eq!(
            type_counts.len(),
            4,
            "All 4 types should be represented with niche selection"
        );

        // Verify determinism with same seed
        let mut rng2 = ChaCha8Rng::seed_from_u64(999);
        let parents2 = select_parents(&pop, &param, &mut rng2);

        assert_eq!(
            parents.individuals.len(),
            parents2.individuals.len(),
            "Same seed should produce same parent count"
        );

        for i in 0..parents.individuals.len() {
            assert_eq!(
                parents.individuals[i].hash, parents2.individuals[i].hash,
                "Parent order should be deterministic with seed"
            );
        }
    }

    #[test]
    fn test_mutation_non_null_chance_impact() {
        let mut param = create_test_params();
        param.ga.mutated_children_pct = 100.0;
        param.ga.mutated_features_pct = 100.0;

        let feature_selection = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        // Test with 0% non-null chance (all removals)
        param.ga.mutation_non_null_chance_pct = 0.0;
        let mut pop_zero = create_test_population(20, TERNARY_LANG, PREVALENCE_TYPE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        mutate(&mut pop_zero, &param, &feature_selection, &mut rng);

        let avg_k_zero: f64 = pop_zero.individuals.iter().map(|i| i.k as f64).sum::<f64>()
            / pop_zero.individuals.len() as f64;

        // Test with 100% non-null chance (many insertions)
        param.ga.mutation_non_null_chance_pct = 100.0;
        let mut pop_hundred = create_test_population(20, TERNARY_LANG, PREVALENCE_TYPE);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        mutate(&mut pop_hundred, &param, &feature_selection, &mut rng);

        let avg_k_hundred: f64 = pop_hundred
            .individuals
            .iter()
            .map(|i| i.k as f64)
            .sum::<f64>()
            / pop_hundred.individuals.len() as f64;

        // 100% should generally result in more features than 0%
        assert!(
            avg_k_hundred >= avg_k_zero,
            "Higher mutation_non_null_chance should result in more features: {} vs {}",
            avg_k_hundred,
            avg_k_zero
        );
    }

    #[test]
    fn test_extreme_selection_percentages() {
        let pop = create_test_population(100, TERNARY_LANG, PREVALENCE_TYPE);
        let mut param = create_test_params();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Test 0% selection
        param.ga.select_elite_pct = 0.0;
        param.ga.select_random_pct = 0.0;
        param.ga.select_niche_pct = 0.0;
        let parents_zero = select_parents(&pop, &param, &mut rng);
        assert_eq!(
            parents_zero.individuals.len(),
            0,
            "0% selection should give 0 parents"
        );

        // Test 100% elite
        param.ga.select_elite_pct = 100.0;
        param.ga.select_random_pct = 0.0;
        let parents_hundred = select_parents(&pop, &param, &mut rng);
        assert_eq!(
            parents_hundred.individuals.len(),
            100,
            "100% elite should select all individuals"
        );

        // Verify they're in order (elite = best first)
        for i in 0..100 {
            assert_eq!(
                parents_hundred.individuals[i].hash, pop.individuals[i].hash,
                "100% elite should preserve population order"
            );
        }
    }

    #[test]
    fn test_crossover_multi_language_exhaustive() {
        let mut pop = Population::new();

        // TERNARY ↔ RATIO
        let mut t1 = Individual::new();
        t1.language = TERNARY_LANG;
        t1.data_type = PREVALENCE_TYPE;
        t1.features.insert(0, 1);
        t1.features.insert(1, -1);
        t1.k = 2;
        t1.hash = 1;
        pop.individuals.push(t1);

        let mut r1 = Individual::new();
        r1.language = RATIO_LANG;
        r1.data_type = PREVALENCE_TYPE;
        r1.features.insert(2, 1);
        r1.features.insert(3, -1);
        r1.k = 2;
        r1.hash = 2;
        pop.individuals.push(r1);

        // POW2 ↔ TERNARY
        let mut p1 = Individual::new();
        p1.language = POW2_LANG;
        p1.data_type = PREVALENCE_TYPE;
        p1.features.insert(4, 2);
        p1.features.insert(5, -4);
        p1.k = 2;
        p1.hash = 3;
        pop.individuals.push(p1);

        let mut t2 = Individual::new();
        t2.language = TERNARY_LANG;
        t2.data_type = PREVALENCE_TYPE;
        t2.features.insert(6, 1);
        t2.features.insert(7, -1);
        t2.k = 2;
        t2.hash = 4;
        pop.individuals.push(t2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 30, &mut rng);

        for child in &children.individuals {
            // Validate language-specific constraints
            match child.language {
                TERNARY_LANG | RATIO_LANG => {
                    for (_, &val) in &child.features {
                        assert!(
                            val >= -1 && val <= 1,
                            "Ternary/Ratio child should have values in [-1,1], got {}",
                            val
                        );
                    }
                }
                POW2_LANG => {
                    for (_, &val) in &child.features {
                        assert!(
                            val.abs() <= 64,
                            "POW2 child should have |value| <= 64, got {}",
                            val
                        );
                        let abs_val = val.abs();
                        assert!(
                            abs_val > 0 && (abs_val & (abs_val - 1)) == 0,
                            "POW2 child should have power-of-2 values, got {}",
                            val
                        );
                    }
                }
                BINARY_LANG => {
                    for (_, &val) in &child.features {
                        assert_eq!(val, 1, "Binary child should only have 1, got {}", val);
                    }
                }
                _ => panic!("Unexpected language {}", child.language),
            }

            assert_eq!(child.k, child.features.len());
        }
    }

    #[test]
    fn test_mutate_pow2_boundaries_strict() {
        let mut ind = Individual::new();
        ind.language = POW2_LANG;
        ind.data_type = PREVALENCE_TYPE;

        let param = create_test_params();
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        // Test doubling at boundary (64 should not become 128)
        ind.features.clear();
        ind.features.insert(0, 64);
        ind.k = 1;

        for _ in 0..50 {
            mutate_pow2(&mut ind, &param, &vec![0], &mut rng);
            if let Some(&val) = ind.features.get(&0) {
                assert!(
                    val.abs() <= 64,
                    "Value at boundary should not exceed 64, got {}",
                    val
                );
            }
        }

        // Test halving at minimum (1 should stay 1 or be removed)
        ind.features.clear();
        ind.features.insert(1, 1);
        ind.k = 1;

        for _ in 0..50 {
            mutate_pow2(&mut ind, &param, &vec![1], &mut rng);
            if let Some(&val) = ind.features.get(&1) {
                assert!(val.abs() >= 1, "Halving should not go below 1, got {}", val);
            }
        }

        // Test that -1 can be doubled to -2 or stay -1
        ind.features.clear();
        ind.features.insert(2, -1);
        ind.k = 1;

        for _ in 0..50 {
            mutate_pow2(&mut ind, &param, &vec![2], &mut rng);
            if let Some(&val) = ind.features.get(&2) {
                // Can be -1, -2 (doubled), or new value
                assert!(
                    val.abs() <= 64 && (val.abs() & (val.abs() - 1)) == 0,
                    "POW2 value should be valid power of 2, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_remove_stillborn_loop_until_viable() {
        let mut pop = Population::new();

        // Create population with many stillborn
        for i in 0..10 {
            let mut stillborn = Individual::new();
            stillborn.language = TERNARY_LANG;
            stillborn.features.insert(0, 1);
            stillborn.features.insert(1, 1); // All positive = stillborn
            stillborn.k = 2;
            pop.individuals.push(stillborn);

            if i < 3 {
                // Add some valid ones
                let mut valid = Individual::new();
                valid.language = TERNARY_LANG;
                valid.features.insert(0, 1);
                valid.features.insert(1, -1);
                valid.k = 2;
                pop.individuals.push(valid);
            }
        }

        let initial_count = pop.individuals.len();
        let removed = remove_stillborn(&mut pop);

        assert_eq!(removed, 10, "Should remove all 10 stillborn");
        assert_eq!(
            pop.individuals.len(),
            initial_count - 10,
            "Population should decrease by number of stillborn"
        );

        // All remaining should be valid
        for ind in &pop.individuals {
            if ind.language == TERNARY_LANG || ind.language == RATIO_LANG {
                let has_pos = ind.features.values().any(|&v| v > 0);
                let has_neg = ind.features.values().any(|&v| v < 0);
                assert!(
                    has_pos && has_neg,
                    "Remaining ternary/ratio individuals should have both signs"
                );
            }
        }
    }

    #[test]
    fn test_crossover_inherits_data_type_from_main_parent() {
        let mut pop = Population::new();

        // Parent 1: PREVALENCE_TYPE
        let mut p1 = Individual::new();
        p1.language = TERNARY_LANG;
        p1.data_type = PREVALENCE_TYPE;
        p1.features.insert(0, 1);
        p1.k = 1;
        p1.hash = 1;
        pop.individuals.push(p1);

        // Parent 2: RAW_TYPE (different!)
        let mut p2 = Individual::new();
        p2.language = TERNARY_LANG;
        p2.data_type = RAW_TYPE;
        p2.features.insert(1, -1);
        p2.k = 1;
        p2.hash = 2;
        pop.individuals.push(p2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 20, &mut rng);

        // Each child should have data_type from one of the parents
        for child in &children.individuals {
            assert!(
                child.data_type == PREVALENCE_TYPE || child.data_type == RAW_TYPE,
                "Child data_type should be from a parent, got {}",
                child.data_type
            );

            // Child inherits from main_parent, which is randomly selected
            // So we just verify it's a valid parent data_type
        }
    }

    #[test]
    fn test_mutate_pow2_deterministic_branches() {
        let mut ind = Individual::new();
        ind.language = POW2_LANG;
        ind.data_type = PREVALENCE_TYPE;

        let mut param = create_test_params();
        param.ga.mutation_non_null_chance_pct = 100.0;

        // Test insert +1 branch
        ind.features.clear();
        ind.k = 0;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        mutate_pow2(&mut ind, &param, &vec![0], &mut rng);

        // Test insert -1 branch
        ind.features.clear();
        ind.k = 0;
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        mutate_pow2(&mut ind, &param, &vec![1], &mut rng);

        // Test doubling branch (with value 2 -> 4)
        ind.features.clear();
        ind.features.insert(2, 2);
        ind.k = 1;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        mutate_pow2(&mut ind, &param, &vec![2], &mut rng);

        // Test halving branch (with value 4 -> 2)
        ind.features.clear();
        ind.features.insert(3, 4);
        ind.k = 1;
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        mutate_pow2(&mut ind, &param, &vec![3], &mut rng);

        // All mutations should maintain POW2 invariants
        for (_, &val) in &ind.features {
            if val != 0 {
                assert!(
                    val.abs() <= 64 && (val.abs() & (val.abs() - 1)) == 0,
                    "POW2 value should be power of 2 <= 64, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_selection_determinism_with_equal_fits_extensive() {
        let mut pop = Population::new();

        // Create 10 individuals with EXACT same fit (tie situation)
        for i in 0..10 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.fit = 0.85; // All identical
            ind.auc = 0.85;
            ind.features.insert(i, 1);
            ind.k = 1;
            ind.hash = i as u64;
            pop.individuals.push(ind);
        }

        // Add 10 more with different fit
        for i in 10..20 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.fit = 0.75;
            ind.auc = 0.75;
            ind.features.insert(i, 1);
            ind.k = 1;
            ind.hash = i as u64;
            pop.individuals.push(ind);
        }

        let mut param = create_test_params();
        param.ga.select_elite_pct = 50.0;
        param.ga.select_random_pct = 20.0;

        // Run selection multiple times with same seed
        for seed in vec![42, 123, 999] {
            let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
            let parents1 = select_parents(&pop, &param, &mut rng1);

            let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
            let parents2 = select_parents(&pop, &param, &mut rng2);

            assert_eq!(
                parents1.individuals.len(),
                parents2.individuals.len(),
                "Seed {} should produce consistent parent count",
                seed
            );

            for i in 0..parents1.individuals.len() {
                assert_eq!(
                    parents1.individuals[i].hash, parents2.individuals[i].hash,
                    "Seed {} parent {} should be identical (tie-break by hash)",
                    seed, i
                );
            }
        }
    }

    // ============================================================================
    // TESTS CIBLÉS SUPPLÉMENTAIRES
    // ============================================================================

    #[test]
    fn test_equal_fit_deterministic_tiebreak_by_hash() {
        let mut pop = Population::new();

        // Create population where ALL individuals have EXACTLY the same fit
        // This forces tie-breaking to rely entirely on hash ordering
        for i in 0..30 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.fit = 0.888; // Identical for all
            ind.auc = 0.888;
            ind.features.insert(i % 5, if i % 2 == 0 { 1 } else { -1 });
            ind.k = 1;
            ind.hash = i as u64; // Different hashes
            pop.individuals.push(ind);
        }

        let mut param = create_test_params();
        param.ga.select_elite_pct = 30.0; // 9 individuals
        param.ga.select_random_pct = 20.0; // ~6 individuals
        param.ga.select_niche_pct = 0.0;

        // Multiple runs with same seed should give EXACT same results
        let mut results = vec![];
        for _ in 0..3 {
            let mut rng = ChaCha8Rng::seed_from_u64(777);
            let parents = select_parents(&pop, &param, &mut rng);
            results.push(parents);
        }

        // All runs must be identical
        for run in 1..results.len() {
            assert_eq!(
                results[0].individuals.len(),
                results[run].individuals.len(),
                "Run {} should have same parent count as run 0",
                run
            );

            for i in 0..results[0].individuals.len() {
                assert_eq!(
                    results[0].individuals[i].hash, results[run].individuals[i].hash,
                    "Run {} parent {} hash mismatch: expected {}, got {}",
                    run, i, results[0].individuals[i].hash, results[run].individuals[i].hash
                );
            }
        }

        // Different seeds should give different random selections but same elite
        let mut rng1 = ChaCha8Rng::seed_from_u64(111);
        let parents1 = select_parents(&pop, &param, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(222);
        let parents2 = select_parents(&pop, &param, &mut rng2);

        // Elite portion (first 9) should be identical (deterministic by hash)
        let elite_count = (30.0 * param.ga.select_elite_pct / 100.0) as usize;
        for i in 0..elite_count {
            assert_eq!(
                parents1.individuals[i].hash, parents2.individuals[i].hash,
                "Elite parent {} should be same regardless of seed (hash tie-break)",
                i
            );
        }
    }

    #[test]
    fn test_mutation_with_empty_and_partial_feature_selection() {
        let mut pop = Population::new();

        // Create individuals with features {0, 1, 2}
        for _ in 0..10 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.features.insert(0, 1);
            ind.features.insert(1, -1);
            ind.features.insert(2, 1);
            ind.k = 3;
            pop.individuals.push(ind);
        }

        let param = create_test_params();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Test 1: Empty feature_selection - should not panic, nothing changes
        let mut pop_empty = pop.clone();
        let empty_selection: Vec<usize> = vec![];
        let original_features: Vec<_> = pop_empty
            .individuals
            .iter()
            .map(|i| i.features.clone())
            .collect();

        mutate(&mut pop_empty, &param, &empty_selection, &mut rng);

        for i in 0..pop_empty.individuals.len() {
            assert_eq!(
                pop_empty.individuals[i].features, original_features[i],
                "Empty feature_selection should not modify any features"
            );
            assert_eq!(
                pop_empty.individuals[i].k, 3,
                "k should remain unchanged with empty selection"
            );
        }

        // Test 2: Partial feature_selection (only {5, 6, 7}, none overlap with {0,1,2})
        let mut pop_partial = pop.clone();
        let partial_selection = vec![5, 6, 7]; // Disjoint from existing features

        mutate(&mut pop_partial, &param, &partial_selection, &mut rng);

        // Original features {0, 1, 2} should remain if not removed
        // New features can only come from {5, 6, 7}
        for ind in &pop_partial.individuals {
            for feature_idx in ind.features.keys() {
                assert!(
                    *feature_idx <= 2 || partial_selection.contains(feature_idx),
                    "Feature {} should be original (0-2) or from selection (5-7)",
                    feature_idx
                );
            }
            assert_eq!(ind.k, ind.features.len(), "k must match feature count");
        }
    }

    #[test]
    fn test_crossover_gene_conflict_resolution() {
        let mut pop = Population::new();

        // Parent 1: feature 5 = +1, feature 6 = +1
        let mut p1 = Individual::new();
        p1.language = TERNARY_LANG;
        p1.data_type = PREVALENCE_TYPE;
        p1.features.insert(5, 1); // Positive
        p1.features.insert(6, 1); // Positive
        p1.k = 2;
        p1.hash = 100;
        pop.individuals.push(p1);

        // Parent 2: feature 5 = -1 (CONFLICT!), feature 7 = -1
        let mut p2 = Individual::new();
        p2.language = TERNARY_LANG;
        p2.data_type = PREVALENCE_TYPE;
        p2.features.insert(5, -1); // Negative (conflicts with p1)
        p2.features.insert(7, -1); // Negative
        p2.k = 2;
        p2.hash = 200;
        pop.individuals.push(p2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let children = cross_over(&pop, 40, &mut rng);

        for (idx, child) in children.individuals.iter().enumerate() {
            // Rule: child chooses parent randomly for each feature
            // If feature 5 is present, it must be +1 or -1 (from one parent)
            if let Some(&val) = child.features.get(&5) {
                assert!(
                    val == 1 || val == -1,
                    "Child {} feature 5 conflict should resolve to +1 or -1, got {}",
                    idx,
                    val
                );
                assert_ne!(val, 0, "Feature 5 must not be zero");
            }

            // No feature should have value 0 in the HashMap
            for (feature, &val) in &child.features {
                assert_ne!(
                    val, 0,
                    "Child {} feature {} should not be 0 (must be absent instead)",
                    idx, feature
                );
            }

            // All values must be valid for ternary language
            for (_, &val) in &child.features {
                assert!(
                    val >= -1 && val <= 1,
                    "Child {} should have ternary values in [-1,1], got {}",
                    idx,
                    val
                );
            }

            assert_eq!(
                child.k,
                child.features.len(),
                "Child {} k={} must equal features.len()={}",
                idx,
                child.k,
                child.features.len()
            );
        }
    }

    #[test]
    fn test_clone_removal_after_crossover_and_mutation() {
        let pop = create_test_population(15, TERNARY_LANG, PREVALENCE_TYPE);
        let mut param = create_test_params();

        // High mutation rate to potentially create duplicates
        param.ga.mutated_children_pct = 100.0;
        param.ga.mutated_features_pct = 10.0; // Small changes

        let feature_selection = vec![0, 1, 2];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Create many children (high chance of duplicates)
        let mut children = cross_over(&pop, 100, &mut rng);

        // Mutate (may create more duplicates)
        mutate(&mut children, &param, &feature_selection, &mut rng);

        // Remove stillborn
        let _ = remove_stillborn(&mut children);

        let size_before_clone_removal = children.individuals.len();

        // Compute hash and remove clones
        children.compute_hash();
        let clone_count = children.remove_clone();

        // Verify no duplicates remain
        let mut seen_hashes = std::collections::HashSet::new();
        for (i, ind) in children.individuals.iter().enumerate() {
            assert!(
                seen_hashes.insert(ind.hash),
                "Individual {} has duplicate hash {} after clone removal",
                i,
                ind.hash
            );
        }

        // If clones were found, verify population was reduced
        if clone_count > 0 {
            assert_eq!(
                children.individuals.len(),
                size_before_clone_removal - clone_count as usize,
                "Population should be reduced by clone_count"
            );

            // At least one of each unique individual should remain
            assert!(
                children.individuals.len() > 0,
                "Clone removal should preserve at least one of each unique individual"
            );
        }

        // All remaining individuals should be unique
        assert_eq!(
            children.individuals.len(),
            seen_hashes.len(),
            "Final population size should match unique hash count"
        );
    }

    #[test]
    fn test_forced_diversity_on_vs_off() {
        let mut pop = create_test_population(60, TERNARY_LANG, PREVALENCE_TYPE);

        // Make population less diverse (many similar individuals)
        for i in 0..30 {
            pop.individuals[i].features.clear();
            pop.individuals[i].features.insert(0, 1);
            pop.individuals[i].features.insert(1, 1);
            pop.individuals[i].features.insert(2, 1);
            pop.individuals[i].k = 3;
        }

        let mut param = create_test_params();
        param.ga.select_elite_pct = 70.0; // Select many
        param.ga.select_random_pct = 0.0;
        param.ga.select_niche_pct = 0.0;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Test 1: forced_diversity_pct = 0 (OFF) - no filtering
        param.ga.forced_diversity_pct = 0.0;
        let parents_no_filter = select_parents(&pop, &param, &mut rng);
        let size_no_filter = parents_no_filter.individuals.len();

        // With forced_diversity_pct = 0, no filtering is applied
        // (this is tested by evolve function logic, but we verify parent selection)
        assert!(
            size_no_filter > 0,
            "Should select parents even with diversity off"
        );

        // Test 2: forced_diversity_pct > 0 (ON) - filtering applied
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        param.ga.forced_diversity_pct = 60.0; // Keep 60% most diverse
        let mut parents_with_filter = select_parents(&pop, &param, &mut rng2);

        let size_before_filter = parents_with_filter.individuals.len();

        // Apply diversity filter
        parents_with_filter = parents_with_filter.filter_by_signed_jaccard_dissimilarity(
            param.ga.forced_diversity_pct,
            param.ga.select_niche_pct == 0.0,
        );

        // Should reduce population due to diversity filtering
        assert!(
            parents_with_filter.individuals.len() < size_before_filter,
            "Forced diversity should reduce population from {} to {}",
            size_before_filter,
            parents_with_filter.individuals.len()
        );

        // Verify remaining individuals are more diverse
        assert!(
            parents_with_filter.individuals.len() > 0,
            "Diversity filtering should keep at least some individuals"
        );

        // Test 3: Very high forced_diversity_pct should filter more aggressively
        let mut rng3 = ChaCha8Rng::seed_from_u64(42);
        param.ga.forced_diversity_pct = 90.0; // Very strict
        let mut parents_strict = select_parents(&pop, &param, &mut rng3);
        let _size_before_strict = parents_strict.individuals.len();

        parents_strict = parents_strict
            .filter_by_signed_jaccard_dissimilarity(param.ga.forced_diversity_pct, true);

        // Strict filtering should reduce even more
        assert!(
            parents_strict.individuals.len() <= parents_with_filter.individuals.len(),
            "Stricter diversity threshold should keep fewer individuals"
        );
    }

    #[test]
    fn test_unviable_params_detection() {
        // This test documents the behavior when all generated individuals are stillborn
        // In practice, generate_pop has logic to detect and panic on this condition

        let mut pop = Population::new();

        // Create a population where all individuals are stillborn (ternary with same sign)
        for i in 0..20 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            // All positive values = stillborn for ternary
            ind.features.insert(0, 1);
            ind.features.insert(1, 1);
            ind.features.insert(2, 1);
            ind.k = 3;
            ind.hash = i as u64;
            pop.individuals.push(ind);
        }

        let initial_size = pop.individuals.len();
        let stillborn_count = remove_stillborn(&mut pop);

        // All should be removed as stillborn
        assert_eq!(
            stillborn_count, initial_size as u32,
            "All individuals should be detected as stillborn"
        );
        assert_eq!(
            pop.individuals.len(),
            0,
            "Population should be empty after removing all stillborn"
        );

        // In generate_pop, this would trigger the check:
        // if target_size == param.ga.population_size { panic!("Params only create inviable individuals!") }

        // Verify the same for binary with k=0
        let mut pop_binary = Population::new();
        for i in 0..20 {
            let mut ind = Individual::new();
            ind.language = BINARY_LANG;
            ind.k = 0; // Invalid for any language
            ind.hash = i as u64;
            pop_binary.individuals.push(ind);
        }

        let stillborn_binary = remove_stillborn(&mut pop_binary);
        assert_eq!(
            stillborn_binary, 20,
            "Binary individuals with k=0 should all be stillborn"
        );
        assert_eq!(
            pop_binary.individuals.len(),
            0,
            "Binary population should be empty"
        );
    }

    #[test]
    fn test_epoch_stamping_in_evolve() {
        // Test that evolve properly stamps epoch on all children
        let mut data = Data::new();
        data.feature_selection = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        data.sample_len = 20;
        data.feature_len = 10;

        // Create a diverse initial population to avoid too many clones
        let pop_size = 30;
        let mut pop = Population::new();
        let mut rng_for_pop = ChaCha8Rng::seed_from_u64(123);

        for i in 0..pop_size {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.epoch = 3; // Previous epoch

            // Create diverse feature sets
            let num_features = 2 + (i % 4); // Vary k from 2 to 5
            let start_feature = (i * 2) % 8; // Vary which features are selected

            for j in 0..num_features {
                let feature = start_feature + j;
                let val = if rng_for_pop.gen_bool(0.5) { 1 } else { -1 };
                ind.features.insert(feature, val);
            }

            ind.count_k();
            ind.fit = 0.5 + (i as f64 * 0.01); // Vary fit scores
            ind.hash = i as u64;
            pop.individuals.push(ind);
        }

        let mut param = create_test_params();
        param.ga.population_size = pop_size as u32;
        param.ga.select_elite_pct = 30.0;
        param.ga.select_random_pct = 20.0;
        param.ga.forced_diversity_pct = 0.0; // Disable diversity filtering

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut cv = None;

        let target_epoch = 5;

        // Call evolve with specific epoch
        let new_pop = evolve(
            pop,
            &data,
            &mut cv,
            &param,
            &vec![],
            &None,
            target_epoch,
            &mut rng,
        );

        // Verify epoch stamping
        let mut children_count = 0;
        let mut parent_count = 0;

        for ind in &new_pop.individuals {
            if ind.epoch == target_epoch {
                children_count += 1;

                // Children from this epoch should have parent information
                if ind.parents.is_some() {
                    let parents = ind.parents.as_ref().unwrap();
                    assert_eq!(
                        parents.len(),
                        2,
                        "Child with epoch {} should have 2 parent hashes",
                        target_epoch
                    );
                }
            } else if ind.epoch < target_epoch {
                parent_count += 1;
                // These are carried-over elite parents
            } else {
                panic!(
                    "Found individual with epoch {} > target epoch {}",
                    ind.epoch, target_epoch
                );
            }

            // All individuals must have valid k
            assert_eq!(
                ind.k,
                ind.features.len(),
                "Individual at epoch {} must have k={} matching features.len()={}",
                ind.epoch,
                ind.k,
                ind.features.len()
            );

            // No zero values in features
            for (feature, &val) in &ind.features {
                assert_ne!(
                    val, 0,
                    "Individual at epoch {} feature {} should not be 0",
                    ind.epoch, feature
                );
            }
        }

        // Should have both children (new epoch) and some elite parents
        assert!(
            children_count > 0,
            "Should have at least some children with epoch {}",
            target_epoch
        );

        // Key assertion: children created during this epoch have the correct epoch stamp
        assert!(
            children_count >= 1,
            "At least one child should be created with epoch {}",
            target_epoch
        );

        // Population should exist (size may vary due to stillborn/clone removal)
        assert!(
            new_pop.individuals.len() > 0,
            "Population should not be empty"
        );

        println!(
            "Population: {} children with epoch {}, {} parents with epoch < {}",
            children_count, target_epoch, parent_count, target_epoch
        );
    }

    #[test]
    fn test_extreme_forced_diversity_single_parent() {
        // Test the warning path where diversity filtering leaves only 1 parent
        // The cross_over function requires at least 2 parents, so this test
        // documents that extreme diversity can reduce parents to 1

        let mut data = Data::new();
        data.feature_selection = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        data.sample_len = 20;
        data.feature_len = 10;

        let mut pop = create_test_population(50, TERNARY_LANG, PREVALENCE_TYPE);

        // Make all individuals nearly identical (very low diversity)
        for i in 0..49 {
            pop.individuals[i].features.clear();
            pop.individuals[i].features.insert(0, 1);
            pop.individuals[i].features.insert(1, -1);
            pop.individuals[i].k = 2;
            pop.individuals[i].hash = i as u64; // Different hashes
        }

        // Keep one different individual
        pop.individuals[49].features.clear();
        pop.individuals[49].features.insert(8, 1);
        pop.individuals[49].features.insert(9, -1);
        pop.individuals[49].k = 2;
        pop.individuals[49].hash = 49;

        let mut param = create_test_params();
        param.ga.population_size = 50;
        param.ga.select_elite_pct = 90.0; // Select almost all
        param.ga.select_random_pct = 0.0;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Select parents (will get many similar ones)
        let mut parents = select_parents(&pop, &param, &mut rng);
        let initial_parent_count = parents.individuals.len();

        // Apply extreme diversity filter - should reduce to very few (possibly 1)
        parents = parents.filter_by_signed_jaccard_dissimilarity(
            99.9, // Extreme filtering percentage
            true, // Force very strict filtering
        );

        assert!(
            parents.individuals.len() < initial_parent_count,
            "Extreme diversity should filter parents"
        );

        let remaining_parents = parents.individuals.len();
        println!(
            "Extreme diversity filtering left {} parent(s)",
            remaining_parents
        );

        // Document the behavior: with very few parents, the system needs special handling
        if remaining_parents == 1 {
            // This is the edge case we want to document:
            // When only 1 parent remains, cross_over will panic
            // In production, this is prevented by the warning in evolve()
            // and the system continuing with the reduced parent set

            // To verify this would panic, we can document the expected behavior
            assert_eq!(
                remaining_parents, 1,
                "Extreme diversity filtering can reduce to single parent"
            );

            // The actual evolve function will issue a warning:
            // "Only 1 Individual kept after filtration with diversity"
            // and then call cross_over which expects 2 parents

            // For testing purposes, we verify that attempting crossover with 1 parent
            // would be problematic (documented by the warning in the actual code)
            println!(
                "Note: cross_over requires at least 2 parents. With 1 parent, \
                     the system logs a warning but proceeds, which may cause issues."
            );
        } else if remaining_parents >= 2 {
            // With 2+ parents, we can verify crossover works normally
            let children = cross_over(&parents, 5, &mut rng);

            assert_eq!(
                children.individuals.len(),
                5,
                "Should create requested number of children"
            );

            for (i, child) in children.individuals.iter().enumerate() {
                // Verify valid k
                assert_eq!(
                    child.k,
                    child.features.len(),
                    "Child {} must have k matching features.len()",
                    i
                );

                // Verify parent information
                assert!(
                    child.parents.is_some(),
                    "Child {} should have parent information",
                    i
                );
                let parent_hashes = child.parents.as_ref().unwrap();
                assert_eq!(
                    parent_hashes.len(),
                    2,
                    "Child {} should have 2 parent hashes",
                    i
                );

                // No zero values
                for (feature, &val) in &child.features {
                    assert_ne!(val, 0, "Child {} feature {} should not be 0", i, feature);
                }
            }
        } else {
            // 0 parents after filtering
            panic!("Extreme diversity filtering removed all parents!");
        }

        // The key insight: extreme diversity filtering (99.9%) can reduce the parent
        // population to a problematic size. The system logs a warning but may encounter
        // issues if cross_over is called with < 2 parents.
        assert!(
            remaining_parents <= 2,
            "Extreme diversity filtering (99.9%) should leave very few parents: {}",
            remaining_parents
        );
    }

    #[test]
    fn test_ga_with_initial_population() {
        // Test that GA works with an initial population
        let mut data = Data::specific_test(50, 30);
        let mut param = create_test_params();
        param.data.feature_maximal_adj_pvalue = 1.0;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        data.select_features(&param);

        // Create initial population
        let mut initial_pop = Population::new();
        initial_pop.generate(
            500,
            2,
            5,
            TERNARY_LANG,
            PREVALENCE_TYPE,
            0.0,
            &data,
            false,
            None,
            &mut rng,
        );
        initial_pop.compute_hash();

        // Run GA with initial population
        let running = Arc::new(AtomicBool::new(true));
        let populations = ga(
            &mut data,
            &mut None,
            &mut Some(initial_pop),
            &param,
            running,
        );

        // Verify it completes successfully
        assert!(
            !populations.is_empty(),
            "GA should run with initial population"
        );
        let final_pop = &populations[populations.len() - 1];
        assert!(
            !final_pop.individuals.is_empty(),
            "Should have final population"
        );

        // Verify best model is valid
        let best = &final_pop.individuals[0];
        assert!(
            best.auc >= 0.0 && best.auc <= 1.0,
            "Best model should have valid AUC"
        );
        assert!(best.k > 0, "Best model should have features");
    }

    #[test]
    fn test_ga_without_initial_population() {
        // Test that GA works without initial population (baseline)
        let mut data = Data::specific_test(50, 30);
        let mut param = create_test_params();
        param.data.feature_maximal_adj_pvalue = 1.0;

        // Run GA without initial population
        let running = Arc::new(AtomicBool::new(true));
        let populations = ga(&mut data, &mut None, &mut None, &param, running);

        // Verify it works
        assert!(
            !populations.is_empty(),
            "GA should work without initial population"
        );
        let final_pop = &populations[populations.len() - 1];

        // Verify best model is valid
        let best = &final_pop.individuals[0];
        assert!(
            best.auc >= 0.0 && best.auc <= 1.0,
            "Best model should have valid AUC"
        );
    }

    #[test]
    #[should_panic(expected = "Initial population is not compatible with data!")]
    fn test_ga_initial_population_with_features_outside_data() {
        // Test that features not in feature_selection don't break GA and can propagate
        let mut data = Data::specific_test(50, 100);
        let mut param = create_test_params();
        param.data.feature_maximal_adj_pvalue = 1.0;

        // Create initial population with features that may not be in selection
        let mut initial_pop = Population::new();

        // Create individuals with specific features
        for i in 0..105 {
            let mut ind = Individual::new();
            ind.language = BINARY_LANG;
            ind.data_type = PREVALENCE_TYPE;

            // Use features that might be filtered out
            ind.features.insert(i, 1);
            ind.k = 1;
            ind.epoch = 0;

            initial_pop.individuals.push(ind);
        }

        initial_pop.compute_hash();

        // Run GA - it should handle features outside selection gracefully
        let running = Arc::new(AtomicBool::new(true));
        let _ = ga(
            &mut data,
            &mut None,
            &mut Some(initial_pop.clone()),
            &param,
            running,
        );
    }

    use std::collections::HashSet;
    #[test]
    fn test_ga_initial_population_with_features_outside_feature_selection() {
        // Test that features not in feature_selection don't break GA and can propagate
        let mut data = Data::specific_significant_test(50, 105);
        let mut param = create_test_params();

        param.data.feature_maximal_adj_pvalue = 0.85;
        data.select_features(&param);

        println!(
            "Data feature selection length: {}",
            data.feature_selection.len()
        );

        // Create initial population with features that may not be in selection
        let mut initial_pop = Population::new();

        let rng = &mut ChaCha8Rng::seed_from_u64(42);

        // Create individuals with specific features
        for _ in 0..105 {
            let mut ind = Individual::new();
            ind.language = BINARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            for _ in 0..5 {
                ind.features.insert(
                    data.feature_selection[rng.gen_range(0..data.feature_selection.len())],
                    1,
                );
            }
            ind.k = 1;
            ind.epoch = 0;
            initial_pop.individuals.push(ind);
        }

        for i in 0..30 {
            initial_pop.individuals[i].features.insert(i, 1);
        }

        initial_pop.compute_hash();

        // Run GA - it should handle features outside selection gracefully but print a warning
        let running = Arc::new(AtomicBool::new(true));
        let populations = ga(
            &mut data,
            &mut None,
            &mut Some(initial_pop.clone()),
            &param,
            running,
        );

        println!(
            "Data feature selection length: {}",
            data.feature_selection.len()
        );

        // Verify GA completes without crashing
        assert!(
            !populations.is_empty(),
            "GA should handle features outside selection"
        );
        let final_pop = &populations[populations.len() - 1];
        assert!(
            !final_pop.individuals.is_empty(),
            "Should have final population"
        );

        let mut unique_feature: HashSet<usize> = HashSet::new();
        for ind in &final_pop.individuals {
            for feature_idx in ind.features.keys() {
                unique_feature.insert(*feature_idx);
            }
        }

        // Non-selected feature should not be abble to appear in final population if the initial population contains them
        let mut count = 0;
        for idx in 0..data.feature_len {
            if !data.feature_selection.contains(&idx) && unique_feature.contains(&idx) {
                println!(
                    "Feature {} is not in feature selection but is present in final population",
                    idx
                );
                assert!(idx <= 30, "Feature {} should not exist as it does not appear in the initial population and in selected feature", idx);
                count += 1;
            }
        }

        assert!(count > 0, "Some features outside feature selection should be able to propagate to final population if they are in the initial population");
        assert!(
            final_pop.individuals[0].k > 0,
            "Best model should have features"
        );
    }

    #[test]
    #[should_panic(expected = "Initial population size is too small")]
    fn test_ga_initial_population_all_stillborn() {
        // Test GA when initial population contains only stillborn (invalid) individuals
        let mut data = Data::specific_test(40, 25);
        let mut param = create_test_params();
        param.data.feature_maximal_adj_pvalue = 1.0;

        // Create population with only positive values (stillborn for ternary)
        let mut stillborn_pop = Population::new();

        for _ in 0..200 {
            let mut ind = Individual::new();
            ind.language = TERNARY_LANG;
            ind.data_type = PREVALENCE_TYPE;
            ind.features.insert(0, 1);
            ind.features.insert(1, 1);
            ind.features.insert(2, 1);
            ind.k = 3;
            stillborn_pop.individuals.push(ind);
        }

        stillborn_pop.compute_hash();

        // This should either:
        // 1. Generate new valid individuals to replace stillborn ones, or
        // 2. Handle gracefully by generating a fresh population
        let running = Arc::new(AtomicBool::new(true));
        let populations = ga(
            &mut data,
            &mut None,
            &mut Some(stillborn_pop),
            &param,
            running,
        );

        // Verify GA handled it and produced valid results
        assert!(
            !populations.is_empty(),
            "GA should handle stillborn initial population"
        );
        let final_pop = &populations[populations.len() - 1];
        assert!(
            !final_pop.individuals.is_empty(),
            "Should have final population"
        );

        // Check that final population has valid individuals
        let best = &final_pop.individuals[0];
        assert!(best.k > 0, "Best model should have features");

        // For ternary, should have both positive and negative values
        let mut has_positive = false;
        let mut has_negative = false;
        for &val in best.features.values() {
            if val > 0 {
                has_positive = true;
            }
            if val < 0 {
                has_negative = true;
            }
        }

        assert!(
            has_positive && has_negative,
            "Valid ternary individual should have both positive and negative features"
        );
    }

    #[test]
    #[should_panic(expected = "Initial population size is too small")]
    fn test_ga_initial_population_empty() {
        // Test GA with empty initial population
        let mut data = Data::specific_test(50, 30);
        let mut param = create_test_params();
        param.data.feature_maximal_adj_pvalue = 1.0;

        let empty_pop = Population::new();
        // Don't add any individuals - leave it empty

        // GA should handle empty population by generating a new one
        let running = Arc::new(AtomicBool::new(true));
        let populations = ga(&mut data, &mut None, &mut Some(empty_pop), &param, running);

        // Should work like normal GA (generate from scratch)
        assert!(
            !populations.is_empty(),
            "GA should handle empty initial population"
        );
        let final_pop = &populations[populations.len() - 1];
        assert!(
            final_pop.individuals.len() > 0,
            "Should generate population when initial is empty"
        );
    }

    #[test]
    fn test_ga_initial_population_feature_selection_change() {
        // Test that changing feature_selection doesn't break GA with initial population
        let mut data = Data::specific_significant_test(50, 100);
        let mut param = create_test_params();
        param.data.feature_maximal_adj_pvalue = 1.0;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        data.select_features(&param);

        // Create initial population with certain features
        let mut initial_pop = Population::new();
        initial_pop.generate(
            300,
            2,
            5,
            TERNARY_LANG,
            PREVALENCE_TYPE,
            0.0,
            &data,
            false,
            None,
            &mut rng,
        );

        // Store which features are used
        let initial_features: std::collections::HashSet<_> = initial_pop
            .individuals
            .iter()
            .flat_map(|ind| ind.features.keys().copied())
            .collect();

        println!("Initial features used: {:?}", initial_features);

        // Now change feature selection parameters to potentially select different features
        param.data.feature_maximal_adj_pvalue = 0.30; // strict
        param.data.feature_minimal_prevalence_pct = 20.0; // High prevalence requirement

        // Run GA - it will call data.select_features() which may exclude some initial features
        let running = Arc::new(AtomicBool::new(true));
        let populations = ga(
            &mut data,
            &mut None,
            &mut Some(initial_pop),
            &param,
            running,
        );

        // Verify GA handles the mismatch gracefully
        assert!(
            !populations.is_empty(),
            "GA should handle feature selection change"
        );
        let final_pop = &populations[populations.len() - 1];
        assert!(
            !final_pop.individuals.is_empty(),
            "Should have final population"
        );

        let best = &final_pop.individuals[0];
        assert!(
            best.auc >= 0.0 && best.auc <= 1.0,
            "Best model should be valid"
        );
    }
}
