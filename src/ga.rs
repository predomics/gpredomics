use crate::population::Population;
use crate::data::Data;
use crate::individual::{self, RATIO_LANG, TERNARY_LANG};
use crate::individual::Individual;
use crate::param::{Param};
use crate::gpu::GpuAssay;
use rand::prelude::SliceRandom;
use rand::Rng;
use rand::seq::index::sample;
use rand::prelude::*;
use rand_chacha::{ChaCha8Rng};
use log::{debug,info,warn,error};
use std::cmp::min;
use std::collections::HashMap;
use std::mem;
use std::time::Instant;
use crate::cv::CV;
use crate::utils::{display_epoch,display_epoch_legend};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

//-----------------------------------------------------------------------------
// Genetic Algorithm core functions
//-----------------------------------------------------------------------------

pub fn ga(data: &mut Data, _test_data: &mut Option<Data>, param: &Param, running: Arc<AtomicBool>) -> Vec<Population> {   
    let time = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    data.select_features(param);
    debug!("FEATURES {:?}",data.feature_class);

    let gpu_assay = get_gpu_assay(data, param);

    // Initialize first population
    let base_pop = generate_pop(data, param, &mut rng);
    info!("Population size: {}, kmin {}, kmax {}",base_pop.individuals.len(),base_pop.individuals.iter().map(|i| {i.k}).min().unwrap_or(0),base_pop.individuals.iter().map(|i| {i.k}).max().unwrap_or(0));

    display_epoch_legend(param);
    let populations = iterative_evolution(&base_pop, data, &gpu_assay, param, running, &mut rng);

    let elapsed = time.elapsed();
    info!("Genetic algorithm computed {:?} generations in {:.2?}", populations.len(), elapsed);

    populations
    
}

pub fn generate_pop(data: &Data, param : &Param, rng: &mut ChaCha8Rng) -> Population {
    let mut pop = Population::new();
    let languages: Vec<u8> = param.general.language.split(",").map(individual::language).collect();
    let data_types: Vec<u8> = param.general.data_type.split(",").map(individual::data_type).collect();
    let sub_population_size = param.ga.population_size / (languages.len() * data_types.len()) as u32;
    for data_type in &data_types {
        for language in &languages {
            let mut target_size = sub_population_size;
            while target_size>0 {
                let mut sub_pop = Population::new();
                debug!("generating...");
            
                sub_pop.generate(target_size, param.ga.kmin, 
                    if param.ga.kmax>0 { min(data.feature_selection.len(), param.ga.kmax) } else { data.feature_selection.len() }, 
                    *language,
                    *data_type,
                    param.general.data_type_epsilon,
                    data, param.experimental.threshold_ci, rng);
                debug!("generated for {} {}...",sub_pop.individuals[0].get_language(),sub_pop.individuals[0].get_data_type());
            
                target_size = remove_stillborn(&mut sub_pop);
                if target_size>0 { debug!("Some still born are present {} (with healthy {})",target_size,sub_pop.individuals.len());
                    if target_size==param.ga.population_size { 
                        error!("Params only create inviable individuals!");
                        panic!("Params only create inviable individuals!") } 
                }
        
                pop.add(sub_pop);
            }
        }
    }
    pop.compute_hash();
    let clone_number = pop.remove_clone();
    if clone_number>0 { debug!("Some clones were removed : {}.", clone_number) } ;
    pop
}

pub fn iterative_evolution(base_pop: &Population, data: &mut Data, gpu_assay: &Option<GpuAssay>, param: &Param, running: Arc<AtomicBool>, rng: &mut ChaCha8Rng) -> Vec<Population> {
    let mut epoch: usize= 0;
    let mut cv: Option<CV> = None;
    let mut populations: Vec<Population> = vec![];

    let mut data_rng = rng.clone();
    let mut evolution_rng = rng.clone();

    // Prepare epoch associated data
    let mut epoch_data = if param.ga.random_sampling_pct > 0.0 {
        let random_samples = (data.sample_len as f64 * (param.ga.random_sampling_pct/100.0)) as usize;
        data.subset(data.random_subset(random_samples, &mut data_rng))
    } else if param.cv.overfit_penalty != 0.0 {
        let folds_nb = if param.cv.inner_folds > 1 { param.cv.inner_folds } else { 10 };
        info!("Learning on {:?}-folds.", folds_nb);
        cv = Some(CV::new(&data, folds_nb, &mut data_rng));
        Data::new()
    } else {
        data.clone()
    };

    // Clean data before process
    let mut pop = base_pop.clone() ;
    pop.compute_hash();
    let _clone_number = pop.remove_clone();

    // Fitting base population on data
    if let Some(ref cv) = cv {
        debug!("Fitting population on folds...");
        pop.fit_on_folds(cv, &param, &vec![None; param.cv.inner_folds]);
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
            let random_samples = (data.sample_len as f64 * (param.ga.random_sampling_pct/100.0)) as usize;
            debug!("Re-sampling {} samples...", random_samples);
            epoch_data = data.subset(data.random_subset(random_samples, &mut data_rng));

            pop.fit(&epoch_data, &mut None, &gpu_assay, &None, param);
            if param.general.keep_trace { pop.compute_all_metrics(&epoch_data, &param.general.fit); }
            pop = pop.sort();
        } else if param.cv.overfit_penalty > 0.0 && param.cv.resampling_inner_folds_epochs > 0 && epoch % param.cv.resampling_inner_folds_epochs == 0 {
            debug!("Re-sampling folds...");
            let folds_nb = if param.cv.inner_folds > 1 { param.cv.inner_folds } else { 3 };
            let gpu_assays_per_fold: Vec<Option<GpuAssay>> = vec![None; folds_nb];
            cv  = Some(CV::new(&data, folds_nb, &mut data_rng));

            if let Some(ref cv) = cv {
                pop.fit_on_folds(cv, &param, &gpu_assays_per_fold); 
                pop = pop.sort();
            }
        } 

        // Evolution and ranking
        pop = evolve(pop, &epoch_data, &mut cv, param, gpu_assay, epoch, &mut evolution_rng);

        display_epoch(&pop, param, epoch);

        // Stop critera
        let mut need_to_break= false;

        let best_model = &pop.individuals[0];
        if epoch>=param.ga.min_epochs {
            if epoch-best_model.epoch+1>param.ga.max_age_best_model {
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

        if param.general.keep_trace { populations.push(pop.clone()) }

        if need_to_break {
            if populations.len() == 0 { populations = vec![pop]; } 

            if param.ga.random_sampling_pct > 0.0 {
                if let Some(last_population) = populations.last_mut() {
                    warn!("Random sampling: models optimized on samples ({} samples), metrics shown on full dataset. \n\
                    NOTE: Fit values reflect sample-based optimization, not full dataset performance.", (param.ga.random_sampling_pct * 100.0) as u8);
                    last_population.compute_all_metrics(&data, &param.general.fit);
                    //(&mut *last_population).fit(&data, &mut None, &gpu_assay, &None, param);
                }
            }       

            break
        }
    }

    populations
    
    }

// Evolve the population one time: filter, cross-over and mutate
#[inline]
pub fn evolve(pop: Population, data: &Data, cv: &mut Option<CV>, param: &Param, gpu_assay: &Option<GpuAssay>, epoch: usize, rng: &mut ChaCha8Rng) -> Population {
    let mut new_pop = Population::new();

    new_pop.add(select_parents(&pop, param, rng));

    // Filter before cross-over to improve diversity 
    if param.ga.forced_diversity_pct != 0.0 && epoch % param.ga.forced_diversity_epochs == 0 {
        let n = new_pop.individuals.len();
        new_pop = new_pop.filter_by_signed_jaccard_dissimilarity(param.ga.forced_diversity_pct, param.ga.select_niche_pct == 0.0);
        if new_pop.individuals.len() > 1 {
            debug!("Parents filtered for diversity: {}/{} individuals retained", new_pop.individuals.len(), n);
        } else {
            warn!("Only 1 Individual kept after filtration with diversity");
        }
    }

    // Generate children
    let mut children_to_create = param.ga.population_size as usize-new_pop.individuals.len();

    let mut children=Population::new();
    
    while children_to_create > 0 {
        let mut some_children = cross_over(&new_pop, children_to_create, rng);
        mutate(&mut some_children, param, &data.feature_selection, rng);
        children_to_create = remove_stillborn(&mut some_children) as usize;
        if children_to_create > 0 { debug!("Some stillborn are presents: {}", children_to_create) }

        children.add(some_children);
    }

    for i in children.individuals.iter_mut() {
        i.epoch = epoch;
    }

    // Fit children and clean population
    if let Some(ref cv) = cv {
        debug!("Fitting children on folds...");
        children.fit_on_folds(cv, &param,  &vec![None; param.cv.inner_folds]);
    }  else {
        debug!("Fitting children...");
        children.fit(&data, &mut None, gpu_assay, &None, param);    
    }

    new_pop.add(children);

    new_pop.compute_hash();
    let clone_number = new_pop.remove_clone();
    if clone_number>0 { debug!("Some clones were removed : {}.", clone_number) } ;
    new_pop = new_pop.sort(); 

    new_pop
}

/// pick params.ga_select_elite_pct% of the best individuals and params.ga_select_random_pct%  
fn select_parents(pop: &Population, param: &Param, rng: &mut ChaCha8Rng) -> Population {
    
    // order pop by fit and select params.ga_select_elite_pct
    let (mut parents, n) = pop.select_first_pct(param.ga.select_elite_pct);

    let mut individual_by_types : HashMap<(u8,u8),Vec<&Individual>> = HashMap::new();
    for individual in pop.individuals[n..].iter() {
        let i_type = (individual.language, individual.data_type);
        if !individual_by_types.contains_key(&i_type) {
            individual_by_types.insert(i_type, vec![individual]);
        } else {
            individual_by_types.get_mut(&i_type).unwrap().push(individual);
        }
    }

    // adding best models of each language / data type
    if param.ga.select_niche_pct > 0.0 {
        let types = individual_by_types.keys().cloned().collect::<Vec<(u8,u8)>>();
        let target = (pop.individuals.len() as f64 * param.ga.select_niche_pct / 100.0 / types.len() as f64) as usize;
        let mut type_count : HashMap<(u8,u8), usize> = types.iter().map(|x| { (*x,0) }).collect();
        for i in &pop.individuals[n..] {
            let i_type = (i.language,i.data_type);
            let current_count = *type_count.get(&i_type).unwrap_or(&target);
            if current_count<target {
                type_count.insert(i_type, current_count+1);
                parents.individuals.push(i.clone())
            }
        }
    }


    // add a random part of the others
    // parents.add(pop.select_random_above_n(param.ga.select_random_pct, n, rng));
    let n2 = (pop.individuals.len() as f64 * param.ga.select_random_pct / 100.0 / individual_by_types.keys().len() as f64) as usize;

    let mut sorted_keys: Vec<_> = individual_by_types.keys().collect();
    sorted_keys.sort(); 

    for i_type in sorted_keys {
        debug!("Adding {}:{} Individuals {} ", individual_by_types[i_type][0].get_language(), individual_by_types[i_type][0].get_data_type(), n2);
        parents.individuals.extend(individual_by_types[i_type].choose_multiple(rng, n2).map(|i| (*i).clone()));
    }
    parents

}

/// create children from parents
pub fn cross_over(parents: &Population, children_number: usize, rng: &mut ChaCha8Rng) -> Population {
    let mut children = Population::new();
    
    for _i in 0..children_number {
        let [p1, p2] = parents.individuals.choose_multiple(rng, 2)
            .collect::<Vec<_>>()
            .try_into()
            .expect("Vec must have exactly 2 elements");
        
        let main_parent = *[p1, p2].choose(rng).unwrap();
        let mut child = Individual::child(main_parent);
        
        let mut all_features: Vec<usize> = p1.features.keys()
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

/// change a sign, remove a variable, add a new variable
pub fn mutate(children: &mut Population, param: &Param, feature_selection: &Vec<usize>, rng: &mut ChaCha8Rng) {
    let feature_len = feature_selection.len();

    if param.ga.mutated_children_pct > 0.0 {
        let num_mutated_individuals = (children.individuals.len() as f64 
            * param.ga.mutated_children_pct / 100.0) as usize;

        let num_mutated_features = (feature_len as f64 
            * param.ga.mutated_features_pct / 100.0) as usize;

        // Select indices of the individuals to mutate
        let individuals_to_mutate = sample(rng, children.individuals.len(), num_mutated_individuals);

        for idx in individuals_to_mutate {
            // Mutate features for each selected individual
            let individual = &mut children.individuals[idx]; // Get a mutable reference
            let feature_indices = sample(rng, feature_len, num_mutated_features)
                            .iter()
                            .map(|i| {feature_selection[i]}).collect::<Vec<usize>>();

            match individual.language {
                individual::TERNARY_LANG|individual::RATIO_LANG => { 
                    mutate_ternary(individual, param, &feature_indices, rng);
                },
                individual::POW2_LANG => { mutate_pow2(individual, param, &feature_indices, rng); },
                individual::BINARY_LANG => { mutate_binary(individual, param, &feature_indices, rng); },
                other => { panic!("Unsupported language {}", other); }
            };
        }
    }
}

pub fn remove_stillborn(children: &mut Population) -> u32 {
    let mut stillborn_children: u32 =0;
    let mut valid_individuals: Vec<Individual> = Vec::new();
    let individuals = mem::take(&mut children.individuals);
    for individual in individuals.into_iter() {
        if individual.language==TERNARY_LANG || individual.language==RATIO_LANG { 
            let mut has_positive: bool=false;
            let mut has_negative: bool=false;
            let mut stillborn_child: bool = true;
            for feature in individual.features.values() {
                if *feature<0 { has_negative=true; if has_positive { stillborn_child=false; break; } }
                if *feature>0 { has_positive=true; if has_negative { stillborn_child=false; break; } }
            }
            if stillborn_child {
                stillborn_children += 1;
                // println!("still {:?}",individual.features);
            }
            else {
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


/// change a sign, remove a variable, add a new variable
pub fn mutate_ternary(individual: &mut Individual, param: &Param, feature_indices: &Vec<usize>, rng: &mut ChaCha8Rng) {
    let p1 = param.ga.mutation_non_null_chance_pct/200.0;
    let p2= 2.0*p1;

    for i in feature_indices {
        if individual.features.contains_key(&i) { individual.k-=1; individual.features.remove(&i); }
        match rng.gen::<f64>() {
            r if r < p1 => { individual.k+=1; individual.features.insert(*i, 1); },
            r if r < p2 => { individual.k+=1; individual.features.insert(*i, -1); },
            _ => {}
        };
    }
}

/// change a sign, remove a variable, add a new variable
pub fn mutate_binary(individual: &mut Individual, param: &Param, feature_indices: &Vec<usize>, rng: &mut ChaCha8Rng) {
    let p1 = param.ga.mutation_non_null_chance_pct/100.0;

    for i in feature_indices {
        if individual.features.contains_key(&i) { individual.k-=1; individual.features.remove(&i); }
        match rng.gen::<f64>() {
            r if r < p1 => { individual.k+=1; individual.features.insert(*i, 1); },
            _ => {}
        };
    }
}


/// change a sign, remove a variable, add a new variable can also double a variable or divide it by two
pub fn mutate_pow2(individual: &mut Individual, param: &Param, feature_indices: &Vec<usize>, rng: &mut ChaCha8Rng) {
    let p1 = param.ga.mutation_non_null_chance_pct/200.0;
    let p2= 2.0*p1;
    let p3= 2.0*p2;
    let p4= 3.0*p2;

    for i in feature_indices {
        let value = if individual.features.contains_key(&i) { individual.k-=1; individual.features.remove(&i).unwrap() } else { 0 };
        match rng.gen::<f64>() {
            r if r < p1 => { individual.k+=1; individual.features.insert(*i, 1); },
            r if r < p2 => { individual.k+=1; individual.features.insert(*i, -1); },
            r if r < p3 => { if value!=0 { individual.features.insert(*i,if value.abs()<64 {2*value} else {value});
                                        individual.k+=1; } },
            r if r < p4 => { if value!=0 { individual.features.insert(*i, if value.abs()==1 {value} else {value/2});
                                        individual.k+=1; } },
                _ => {}
        };
    }
}

fn get_gpu_assay(data: &Data, param: &Param) -> Option<GpuAssay> {
    let gpu_assay = if param.general.gpu && param.ga.random_sampling_pct == 0.0 && param.cv.overfit_penalty == 0.0 {
        let buffer_binding_size = GpuAssay::get_max_buffer_size(&param.gpu) as usize;
        let gpu_max_nb_models = buffer_binding_size / (data.sample_len * std::mem::size_of::<f32>());
        let assay = if gpu_max_nb_models < param.ga.population_size as usize {
            warn!("GPU requires a maximum number of models (<=> Population size). \
            \nAccording to your configuration, param.ga.population_size must not exceed {}. \
            \nIf your configuration supports it and you know what you're doing, consider alternatively increasing the size of the buffers to {:.0} MB (do not forget to adjust the total size accordingly) \
            \nThis Gpredomics session will therefore be launched without a GPU.", gpu_max_nb_models,
            ((param.ga.population_size as usize * data.sample_len * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)+1.0));
            None
        } else {
            Some(GpuAssay::new(&data.X, &data.feature_selection, data.sample_len, param.ga.population_size as usize, &param.gpu))
        }; 
        assay
    } else {
        None
    };

    gpu_assay
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param;
    use crate::param::FitFunction;

    fn create_test_data_disc() -> Data {
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        let mut feature_class: HashMap<usize, u8> = HashMap::new();

        // Simulate data
        X.insert((0, 0), 0.9); // Sample 0, Feature 0
        X.insert((0, 1), 0.01); // Sample 0, Feature 1
        X.insert((1, 0), 0.91); // Sample 1, Feature 0
        X.insert((1, 1), 0.12); // Sample 1, Feature 1
        X.insert((2, 0), 0.75); // Sample 2, Feature 0
        X.insert((2, 1), 0.32); // Sample 2, Feature 1
        X.insert((3, 0), 0.03); // Sample 3, Feature 0
        X.insert((3, 1), 0.92); // Sample 3, Feature 1
        X.insert((4, 0), 0.9);  // Sample 4, Feature 0
        X.insert((4, 1), 0.01); // Sample 4, Feature 1
        feature_class.insert(0, 0);
        feature_class.insert(1, 1);

        Data {
            X,
            y: vec![0, 0, 1, 1, 0], // Vraies étiquettes
            features: vec!["feature1".to_string(), "feature2".to_string()],
            samples: vec!["sample1".to_string(), "sample2".to_string(), "sample3".to_string(),
            "sample4".to_string(), "sample5".to_string()],
            feature_class,
            feature_significance: HashMap::new(),
            feature_selection: vec![0, 1],
            feature_len: 2,
            sample_len: 5,
            classes: vec!["a".to_string(),"b".to_string()]
        }
    }

    fn create_test_data_valid() -> Data {
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        let mut feature_class: HashMap<usize, u8> = HashMap::new();

        // Simulate data
        X.insert((0, 0), 0.91); // Sample 5, Feature 0
        X.insert((0, 1), 0.12); // Sample 5, Feature 1
        X.insert((1, 0), 0.75); // Sample 6, Feature 0
        X.insert((1, 1), 0.01); // Sample 6, Feature 1
        X.insert((2, 0), 0.19); // Sample 7, Feature 0
        X.insert((2, 1), 0.92); // Sample 7, Feature 1
        X.insert((3, 0), 0.9);  // Sample 8, Feature 0
        X.insert((3, 1), 0.01); // Sample 8, Feature 1
        X.insert((4, 0), 0.91); // Sample 9, Feature 0
        X.insert((4, 1), 0.12); // Sample 9, Feature 1
        feature_class.insert(0, 0);
        feature_class.insert(1, 1);

        Data {
            X,
            y: vec![0, 0, 0, 1, 0], // Vraies étiquettes
            features: vec!["feature1".to_string(), "feature2".to_string()],
            samples: vec!["sample6".to_string(), "sample7".to_string(), 
            "sample8".to_string(), "sample9".to_string(), "sample10".to_string()],
            feature_class,
            feature_significance: HashMap::new(),
            feature_selection: vec![0, 1],
            feature_len: 2,
            sample_len: 5,
            classes: vec!["a".to_string(),"b".to_string()]
        }
    }

    use crate::individual::AdditionalMetrics;

    fn create_test_population() -> Population {
        let mut pop = Population {
            individuals: Vec::new(),
        };
    
        for i in 0..10 {
            let ind = Individual {
                features: vec![(0, i%2), (1, -i%2)].into_iter().collect(),
                auc: 0.4 + (i as f64 * 0.05),
                fit: 0.8 - (i as f64 * 0.02),
                specificity: 0.15 + (i as f64 * 0.01),
                sensitivity: 0.16 + (i as f64 * 0.01),
                accuracy: 0.23 + (i as f64 * 0.03),
                threshold: 42.0 + (i as f64),
                k: (42 + i) as usize,
                epoch: (42 + i) as usize,
                language: (i % 4) as u8,
                data_type: (i % 3) as u8,
                hash: i as u64,
                epsilon: f64::MIN_POSITIVE + (i as f64 * 0.001),
                parents : None,
                betas: None, threshold_ci: None,
                metrics: AdditionalMetrics { mcc:None, f1_score: None, npv: None, ppv: None, g_means: None}
            };
            pop.individuals.push(ind);
        }
    
        pop.individuals.push(pop.individuals[9].clone());

        pop
    }

    #[test]
    fn test_fit_no_overfit_penalty() {
        let mut pop = create_test_population();
        let mut data=  create_test_data_disc();
        let mut param = param::get("param.yaml".to_string()).unwrap();

        for ind in pop.individuals.iter_mut() {
            ind.epsilon = param.general.data_type_epsilon;
        }

        const EPSILON: f64 = 1e-5; // mandatory to compare threshold which could be a little different because of GPU f32
        param.cv.overfit_penalty = 0.0;
        let fit_vec = vec![FitFunction::auc, FitFunction::sensitivity, FitFunction::specificity];

        for fit_method in fit_vec {
            let mut populations: Vec<Vec<Individual>> = vec![];
            param.general.fit = fit_method.clone();
            // gpu = false
            param.general.keep_trace = true;
            pop.fit(&mut data, &mut None, &None, &mut None, &param);
            populations.push(pop.individuals.clone());
            param.general.keep_trace = false;
            pop.fit(&mut data, &mut None, &None, &mut None, &param);
            populations.push(pop.individuals.clone());
            // gpu = true
            let gpu_assay = Some(GpuAssay::new(&data.X, &data.feature_selection, data.sample_len, pop.individuals.len() as usize, &param.gpu));
            param.general.keep_trace = true;
            pop.fit(&mut data, &mut None, &gpu_assay, &mut None, &param);
            populations.push(pop.individuals.clone());
            param.general.keep_trace = false;
            pop.fit(&mut data, &mut None, &gpu_assay, &mut None, &param);
            populations.push(pop.individuals.clone());
            
            for p in 1..populations.len() {
                for i in 0..populations[0].len() {
                    assert_eq!(populations[0][i].auc, populations[p][i].auc, "Calculated AUCs for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                    assert_eq!(populations[0][i].sensitivity, populations[p][i].sensitivity, "Calculated Sensitivities for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                    assert_eq!(populations[0][i].specificity, populations[p][i].specificity, "Calculated Specificities for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                    assert_eq!(populations[0][i].accuracy, populations[p][i].accuracy, "Calculated Accuracies for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                    assert!((populations[0][i].threshold - populations[p][i].threshold).abs() < EPSILON, "Calculated Thresholds for parameters 0 and parameters {} are differents (ind = {}, fit on {:?}) : {} VS {}", p, i, fit_method, populations[0][i].threshold, populations[p][i].threshold);
                    assert_eq!(populations[0][i].fit, populations[p][i].fit, "Calculated Fits for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                }
            }
        }
    }

    #[test]
    fn test_fit_overfit_penalty() {
        let mut pop = create_test_population();
        
        let mut data=  create_test_data_disc();
        let test_data = create_test_data_valid();
        let mut param = param::get("param.yaml".to_string()).unwrap();

        for ind in pop.individuals.iter_mut() {
            ind.epsilon = param.general.data_type_epsilon;
        }

        // mandatory to compare threshold which could be a little different because of GPU f32
        const EPSILON: f64 = 1e-5; 
        let fit_vec = vec![FitFunction::auc, FitFunction::sensitivity, FitFunction::specificity];
        param.cv.overfit_penalty = 1.0;
        
        for fit_method in fit_vec {
            let mut populations: Vec<Vec<Individual>> = vec![];
            param.general.fit = fit_method.clone();
            // gpu = false
            param.general.keep_trace = true;
            pop.fit(&mut data, &mut Some(test_data.clone()), &None, &mut None, &param);
            populations.push(pop.individuals.clone());
            param.general.keep_trace = false;
            pop.fit(&mut data, &mut Some(test_data.clone()), &None, &mut None, &param);
            populations.push(pop.individuals.clone());
            // gpu = true
            let gpu_assay = Some(GpuAssay::new(&data.X, &data.feature_selection, data.sample_len, pop.individuals.len() as usize, &param.gpu));
            let test_assay = Some(GpuAssay::new(&test_data.X, &test_data.feature_selection, test_data.sample_len, pop.individuals.len() as usize, &param.gpu));
            param.general.keep_trace = true;
            pop.fit(&mut data, &mut Some(test_data.clone()), &gpu_assay, &test_assay, &param);
            populations.push(pop.individuals.clone());
            param.general.keep_trace = false;
            pop.fit(&mut data, &mut Some(test_data.clone()), &gpu_assay, &test_assay, &param);
            populations.push(pop.individuals.clone());
            
            for p in 1..populations.len() {
                for i in 0..populations[0].len() {
                    assert_eq!(populations[0][i].auc, populations[p][i].auc, "Calculated AUCs for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                    assert_eq!(populations[0][i].sensitivity, populations[p][i].sensitivity, "Calculated Sensitivities for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                    assert_eq!(populations[0][i].specificity, populations[p][i].specificity, "Calculated Specificities for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                    assert_eq!(populations[0][i].accuracy, populations[p][i].accuracy, "Calculated Accuracies for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                    assert!((populations[0][i].threshold - populations[p][i].threshold).abs() < EPSILON, "Calculated Thresholds for parameters 0 and parameters {} are differents (ind = {}, fit on {:?}) : {} VS {} {:?}", p, i, fit_method, populations[0][i].threshold, populations[p][i].threshold, populations[p][i]);
                    assert_eq!(populations[0][i].fit, populations[p][i].fit, "Calculated Fits for parameters 0 and parameters {} are differents (ind = {}, fit on {:?})", p, i, fit_method);
                }
            }
        }
    }

}