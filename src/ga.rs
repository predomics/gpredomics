use crate::population::Population;
use crate::data::Data;
use crate::individual::{self, RATIO_LANG, TERNARY_LANG};
use crate::individual::Individual;
use crate::param::{Param, FitFunction};
use crate::gpu::GpuAssay;
use rand::prelude::SliceRandom;
use rand::Rng;
use rand::seq::index::sample;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use log::{debug,info,warn,error};
use std::cmp::min;
use std::collections::HashMap;
use std::mem;
use std::time::Instant;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn fit_fn(pop: &mut Population, data: &mut Data, test_data: &mut Option<Data>, gpu_assay: &Option<GpuAssay>, test_assay: &Option<GpuAssay>, param: &Param) {
    if let Some(assay)=gpu_assay {
        let pool = ThreadPoolBuilder::new()
            .num_threads(param.general.thread_number)
            .build()
            .unwrap();
        if param.general.overfit_penalty == 0.0 {
            let scores:Vec<f64> = assay.compute_scores(&pop.individuals, param.general.data_type_epsilon as f32)
                .into_iter().map(|x| {x as f64}).collect();
            match param.general.fit {
                FitFunction::auc => {
                    pool.install(|| {
                        pop.individuals
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(n,i)| {
                                if param.general.keep_all_generations {
                                    (i.auc, i.threshold, i.accuracy, i.sensitivity, i.specificity) = i.compute_roc_and_metrics_from_value(&scores[n*data.sample_len..(n+1)*data.sample_len], &data.y);
                                } else {
                                    i.auc = i.compute_auc_from_value(&scores[n*data.sample_len..(n+1)*data.sample_len], &data.y);}
                                i.fit = i.auc - i.k as f64 * param.general.k_penalty;
                            });
                    });
                },
                FitFunction::sensitivity => {
                    pool.install(|| {
                        pop.individuals
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(n,i)| {
                                i.fit = i.maximize_objective_with_scores(&scores[n*data.sample_len..(n+1)*data.sample_len], &data, param.general.fr_penalty, 1.0) - i.k as f64 * param.general.k_penalty;
                            });
                    });
                },
                FitFunction::specificity => {
                    pool.install(|| {
                        pop.individuals
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(n,i)| {
                                i.fit = i.maximize_objective_with_scores(&scores[n*data.sample_len..(n+1)*data.sample_len], &data, 1.0, param.general.fr_penalty) - i.k as f64 * param.general.k_penalty;
                            });
                    });
                }
            }
        } else {
            warn!("Be careful: Train AUC is currently wrong calculated when overfit_penalty>0.0 in GA.");
            let t_assay = test_assay.as_ref().unwrap();
            let test_data = test_data.as_mut().unwrap();
            let scores:Vec<f64> = assay.compute_scores(&pop.individuals, param.general.data_type_epsilon as f32)
                .into_iter().map(|x| {x as f64}).collect();
            let t_scores:Vec<f64> = t_assay.compute_scores(&pop.individuals, param.general.data_type_epsilon as f32)
                .into_iter().map(|x| {x as f64}).collect();
            match param.general.fit {
                FitFunction::auc => {
                    pool.install(|| {
                        pop.individuals
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(n,i)| {
                                let test_auc = i.compute_auc_from_value(&t_scores[n*test_data.sample_len..(n+1)*test_data.sample_len], &test_data.y);
                                if param.general.keep_all_generations {
                                    (i.auc, i.threshold, i.accuracy, i.sensitivity, i.specificity) = i.compute_roc_and_metrics_from_value(&scores[n*data.sample_len..(n+1)*data.sample_len], &data.y);
                                } else {
                                    i.auc = i.compute_auc_from_value(&scores[n*data.sample_len..(n+1)*data.sample_len], &data.y);}
                                i.fit = i.auc - i.k as f64 * param.general.k_penalty - (i.auc-test_auc).abs() * param.general.overfit_penalty;
                            });
                    });
                },
                FitFunction::sensitivity => {
                    pool.install(|| {
                        pop.individuals
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(n,i)| {
                                let test_objective = i.maximize_objective_with_scores(&t_scores[n*test_data.sample_len..(n+1)*test_data.sample_len], &test_data, param.general.fr_penalty, 1.0);
                                let objective = i.maximize_objective_with_scores(&scores[n*data.sample_len..(n+1)*data.sample_len], &data, param.general.fr_penalty, 1.0); 
                                 i.fit = objective - i.k as f64 * param.general.k_penalty
                                    - (objective-test_objective).abs() * param.general.overfit_penalty;
                            });
                    });
                },
                FitFunction::specificity => {
                    pool.install(|| {
                        pop.individuals
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(n,i)| {
                                let test_objective = i.maximize_objective_with_scores(&t_scores[n*test_data.sample_len..(n+1)*test_data.sample_len], &test_data, param.general.fr_penalty, 1.0);
                                let objective = i.maximize_objective_with_scores(&scores[n*data.sample_len..(n+1)*data.sample_len], &data, 1.0, param.general.fr_penalty); 
                                 i.fit = objective - i.k as f64 * param.general.k_penalty
                                    - (objective-test_objective).abs() * param.general.overfit_penalty;
                            });
                    });
                }
            }
        }
    }
    else {

        if param.general.overfit_penalty == 0.0 {
            match param.general.fit {
                FitFunction::auc => {
                    pop.auc_fit(data, param.general.k_penalty, param.general.thread_number, param.general.keep_all_generations);
                },
                FitFunction::sensitivity => {
                    pop.objective_fit(data, param.general.fr_penalty,1.0,param.general.k_penalty,
                        param.general.thread_number);
                },
                FitFunction::specificity => {
                    pop.objective_fit(data, 1.0,param.general.fr_penalty,param.general.k_penalty,
                        param.general.thread_number);
                }
            } 
        } 
        else {
            warn!("Be careful: Train AUC is currently wrong calculated when overfit_penalty>0.0 in GA.");
            let test_data = test_data.as_mut().unwrap();
            match param.general.fit {
                FitFunction::auc => {
                    pop.auc_nooverfit_fit(data, param.general.k_penalty, test_data, param.general.overfit_penalty, param.general.thread_number, param.general.keep_all_generations);
                },
                FitFunction::sensitivity => {
                    pop.objective_nooverfit_fit(data, param.general.fr_penalty,1.0,param.general.k_penalty,
                        test_data, param.general.overfit_penalty, param.general.thread_number);
                },
                FitFunction::specificity => {
                    pop.objective_nooverfit_fit(data, 1.0,param.general.fr_penalty,param.general.k_penalty,
                        test_data, param.general.overfit_penalty, param.general.thread_number);
                }
            } 
        }
    }
}

pub fn ga(data: &mut Data, test_data: &mut Option<Data>, param: &Param, running: Arc<AtomicBool>) -> Vec<Population> {   
    let time = Instant::now();
    // generate a random population with a given size  (and evaluate it for selection)
    let mut pop = Population::new();
    let mut epoch:usize = 0;
    let mut populations: Vec<Population> = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    info!("Selecting features...");
    data.select_features(param);
    info!("{} features selected.",data.feature_selection.len());
    debug!("FEATURES {:?}",data.feature_class);

    // generate initial population

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
                    data, &mut rng);
                debug!("generated for {} {}...",sub_pop.individuals[0].get_language(),sub_pop.individuals[0].get_data_type());
            
                target_size = remove_stillborn(&mut sub_pop);
                if target_size>0 { warn!("Some still born are present {} (with healthy {})",target_size,sub_pop.individuals.len());
                    if target_size==param.ga.population_size { 
                        error!("Params only create inviable individuals!");
                        panic!("Params only create inviable individuals!") } 
                }
        
                pop.add(sub_pop);
            }
        }
    }

    info!("pop size {}, kmin {}, kmax {}",pop.individuals.len(),pop.individuals.iter().map(|i| {i.k}).min().unwrap_or(0),pop.individuals.iter().map(|i| {i.k}).max().unwrap_or(0));

    let gpu_assay = if param.general.gpu {
        Some(GpuAssay::new(&data.X, &data.feature_selection, data.sample_len, param.ga.population_size as usize))
    } else { None };
    let test_assay = if param.general.gpu && param.general.overfit_penalty>0.0 {
        let test_data = test_data.as_mut().unwrap();
        Some(GpuAssay::new(&test_data.X, &data.feature_selection, test_data.sample_len, param.ga.population_size as usize))
    } else { None };

    fit_fn(&mut pop, data, test_data, &gpu_assay, &test_assay, param);

    
    loop {
        epoch += 1;
        debug!("Starting epoch {}",epoch);


        // we create a new generation
        let mut new_pop = Population::new();

        // select some parents (according to elitiste/random params)
        // these individuals are flagged with the parent attribute
        // this populate half the new generation new_pop
        pop.compute_hash();
        let clone_number = pop.remove_clone();
        if clone_number>0 { debug!("Some clones were removed : {}.",clone_number) } ;
        
        pop = pop.sort(); 
        let best_model = &pop.individuals[0];
        debug!("best model so far AUC:{:.3} ({}:{} fit:{:.3}, k={}, gen#{}, specificity:{:.3}, sensitivity:{:.3}), average AUC {:.3}, fit {:.3}, k:{:.1}", 
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
            pop.individuals.iter().map(|i| {i.k}).sum::<usize>() as f64/param.ga.population_size as f64
        );

        //auc_values.push(best_model.fit);

        let mut need_to_break= false;
        if epoch>=param.ga.min_epochs {
            if epoch-best_model.epoch+1>param.ga.max_age_best_model {
                info!("Best model has reached limit age...");
                need_to_break = true;
            }
        }

        if epoch>=param.ga.max_epochs {
            info!("Reach max epoch");
            need_to_break = true;
        }

        if !running.load(Ordering::Relaxed) {
            info!("Signal received");
            need_to_break = true;
        }

        if need_to_break {
            if populations.len()==0 {
                populations = vec![pop];
            }
            break
        }



        // Creating new generation
        new_pop.add(select_parents(&pop, param, &mut rng));
        let mut children_to_create = param.ga.population_size as usize-new_pop.individuals.len();
        let mut children=Population::new();
        while children_to_create>0 {
            let mut some_children = cross_over(&new_pop,data.feature_len, children_to_create, &mut rng);
            mutate(&mut some_children, param, &data.feature_selection, &mut rng);
            children_to_create = remove_stillborn(&mut some_children) as usize;
            if children_to_create>0 { warn!("Some stillborn are presents: {}", children_to_create) }

            children.add(some_children);
        }
        
        fit_fn(&mut children, data, test_data, &gpu_assay, &test_assay, param);
        
        for i in children.individuals.iter_mut() {
            i.epoch = epoch;
        }
        new_pop.add(children);

        if param.general.keep_all_generations {
            populations.push(pop);
        }
        else {
            populations = vec![pop];
        }
        pop = new_pop;

    }

    let elapsed = time.elapsed();
        info!("Genetic algorithm computed {:?} generations in {:.2?}", populations.len(), elapsed);

    populations
    
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
pub fn cross_over(parents: &Population, feature_len: usize, children_number: usize, rng: &mut ChaCha8Rng) -> Population {
    let mut children=Population::new();

    for _i in 0..children_number {
        let [p1,p2] = parents.individuals.choose_multiple(rng, 2)
                        .collect::<Vec<&Individual>>()
                        .try_into()
                        .expect("Vec must have exactly 2 elements");
        let main_parent = *vec![p1,p2].choose(rng).unwrap();
        let mut child=Individual::child(main_parent);    
        let x=rng.gen_range(1..feature_len-1);
        if individual::needs_conversion(p1.language, child.language) {
            for (i,val) in p1.features.iter() {
                if i<&x {
                    child.features.insert(*i, individual::gene_convert_from_to(p1.language,
                                                     child.language, 
                                                     *val));
                }
            }
        }
        else {
            for (i,val) in p1.features.iter() {
                if i<&x {
                    child.features.insert(*i, *val);
                }
            }
        }
        if individual::needs_conversion(p2.language, child.language) {
            for (i,val) in p2.features.iter() {
                if i>=&x {
                    child.features.insert(*i, individual::gene_convert_from_to(p1.language,
                        child.language, 
                        *val));
                }
            }
        }
        else {
            for (i,val) in p2.features.iter() {
                if i>=&x {
                    child.features.insert(*i, *val);
                }
            }
        }

        child.count_k();

        child.parents = Some(vec!(p1.hash, p2.hash));
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
        } else {
            valid_individuals.push(individual);
        }
    }
    children.individuals = valid_individuals;

    stillborn_children
}


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
    let p3=2.0*p2;
    let p4=3.0*p2;

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