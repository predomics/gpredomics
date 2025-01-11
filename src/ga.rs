use crate::population::Population;
use crate::data::Data;
use crate::individual::{self, RATIO_LANG, TERNARY_LANG};
use crate::individual::Individual;
use crate::param::Param;
use rand::prelude::SliceRandom;
use rand::Rng;
use rand::seq::index::sample;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use log::{debug,info,warn,error};
use std::cmp::min;
use std::mem;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub fn ga<F>(data: &mut Data, param: &Param, running: Arc<AtomicBool>, mut fit_fn: F) -> Vec<Population> 
where
    F: FnMut(&mut Population, &Data)
{
    // generate a random population with a given size  (and evaluate it for selection)
    let mut pop = Population::new();
    let mut epoch:usize = 0;
    let mut populations: Vec<Population> = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    info!("Selecting features...");
    data.select_features(param);
    info!("{} features selected.",data.feature_selection.len());
    debug!("FEATURES {:?}",data.feature_class_sign);

    // generate initial population
    let mut target_size = param.ga.population_size;
    while target_size>0 {
        let mut sub_pop = Population::new();
        debug!("generating:");
        sub_pop.generate(target_size, param.ga.kmin, 
            if param.ga.kmax>0 { min(data.feature_selection.len(), param.ga.kmax) } else { data.feature_selection.len() }, 
            individual::language(&param.general.language),
            individual::data_type(&param.general.data_type),
             data, &mut rng);
        debug!("generated...");
        
        target_size = remove_stillborn(&mut sub_pop);
        if target_size>0 { warn!("Some still born are present {}",target_size);
            if target_size==param.ga.population_size { 
                error!("Params only create inviable individuals!");
                panic!("Params only create inviable individuals!") } 
        }

        pop.add(sub_pop);
    }

    info!("pop size {}, kmin {}, kmax {}",pop.individuals.len(),pop.individuals.iter().map(|i| {i.k}).min().unwrap_or(0),pop.individuals.iter().map(|i| {i.k}).max().unwrap_or(0));

    fit_fn(&mut pop, data);

    
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
        debug!("best model so far AUC:{:.3} (fit:{:.3}, k={}, gen#{}, specificity:{:.3}, sensitivity:{:.3}), average AUC {:.3}, fit {:.3}, k:{:.1}", 
            best_model.auc,
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

        if epoch>param.ga.min_epochs {
            if epoch-best_model.epoch+1>param.ga.max_age_best_model {
                info!("Best model has reached limit age...");
                break
            }
        }

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
        
        fit_fn(&mut children, data);
        
        for i in children.individuals.iter_mut() {
            i.epoch = epoch;
        }
        new_pop.add(children);

        if param.ga.keep_all_generations {
            populations.push(pop);
        }
        else {
            populations = vec![pop];
        }
        pop = new_pop;

        if epoch>=param.ga.max_epochs {
            info!("Reach max epoch");
            break
        }

        if !running.load(Ordering::Relaxed) {
            info!("Signal received");
            break
        }
    }

    populations
    
}


/// pick params.ga_select_elite_pct% of the best individuals and params.ga_select_random_pct%  
fn select_parents(pop: &Population, param: &Param, rng: &mut ChaCha8Rng) -> Population {
    
    // order pop by fit and select params.ga_select_elite_pct
    let (mut parents, n) = pop.select_first_pct(param.ga.select_elite_pct);

    // add a random part of the others
    parents.add(pop.select_random_above_n(param.ga.select_random_pct, n, rng));

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

fn remove_stillborn(children: &mut Population) -> u32 {
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