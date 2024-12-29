use crate::population::Population;
use crate::data::Data;
use crate::individual::Individual;
use crate::param::Param;
use rand::prelude::SliceRandom;
use rand::Rng;
use rand::seq::index::sample;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use log::{debug,info,trace};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub fn ga(data: &mut Data, param: &Param, running: Arc<AtomicBool>) -> Vec<Population> {
    // generate a random population with a given size  (and evaluate it for selection)
    let mut pop = Population::new();
    let mut epoch:usize = 0;
    let mut populations: Vec<Population> = Vec::new();
    let mut auc_values: Vec<f64> = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    info!("Selecting features...");
    data.select_features(param);
    info!("{} features selected.",data.feature_selection.len());

    pop.generate(param.ga.population_size,param.ga.kmin, param.ga.kmax, data, &mut rng);
    pop.evaluate_with_k_penalty(data, param.ga.kpenalty);
    
    let use_mutate2 = param.general.algo.contains("ga2");
    loop {
        epoch += 1;
        debug!("Starting epoch {}",epoch);


        // we create a new generation
        let mut new_pop = Population::new();

        // select some parents (according to elitiste/random params)
        // these individuals are flagged with the parent attribute
        // this populate half the new generation new_pop
        let sorted_pop = pop.sort();  
        debug!("best AUC so far {:.3} (k={}, gen#{}) , average AUC {:.3}, k:{:.3}", 
            &sorted_pop.individuals[0].auc, &sorted_pop.individuals[0].k, &sorted_pop.individuals[0].n,
            &sorted_pop.individuals.iter().map(|i| {i.auc}).sum::<f64>()/param.ga.population_size as f64,
            sorted_pop.individuals.iter().map(|i| {i.k}).sum::<usize>() as f64/param.ga.population_size as f64
        );

        auc_values.push(sorted_pop.fit[0]);

        if auc_values.len()>param.ga.min_epochs {
            if epoch-sorted_pop.individuals[0].n+1>param.ga.max_age_best_model {
                info!("Best model has reached limit age...");
                break
            }
        }

        new_pop.add(select_parents(&sorted_pop, param, &mut rng));

        let mut children = cross_over(&new_pop,param,data.feature_len, &mut rng);

        if use_mutate2 
            { mutate(&mut children, param, &data.feature_selection, &mut rng); }
        else
            { mutate2(&mut children, param, &data.feature_selection, &mut rng); }
        
        children.evaluate_with_k_penalty(data, param.ga.kpenalty);
        
        for i in children.individuals.iter_mut() {
            i.n = epoch;
        }
        new_pop.add(children);

        if param.ga.keep_all_generations {
            populations.push(sorted_pop);
        }
        else {
            populations = vec![sorted_pop];
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


pub fn ga_no_overfit(data: &mut Data, test_data: & Data, param: &Param, running: Arc<AtomicBool>) -> Vec<Population> {
    // generate a random population with a given size  (and evaluate it for selection)
    let mut pop = Population::new();
    let mut epoch:usize = 0;
    let mut populations: Vec<Population> = Vec::new();
    let mut auc_values: Vec<f64> = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    info!("Selecting features...");
    data.select_features(param);
    info!("{} features selected.",data.feature_selection.len());
    trace!("FEATURES:{:?}", data.feature_selection);


    pop.generate(param.ga.population_size,param.ga.kmin, param.ga.kmax, data, &mut rng);
    pop.evaluate_with_kno_penalty(data, param.ga.kpenalty, test_data, param.cv.overfit_penalty);
    
    let use_mutate2 = param.general.algo.contains("ga2");
    loop {
        epoch += 1;
        debug!("Starting epoch {}",epoch);


        // we create a new generation
        let mut new_pop = Population::new();

        // select some parents (according to elitiste/random params)
        // these individuals are flagged with the parent attribute
        // this populate half the new generation new_pop
        let sorted_pop = pop.sort();  
        debug!("best AUC so far {:.3} (k={}, gen#{}) , average AUC {:.3}, k:{:.3}", 
            &sorted_pop.individuals[0].auc, &sorted_pop.individuals[0].k, &sorted_pop.individuals[0].n,
            &sorted_pop.individuals.iter().map(|i| {i.auc}).sum::<f64>()/param.ga.population_size as f64,
            sorted_pop.individuals.iter().map(|i| {i.k}).sum::<usize>() as f64/param.ga.population_size as f64
    );

        auc_values.push(sorted_pop.fit[0]);

        if auc_values.len()>param.ga.min_epochs {
            if epoch-sorted_pop.individuals[0].n+1>param.ga.max_age_best_model {
                info!("Best model has reached limit age...");
                break
            }
        }

        new_pop.add(select_parents(&sorted_pop, param, &mut rng));

        let mut children = cross_over(&new_pop,param,data.feature_len, &mut rng);
        if use_mutate2 
            { mutate(&mut children, param, &data.feature_selection, &mut rng); }
        else
            { mutate2(&mut children, param, &data.feature_selection, &mut rng); }
        
        children.evaluate_with_kno_penalty(data, param.ga.kpenalty, test_data, param.cv.overfit_penalty);
        for i in children.individuals.iter_mut() {
            i.n = epoch;
        }
        new_pop.add(children);

        if param.ga.keep_all_generations {
            populations.push(sorted_pop);
        }
        else {
            populations = vec![sorted_pop];
        }
        pop = new_pop;

        if epoch>=param.ga.max_epochs {
            debug!("Max epoch reached");
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
pub fn cross_over(parents: &Population, param: &Param, feature_len: usize, rng: &mut ChaCha8Rng) -> Population {
    let mut children=Population::new();

    for _i in 0..(param.ga.population_size as usize-parents.individuals.len()) {
        let mut child=Individual::new();        
        let [p1,p2] = parents.individuals.choose_multiple(rng, 2)
                        .collect::<Vec<&Individual>>()
                        .try_into()
                        .expect("Vec must have exactly 2 elements");
        let x=rng.gen_range(1..feature_len-1);
        for (i,val) in p1.features.iter() {
            if i<&x {
                child.features.insert(*i, *val);
            }
        }
        for (i,val) in p2.features.iter() {
            if i>=&x {
                child.features.insert(*i, *val);
            }
        }

        child.count_k();

        children.individuals.push(child);
    }

    children

}

/// change a sign, remove a variable, add a new variable
pub fn mutate(children: &mut Population, param: &Param, feature_selection: &Vec<usize>, rng: &mut ChaCha8Rng) {
    let p1 = param.ga.mutation_non_null_chance_pct/200.0;
    let p2= 2.0*p1;
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


            for i in feature_indices {
                if individual.features.contains_key(&i) { individual.k-=1; individual.features.remove(&i); }
                match rng.gen::<f64>() {
                    r if r < p1 => { individual.k+=1; individual.features.insert(i, 1); },
                    r if r < p2 => { individual.k+=1; individual.features.insert(i, -1); },
                    _ => {}
                };
            }
        }
    }
}


/// change a sign, remove a variable, add a new variable can also double a variable or divide it by two
pub fn mutate2(children: &mut Population, param: &Param, feature_selection: &Vec<usize>, rng: &mut ChaCha8Rng) {
    let p1 = param.ga.mutation_non_null_chance_pct/200.0;
    let p2= 2.0*p1;
    let p3=2.0*p2;
    let p4=3.0*p2;
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


            for i in feature_indices {
                let value = if individual.features.contains_key(&i) { individual.k-=1; individual.features.remove(&i).unwrap() } else { 0 };
                match rng.gen::<f64>() {
                    r if r < p1 => { individual.k+=1; individual.features.insert(i, 1); },
                    r if r < p2 => { individual.k+=1; individual.features.insert(i, -1); },
                    r if r < p3 => { individual.features.insert(i,if value.abs()<64 {2*value} else {value});
                                                individual.k+=1; },
                    r if r < p4 => { individual.features.insert(i, if value.abs()==1 {value} else {value/2});
                                                individual.k+=1; },
                        _ => {}
                };
            }
        }
    }
}