use crate::population::Population;
use crate::data::Data;
use crate::individual::Individual;
use crate::param::Param;
use rand::prelude::SliceRandom;
use rand::Rng;
use rand::seq::index::sample;


pub fn ga<'a>(mut data: &'a Data, param: &'a Param) -> Vec<Population<'a>> {
    // generate a random population with a given size  (and evaluate it for selection)
    let mut pop = Population::new(data);
    let mut epoch:u32 = 0;
    let mut populations: Vec<Population> = Vec::new();

    data.select_features(param);
    pop.generate(param.ga.population_size,param.ga.kmin, param.ga.kmax);
    
    loop {
        epoch += 1;


        // we create a new generation
        let mut new_pop = Population::new(data);

        // select some parents (according to elitiste/random params)
        // these individuals are flagged with the parent attribute
        // this populate half the new generation new_pop
        pop.evaluate();
        let sorted_pop = pop.sort();  
        new_pop.add(select_parents(&sorted_pop, param));

        let mut children = cross_over(&new_pop,param);
        mutate(&mut children, param);
        children.evaluate();
        new_pop.add(children);

        populations.push(pop);
        pop = new_pop;

        if (epoch>=param.ga.epochs) {
            break
        }
    }

    populations
    
}

/// pick params.ga_select_elite_pct% of the best individuals and params.ga_select_random_pct%  
fn select_parents<'a>(pop: &'a Population, param: &Param) -> Population<'a> {
    
    // order pop by fit and select params.ga_select_elite_pct
    let (mut parents, n) = pop.select_first_pct(param.ga.select_elite_pct);

    // add a random part of the others
    parents.add(pop.select_random_above_n(param.ga.select_random_pct, n));

    parents

}


/// create children from parents
fn cross_over<'a>(parents: &'a Population, param: &Param) -> Population<'a> {
    let mut rng = rand::thread_rng();

    let mut children=Population::new(parents.data);

    for i in 0..(param.ga.population_size as usize-parents.individuals.len()) {
        let mut child=Individual::new();        
        let [p1,p2] = parents.individuals.choose_multiple(&mut rng, 2)
                        .collect::<Vec<&Individual>>()
                        .try_into()
                        .expect("Vec must have exactly 2 elements");
        let x=rng.gen_range(1..parents.data.feature_len-1);
        child.features = p1.features[..x].to_vec();
        child.features.extend_from_slice(&p2.features[x..]);

        children.individuals.push(child);
    }

    children

}

/// change a sign, remove a variable, add a new variable
fn mutate<'a>(children: &mut Population<'a>, param: &Param) {
    let mut rng = rand::thread_rng();

    if param.ga.mutated_individuals_pct > 0.0 {
        let num_mutated_individuals = (children.individuals.len() as f64 
            * param.ga.mutated_individuals_pct / 100.0) as usize;

        let num_mutated_features = (children.data.feature_len as f64 
            * param.ga.mutated_features_pct / 100.0) as usize;

        // Select indices of the individuals to mutate
        let individuals_to_mutate = sample(&mut rng, children.individuals.len(), num_mutated_individuals);

        for idx in individuals_to_mutate {
            // Mutate features for each selected individual
            let individual = &mut children.individuals[idx]; // Get a mutable reference
            let feature_indices = sample(&mut rng, children.data.feature_len, num_mutated_features);

            for i in feature_indices {
                individual.features[i] = match rng.gen_range(0..3) {
                    0 => 1,
                    1 => -1,
                    _ => 0,
                };
            }
        }
    }
}