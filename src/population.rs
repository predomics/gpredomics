use crate::data::Data;
use crate::individual::Individual;
use rand::prelude::SliceRandom;
use rand_chacha::ChaCha8Rng;
use std::mem;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

pub struct Population {
    pub individuals: Vec<Individual>
}

impl Population {
    /// Provides a help message describing the `Population` struct and its fields.
    pub fn help() -> &'static str {
        "
        Population Struct:
        -----------------
        Represents a population consisting of multiple individuals, 
        with associated feature metadata.

        Fields:
        - individuals: Vec<Individual>
            A vector containing the individuals in the population. 
            Each individual represents an entity with a set of attributes or features.

        - feature_names: HashMap<u32, String>
            A map between feature indices (u32) and their corresponding names (String).
            This provides a human-readable label for each feature in the population.
        "
    }


    pub fn new() -> Population {
        Population {
            individuals: Vec::new()
        }
    }

    pub fn compute_hash(&mut self) {
        for individual in &mut self.individuals {
            individual.compute_hash();
        }
    }

    pub fn remove_clone(&mut self) -> u32 {
        let mut clone_number: u32 =0;
        let mut unique_individuals: Vec<Individual> = Vec::new();
        let mut hash_vector: Vec<u64> = Vec::new();

        let individuals = mem::take(&mut self.individuals);
        for individual in individuals.into_iter() { 
            if hash_vector.contains(&individual.hash) {
                clone_number +=1;
            } else {
                hash_vector.push(individual.hash);
                unique_individuals.push(individual);
            } 
        }
        self.individuals = unique_individuals;
    
        clone_number
    }
    

    pub fn auc_fit(&mut self, data: &Data, min_value: f64, k_penalty: f64, thread_number: usize) {
        // Create a custom thread pool with 4 threads
        let pool = ThreadPoolBuilder::new()
            .num_threads(thread_number)
            .build()
            .unwrap();

        // Use the custom thread pool for parallel processing
        pool.install(|| {
            self.individuals
                .par_iter_mut()
                .for_each(|i| {
                    i.fit = i.compute_auc(data, min_value) - i.k as f64 * k_penalty;
                });
        });
    }
    
    pub fn auc_nooverfit_fit(& mut self, data: &Data, min_value: f64, k_penalty: f64, test_data: &Data, overfit_penalty: f64,
            thread_number: usize) {
        // Create a custom thread pool with 4 threads
        let pool = ThreadPoolBuilder::new()
            .num_threads(thread_number)
            .build()
            .unwrap();

        // Use the custom thread pool for parallel processing
        pool.install(|| {
            self.individuals
                .par_iter_mut()
                .for_each(|i| {
                    let test_auc = i.compute_auc(test_data, min_value);
                    let auc= i.compute_auc(data, min_value);
                    i.fit = auc - i.k as f64 * k_penalty - (auc-test_auc).abs() * overfit_penalty;
                });
        });
    }

    pub fn objective_fit(&mut self, data: &Data, min_value: f64, fpr_penalty: f64, fnr_penalty: f64, k_penalty: f64,
                            thread_number: usize) 
    {
                // Create a custom thread pool with 4 threads
                let pool = ThreadPoolBuilder::new()
                .num_threads(thread_number)
                .build()
                .unwrap();
    
            // Use the custom thread pool for parallel processing
            pool.install(|| {
                self.individuals
                    .par_iter_mut()
                    .for_each(|i| {
                        i.fit = i.maximize_objective(data, min_value, fpr_penalty, fnr_penalty) - i.k as f64 * k_penalty;
                    });
            });
    }

    pub fn objective_nooverfit_fit(& mut self, data: &Data, min_value: f64, fpr_penalty: f64, fnr_penalty: f64, 
                                    k_penalty: f64, test_data: &Data, overfit_penalty: f64, thread_number: usize) {
        // Create a custom thread pool with 4 threads
        let pool = ThreadPoolBuilder::new()
        .num_threads(thread_number)
        .build()
        .unwrap();

        // Use the custom thread pool for parallel processing
        pool.install(|| {
            self.individuals
                .par_iter_mut()
                .for_each(|i| {
                    let test_objective = i.maximize_objective(test_data, min_value, fpr_penalty, fnr_penalty);
                    let objective= i.maximize_objective(data, min_value, fpr_penalty, fnr_penalty);
                    i.fit = objective - i.k as f64 * k_penalty - (objective-test_objective).abs() * overfit_penalty;
                });
        });
    }

    pub fn sort(mut self) -> Self {
        self.individuals.sort_by(|i,j| j.fit.partial_cmp(&i.fit).unwrap());
        self
    }

    /// populate the population with a set of random individuals
    /// populate the population with a set of random individuals
    pub fn generate(&mut self, population_size: u32, kmin:usize, kmax:usize, language: u8, data_type: u8, data: &Data, rng: &mut ChaCha8Rng) {
        for _ in 0..population_size {
            self.individuals.push(Individual::random_select_k(kmin,
                                    kmax,
                                    &data.feature_selection,
                                    &data.feature_class_sign,
                                    language,
                                    data_type,
                                rng))
        }
    }

    /// add some individuals in the population
    pub fn add(&mut self, population: Population) {
        self.individuals.extend(population.individuals);
    }
    

    /// select first element of a (sorted) population
    pub fn select_first_pct(&self, pct: f64) -> (Population,usize) {
        let n: usize = (self.individuals.len() as f64 * pct/100.0) as usize;

        (
            Population {
                individuals: self.individuals.iter().take(n).map(|i|i.clone()).collect(),
            },
            n
        )
    }

    pub fn select_random_above_n(&self, pct: f64, n: usize, rng: &mut ChaCha8Rng) -> Population {
        let k = ( self.individuals.len() as f64 * pct / 100.0 ) as usize;
        
        Population {
            individuals: self.individuals[n..].choose_multiple(rng, k).cloned().collect()
        }

    }
}