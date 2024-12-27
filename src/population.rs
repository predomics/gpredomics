use crate::data::Data;
use crate::individual::Individual;
use rand::prelude::SliceRandom;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "extendr-support")]
use crate::extendr_api::extendr;

#[cfg_attr(feature = "extendr-support", extendr)]
pub struct Population {
    pub individuals: Vec<Individual>,
    pub fit: Vec<f64>
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
            individuals: Vec::new(),
            fit: Vec::new()
        }
    }

    pub fn evaluate(&mut self, data: &Data) -> & mut Self {
        self.fit = self.individuals.iter_mut().map(|i|{
            i.evaluate(data); 
            i.compute_auc(data)
        }).collect();
        self
    }

    pub fn evaluate_with_k_penalty(&mut self, data: &Data, k_penalty: f64) -> & mut Self {
        self.fit = self.individuals.iter_mut().map(|i|{
            let k=i.compute_auc(data) - i.k as f64 * k_penalty;
            k

        }).collect();
        self
    }
    
    pub fn evaluate_with_kno_penalty(& mut self, data: &Data, k_penalty: f64, test_data: &Data, overfit_penalty: f64) -> & mut Self {
        self.fit = self.individuals.iter_mut().map(|i|{
            //i.evaluate(data); 
            let test_auc = i.compute_auc(test_data);
            let auc=i.compute_auc(data);
            auc - i.k as f64 * k_penalty - (auc-test_auc).abs() * overfit_penalty
        }).collect();
        self
    }

    pub fn sort(mut self) -> Self {

        let mut combined = self.fit.iter()
            .zip(self.individuals.into_iter())
            .collect::<Vec<_>>();


        combined.sort_by(|f, i| i.0.partial_cmp(f.0).unwrap());

        (self.fit, self.individuals)=combined
            .into_iter()
            .unzip();

        self
    }

    /// populate the population with a set of random individuals
    /// populate the population with a set of random individuals
    pub fn generate(&mut self, population_size: u32, kmin:usize, kmax:usize, data: &Data, rng: &mut ChaCha8Rng) {
        for i in 0..population_size {
            self.individuals.push(Individual::random_select_k(data.feature_len, 
                                    kmin,
                                    kmax,
                                    &data.feature_selection,
                                    &data.feature_class_sign,
                                rng))
        }
    }
    /// add some individuals in the population (you may need to evaluate after, prefer add)
    pub fn extend(&mut self, individuals: Vec<Individual>) {
        for i in individuals { self.individuals.push(i) };
    }

    /// add some individuals in the population
    pub fn add(&mut self, population: Population) {
        for (i,f) in population.individuals.into_iter().zip(population.fit.into_iter()) {
             self.individuals.push(i);
             self.fit.push(f);
            };
    }
    

    /// select first element of a (sorted) population
    pub fn select_first_pct(&self, pct: f64) -> (Population,usize) {
        let n: usize = (self.individuals.len() as f64 * pct/100.0) as usize;

        (
            Population {
                individuals: self.individuals.iter().take(n).map(|i|i.clone()).collect(),
                fit: self.fit.iter().take(n).cloned().collect()
            },
            n
        )
    }

    pub fn select_random_above_n(&self, pct: f64, n: usize, rng: &mut ChaCha8Rng) -> Population {
        let k = ( self.individuals.len() as f64 * pct / 100.0 ) as usize;

        let combined = self.individuals.iter()
            .zip(self.fit.iter())
            .skip(n)
            .collect::<Vec<(&Individual,&f64)>>();
            
            
        let selected = combined.choose_multiple(rng, k);

        let (individuals,fit):(Vec<Individual>,Vec<f64>) = selected.map(|(x,f)| ((**x).clone(),*f))
            .unzip();

        Population {
            individuals: individuals,
            fit: fit
        }



    }
}