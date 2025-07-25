use serde::{Serialize, Deserialize};
use crate::param::{FitFunction, ImportanceAggregation};
use crate::utils::{conf_inter_binomial,shuffle_row, compute_roc_and_metrics_from_value, compute_auc_from_value};
use crate::cv::CV;
use crate::data::Data;
use crate::individual::Individual;
use crate::param::Param;
use rand::RngCore;
use rand::SeedableRng;
use rand::prelude::SliceRandom;
use rand_chacha::ChaCha8Rng;
use std::mem;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::cmp::max;
use std::collections::{HashMap, HashSet};
#[derive(Clone, Serialize, Deserialize)]
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

    pub fn display(&mut self, data: &Data, data_to_test: Option<&Data>, param: &Param) -> String {
        if param.general.algo == "mcmc" {
            let mut str: String = format!("Displaying bayesian model with the greatest log evidence. Metrics are shown in the following order: Train/Test.");
            let (train_auc, train_best_threshold, train_best_acc, train_best_sens, train_best_spec, _) = self.bayesian_compute_roc_and_metrics(&data);
            if let Some(data_to_test) = data_to_test {
                let (test_auc, test_best_acc, test_best_sens, test_best_spec) = self.bayesian_compute_metrics(&data_to_test, train_best_threshold);
                str = format!("{}\n\nBayesian model {}:{} [n_it = {}] AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}", str,
                self.individuals[0].get_language(), self.individuals[0].get_data_type(), self.individuals.len(), train_auc, test_auc, train_best_acc, test_best_acc, 
                train_best_sens, test_best_sens, train_best_spec, test_best_spec)
            } else {
                str = format!("{}\nBayesian model {}:{} [n_it = {}] AUC {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3}", str,
                self.individuals[0].get_language(), self.individuals[0].get_data_type(), self.individuals.len(), train_auc, train_best_acc, train_best_sens,
                train_best_sens)
            }   
            str = format!("{}\n\nThese metrics were calculated by optimizing the probability of decision (threshold={:.3} instead of 0.5)", str, train_best_threshold);
            str = format!("{}\nNote: the number of iterations corresponds to the sum of the number of betas and feature coefficient variations, resulting in a number greater than n_iter-n_burn)", str);
            format!("{}\nTo reproduce these results, relaunch this exact training with data to be predicted and/or export the MCMC trace with save_trace_outdir", str)
        }
        else {
            let limit:u32;
            if (self.individuals.len() as u32) < param.general.nb_best_model_to_test || param.general.nb_best_model_to_test == 0 {
                limit = self.individuals.len() as u32
            } else {
                limit = param.general.nb_best_model_to_test
            }

            let mut str: String = format!("Displaying {} models. Metrics are shown in the following order: Train/Test.", limit);
            for i in 0..=(limit-1) as usize {
                if param.general.keep_trace == false {
                    match param.general.fit {
                        FitFunction::sensitivity =>  {(self.individuals[i].auc, self.individuals[i].threshold, self.individuals[i].accuracy, self.individuals[i].sensitivity, self.individuals[i].specificity, _) = self.individuals[i].compute_roc_and_metrics(data, Some(&vec![param.general.fr_penalty, 1.0]));},
                        FitFunction::specificity => {(self.individuals[i].auc, self.individuals[i].threshold, self.individuals[i].accuracy, self.individuals[i].sensitivity, self.individuals[i].specificity, _) = self.individuals[i].compute_roc_and_metrics(data, Some(&vec![1.0, param.general.fr_penalty]));},
                        _ => {(self.individuals[i].auc, self.individuals[i].threshold, self.individuals[i].accuracy, self.individuals[i].sensitivity, self.individuals[i].specificity, _) = self.individuals[i].compute_roc_and_metrics(data, None);}
                    }
                }
                if param.general.display_colorful == true && param.general.log_base == "" {
                    str = format!("{}\nModel \x1b[1;93m#{:?}\x1b[0m {}\n ", str, i+1, self.individuals[i].display(data, data_to_test, &param.general.algo, param.general.display_level, param.general.display_colorful));
                } else if param.general.display_colorful == false && param.general.log_base == "" {
                    str = format!("{}\nModel #{:?} {}", str, i+1, self.individuals[i].display(data, data_to_test, &param.general.algo, param.general.display_level, param.general.display_colorful));
                } else {
                    // avoid ASCII symbols and newlines in log file
                    str = format!("{}\nModel #{:?} {}", str, i+1, self.individuals[i].display(data, data_to_test, &param.general.algo, param.general.display_level, false));
                }
            }
        return str
        }
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
    

    pub fn auc_fit(&mut self, data: &Data, k_penalty: f64, thread_number: usize, compute_metrics: bool) {
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
                    if compute_metrics {
                        (i.auc, i.threshold, i.accuracy, i.sensitivity, i.specificity, _) = i.compute_roc_and_metrics(&data, None);
                    } else {
                        i.auc = i.compute_auc(data);
                    }
                    i.fit = i.auc - i.k as f64 * k_penalty;
                });
        });
    }
    
    pub fn auc_nooverfit_fit(& mut self, data: &Data, k_penalty: f64, test_data: &Data, overfit_penalty: f64,
            thread_number: usize, compute_metrics: bool) {
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
                    let test_auc = i.compute_new_auc(test_data);
                    if compute_metrics {
                        (i.auc, i.threshold, i.accuracy, i.sensitivity, i.specificity, _) = i.compute_roc_and_metrics(&data, None);
                    } else {
                        i.auc = i.compute_auc(data);
                    }
                    i.fit = i.auc - i.k as f64 * k_penalty - (i.auc-test_auc).abs() * overfit_penalty;
                });
        });
    }

    // Do we need to write it directly in FitFunction as another fit method? or factorise others?
    pub fn objective_fit(&mut self, data: &Data, fpr_penalty: f64, fnr_penalty: f64, k_penalty: f64,
                            thread_number: usize, compute_metrics: bool) 
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
                        let objective;
                        if compute_metrics {
                            (i.auc, i.threshold, i.accuracy, i.sensitivity, i.specificity, objective) = i.compute_roc_and_metrics(&data, Some(&vec![fpr_penalty, fnr_penalty]));
                        } else {
                            objective = i.maximize_objective(data, fpr_penalty, fnr_penalty)
                        }
                        i.fit = objective - i.k as f64 * k_penalty;        
                    });
            });
    }

    pub fn objective_nooverfit_fit(& mut self, data: &Data, fpr_penalty: f64, fnr_penalty: f64, 
                                    k_penalty: f64, test_data: &Data, overfit_penalty: f64, thread_number: usize,
                                    compute_metrics: bool) {
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
                    let objective;
                    let test_objective = i.maximize_objective(test_data, fpr_penalty, fnr_penalty);
                    if compute_metrics {
                        (i.auc, i.threshold, i.accuracy, i.sensitivity, i.specificity, objective) = i.compute_roc_and_metrics(&data, Some(&vec![fpr_penalty, fnr_penalty]));
                    } else {
                        objective = i.maximize_objective(data, fpr_penalty, fnr_penalty)
                    }
                    i.fit = objective - i.k as f64 * k_penalty - (objective-test_objective).abs() * overfit_penalty;
                });
        });
    }

    pub fn sort(mut self) -> Self {
        self.individuals.sort_by(|i,j| j.fit.partial_cmp(&i.fit).unwrap());
        self
    }

    // Function for cross-validation
    pub fn fit_on_folds(&mut self, data: &Data, cv: &CV, param: &Param) {
        let chunk_size = max(1, self.individuals.len() / (4 * rayon::current_num_threads()));
        
        self.individuals.par_chunks_mut(chunk_size).for_each(|chunk| {
            for individual in chunk {
                let (auc_sum, auc_squared_sum) = cv.validation_folds.par_iter()
                    .map(|fold| {
                        let auc = individual.compute_new_auc(fold);
                        (auc, auc * auc)
                    })
                    .reduce(
                        || (0.0, 0.0),
                        |(auc_sum1, auc_squared_sum1), (auc_sum2, auc_squared_sum2)| {
                            (auc_sum1 + auc_sum2, auc_squared_sum1 + auc_squared_sum2)
                        }
                    );
    
                let num_folds = cv.validation_folds.len() as f64;
                let average_auc = auc_sum / num_folds;
                let variance = (auc_squared_sum / (num_folds - 1.0)) - (average_auc * average_auc);
                individual.auc = individual.compute_auc(&data);
                individual.fit = average_auc - (param.general.overfit_penalty * variance) - (individual.k as f64 * param.general.k_penalty);
            }
        });
    }
    

    pub fn select_best_population(&self, alpha:f64) -> Population { 
        // (Family of best models)
        // This function return a population containing only individuals with a Fit greater than the best individual evaluation metric lower bound
        // it require a sorted population and a fit between 0 et 1. 
        let mut best_pop = Population::new();    
        let mut eval: Vec<f64> = vec![];

        // Collect evaluation score == ind.fit
        for ind in &self.individuals {
            eval.push(ind.fit)
            //eval = ind.fit - (k_penalty * spar) -> it is useful as fit already take k_penalty in account ? 
        }

        // Control the distribution of evaluation metric
        if eval[0] > 1.0 && eval.last().unwrap_or(&0.0) < &0.0 {
            panic!("Evaluation metric should be in the [0, 1] interval");
        } 

        let (lower_bound, _, _) = conf_inter_binomial(eval[0], self.individuals.len(), alpha);

        for ind in &self.individuals {
            if ind.fit > lower_bound {
                best_pop.individuals.push(ind.clone());
            } 
            // indivduals are theoricaly sorted by fit
            else {
                break
            }
        }

        best_pop
        
    }

    /// populate the population with a set of random individuals
    pub fn generate(&mut self, population_size: u32, kmin:usize, kmax:usize, language: u8, data_type: u8, epsilon: f64, data: &Data, rng: &mut ChaCha8Rng) {
        for _ in 0..population_size {
            self.individuals.push(Individual::random_select_k(kmin,
                                    kmax,
                                    &data.feature_selection,
                                    &data.feature_class,
                                    language,
                                    data_type,
                                    epsilon,
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

    pub fn compute_all_metrics(&mut self, data: &Data) {
        self.individuals
            .par_iter_mut()
            .for_each(|ind| {
                (ind.auc, ind.threshold, ind.accuracy, ind.sensitivity, ind.specificity, _) = ind.compute_roc_and_metrics(data, None);
            });
    }

    // Genealogy functions
    pub fn get_ind_from_hash(&self, hash: u64) -> Option<&Individual> {
        assert!(self.individuals[0].hash!=0, "Hash should be computed to allow Individual selection");
        self.individuals
            .par_iter()
            .find_any(|ind| ind.hash == hash)
    }

    /// Compute OOB feature importance by doing N permutations on samples on a feature (for each feature)
    /// uses mean decreased AUC
    pub fn compute_pop_oob_feature_importance( 
        &mut self, 
        data: &Data, 
        permutations: usize, 
        main_rng: &mut ChaCha8Rng, 
        aggregation_method: &ImportanceAggregation,
        scaled_importance: bool
    ) -> HashMap<usize, (f64, f64)> { 
        let mut all_features: Vec<usize> = self.individuals
            .iter()
            .flat_map(|ind| ind.features_index())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        all_features.sort_unstable();
        
        let mut feature_seeds: HashMap<usize, Vec<u64>> = HashMap::new();
        for &feature_idx in &all_features {
            let seeds: Vec<u64> = (0..permutations)
                .map(|_| main_rng.next_u64())
                .collect();
            feature_seeds.insert(feature_idx, seeds);
        }
        
        self.individuals.sort_by_key(|ind| ind.hash);
        
        let individual_results: Vec<Vec<(usize, f64)>> = self.individuals
            .par_iter_mut()
            .map(|ind| {
                let baseline_auc = ind.compute_new_auc(data);
                
                let mut model_features = ind.features_index();
                model_features.sort_unstable(); 
                
                model_features.iter().map(|&feature_idx| {
                    let seeds = &feature_seeds[&feature_idx];
                    
                    let permuted_auc_sum: f64 = seeds.iter().take(permutations)
                        .map(|&seed| {
                            let mut permutation_rng = ChaCha8Rng::seed_from_u64(seed);
                            
                            let mut X_permuted = data.X.clone();
                            shuffle_row(&mut X_permuted, data.sample_len, feature_idx, &mut permutation_rng);
                            
                            ind.compute_auc_from_features(&X_permuted, data.sample_len, &data.y)
                        })
                        .sum();
                    
                    let mean_permuted_auc = permuted_auc_sum / permutations as f64;
                    let importance = baseline_auc - mean_permuted_auc;
                    
                    (feature_idx, importance)
                }).collect()
            })
            .collect();
        
        let mut feature_importances: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut feature_counts: HashMap<usize, usize> = HashMap::new();
        
        for importance_list in individual_results {
            for (feature_idx, importance) in importance_list {
                feature_importances.entry(feature_idx)
                    .or_insert_with(Vec::new)
                    .push(importance);
                
                *feature_counts.entry(feature_idx).or_insert(0) += 1;
            }
        }
        
        let mut result = HashMap::new();
        for (feature_idx, values) in feature_importances {
            let aggregated_value = match aggregation_method {
                ImportanceAggregation::Mean => values.iter().sum::<f64>() / values.len() as f64,
                _ => {
                    let mut v = values.clone();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    if v.len() % 2 == 1 {
                        v[v.len() / 2]
                    } else {
                        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
                    }
                }
            };
            
            let dispersion = match aggregation_method {
                ImportanceAggregation::Median => {
                    let mut v = values.clone();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = if v.len() % 2 == 1 {
                        v[v.len() / 2]
                    } else {
                        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
                    };
                    
                    let mut deviations: Vec<f64> = v.iter()
                        .map(|&val| (val - median).abs())
                        .collect();
                    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                    let mad = if deviations.len() % 2 == 1 {
                        deviations[deviations.len() / 2]
                    } else {
                        (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
                    };
                    
                    1.4826 * mad
                },
                _ => {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter()
                        .map(|&val| (val - mean).powi(2))
                        .sum::<f64>() / values.len() as f64;
                    variance.sqrt()
                }
            };
            
            let (final_value, final_dispersion) = if scaled_importance {
                let prevalence = feature_counts[&feature_idx] as f64 / self.individuals.len() as f64;
                (aggregated_value * prevalence, dispersion * prevalence)
            } else {
                (aggregated_value, dispersion)
            };
            
            result.insert(feature_idx, (final_value, final_dispersion));
        }
        
        result
    }
    
    pub fn bayesian_predict(&self, data: &Data) -> Vec<f64> {
        let mut bayesian_prob: Vec<f64> = vec![0.0; data.sample_len];
        for ind in &self.individuals {
            let sample_prob = ind.evaluate(&data);
            for (sample_sum, &val) in bayesian_prob.iter_mut().zip(sample_prob.iter()) {
                *sample_sum += val;
            }
        }

        let pop_size = self.individuals.len() as f64;
        for sample in bayesian_prob.iter_mut() {
            *sample /= pop_size;
        }

        bayesian_prob
    }

    pub fn bayesian_class(&self, data: &Data, threshold: f64) -> Vec<u8> {
        let probs = self.bayesian_predict(data);
        probs.iter()
            .map(|p| if *p > threshold { 1 } else { 0 })
            .collect()
    }

    pub fn bayesian_compute_roc_and_metrics(&self, data: &Data) -> (f64, f64, f64, f64, f64, f64) {
        compute_roc_and_metrics_from_value(&self.bayesian_predict(data), &data.y, None) 
    }

    pub fn bayesian_compute_metrics(&self, data: &Data, threshold: f64) -> (f64, f64, f64, f64) {
        let ind = Individual::new();
        let (acc, se, sp) = ind.compute_metrics_from_classes(&self.bayesian_class(data, threshold), &data.y);
        (compute_auc_from_value(&self.bayesian_predict(&data), &data.y), acc, se, sp)
    }
    
}

use std::fmt;
impl fmt::Debug for Population {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Population")
            .field("individuals", &self.individuals)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::{collections::HashMap, f64::MAX};
    use crate::individual::{DEFAULT_MINIMUM, RAW_TYPE, TERNARY_LANG};

    fn create_test_population() -> Population {
        let mut pop = Population {
            individuals: Vec::new(),
        };
    
        for i in 0..10 {
            let ind = Individual {
                features: vec![(0, i), (1, -i), (2, i * 2), (3, i % 3)].into_iter().collect(),
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
                betas: None 
            };
            pop.individuals.push(ind);
        }
    
        pop.individuals.push(pop.individuals[9].clone());

        pop
    }

    fn create_test_data_disc() -> Data {
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        let mut feature_class: HashMap<usize, u8> = HashMap::new();

        // Simulate data
        X.insert((0, 0), 0.9); // Sample 0, Feature 0
        X.insert((0, 1), 0.01); // Sample 0, Feature 1
        X.insert((1, 0), 0.91); // Sample 1, Feature 0
        X.insert((1, 1), 0.12); // Sample 1, Feature 1
        X.insert((2, 0), 0.75); // Sample 2, Feature 0
        X.insert((2, 1), 0.01); // Sample 2, Feature 1
        X.insert((3, 0), 0.19); // Sample 3, Feature 0
        X.insert((3, 1), 0.92); // Sample 3, Feature 1
        X.insert((4, 0), 0.9);  // Sample 4, Feature 0
        X.insert((4, 1), 0.01); // Sample 4, Feature 1
        feature_class.insert(0, 0);
        feature_class.insert(1, 1);

        Data {
            X,
            y: vec![1, 0, 1, 0, 0], // Vraies étiquettes
            features: vec!["feature1".to_string(), "feature2".to_string()],
            samples: vec!["sample1".to_string(), "sample2".to_string(), "sample3".to_string(),
            "sample4".to_string(), "sample5".to_string()],
            feature_class,
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
        X.insert((5, 0), 0.91); // Sample 5, Feature 0
        X.insert((5, 1), 0.12); // Sample 5, Feature 1
        X.insert((6, 0), 0.75); // Sample 6, Feature 0
        X.insert((6, 1), 0.01); // Sample 6, Feature 1
        X.insert((7, 0), 0.19); // Sample 7, Feature 0
        X.insert((7, 1), 0.92); // Sample 7, Feature 1
        X.insert((8, 0), 0.9);  // Sample 8, Feature 0
        X.insert((8, 1), 0.01); // Sample 8, Feature 1
        X.insert((9, 0), 0.91); // Sample 9, Feature 0
        X.insert((9, 1), 0.12); // Sample 9, Feature 1
        feature_class.insert(0, 0);
        feature_class.insert(1, 1);

        Data {
            X,
            y: vec![0, 0, 0, 1, 0], // Vraies étiquettes
            features: vec!["feature1".to_string(), "feature2".to_string()],
            samples: vec!["sample6".to_string(), "sample7".to_string(), 
            "sample8".to_string(), "sample9".to_string(), "sample10".to_string()],
            feature_class,
            feature_selection: vec![0, 1],
            feature_len: 2,
            sample_len: 5,
            classes: vec!["a".to_string(),"b".to_string()]
        }
    }

    // #[test] outdated - betas are now used to compute hash
    // fn test_hash_reproductibility() {
    //     let mut pop = create_test_population();
    //     let test_stdout ="hashes of Individuals of the Population are not the same as generated in the past, indicating a reproducibility problem.";
    //     pop.compute_hash();
    //     assert_eq!(8828241517069783773, pop.individuals[0].hash, "{}", test_stdout);
    //     assert_eq!(10219684453491833036, pop.individuals[1].hash, "{}", test_stdout);
    //     assert_eq!(15701002319609139959, pop.individuals[2].hash, "{}", test_stdout);
    //     assert_eq!(8712745243790315600, pop.individuals[3].hash, "{}", test_stdout);
    //     assert_eq!(5577107020388865403, pop.individuals[4].hash, "{}", test_stdout);
    //     assert_eq!(573762283934476040, pop.individuals[5].hash, "{}", test_stdout);
    //     assert_eq!(8225502949628013706, pop.individuals[6].hash, "{}", test_stdout);
    //     assert_eq!(212344198819259482, pop.individuals[7].hash, "{}", test_stdout);
    //     assert_eq!(16067238433084305925, pop.individuals[8].hash, "{}", test_stdout);
    //     assert_eq!(9859973499533993323, pop.individuals[9].hash, "{}", test_stdout);
    //     assert_eq!(9859973499533993323, pop.individuals[10].hash, "same Individual of the Population should have the same hash");
    // }

    #[test]
    fn test_remove_clone() {
        let mut pop = create_test_population();
        assert_eq!(1, pop.remove_clone(), "remove_clone() should return the number of clone eliminated");
        assert_eq!(10, pop.individuals.len(), "remove_clone() should eliminate clones");
        assert_eq!(0, pop.remove_clone(), "remove_clone() should return 0 when there is no clone");
        assert_eq!(10, pop.individuals.len(), "remove_clone() should return the same Population if it does not contain clone");
    }

    #[test]
    fn test_auc_fit() {
        let data = &&create_test_data_disc();
        let mut pop = create_test_population();
        let mut pop_clone = create_test_population();
        let mut pop_1ind = Population::new();
        pop_1ind.individuals.push(pop.individuals[0].clone());

        pop.auc_fit(&data, 10.0, 4, false);
        pop_clone.auc_fit(&data, 10.0, 1, false);
        
        for i in 0..pop.individuals.len(){
            assert_eq!(pop.individuals[i].compute_auc(&data)- pop.individuals[i].k as f64 * 10.0, pop.individuals[i].fit, 
            "each Population's Individual should have a fit equal to Individual.compute_auc(&data) - Individual.k * k_penalty");
            assert_eq!(pop.individuals[i].fit, pop_clone.individuals[i].fit, 
            "mono- and multi- thread should lead to the same final result");
        }
        
        pop_1ind.auc_fit(&data, 10.0, 4, false);
        assert_eq!(pop_1ind.individuals[0].fit, pop.individuals[0].fit, 
        "fitting should work for a Population composed of only one Individual");
    }

    #[test]
    fn test_auc_nooverfit_fit() {
        let data_disc = create_test_data_disc();
        let data_valid = create_test_data_valid();
        let mut pop = create_test_population();
        let mut pop_clone = create_test_population();
        let mut pop_1ind = Population::new();
        pop_1ind.individuals.push(pop.individuals[0].clone());

        pop.auc_nooverfit_fit(&data_disc, 10.0, &data_valid, 20.0, 4, false);
        pop_clone.auc_nooverfit_fit(&data_disc, 10.0, &data_valid, 20.0, 1, false);
        
        for i in 0..pop.individuals.len(){
            assert_eq!(pop.individuals[i].compute_auc(&data_disc) - pop.individuals[i].k as f64 * 10.0 - (pop.individuals[i].compute_auc(&data_disc) - pop.individuals[i].compute_auc(&data_valid)).abs() * 20.0,
            pop.individuals[i].fit, "each Population's Individual should have a fit equal to AUC - Individual.k * k_penalty - (AUC - AUC_test * overfit_penalty");
            assert_eq!(pop.individuals[i].fit, pop_clone.individuals[i].fit, 
            "mono- and multi- thread should lead to the same final result");
        }
        
        pop_1ind.auc_nooverfit_fit(&data_disc, 10.0, &data_valid, 20.0, 1, false);
        assert_eq!(pop_1ind.individuals[0].fit, pop.individuals[0].fit, 
        "fitting should work for a Population composed of only one Individual");
    }

    #[test]
    fn test_auc_fit_vs_aucnooverfit_fit() {
        let data_disc = create_test_data_disc();
        let data_valid = create_test_data_valid();
        let mut pop1 = create_test_population();
        let mut pop2 = create_test_population();

        pop1.auc_fit(&data_disc, 10.0, 4, false);
        pop2.auc_nooverfit_fit(&data_disc, 10.0, &data_valid, 20.0, 4, false);

        for i in 1..pop1.individuals.len(){
            assert!(pop1.individuals[i].fit >= pop2.individuals[i].fit, "reducing the overfit should lead to better fit")
        }
    }

    #[test]
    fn test_objective_fit() {
        let data = &&create_test_data_disc();
        let mut pop = create_test_population();
        let mut pop_clone = create_test_population();
        let mut pop_1ind = Population::new();
        pop_1ind.individuals.push(pop.individuals[0].clone());

        pop.objective_fit(&data, 1.0, 1.0, 10.0, 4, false);
        pop_clone.objective_fit(&data, 1.0, 1.0, 10.0, 1, false);
        
        for i in 0..pop.individuals.len(){
            assert_eq!(pop.individuals[i].maximize_objective(&data, 1.0, 1.0) - pop.individuals[i].k as f64 * 10.0, pop.individuals[i].fit, 
            "bad calculation for objective_fit()");
            assert_eq!(pop.individuals[i].fit, pop_clone.individuals[i].fit, 
            "mono- and multi- thread should lead to the same final result");
        }
        
        pop_1ind.objective_fit(&data, 1.0, 1.0, 10.0, 4, false);
        assert_eq!(pop_1ind.individuals[0].fit, pop.individuals[0].fit, 
        "fitting should work for a Population composed of only one Individual");
    }

    #[test]
    fn test_objective_nooverfit_fit() {
        let data_disc = create_test_data_disc();
        let data_valid = create_test_data_valid();
        let mut pop = create_test_population();
        let mut pop_clone = create_test_population();
        let mut pop_1ind = Population::new();
        pop_1ind.individuals.push(pop.individuals[0].clone());

        pop.objective_nooverfit_fit(&data_disc, 10.0, 1.0, 10.0, &data_valid, 20.0, 4, false);
        pop_clone.objective_nooverfit_fit(&data_disc, 10.0, 1.0, 10.0, &data_valid, 20.0, 1, false);
        
        for i in 0..pop.individuals.len(){
            //the below test should work ?
            //assert_eq!(pop.individuals[i].maximize_objective(&data_disc, 1.0, 1.0) - pop.individuals[i].k as f64 * 10.0 - (pop.individuals[i].maximize_objective(&data_disc, 1.0, 1.0) - pop.individuals[i].maximize_objective(&data_valid, 1.0, 1.0)).abs() * 20.0,
            //pop.individuals[i].fit, "bad calculation for test_objective_nooverfit_fit()");
            assert_eq!(pop.individuals[i].fit, pop_clone.individuals[i].fit, 
            "mono- and multi- thread should lead to the same final result");
        }
        
        pop_1ind.objective_nooverfit_fit(&data_disc, 10.0, 1.0, 10.0, &data_valid, 20.0, 1, false);
        assert_eq!(pop_1ind.individuals[0].fit, pop.individuals[0].fit, 
        "fitting should work for a Population composed of only one Individual");
    }

    #[test]
    fn test_objective_fit_vs_objective_nooverfit_fit() {
        let data_disc = create_test_data_disc();
        let data_valid = create_test_data_valid();
        let mut pop1 = create_test_population();
        let mut pop2 = create_test_population();

        pop1.objective_fit(&data_disc, 1.0, 1.0, 10.0, 4, false);
        pop2.objective_nooverfit_fit(&data_disc, 1.0, 1.0, 10.0, &data_valid, 20.0, 4, false);

        for i in 1..pop1.individuals.len(){
            assert!(pop1.individuals[i].fit >= pop2.individuals[i].fit, "reducing the overfit should lead to better fit")
        }
    }

    // smaller, better ? 
    #[test]
    fn test_sort() {
        let mut pop = create_test_population();
        let mut previous_fit = MAX as f64;
        pop = pop.sort();
        for i in 1..pop.individuals.len(){
            assert!(pop.individuals[i].fit <= previous_fit);
            previous_fit = pop.individuals[i].fit;
        }
    }

    #[test]
    fn test_generate() {
        let data = &create_test_data_disc();
        let mut pop = Population::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let expected_features:Vec<(usize, i8)> = vec![(1, 1), (1, 1), (0, -1), (1, 1), (0, -1), (0, -1), (0, -1), (1, 1), (1, 1), (0, -1)];

        pop.generate(10, 1, 3, TERNARY_LANG, RAW_TYPE, DEFAULT_MINIMUM, data, &mut rng);
        assert_eq!(pop.individuals.len(), 10, "generated Population should be composed of population_size Individuals");

        for i in 0..pop.individuals.len(){
            assert_eq!(pop.individuals[i].language, TERNARY_LANG, "each Individual of the generated Population should respect the input language");
            assert_eq!(pop.individuals[i].data_type, RAW_TYPE, "each Individual of the generated Population should respect the input data_type");
            assert_eq!(pop.individuals[i].epsilon, DEFAULT_MINIMUM, "each Individual of the generated Population should respect the input epsilon");
            assert!(pop.individuals[i].k >= 1, "each Individual of the generated Population should have k_min or more k");
            assert!(pop.individuals[i].k <= 3, "each Individual of the generated Population should have k_max or less k");
            assert_eq!(expected_features.get(i).copied(), pop.individuals[i].features.iter().next().map(|(&k, &v)| (k, v)),
            "the generated Individual' features part of the generated Population are not the same as generated in the past for a same seed, indicating a reproducibility problem");
        }

        let mut empty_pop = Population::new();
        empty_pop.generate(0, 1, 3, TERNARY_LANG, RAW_TYPE, DEFAULT_MINIMUM, data, &mut rng);
        assert_eq!(empty_pop.individuals.len(), 0, "generating an empty population (population_size=0) should return an empty population");
    }
    
    #[test]
    fn test_add(){
        let mut pop = create_test_population();
        let mut pop_to_add= Population::new();
        let ind1 = Individual  {features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(), auc: 0.4, fit: 0.8, 
            specificity: 0.15, sensitivity:0.16, accuracy: 0.23, threshold: 42.0, k: 42, epoch:42, language: 0, data_type: 0, hash: 0, 
            epsilon: f64::MIN_POSITIVE, parents: None, betas: None};
        let ind2 = Individual  {features: vec![(0, -1), (1, 1), (2, 1), (3, 1)].into_iter().collect(), auc: 0.2, fit: 0.4, 
            specificity: 0.6, sensitivity:0.8, accuracy: 0.12, threshold: 24.0, k: 48, epoch:96, language: 0, data_type: 0, hash: 0, 
            epsilon: f64::MIN_POSITIVE, parents: None, betas: None};
        let ind_vec = vec![ind1, ind2];
        pop_to_add.individuals = ind_vec.clone();

        pop.add(pop_to_add);
        assert_eq!(pop.individuals.len(), 13);

        for i in 0..1{
            assert_eq!(pop.individuals[11+i].features, ind_vec[i].features, "adding an Individual should not change its features");
            assert_eq!(pop.individuals[11+i].auc, ind_vec[i].auc, "adding an Individual should not change its auc");
            assert_eq!(pop.individuals[11+i].fit, ind_vec[i].fit, "adding an Individual should not change its fit");
            assert_eq!(pop.individuals[11+i].specificity, ind_vec[i].specificity, "adding an Individual should not change its specificity");
            assert_eq!(pop.individuals[11+i].sensitivity, ind_vec[i].sensitivity, "adding an Individual should not change its sensitivity");
            assert_eq!(pop.individuals[11+i].accuracy, ind_vec[i].accuracy, "adding an Individual should not change its accuracy");
            assert_eq!(pop.individuals[11+i].threshold, ind_vec[i].threshold, "adding an Individual should not change its threshold");
            assert_eq!(pop.individuals[11+i].language, ind_vec[i].language, "adding an Individual should not change its language");
            assert_eq!(pop.individuals[11+i].data_type, ind_vec[i].data_type, "adding an Individual should not change its data_type");
            assert_eq!(pop.individuals[11+i].hash, ind_vec[i].hash, "adding an Individual should not change its hash");
            assert_eq!(pop.individuals[11+i].epsilon, ind_vec[i].epsilon, "adding an Individual should not change its epsilon");
        }

        pop.add(Population::new());
        assert_eq!(pop.individuals.len(), 13, "populating a Population with the Individuals of an empty Population should not change it");
    }

    #[test]
    fn test_select_first_pct() {
        let mut pop = create_test_population();
        
        let (selected_pop, n) = pop.select_first_pct(0.0);
        assert_eq!(selected_pop.individuals.len(), 0, "test_select_first_pct() should return the reduced Population and its number of Individuals");
        assert_eq!(n, 0, "selecting 0% of a Population should return an empty Population");

        let (_selected_pop2, n2) = pop.select_first_pct(100.0);
        assert_eq!(n2, 11, "selecting 100% of a Population should return the same Population");

        let (_selected_pop3, n3) = pop.select_first_pct(50.0);
        assert_eq!(n3, 5, "selecting 50% of a Population of 11 Individuals should return a Population of 5 Individuals (incomplete individual is not included)");
        pop.remove_clone();
        let (_selected_pop4, n4) = pop.select_first_pct(50.0);
        assert_eq!(n4, 5, "selecting 50% of a Population of 10 Individuals should return a Population of 5 Individuals");

        let (selected_pop5, n5) = pop.select_first_pct(10.0);
        assert_eq!(n5, 1);
        assert_eq!(selected_pop5.individuals[0].accuracy, 0.23, 
        "select_first_pct() should be deterministic : selecting 1 Individual (10% of 10 Individuals) should lead to keep only the first Individual of the Population");

        // to change : return n=10 instead of n=100 (currently pop.individuals.len() != n in this case) when pct>100 or panic
        // let (selected_pop5, n5) = pop.select_first_pct(1000.0);
        // println!("{:?}", selected_pop5.individuals.len());
        // assert_eq!(n5, 100, "selecting 1000% of a Population of 10 Individuals should return a Population of 10 Individuals");
    }

    #[test]
    fn test_select_random_above_n() {
        let mut pop = create_test_population();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let selected_pop = pop.select_random_above_n(0.0, 0, &mut rng);
        assert_eq!(selected_pop.individuals.len(), 0, "test_select_first_pct() should return the reduced Population and its number of Individuals");
 
        let selected_pop2 = pop.select_random_above_n(100.0, 0, &mut rng);
        assert_eq!(selected_pop2.individuals.len(), 11, "selecting 100% of a Population should return the same Population");

        let selected_pop3 = pop.select_random_above_n(50.0, 0, &mut rng);
        assert_eq!(selected_pop3.individuals.len(), 5, "selecting 50% of a Population of 11 Individuals should return a Population of 5 Individuals (incomplete individual is not included)");
        pop.remove_clone();
        let selected_pop4 = pop.select_random_above_n(50.0, 0, &mut rng);
        assert_eq!(selected_pop4.individuals.len(), 5, "selecting 50% of a Population of 10 Individuals should return a Population of 5 Individuals");

        let selected_pop5 = pop.select_random_above_n(10.0, 0, &mut rng);
        assert_eq!(selected_pop5.individuals.len(), 1);
        assert_eq!(selected_pop5.individuals[0].accuracy, 0.32, 
        "the selected Individual is not the same as selected in the past, indicating a reproductibility problem probably linked to the seed interpretation");

        let selected_pop6 = pop.select_random_above_n(100.0, 8, &mut rng);
        assert_eq!(selected_pop6.individuals.len(), 2, "selecting 100% of a 10-Individuals Population since index 8 should return a Population fo 2 Individuals");

        let selected_pop7 = pop.select_random_above_n(100.0, 10, &mut rng);
        assert_eq!(selected_pop7.individuals.len(), 0, "selecting 100% of a 10-Individuals Population since index 10 should return a Population fo 0 Individuals");
    }

    #[test]
    fn test_get_ind_from_hash() {
        let mut pop = create_test_population();
        pop.individuals[0].hash = 42;
        assert_eq!(pop.get_ind_from_hash(3).unwrap().hash, 3, "Wrong individual selected");
        assert_eq!(pop.get_ind_from_hash(3).unwrap().auc, 0.55, "Selected individual has wrong auc");
        assert!(pop.get_ind_from_hash(155).is_none(), "Unreachable hash should return None");
    }

    #[test]
    #[should_panic(expected = "Hash should be computed to allow Individual selection")]
    fn test_get_ind_from_hash_zero() {
        let pop = create_test_population();
        assert_eq!(pop.get_ind_from_hash(3).unwrap().hash, 3, "Wrong individual selected");
    }

    // #[test]
    // fn test_compute_pop_oob_feature_importance_reproductibility() {
    //     let mut pop = create_test_population();
    //     let mut rng = ChaCha8Rng::seed_from_u64(42);
    //     let data = create_test_data_disc();
    
    //     let mut expected_map: HashMap<usize, f64> = HashMap::new();
    //     expected_map.insert(0, -0.061363636363636315);
    //     expected_map.insert(1, 0.13484848484848488);
    //     expected_map.insert(2, 3.027880976250427e-17);
    //     expected_map.insert(3, 3.027880976250427e-17);
    
    //     let result_map = pop.compute_pop_oob_feature_importance(&data, 10, &mut rng, &"mean".to_string(), false);
    
    //     // Tiny variations can be observed due to floating point and depending on the execution order of the threads
    //     let tolerance = 1e-10;
    //     for (key, expected_value) in &expected_map {
    //         let result_value = result_map.get(key).unwrap_or(&(0.0 as f64, 0.0 as f64));
    //         assert!((expected_value - result_value).abs() < tolerance,
    //                 "Mismatch at key {} indicating a reproducibility problem: expected = {:?}, found = {:?}", key, expected_value, result_value);
    //     }
    // }

    #[test]
    fn test_compute_pop_oob_feature_importance_basic() {
        let mut population = create_test_population();
        
        let data = create_test_data_disc();
        let mut rng = ChaCha8Rng::seed_from_u64(42); // Seed fixe pour reproductibilité
        
        let importance = population.compute_pop_oob_feature_importance(&data, 10, &mut rng, &ImportanceAggregation::Mean, false);
        println!("{:?}", importance);
        assert!(importance.contains_key(&0));
        assert!(importance.contains_key(&1));
        assert!(importance.contains_key(&2));
        assert!(importance.contains_key(&3));
        assert!(importance.values().any(|&val| val != (0.0, 0.0)));
    }

}