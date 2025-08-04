use rayon::prelude::*;
use crate::{data::Data, experiment::ImportanceAggregation};
use crate::population::Population;
use crate::param::Param;
use crate::utils;
use std::sync::{Arc};
use rand_chacha::ChaCha8Rng;
use log::info;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::experiment::{Importance, ImportanceCollection, ImportanceScope, ImportanceType};
use crate::ga::fit_fn;
use crate::beam;
use crate::utils::{mean_and_std, median, mad};

use std::sync::atomic::AtomicBool;

/// Cross-validation dataset implementation for machine learning workflows.
///
/// Splits data into N validation folds and creates N training subsets, each excluding
/// its corresponding validation fold. Supports stratified sampling to maintain class
/// distribution balance across folds. May contain population collections derived from algorithms.
 #[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CV {
    pub validation_folds: Vec<Data>,
    pub training_sets: Vec<Data>,
    pub fold_collections: Vec<Vec<Population>>
}

impl CV {

    /// Creates a new cross-validation instance with stratified folds.
    ///
    /// Splits the dataset into balanced validation folds, ensuring each fold
    /// maintains the same class distribution as the original dataset.
    ///
    /// # Arguments
    /// * `data` - The dataset to split into folds  
    /// * `outer_folds` - Number of validation folds to create
    /// * `rng` - Random number generator for stratified sampling
    pub fn new(data: &Data, outer_folds: usize, rng: &mut ChaCha8Rng) -> CV {
        let mut indices_class0:Vec<usize> = Vec::new();
        let mut indices_class1:Vec<usize> = Vec::new();

        for (i,f) in data.y.iter().enumerate() {
            if *f==0 { indices_class0.push(i) } else if *f==1 { indices_class1.push(i) }
        }

        let indices_class0_folds = utils::split_into_balanced_random_chunks(indices_class0, outer_folds, rng);
        let indices_class1_folds = utils::split_into_balanced_random_chunks(indices_class1, outer_folds, rng);

        let validation_folds: Vec<Data>  = indices_class0_folds.into_iter().zip(indices_class1_folds.into_iter())
                    .map( |(i1,i2)| {i1.into_iter().chain(i2).collect::<Vec<usize>>()} )
                    .map( |i| { data.subset(i) } )
                    .collect();

        let mut training_sets: Vec<Data> = Vec::new();
        for i in 0..outer_folds {
            let mut dataset = data.subset(vec![]);

            for j in 0..outer_folds {
                if j==i { continue }
                else {
                    dataset.add(&validation_folds[j]);
                }
            }

            training_sets.push(dataset);
        }

        CV {
            validation_folds: validation_folds,
            training_sets: training_sets,
            fold_collections: vec![]
        }
    }

    /// Runs an algorithm on each training fold in parallel.
    ///
    /// Executes the provided algorithm function on each fold's training data using
    /// a thread pool. Collects population results and displays progress information
    /// including AUC scores for training and validation sets.
    ///
    /// # Arguments
    /// * `algo` - Algorithm function to execute on each training fold
    /// * `param` - Parameters to pass to the algorithm
    /// * `thread_number` - Number of threads to use in the thread pool
    /// * `running` - Atomic boolean flag for early termination control
    pub fn pass<F>(&mut self, algo: F, param: &Param, thread_number: usize, running: Arc<AtomicBool>)
        where F: Fn(&mut Data, &Param, Arc<AtomicBool>) -> Vec<Population> + Send + Sync 
        {
            let thread_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread_number)
                .build()
                .unwrap();

            let collections: Vec<Vec<Population>> = thread_pool.install(|| {
                self.training_sets
                    .par_iter_mut()
                    .zip(self.validation_folds.par_iter_mut())
                    .enumerate()
                    .filter_map(|(i, (train, valid))| {
                        info!("\x1b[1;93mCompleting fold #{}...\x1b[0m", i+1);

                        let collection: Vec<Population> = algo(train, param, Arc::clone(&running));
                        
                        if collection.len() > 0 {
                            let final_population = collection.last().unwrap();

                            if final_population.individuals.len() > 0 {
                                let best_model = final_population.individuals.clone().into_iter().take(1).next().unwrap();
                                let train_auc = best_model.auc;
                                let valid_auc = best_model.compute_new_auc(valid);

                                info!(
                                    "\x1b[1;93mFold #{} completed | Best train AUC: {:.3} | Associated validation fold AUC: {:.3}\x1b[0m",
                                    i+1, train_auc, valid_auc
                                );

                                Some(collection)
                                
                            } else {
                                info!("\x1b[1;93mFold #{} skipped - no individuals found\x1b[0m", i+1);
                                Some(vec![])
                            }
                           
                        } else {
                            info!("\x1b[1;93mFold #{} skipped - algorithm did not return any populations.\x1b[0m", i+1);
                            Some(vec![])
                        }
                    })
                    .collect()
                    
            });

            self.fold_collections = collections;
        }

    /// Computes out-of-bag feature importance across cross-validation folds.
    ///
    /// Calculates feature importance for each fold's `family of best models` population and
    /// aggregates the results using the specified method (mean or median).
    /// Returns importance scores at individual, population, and collection levels.
    ///
    /// # Arguments
    /// * `cv_param` - Cross-validation parameters
    /// * `permutations` - Number of permutations for importance calculation
    /// * `main_rng` - Random number generator for permutations
    /// * `aggregation_method` - Method to aggregate importance across folds (mean/median)
    /// * `scaled_importance` - Whether to scale importance values
    /// * `cascade` - Whether to include individual-level importance scores
    ///
    /// # Returns
    /// Returns `Ok(ImportanceCollection)` containing importance scores, or `Err(String)` 
    /// if fold collections are inconsistent.
    ///
    /// # Panics
    /// Panics if `pass()` has not been called first to populate fold collections.
    ///
    /// # Errors
    /// Returns an error if the number of fold collections, training sets, and 
    /// validation folds are inconsistent.
    pub fn compute_cv_oob_feature_importance(&self, cv_param: &Param, permutations: usize, main_rng: &mut ChaCha8Rng, 
        aggregation_method: &ImportanceAggregation, scaled_importance: bool, cascade: bool) -> Result<ImportanceCollection, String> {
             
        assert!(self.fold_collections.len()>0, "No population available. Run pass() first.");
        
        if self.fold_collections.len() != self.training_sets.len() || self.fold_collections.len() != self.validation_folds.len() {
                return Err(format!(
                    "Inconsistency: {} populations, {} training sets, {} validation folds",
                    self.fold_collections.len(),
                    self.training_sets.len(),
                    self.validation_folds.len()
                ));
            }
        
        let mut importance_values: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut features_significant_observations: HashMap<usize, usize> = HashMap::new();
        let mut importances = Vec::new();
        
        for i in 0..self.fold_collections.len() {
            let fold_last_fbm = self.extract_fold_fbm(i, cv_param);      
            // Extracted FBM is already fit and sort based on validation if required      
            let importance_data = if cv_param.cv.fit_on_valid { &self.validation_folds[i] } else { &self.training_sets[i] };
            
            let fold_imp = fold_last_fbm.compute_pop_oob_feature_importance(
                importance_data,
                permutations,
                main_rng,
                aggregation_method,
                scaled_importance,
                cascade,
                Some(i),
            );
            
            for imp in &fold_imp.importances {
                match imp.scope {
                    ImportanceScope::Individual { .. } => importances.push(imp.clone()),
                    ImportanceScope::Population { .. } => {
                        importance_values
                            .entry(imp.feature_idx)
                            .or_default()
                            .push(imp.importance);
                        importances.push(imp.clone());
                        if self.training_sets[i].feature_class.contains_key(&imp.feature_idx) {
                            *features_significant_observations
                                .entry(imp.feature_idx)
                                .or_insert(0) += 1;
                        }
                    }
                    _ => {}
                }
            }
            
        }
        
        for (feature_idx, values) in importance_values {
            if values.is_empty() { continue; }
            
            let (agg_importance, agg_dispersion) = match aggregation_method {
                ImportanceAggregation::mean   => mean_and_std(&values),
                ImportanceAggregation::median => {
                    let mut buf = values.clone();
                    let med = median(&mut buf);
                    (med, mad(&buf))
                }
            };
            
            let fold_count = features_significant_observations
                .get(&feature_idx)
                .copied()
                .unwrap_or(0);
            let scope_pct = fold_count as f64 / self.fold_collections.len() as f64;
            
            let collection_importance = Importance {
                importance_type: ImportanceType::OOB,
                feature_idx,
                scope: ImportanceScope::Collection,
                aggreg_method: Some(aggregation_method.clone()),
                importance: agg_importance,
                is_scaled: scaled_importance,
                dispersion: agg_dispersion,
                scope_pct, 
                direction: None
            };
            
            importances.push(collection_importance);
        }
        
        Ok(ImportanceCollection { importances: importances })
    }

    /// Extracts the `Family of Best Models` population for a specific fold.
    ///
    /// Returns the `Family of Best Models` population from the specified fold. 
    /// For beam search algorithms, selects the N best models within the collection. 
    /// For other algorithms, returns the last population from the fold's evolution.
    /// Optionally refits the model on validation data.
    ///
    /// # Arguments
    /// * `fold_idx` - Index of the fold to extract the model from
    /// * `param` - Parameters containing algorithm type and fitting options
    ///
    /// # Returns
    /// Returns the final best model population for the specified fold.
    ///
    /// # Panics
    /// Panics if `pass()` has not been called first or if fold_idx is invalid.
    pub fn extract_fold_fbm(&self, fold_idx: usize, param: &Param) -> Population {
        assert_ne!(self.fold_collections.len(), 0, "No population available. Run pass() first.");

        let mut pop: Population;

        // For Beam, compute FBM on [kmin; kmax]
        if param.general.algo == "beam" {
            pop = beam::keep_n_best_model_within_collection(&self.fold_collections[fold_idx], param.beam.max_nb_of_models as usize);
        } else {
            pop = self.fold_collections[fold_idx].last().unwrap().clone();
        }
        
        // Fit on valid if required and sort
        if param.cv.fit_on_valid {
            fit_fn(&mut pop, &self.validation_folds[fold_idx], &mut None, &None, &None, param);
            pop = pop.sort();
        }
        
        pop

    }

    /// Gets f`Family of Best Models` from all folds and merges them into a single population.
    ///
    /// Collects the `Family of Best Models` from each fold, displays their
    /// performance information, and merges all individuals into a single population
    /// for overall analysis.
    ///
    /// # Arguments
    /// * `param` - Parameters for model extraction and display
    ///
    /// # Returns
    /// Returns a merged population containing individuals from all `Family of Best Models`.
    ///
    /// # Panics
    /// Panics if `pass()` has not been called first to populate fold collections.
    pub fn get_fbm(&self, param: &Param) -> Population {
        assert_ne!(self.fold_collections.len(), 0, "No population available. Run pass() first.");

        let mut fold_fbms: Vec<Population> = vec![];
        let mut merged_fbms = Population::new();

        for fold_idx in 0..self.training_sets.len() {
            if self.fold_collections[fold_idx].len() > 0 && self.fold_collections[fold_idx].last().unwrap().individuals.len() > 0 {
                let fbm = self.extract_fold_fbm(fold_idx, &param);

                info!("\x1b[1;93mFold #{}\x1b[0m", fold_idx+1);
                info!("{}", fbm.clone().display(&self.training_sets[fold_idx], Some(&self.validation_folds[fold_idx]), &param));

                fold_fbms.push(fbm);
            
            } else {
                info!("\x1b[1;93mFold #{}: empty population\x1b[0m", fold_idx+1);
            }
        }

        for fold_fbm in &fold_fbms {
            merged_fbms.individuals.extend(fold_fbm.individuals.clone());
        }

        merged_fbms

    }

   
    /// Returns sample identifiers for serialization optimization.
    ///
    /// Extracts sample names from training and validation sets for each fold,
    /// returning them as tuples. This enables lightweight serialization by storing
    /// only sample identifiers instead of full datasets.
    ///
    /// # Returns
    /// Returns a vector of tuples, each containing (training_sample_names, validation_sample_names)
    /// for the corresponding fold.
    pub fn get_ids(&self) -> Vec<(Vec<String>, Vec<String>)> {
        self.training_sets.iter()
        .zip(self.validation_folds.iter())
        .map(|(train_data, valid_data)| { (train_data.samples.clone(), valid_data.samples.clone()) })
        .collect()
    }

    /// Reconstructs a CV instance from sample names and fold collections.
    ///
    /// Rebuilds the cross-validation structure from the original dataset using
    /// sample name pairs and pre-existing fold collections. Validates that all
    /// sample names exist in the original dataset and that collection counts match.
    ///
    /// # Arguments
    /// * `data` - Original dataset containing all samples
    /// * `fold_train_valid_names` - Tuples of (training_names, validation_names) for each fold
    /// * `fold_collections` - Pre-existing population collections for each fold
    ///
    /// # Returns
    /// Returns `Ok(CV)` with reconstructed cross-validation structure, or `Err(String)`
    /// if reconstruction fails.
    ///
    /// # Errors
    /// * Returns error if the number of name pairs doesn't match the number of collections
    /// * Returns error if any sample name cannot be found in the original dataset
    pub fn reconstruct(data: &Data, fold_train_valid_names: Vec<(Vec<String>, Vec<String>)>, fold_collections: Vec<Vec<Population>>) -> Result<CV, String> {
        let mut validation_folds = Vec::new();
        let mut training_sets = Vec::new();
        
        if fold_train_valid_names.len() != fold_collections.len() {
            return Err(format!(
                "Mismatch: {} folds vs {} collections", 
                fold_train_valid_names.len(), 
                fold_collections.len()
            ));
        }
        
        for (train_names, test_names) in &fold_train_valid_names {
            let train_indices: Vec<usize> = train_names
                .iter()
                .map(|name| data.samples.iter().position(|s| s == name))
                .collect::<Option<Vec<_>>>()
                .ok_or("Sample name can not be found (Train)")?;
            
            let test_indices: Vec<usize> = test_names
                .iter()
                .map(|name| data.samples.iter().position(|s| s == name))
                .collect::<Option<Vec<_>>>()
                .ok_or("Sample name can not be found (Validation)")?;
            
            let training_set = data.subset(train_indices);
            let validation_fold = data.subset(test_indices);
            
            training_sets.push(training_set);
            validation_folds.push(validation_fold);
        }
        
        Ok(CV {
            validation_folds,
            training_sets,
            fold_collections: fold_collections,
        })
    }

}

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    impl CV {
        pub fn test() -> CV {
            let cv = CV {
                fold_collections: vec![
                    vec![Population::test_with_these_features(&[0, 1, 2])], 
                    vec![Population::test_with_these_features(&[1, 2, 3])], 
                    vec![Population::test_with_these_features(&[0, 2, 3])], 
                ],
                
                training_sets: vec![
                    Data::test_with_these_features(&[0, 1, 2, 3]),
                    Data::test_with_these_features(&[0, 1, 2, 3]),
                    Data::test_with_these_features(&[0, 1, 2, 3]),
                ],

                validation_folds: vec![
                    Data::test_with_these_features(&[0, 1, 2, 3]),
                    Data::test_with_these_features(&[0, 1, 2, 3]),
                    Data::test_with_these_features(&[0, 1, 2, 3]),
                ]
            };
            
            cv
        }

    }

    #[test]
    fn test_cv_new_creates_correct_number_of_folds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);
        assert_eq!(cv.validation_folds.len(), outer_folds);
        assert_eq!(cv.training_sets.len(), outer_folds);
    }

    #[test]
    fn test_cv_new_distributes_y_correctly_and_preserve_them() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);

        // Check that y are correctly splitted
        let expected_size = (data.y.len() + outer_folds - 1) / outer_folds;
        for fold in &cv.validation_folds {
            let fold_size = fold.y.len();
            assert!((fold_size as isize - expected_size as isize).abs() <= 1);
        }

        // Check that all data is preserved across all validation_folds
        let mut real_y: Vec<usize> = data.y.iter().map(|&x| x as usize).collect();
        let mut collected_y = Vec::new();
        for fold in &cv.validation_folds {
            collected_y.extend(fold.y.iter().map(|&x| x as usize));
        }
        real_y.sort();
        collected_y.sort();
 
         assert_eq!(collected_y, real_y);
    }

    // add a unit test to check if y are correctly distribued ? 

    #[test]
    fn test_cv_new_reproductibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);
        
        let mut X1: HashMap<(usize, usize), f64> = HashMap::new();
        X1.insert((1, 1), 0.91);  
        X1.insert((2, 0), 0.01);  
        
        let mut X2: HashMap<(usize, usize), f64> = HashMap::new();
        X2.insert((0, 0), 0.9); 
        X2.insert((0, 1), 0.01);
        X2.insert((1, 0), 0.12);
        X2.insert((1, 1), 0.75);
        
        let mut X3: HashMap<(usize, usize), f64> = HashMap::new();
        X3.insert((0, 1), 0.9);  
        
        
        assert_eq!(cv.validation_folds[0].X, X1);
        assert_eq!(cv.validation_folds[1].X, X2);
        assert_eq!(cv.validation_folds[2].X, X3);
        
        assert_eq!(cv.validation_folds[0].y, [0, 1, 1]);
        assert_eq!(cv.validation_folds[1].y, [0, 1]);
        assert_eq!(cv.validation_folds[2].y, [1]);
        
        assert_eq!(cv.validation_folds[0].samples, ["sample3", "sample2", "sample5"]);
        assert_eq!(cv.validation_folds[1].samples, ["sample1", "sample4"]);
        assert_eq!(cv.validation_folds[2].samples, ["sample6"]);
        
        assert_eq!(cv.validation_folds[0].sample_len, 3);
        assert_eq!(cv.validation_folds[1].sample_len, 2);
        assert_eq!(cv.validation_folds[2].sample_len, 1);

        for fold in &cv.validation_folds {
            assert_eq!(fold.features, ["feature1", "feature2"]);
            assert_eq!(fold.feature_len, 2);
        }
    }

    #[test]
    fn test_cv_new_with_single_class() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();
        data.y = vec![0; data.y.len()];  

        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);

        assert_eq!(cv.validation_folds.len(), outer_folds);
        assert_eq!(cv.training_sets.len(), outer_folds);

        for fold in &cv.validation_folds {
            for &label in &fold.y {
                assert_eq!(label, 0);
            }
        }
    }

    // For tiny dataset, a fold can have the two samples because of stratified distribution
    #[test]
    fn test_cv_new_with_very_small_dataset() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        let indices = vec![0, 1];
        data = data.subset(indices);
        
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);

        assert_eq!(cv.validation_folds.len(), outer_folds);
        assert_eq!(cv.training_sets.len(), outer_folds);

        for fold in &cv.validation_folds {
            assert!(fold.y.len() <= 2);
        }
    }

    #[test]
    fn test_get_ids_returns_correct_sample_names() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);
        
        let ids = cv.get_ids();
        
        assert_eq!(ids.len(), outer_folds);
        
        // Check get_ids main goal
        for (i, (train_names, valid_names)) in ids.iter().enumerate() {
            assert_eq!(train_names, &cv.training_sets[i].samples);
            assert_eq!(valid_names, &cv.validation_folds[i].samples);
        }
        
        // Check that all original samples are present in the validation folds
        let mut all_validation_samples: Vec<String> = Vec::new();
        for (_, valid_names) in &ids {
            all_validation_samples.extend(valid_names.clone());
        }
        all_validation_samples.sort();
        let mut original_samples = data.samples.clone();
        original_samples.sort();
        assert_eq!(all_validation_samples, original_samples);
        
        // Check that no sample appears in multiple validation folds
        let mut seen_samples = std::collections::HashSet::new();
        for (_, valid_names) in &ids {
            for sample in valid_names {
                assert!(seen_samples.insert(sample.clone()), 
                    "Sample {} appears in multiple validation folds", sample);
            }
        }
    }

    #[test]
    fn test_pass_sets_fold_collections() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 2;
        let mut cv = CV::new(&data, outer_folds, &mut rng);
        let param = Param::default();
        let running = Arc::new(AtomicBool::new(true));
        
        let mock_algo = |_train_data: &mut Data, _param: &Param, _running: Arc<AtomicBool>| -> Vec<Population> {
            let mut pop = Population::test();
            pop.compute_hash();
            vec![pop]
        };
        
        cv.pass(mock_algo, &param, 1, running);
        
        assert!(cv.fold_collections.len() > 0);
        assert!(cv.fold_collections.len() <= outer_folds);
    }

    #[test] 
    fn test_pass_handles_empty_populations() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let mut cv = CV::new(&data, 2, &mut rng);
        let param = Param::default();
        let running = Arc::new(AtomicBool::new(true));
        
        let mock_algo = |_train_data: &mut Data, _param: &Param, _running: Arc<AtomicBool>| -> Vec<Population> {
            vec![Population::new()]
        };
        
        cv.pass(mock_algo, &param, 1, running);
        
        assert!(cv.fold_collections.len() > 0);
        let collections = cv.fold_collections;
        assert_eq!(collections[0].len(), 0); 
        assert_eq!(collections[1].len(), 0);
    }

    #[test] 
    fn test_pass_with_algorithm_timeout() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 2;
        let mut cv = CV::new(&data, outer_folds, &mut rng);
        let param = Param::default();

        // Create a flag that is initially true
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        // Mock algorithm that sets running to false (simulate an interruption)
        let mock_algo = |_train_data: &mut Data, _param: &Param, running: Arc<AtomicBool>| -> Vec<Population> {
            running.store(false, std::sync::atomic::Ordering::Relaxed);
            vec![Population::test()]
        };

        cv.pass(mock_algo, &param, 1, running_clone);

        // Check that fold_collections is defined
        assert!(cv.fold_collections.len() > 0);
    }

    // CV test with extreme class imbalance
    #[test]
    fn test_cv_new_class_imbalance_extreme() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        // Define 95% of samples as class 0, 5% as class 1
        let n = data.y.len();
        let n_class1 = (n as f64 * 0.05).round() as usize;
        data.y = vec![0; n];
        for i in 0..n_class1 {
            data.y[i] = 1;
        }

        let outer_folds = 5;
        let cv = CV::new(&data, outer_folds, &mut rng);

        // Each fold should contain a few class 1 samples or be empty if there are too few.
        for fold in &cv.validation_folds {
            let count_class1 = fold.y.iter().filter(|&&y| y == 1).count();
            assert!(count_class1 <= n_class1);
        }

        // Check that the overall distribution is preserved
        let total_class1: usize = cv.validation_folds.iter()
            .map(|fold| fold.y.iter().filter(|&&y| y == 1).count())
            .sum();
        assert_eq!(total_class1, n_class1);
    }

    #[test]
    fn test_reconstruct_success() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 2;
        let cv = CV::new(&data, outer_folds, &mut rng);
        
        let mock_collections = vec![
            vec![Population::test()], 
            vec![Population::test()],
        ];
        
        let fold_names = cv.get_ids();
        let reconstructed_cv = CV::reconstruct(&data, fold_names, mock_collections);
        
        assert!(reconstructed_cv.is_ok());
        let reconstructed = reconstructed_cv.unwrap();
        
        assert_eq!(reconstructed.training_sets.len(), outer_folds);
        assert_eq!(reconstructed.validation_folds.len(), outer_folds);
        assert_eq!(reconstructed.fold_collections.len(), outer_folds);
    }

    #[test]
    fn test_reconstruct_fails_with_mismatched_lengths() {
        let data = Data::test();
        let fold_names = vec![(vec!["sample1".to_string()], vec!["sample2".to_string()])];
        let mock_collections = vec![vec![Population::test()],  vec![Population::test()]]; 
        
        let result = CV::reconstruct(&data, fold_names, mock_collections);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Mismatch"));
    }

    #[test]
    fn test_reconstruct_fails_with_unknown_sample_name() {
        let data =Data::test();
        let fold_names = vec![
            (vec!["unknown_sample".to_string()], vec!["sample1".to_string()]),
        ];
        let mock_populations = vec![vec![Population::test()]];
        
        let result = CV::reconstruct(&data, fold_names, mock_populations);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Sample name can not be found"));
    }

    // #[test]
    // fn test_extract_fold_fbm_without_fit_on_valid() {
    //     let mut rng = ChaCha8Rng::seed_from_u64(42);
    //     let data = Data::test();
    //     let outer_folds = 2;
    //     let mut cv = CV::new(&data, outer_folds, &mut rng);
        
    //     let mock_collections = vec![
    //         vec![Population::test()], 
    //         vec![Population::test()],
    //     ];
    //     cv.fold_collections = mock_collections;
        
    //     let mut param = Param::default();
    //     param.cv.fit_on_valid = false;
    //     param.cv.cv_best_models_ci_alpha = 0.05;
        
    //     let (_, valid_data) = cv.extract_fold_fbm(0, &param);
        
    //     assert_eq!(valid_data.samples, cv.validation_folds[0].samples);
    //     assert_eq!(valid_data.y, cv.validation_folds[0].y);
    // }

    #[test]
    #[should_panic(expected = "No population available")]
    fn test_extract_fold_fbm_panics_without_populations() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let cv = CV::new(&data, 2, &mut rng);
        let param = Param::default();
        cv.extract_fold_fbm(0, &param);
    }

    #[test]
    #[should_panic(expected = "No population available")]
    fn test_compute_cv_oob_feature_importance_missing_fold_collections() {
        let mut cv = CV::test();
        cv.fold_collections = vec![];
        
        let cv_param = Param::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let _ = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng, 
            &ImportanceAggregation::mean, false, false
        );
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_inconsistent_fold_sizes() {
        let mut cv = CV::test();
        cv.fold_collections = vec![vec![Population::test_with_these_features(&[0, 1])]];
        cv.training_sets = vec![Data::test_with_these_features(&[0, 1])];
        cv.validation_folds = vec![
            Data::test_with_these_features(&[0, 1]),
            Data::test_with_these_features(&[0, 1]), // Unconsistent size
        ];
        
        let cv_param = Param::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let result = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng,
            &ImportanceAggregation::mean, false, false
        );
        
        assert!(result.is_err());
        assert!(result.clone().unwrap_err().contains("Inconsistency"));
        assert!(result.unwrap_err().contains("1 populations, 1 training sets, 2 validation folds"));
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_aggregation_across_folds_mean() {
        use crate::ga;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::specific_test(40, 20);
        let mut cv = CV::new(&data, 3, &mut rng);
        let r =  Arc::new(AtomicBool::new(true));
        let mut cv_param = Param::default();
        cv_param.ga.max_epochs = 3;
        cv_param.data.feature_maximal_pvalue = 1.0;
        cv.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| {
            match p.general.algo.as_str() {
                "ga" => ga::ga(d, &mut None, &cv_param, r),
                _ => panic!("Such algorithm is not useful for the test."),
            }
        }, &cv_param, cv_param.general.thread_number, r);
        
        let result = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng,
            &ImportanceAggregation::mean, false, false).unwrap();
        
        // Check that we have (aggregated) Collection weights
        let collection_importances: Vec<_> = result.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .collect();
        
        assert!(!collection_importances.is_empty(), "Should have Collection-level importances");
        
         // Check that the aggregation is marked as mean
        for imp in &collection_importances {
            assert_eq!(imp.aggreg_method, Some(ImportanceAggregation::mean));
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_aggregation_across_folds_median() {
        let cv = CV::test();
        let cv_param = Param::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let result = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng,
            &ImportanceAggregation::median, false, false
        ).unwrap();
        
        let collection_importances: Vec<_> = result.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .collect();
        
        // Check that the aggregation is marked as median
        for imp in &collection_importances {
            assert_eq!(imp.aggreg_method, Some(ImportanceAggregation::median));
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_scope_pct_calculation() {
        let cv = CV::test();
        let cv_param = Param::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let result = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng,
            &ImportanceAggregation::mean, false, false
        ).unwrap();
        
        let collection_importances: Vec<_> = result.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .collect();
        
        for imp in &collection_importances {
            assert!(imp.scope_pct >= 0.0 && imp.scope_pct <= 1.0, 
                "scope_pct should be between 0.0 and 1.0, got {}", imp.scope_pct);
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_cascade_mode_includes_individual_importances() {
        let cv = CV::test();
        let cv_param = Param::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        // cascade = true
        let result_cascade = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng,
            &ImportanceAggregation::mean, false, true
        ).unwrap();
        
        // cascade = false  
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let result_no_cascade = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng2,
            &ImportanceAggregation::mean, false, false
        ).unwrap();
        
        let individual_count_cascade = result_cascade.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Individual { .. }))
            .count();
            
        let individual_count_no_cascade = result_no_cascade.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Individual { .. }))
            .count();
        
        if individual_count_cascade > 0 {
            assert!(individual_count_cascade >= individual_count_no_cascade,
                "Cascade mode should include individual importances");
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_on_validation_vs_training_data() {
        use crate::ga;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::specific_test(40, 20);
        let mut cv = CV::new(&data, 3, &mut rng);
        let r =  Arc::new(AtomicBool::new(true));
        let mut cv_param = Param::default();
        cv_param.ga.max_epochs = 3;
        cv_param.data.feature_maximal_pvalue = 1.0;
        cv.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| {
            match p.general.algo.as_str() {
                "ga" => ga::ga(d, &mut None, &cv_param, r),
                _ => panic!("Such algorithm is not useful for the test."),
            }
        }, &cv_param, cv_param.general.thread_number, r);
        
        // on_validation = true
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        
        cv_param.cv.fit_on_valid = true;
        let result_validation = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng1,
            &ImportanceAggregation::mean, false, false
        ).unwrap();
        
        cv_param.cv.fit_on_valid = false;
        // on_validation = false
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let result_training = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng2,
            &ImportanceAggregation::mean, false, false
        ).unwrap();
        
        // Both should succeed and achieve results.
        assert!(!result_validation.importances.is_empty());
        assert!(!result_training.importances.is_empty());
        
        // The amounts may differ depending on the data used.
        // We are then just checking that the structures are consistent.
        let validation_collection_count = result_validation.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .count();
            
        let training_collection_count = result_training.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .count();
        
        // We can expect to have Collection importances in both cases.
        assert!(validation_collection_count > 0 || training_collection_count > 0);
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_scaled_importance_flag() {
        let cv = CV::test();
        let cv_param = Param::default();
        
        // scaled = true
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let result_scaled = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng1,
            &ImportanceAggregation::mean, true, false
        ).unwrap();
        
        // scaled = false
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let result_unscaled = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng2,
            &ImportanceAggregation::mean, false, false
        ).unwrap();
        
        // Check that the is_scaled flag is correctly propagated
        for imp in &result_scaled.importances {
            if matches!(imp.scope, ImportanceScope::Collection) {
                assert!(imp.is_scaled, "Collection importance should be marked as scaled");
            }
        }
        
        for imp in &result_unscaled.importances {
            if matches!(imp.scope, ImportanceScope::Collection) {
                assert!(!imp.is_scaled, "Collection importance should not be marked as scaled");
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_compute_cv_oob_feature_importance_empty_fold_collections() {
        let mut cv = CV::test();
        cv.fold_collections = vec![vec![]];
        cv.training_sets = vec![];
        cv.validation_folds = vec![];
        
        let cv_param = Param::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let result = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng,
            &ImportanceAggregation::mean, false, false
        ).unwrap();
        
        assert!(result.importances.is_empty(), "Empty folds should produce no importances");
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_feature_filtering_in_aggregation() {
        let cv = CV::test();
        let cv_param = Param::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let result = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng,
            &ImportanceAggregation::mean, false, false
        ).unwrap();
        
        // Check that no importance Collection has empty values
        let collection_importances: Vec<_> = result.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .collect();
        
        for imp in &collection_importances {
            // All Collection amounts must have been calculated 
            // from non-empty values.
            assert!(!imp.importance.is_nan(), 
                "Collection importance should not be NaN for feature {}", imp.feature_idx);
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_population_id_assignment() {
        let cv = CV::test();
        let cv_param = Param::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let result = cv.compute_cv_oob_feature_importance(
            &cv_param, 5, &mut rng,
            &ImportanceAggregation::mean, false, false
        ).unwrap();
        
        // Check that Population amounts have correct IDs
        let population_importances: Vec<_> = result.importances.iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Population { .. }))
            .collect();
        
        for imp in &population_importances {
            if let ImportanceScope::Population { id } = imp.scope {
                assert!(id < cv.fold_collections.len(), 
                    "Population ID {} should be less than number of folds {}", 
                    id, cv.fold_collections.len());
            }
        }
    }

    #[test]
    fn test_full_cv_workflow() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 2;
        let mut cv = CV::new(&data, outer_folds, &mut rng);
        let param = Param::default();
        let running = Arc::new(AtomicBool::new(true));

        // Mock algorithm returing a test population
        let mock_algo = |_train_data: &mut Data, _param: &Param, _running: Arc<AtomicBool>| -> Vec<Population> {
            let mut pop = Population::test();
            pop.compute_hash();
            vec![pop]
        };

        cv.pass(mock_algo, &param, 1, Arc::clone(&running));
        assert!(cv.fold_collections.len() > 0);

        let importance_collection = cv.compute_cv_oob_feature_importance(
            &param, 1, &mut rng, &ImportanceAggregation::mean, false, false
        ).unwrap();

        assert!(!importance_collection.importances.is_empty());

        let fbm = cv.extract_fold_fbm(0, &param);
        assert!(!fbm.individuals.is_empty());

        let fbmMerged = cv.get_fbm(&param);
        assert!(!fbmMerged.individuals.is_empty());
        assert!(fbmMerged.individuals.contains(&fbm.individuals[0]))
    }

    #[test]
    fn test_get_ids_reconstruct_consistency() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);

        let fold_ids = cv.get_ids();

        let mock_collections = vec![
            vec![Population::test()], 
            vec![Population::test()],
            vec![Population::test()],
        ];

        let reconstructed_cv = CV::reconstruct(&data, fold_ids, mock_collections).unwrap();

        assert_eq!(cv.validation_folds.len(), reconstructed_cv.validation_folds.len());
        assert_eq!(cv.training_sets.len(), reconstructed_cv.training_sets.len());

        for i in 0..outer_folds {
            assert_eq!(cv.validation_folds[i].samples, reconstructed_cv.validation_folds[i].samples);
            assert_eq!(cv.training_sets[i].samples, reconstructed_cv.training_sets[i].samples);
        }
    }

    #[test]
    fn test_training_validation_no_overlap() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);

        for i in 0..outer_folds {
            let training = &cv.training_sets[i].samples;
            let validation = &cv.validation_folds[i].samples;

            for sample in training {
                assert!(!validation.contains(sample), 
                    "Sample '{}' found in both training and validation for fold {}", sample, i);
            }
        }
    }

    #[test]
    fn test_fold_feature_consistency() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);

        let reference_features = &data.features;
        for (i, fold) in cv.validation_folds.iter().chain(cv.training_sets.iter()).enumerate() {
            assert_eq!(&fold.features, reference_features, 
                "Features mismatch in fold {}", i);
            assert_eq!(fold.feature_len, data.feature_len,
                "Feature length mismatch in fold {}", i);
        }
    }

    #[test]
    fn test_data_size_preservation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data: Data = Data::test();
        let folds = 4;
        let cv = CV::new(&data, folds, &mut rng);

        // Check that the sum of the sizes of the validation folds equals the original size
        let total_validation_size: usize = cv.validation_folds.iter()
            .map(|fold| fold.sample_len)
            .sum();
        assert_eq!(total_validation_size, data.sample_len);

        // Check that each drive set is the correct size
        for (i, training_set) in cv.training_sets.iter().enumerate() {
            assert_eq!(training_set.sample_len, data.sample_len - cv.validation_folds[i].sample_len);
        }
    }

    // Test for balanced class distribution
    #[test]
    fn test_balanced_class_distribution() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::specific_test(30, 50);
        let outer_folds = 3;
        let cv = CV::new(&data, outer_folds, &mut rng);

        // Count the classes in the original data
        let original_class0 = data.y.iter().filter(|&&y| y == 0).count();
        let original_class1 = data.y.iter().filter(|&&y| y == 1).count();

        // Check the distribution in each fold
        for (i, fold) in cv.validation_folds.iter().enumerate() {
            let fold_class0 = fold.y.iter().filter(|&&y| y == 0).count();
            let fold_class1 = fold.y.iter().filter(|&&y| y == 1).count();

            // The distribution should be roughly balanced
            let expected_class0 = (original_class0 + outer_folds - 1) / outer_folds;
            let expected_class1 = (original_class1 + outer_folds - 1) / outer_folds;

            assert!((fold_class0 as isize - expected_class0 as isize).abs() <= 1,
                "Class 0 distribution imbalanced in fold {}: expected ~{}, got {}", 
                i, expected_class0, fold_class0);
            assert!((fold_class1 as isize - expected_class1 as isize).abs() <= 1,
                "Class 1 distribution imbalanced in fold {}: expected ~{}, got {}", 
                i, expected_class1, fold_class1);
        }
    }

    #[test]
    fn test_cv_new_single_fold() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let outer_folds = 1;
        let cv = CV::new(&data, outer_folds, &mut rng);

        assert_eq!(cv.validation_folds.len(), 1);
        assert_eq!(cv.training_sets.len(), 1);

        // The validation fold should contain all the data.
        assert_eq!(cv.validation_folds[0].sample_len, data.sample_len);
        
        // The training set should be empty.
        assert_eq!(cv.training_sets[0].sample_len, 0);
    }

}