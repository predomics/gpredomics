use rayon::prelude::*;
use crate::{data::Data, param::ImportanceAggregation};
use crate::population::Population;
use crate::param::Param;
use crate::utils;
use std::sync::{Arc, Mutex};
use rand_chacha::ChaCha8Rng;
use log::info;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::experiment::{Importance, ImportanceCollection, ImportanceScope, ImportanceType};
use crate::ga::fit_fn;
use crate::utils::{mean_and_std, median, mad};

use std::sync::atomic::AtomicBool;

/// This class implement Cross Validation dataset, e.g. split the Data in N validation_folds and create N subset of Data each with its test subset.
 #[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CV {
    pub validation_folds: Vec<Data>,
    pub training_sets: Vec<Data>
}

impl CV {
    pub fn new(data: &Data, fold_number: usize, rng: &mut ChaCha8Rng) -> CV {
        let mut indices_class0:Vec<usize> = Vec::new();
        let mut indices_class1:Vec<usize> = Vec::new();

        for (i,f) in data.y.iter().enumerate() {
            if *f==0 { indices_class0.push(i) } else if *f==1 { indices_class1.push(i) }
        }

        let indices_class0_folds = utils::split_into_balanced_random_chunks(indices_class0, fold_number, rng);
        let indices_class1_folds = utils::split_into_balanced_random_chunks(indices_class1, fold_number, rng);

        let validation_folds: Vec<Data>  = indices_class0_folds.into_iter().zip(indices_class1_folds.into_iter())
                    .map( |(i1,i2)| {i1.into_iter().chain(i2).collect::<Vec<usize>>()} )
                    .map( |i| { data.subset(i) } )
                    .collect();

        let mut training_sets: Vec<Data> = Vec::new();
        for i in 0..fold_number {
            let mut dataset: Data = if i==0 { validation_folds[1].clone() } else { validation_folds[0].clone() };

            for j in 1..fold_number {
                if j==i { continue }
                else {
                    dataset.add(&validation_folds[j]);
                }
            }

            training_sets.push(dataset);
        }

        CV {
            validation_folds: validation_folds,
            training_sets: training_sets
        }
    }

    pub fn pass<F>(
        &mut self,
        algo: F,
        param: &Param,
        thread_number: usize,
        running: Arc<AtomicBool>
    ) -> Vec<(Population, Data, Data)>
    where F: Fn(&mut Data, &Param, Arc<AtomicBool>) -> Vec<Population> + std::marker::Send + std::marker::Sync, {
        // Configure the thread pool with the specified thread number
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_number)
            .build()
            .unwrap();

        // Arc-Mutex to collect results from threads safely
        let results_per_fold = Arc::new(Mutex::new(Vec::new()));

        thread_pool.install(|| {
            self.training_sets
                .par_iter_mut()
                .zip(self.validation_folds.par_iter_mut())
                .enumerate()
                .for_each(|(i, (train, test))| {
                    // Train and evaluate model
                    info!("\x1b[1;93mCompleting fold #{}...\x1b[0m", i+1);

                    let last_generation: Population =
                        algo(train, param, Arc::clone(&running)).pop().unwrap();
                    let best_model = last_generation.clone().individuals.into_iter().take(1).next().unwrap();
                    let train_auc = best_model.auc;
                    let test_auc = best_model.compute_new_auc(test);

                    info!(
                        "\x1b[1;93mFold #{} completed | Best train AUC: {:.3} | Associated validation fold AUC: {:.3}\x1b[0m",
                        i+1, train_auc, test_auc
                    );

                    // Store the results
                    results_per_fold
                        .lock()
                        .unwrap()
                        .push((last_generation, train.clone(), test.clone()));
                });
        });

        // Extract results from the Arc-Mutex
        Arc::try_unwrap(results_per_fold)
            .unwrap()
            .into_inner()
            .unwrap()
    }

    pub fn compute_importance_from_cv_results(&self, cv_results: Vec<(Population, Data, Data)>, cv_param: &Param, ci_alpha: f64, permutations: usize, main_rng: &mut ChaCha8Rng, aggregation_method: &ImportanceAggregation, scaled_importance: bool, cascade: bool) -> (ImportanceCollection, Population) {
        let mut cv_fbm_pop = Population::new();
        let mut importance_values: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut features_significant_observations: HashMap<usize, usize> = HashMap::new();
        let mut features_classes: HashMap<usize, i8> = HashMap::new();
        
        let mut importances = Vec::new();
        for (i, (mut fold_last_population, train, mut test)) in cv_results.clone().into_iter().enumerate() {
            fit_fn(&mut fold_last_population, &mut test, &mut None, &None, &None, cv_param);
            
            let mut fold_last_fbm = fold_last_population.select_best_population(ci_alpha);
            fold_last_fbm = fold_last_fbm.sort();
            
            let fold_importance_collection = fold_last_fbm.compute_pop_oob_feature_importance(
                &self.validation_folds[i],
                permutations,
                main_rng,
                aggregation_method,
                scaled_importance,
                cascade,
                Some(i), // fold ID
            );
            
            for importance in &fold_importance_collection.importances {
                match importance.scope {
                    ImportanceScope::Collection { .. } => { panic!("CV.compute_importance_from_cv_results: You should not be here!") }
                    ImportanceScope::Individual { .. } => { importances.push(importance.clone())}
                    ImportanceScope::Population { .. } => {
                    importance_values
                        .entry(importance.feature_idx)
                        .or_insert_with(Vec::new)
                        .push(importance.importance);
                    
                    importances.push(importance.clone());

                    if train.feature_class.contains_key(&importance.feature_idx) {
                        *features_significant_observations
                            .entry(importance.feature_idx)
                            .or_insert(0) += 1;
                        
                        let associated_class = train.feature_class[&importance.feature_idx];
                        let class_value = if associated_class == 0 { 
                            -1 
                        } else if associated_class == 1 { 
                            1 
                        } else { 
                            0 
                        };
                        features_classes.insert(importance.feature_idx, class_value);
                    }      
                }
                }
            }
            cv_fbm_pop.individuals.extend(fold_last_fbm.individuals);
        }
        
        let total_folds = cv_results.len();
        
        for (feature_idx, values) in importance_values {
            if values.is_empty() { continue; }
            
            let (agg_importance, agg_dispersion) = match aggregation_method {
                ImportanceAggregation::Mean   => mean_and_std(&values),
                ImportanceAggregation::Median => {
                    let mut buf = values.clone();
                    let med = median(&mut buf);
                    (med, mad(&buf))
                }
            };
            
            let fold_count = features_significant_observations
                .get(&feature_idx)
                .copied()
                .unwrap_or(0);
            let scope_pct = fold_count as f64 / total_folds as f64;
            
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
        
        (
            ImportanceCollection { 
                importances: importances 
            }, 
            cv_fbm_pop
        )
    }
}

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    fn create_test_data() -> Data {
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        let mut feature_class: HashMap<usize, u8> = HashMap::new();

        // Simulate data
        X.insert((0, 0), 0.1);
        X.insert((0, 1), 0.4);
        X.insert((1, 0), 0.2);
        X.insert((1, 1), 0.5);
        X.insert((2, 0), 0.3);
        X.insert((2, 1), 0.6);
        X.insert((3, 0), 0.7);
        X.insert((3, 1), 0.8);
        X.insert((4, 0), 0.1);
        X.insert((4, 1), 0.2);
        X.insert((5, 0), 0.9);
        X.insert((5, 1), 0.8);
        feature_class.insert(0, 0);
        feature_class.insert(1, 1);

        Data {
            X,
            y: vec![0, 1, 0, 1, 1, 1],
            features: vec!["feature1".to_string(), "feature2".to_string()],
            samples: vec!["sample1".to_string(), "sample2".to_string(), "sample3".to_string(), "sample4".to_string(), "sample5".to_string(), "sample6".to_string()],
            feature_class,
            feature_selection: vec![0, 1],
            feature_len: 2,
            sample_len: 6,
            classes: vec!["a".to_string(),"b".to_string()]
        }
    }

    #[test]
    fn test_cv_new_creates_correct_number_of_folds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = create_test_data();
        let fold_number = 3;
        let cv = CV::new(&data, fold_number, &mut rng);
        assert_eq!(cv.validation_folds.len(), fold_number);
        assert_eq!(cv.training_sets.len(), fold_number);
    }

    #[test]
    fn test_cv_new_distributes_y_correctly_and_preserve_them() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = create_test_data();
        let fold_number = 3;
        let cv = CV::new(&data, fold_number, &mut rng);

        // Check that y are correctly splitted
        let expected_size = (data.y.len() + fold_number - 1) / fold_number;
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
        let data = create_test_data();
        let fold_number = 3;
        let cv = CV::new(&data, fold_number, &mut rng);
        
        let mut X1: HashMap<(usize, usize), f64> = HashMap::new();
        let mut X2: HashMap<(usize, usize), f64> = HashMap::new();
        let mut X3: HashMap<(usize, usize), f64> = HashMap::new();

        X1.insert((0, 0), 0.3);
        X1.insert((0, 1), 0.6);
        X1.insert((1, 0), 0.2);
        X1.insert((1, 1), 0.5);
        X1.insert((2, 0), 0.1);
        X1.insert((2, 1), 0.2);
        X2.insert((0, 0), 0.1);
        X2.insert((0, 1), 0.4);
        X2.insert((1, 0), 0.7);
        X2.insert((1, 1), 0.8);
        X3.insert((0, 0), 0.9);
        X3.insert((0, 1), 0.8);

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

    // tests for pass to add

}