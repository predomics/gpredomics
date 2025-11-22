use crate::beam;
use crate::cinfo;
use crate::experiment::{Importance, ImportanceCollection, ImportanceScope, ImportanceType};
use crate::param::Param;
use crate::population::Population;
use crate::utils;
use crate::utils::{mad, mean_and_std, median, stratify_by_annotation};
use crate::{data::Data, experiment::ImportanceAggregation};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

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
    pub fold_collections: Vec<Vec<Population>>,
}

impl CV {
    /// Creates a new cross-validation instance with stratified folds.
    ///
    /// Splits the dataset into balanced validation folds, ensuring each fold
    /// maintains the same class distribution as the original dataset.
    ///
    /// # Arguments
    /// * `data` - The dataset to split into folds  
    /// * `folds` - Number of validation folds to create
    /// * `rng` - Random number generator for stratified sampling
    pub fn new(data: &Data, folds: usize, rng: &mut ChaCha8Rng) -> CV {
        let (indices_class1, indices_class0) = utils::stratify_indices_by_class(&data.y);

        let indices_class0_folds =
            utils::split_into_balanced_random_chunks(indices_class0, folds, rng);
        let indices_class1_folds =
            utils::split_into_balanced_random_chunks(indices_class1, folds, rng);

        let validation_folds: Vec<Data> = indices_class0_folds
            .into_iter()
            .zip(indices_class1_folds.into_iter())
            .map(|(i1, i2)| i1.into_iter().chain(i2).collect::<Vec<usize>>())
            .map(|i| data.subset(i))
            .collect();

        let mut training_sets: Vec<Data> = Vec::new();
        for i in 0..folds {
            let mut dataset = data.subset(vec![]);

            for j in 0..folds {
                if j == i {
                    continue;
                } else {
                    dataset.add(&validation_folds[j]);
                }
            }

            training_sets.push(dataset);
        }

        CV {
            validation_folds: validation_folds,
            training_sets: training_sets,
            fold_collections: vec![],
        }
    }

    /// Creates a new cross-validation instance with double stratification.
    ///
    /// Splits the dataset into balanced validation folds, ensuring each fold
    /// maintains both the class distribution (y) and the sample annotation distribution.
    /// This performs a two-level stratification: first by class label (y), then by
    /// the specified annotation column within each class.
    ///
    /// # Arguments
    /// * `data` - The dataset to split into folds  
    /// * `folds` - Number of validation folds to create
    /// * `rng` - Random number generator for stratified sampling
    /// * `stratify_by` - Column name in sample annotations to use for secondary stratification
    ///
    /// # Panics
    /// Panics if the stratify_by column is not found in sample annotations or if
    /// sample annotations are not available in the dataset.
    pub fn new_stratified_by(
        data: &Data,
        folds: usize,
        rng: &mut ChaCha8Rng,
        stratify_by: &str,
    ) -> CV {
        let (indices_class1, indices_class0) = utils::stratify_indices_by_class(&data.y);

        let annot = data
            .sample_annotations
            .as_ref()
            .expect("Sample annotations are required for stratified CV");

        // Find the column index
        let col_idx = annot
            .tag_column_names
            .iter()
            .position(|c| c == stratify_by)
            .expect(&format!(
                "Stratification column '{}' not found in sample annotations",
                stratify_by
            ));

        // Ensure all samples have annotations for the stratification column
        for i in 0..data.sample_len {
            let tags = annot.sample_tags.get(&i).unwrap_or_else(|| {
                panic!(
                    "Missing sample annotation for sample index {} while using stratified CV on column '{}'. \
                    Annotation file must cover all samples.",
                    i, stratify_by
                );
            });

            if col_idx >= tags.len() {
                panic!(
                    "Sample index {} has incomplete annotations for column '{}': \
                    expected at least {} columns, found {}.",
                    i,
                    stratify_by,
                    col_idx + 1,
                    tags.len()
                );
            }
        }

        // Stratify each class separately by the annotation column
        let indices_class0_folds =
            stratify_by_annotation(indices_class0, annot, col_idx, folds, rng);
        let indices_class1_folds =
            stratify_by_annotation(indices_class1, annot, col_idx, folds, rng);

        let validation_folds: Vec<Data> = indices_class0_folds
            .into_iter()
            .zip(indices_class1_folds.into_iter())
            .map(|(i1, i2)| i1.into_iter().chain(i2).collect::<Vec<usize>>())
            .map(|i| data.subset(i))
            .collect();

        let mut training_sets: Vec<Data> = Vec::new();
        for i in 0..folds {
            let mut dataset = data.subset(vec![]);

            for j in 0..folds {
                if j == i {
                    continue;
                } else {
                    dataset.add(&validation_folds[j]);
                }
            }

            training_sets.push(dataset);
        }

        CV {
            validation_folds: validation_folds,
            training_sets: training_sets,
            fold_collections: vec![],
        }
    }

    /// Creates a new cross-validation instance based on parameters.    
    pub fn new_from_param(data: &Data, param: &Param, rng: &mut ChaCha8Rng, k_folds: usize) -> CV {
        if param.cv.stratify_by.len() > 0 && data.sample_annotations.is_some() {
            CV::new_stratified_by(data, k_folds, rng, param.cv.stratify_by.as_str())
        } else {
            CV::new(data, k_folds, rng)
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
    pub fn pass<F>(&mut self, algo: F, param: &Param, running: Arc<AtomicBool>)
    where
        F: Fn(&mut Data, &Param, Arc<AtomicBool>) -> Vec<Population> + Send + Sync,
    {
        let collections: Vec<Vec<Population>> = {
            self.training_sets
                    .par_iter_mut()
                    .zip(self.validation_folds.par_iter_mut())
                    .enumerate()
                    .filter_map(|(i, (train, valid))| {
                        cinfo!(param.general.display_colorful, "\x1b[1;93mCompleting fold #{}...\x1b[0m", i+1);

                        let mut i_param = param.clone();
                        i_param.tag = format!("Fold {}", i+1);

                        let collection: Vec<Population> = algo(train, &i_param, Arc::clone(&running));

                        if collection.len() > 0 {
                            let final_population = collection.last().unwrap();

                            if final_population.individuals.len() > 0 {
                                let best_model = final_population.individuals.clone().into_iter().take(1).next().unwrap();
                                let train_auc = best_model.auc;
                                let valid_auc = best_model.compute_new_auc(valid);

                                cinfo!(
                                    param.general.display_colorful,
                                    "\x1b[1;93mFold #{} completed | Best train AUC: {:.3} | Associated validation fold AUC: {:.3}\x1b[0m",
                                    i+1, train_auc, valid_auc
                                );

                                Some(collection)

                            } else {
                                cinfo!(param.general.display_colorful, "\x1b[1;93mFold #{} skipped - no individuals found\x1b[0m", i+1);
                                Some(vec![])
                            }

                        } else {
                            cinfo!(param.general.display_colorful, "\x1b[1;93mFold #{} skipped - algorithm did not return any populations.\x1b[0m", i+1);
                            Some(vec![])
                        }
                    })
                    .collect()
        };

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
    pub fn compute_cv_oob_feature_importance(
        &self,
        cv_param: &Param,
        permutations: usize,
        main_rng: &mut ChaCha8Rng,
        aggregation_method: &ImportanceAggregation,
        scaled_importance: bool,
        cascade: bool,
    ) -> Result<ImportanceCollection, String> {
        assert!(
            self.fold_collections.len() > 0,
            "No population available. Run pass() first."
        );

        if self.fold_collections.len() != self.training_sets.len()
            || self.fold_collections.len() != self.validation_folds.len()
        {
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
            if fold_last_fbm.individuals.len() == 0 {
                panic!(
                    "Fold #{} has an empty Family of Best Models population. Skipping importance computation for this fold.",
                    i + 1
                );
            }
            // Extracted FBM is already fit and sort based on validation if required
            let importance_data = if cv_param.cv.fit_on_valid {
                &self.validation_folds[i]
            } else {
                &self.training_sets[i]
            };

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
                        if self.training_sets[i]
                            .feature_class
                            .contains_key(&imp.feature_idx)
                        {
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
            if values.is_empty() {
                continue;
            }

            let (agg_importance, agg_dispersion) = match aggregation_method {
                ImportanceAggregation::mean => mean_and_std(&values),
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
                importance_type: ImportanceType::MDA,
                feature_idx,
                scope: ImportanceScope::Collection,
                aggreg_method: Some(aggregation_method.clone()),
                importance: agg_importance,
                is_scaled: scaled_importance,
                dispersion: agg_dispersion,
                scope_pct,
                direction: None,
            };

            importances.push(collection_importance);
        }

        Ok(ImportanceCollection {
            importances: importances,
        })
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
        assert_ne!(
            self.fold_collections.len(),
            0,
            "No population available. Run pass() first."
        );

        let mut pop: Population;

        // For Beam, compute FBM on [kmin; kmax]
        if param.general.algo == "beam" {
            pop = beam::keep_n_best_model_within_collection(
                &self.fold_collections[fold_idx],
                param.beam.max_nb_of_models as usize,
            );
        } else {
            pop = self.fold_collections[fold_idx].last().unwrap().clone();
        }

        // Fit on valid if required and sort
        if param.cv.fit_on_valid {
            pop.fit(
                &self.validation_folds[fold_idx],
                &mut None,
                &None,
                &None,
                param,
            );
            pop = pop.sort();
        }

        pop = pop.select_best_population(param.cv.cv_best_models_ci_alpha);

        pop
    }

    /// Gets f`Family of Best Models` from all folds and merges them into a single population.
    ///
    /// Collects the `Family of Best Models` from each fold, displays their
    /// performance information, and merges all individuals into a single population
    /// for overall analysis.
    ///
    /// # Arguments
    ///
    /// # Returns
    /// Returns a merged population containing individuals from all `Family of Best Models`.
    ///
    /// # Panics
    /// Panics if `pass()` has not been called first to populate fold collections.
    pub fn get_fbm(&self, param: &Param) -> Population {
        assert_ne!(
            self.fold_collections.len(),
            0,
            "No population available. Run pass() first."
        );

        let mut fold_fbms: Vec<Population> = vec![];
        let mut merged_fbms = Population::new();

        for fold_idx in 0..self.training_sets.len() {
            if self.fold_collections[fold_idx].len() > 0
                && self.fold_collections[fold_idx]
                    .last()
                    .unwrap()
                    .individuals
                    .len()
                    > 0
            {
                let fbm = self.extract_fold_fbm(fold_idx, &param);

                cinfo!(
                    param.general.display_colorful,
                    "\x1b[1;93mFold #{}\x1b[0m",
                    fold_idx + 1
                );
                cinfo!(
                    param.general.display_colorful,
                    "{}",
                    fbm.clone().display(
                        &self.training_sets[fold_idx],
                        Some(&self.validation_folds[fold_idx]),
                        &param
                    )
                );

                fold_fbms.push(fbm);
            } else {
                cinfo!(
                    param.general.display_colorful,
                    "\x1b[1;93mFold #{}: empty population\x1b[0m",
                    fold_idx + 1
                );
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
        self.training_sets
            .iter()
            .zip(self.validation_folds.iter())
            .map(|(train_data, valid_data)| {
                (train_data.samples.clone(), valid_data.samples.clone())
            })
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
    pub fn reconstruct(
        data: &Data,
        fold_train_valid_names: Vec<(Vec<String>, Vec<String>)>,
        fold_collections: Vec<Vec<Population>>,
    ) -> Result<CV, String> {
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
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    impl CV {
        pub fn test() -> CV {
            // Create training and validation sets
            let training_sets = vec![
                Data::test_with_these_features(&[0, 1, 2, 3]),
                Data::test_with_these_features(&[0, 1, 2, 3]),
                Data::test_with_these_features(&[0, 1, 2, 3]),
            ];

            let validation_folds = vec![
                Data::test_with_these_features(&[0, 1, 2, 3]),
                Data::test_with_these_features(&[0, 1, 2, 3]),
                Data::test_with_these_features(&[0, 1, 2, 3]),
            ];

            // Create populations with properly fitted individuals
            let mut fold_collections = vec![];
            for _i in 0..3 {
                let mut pop = Population::test_with_these_features(&[0, 1, 2, 3]);

                // Compute metrics for each individual so they have fit values
                // Use different fit values to avoid them being filtered out by CI
                for (j, ind) in pop.individuals.iter_mut().enumerate() {
                    ind.fit = 0.95 - (j as f64 * 0.01); // Descending fit values
                    ind.auc = 0.95 - (j as f64 * 0.01);
                }

                // Sort by fitness
                pop = pop.sort();

                fold_collections.push(vec![pop]);
            }

            CV {
                fold_collections,
                training_sets,
                validation_folds,
            }
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

    // Add a unit test to check if y are correctly distributed?

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

        assert_eq!(
            cv.validation_folds[0].samples,
            ["sample3", "sample2", "sample5"]
        );
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
                assert!(
                    seen_samples.insert(sample.clone()),
                    "Sample {} appears in multiple validation folds",
                    sample
                );
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

        let mock_algo = |_train_data: &mut Data,
                         _param: &Param,
                         _running: Arc<AtomicBool>|
         -> Vec<Population> {
            let mut pop = Population::test();
            pop.compute_hash();
            vec![pop]
        };

        cv.pass(mock_algo, &param, running);

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

        let mock_algo = |_train_data: &mut Data,
                         _param: &Param,
                         _running: Arc<AtomicBool>|
         -> Vec<Population> { vec![Population::new()] };

        cv.pass(mock_algo, &param, running);

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
        let mock_algo =
            |_train_data: &mut Data, _param: &Param, running: Arc<AtomicBool>| -> Vec<Population> {
                running.store(false, std::sync::atomic::Ordering::Relaxed);
                vec![Population::test()]
            };

        cv.pass(mock_algo, &param, running_clone);

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
        let total_class1: usize = cv
            .validation_folds
            .iter()
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

        let mock_collections = vec![vec![Population::test()], vec![Population::test()]];

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
        let mock_collections = vec![vec![Population::test()], vec![Population::test()]];

        let result = CV::reconstruct(&data, fold_names, mock_collections);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Mismatch"));
    }

    #[test]
    fn test_reconstruct_fails_with_unknown_sample_name() {
        let data = Data::test();
        let fold_names = vec![(
            vec!["unknown_sample".to_string()],
            vec!["sample1".to_string()],
        )];
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
            &cv_param,
            5,
            &mut rng,
            &ImportanceAggregation::mean,
            false,
            false,
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
            &cv_param,
            5,
            &mut rng,
            &ImportanceAggregation::mean,
            false,
            false,
        );

        assert!(result.is_err());
        assert!(result.clone().unwrap_err().contains("Inconsistency"));
        assert!(result
            .unwrap_err()
            .contains("1 populations, 1 training sets, 2 validation folds"));
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_aggregation_across_folds_mean() {
        use crate::ga;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::specific_test(40, 20);
        let mut cv = CV::new(&data, 3, &mut rng);
        let r = Arc::new(AtomicBool::new(true));
        let mut cv_param = Param::default();
        cv_param.ga.max_epochs = 3;
        cv_param.data.feature_maximal_adj_pvalue = 1.0;
        cv.pass(
            |d: &mut Data, p: &Param, r: Arc<AtomicBool>| match p.general.algo.as_str() {
                "ga" => ga::ga(d, &mut None, &cv_param, r),
                _ => panic!("Such algorithm is not useful for the test."),
            },
            &cv_param,
            r,
        );

        let result = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        // Check that we have (aggregated) Collection weights
        let collection_importances: Vec<_> = result
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .collect();

        assert!(
            !collection_importances.is_empty(),
            "Should have Collection-level importances"
        );

        // Check that the aggregation is marked as mean
        for imp in &collection_importances {
            assert_eq!(imp.aggreg_method, Some(ImportanceAggregation::mean));
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_aggregation_across_folds_median() {
        let cv = CV::test();
        let mut cv_param = Param::default();
        cv_param.cv.cv_best_models_ci_alpha = 0.01; // Include most models in FBM
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng,
                &ImportanceAggregation::median,
                false,
                false,
            )
            .unwrap();

        let collection_importances: Vec<_> = result
            .importances
            .iter()
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
        let mut cv_param = Param::default();
        cv_param.cv.cv_best_models_ci_alpha = 0.01; // Include most models in FBM
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        let collection_importances: Vec<_> = result
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .collect();

        for imp in &collection_importances {
            assert!(
                imp.scope_pct >= 0.0 && imp.scope_pct <= 1.0,
                "scope_pct should be between 0.0 and 1.0, got {}",
                imp.scope_pct
            );
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_cascade_mode_includes_individual_importances() {
        let cv = CV::test();
        let mut cv_param = Param::default();
        cv_param.cv.cv_best_models_ci_alpha = 0.01; // Include most models in FBM
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // cascade = true
        let result_cascade = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng,
                &ImportanceAggregation::mean,
                false,
                true,
            )
            .unwrap();

        // cascade = false
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let result_no_cascade = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng2,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        let individual_count_cascade = result_cascade
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Individual { .. }))
            .count();

        let individual_count_no_cascade = result_no_cascade
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Individual { .. }))
            .count();

        if individual_count_cascade > 0 {
            assert!(
                individual_count_cascade >= individual_count_no_cascade,
                "Cascade mode should include individual importances"
            );
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_on_validation_vs_training_data() {
        use crate::ga;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::specific_test(50, 30);
        let mut cv = CV::new(&data, 3, &mut rng);
        let r = Arc::new(AtomicBool::new(true));
        let mut cv_param = Param::default();
        cv_param.ga.max_epochs = 10;
        cv_param.data.feature_maximal_adj_pvalue = 1.0;
        cv.pass(
            |d: &mut Data, p: &Param, r: Arc<AtomicBool>| match p.general.algo.as_str() {
                "ga" => ga::ga(d, &mut None, &cv_param, r),
                _ => panic!("Such algorithm is not useful for the test."),
            },
            &cv_param,
            r,
        );

        // on_validation = true
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);

        cv_param.cv.fit_on_valid = true;
        let result_validation = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng1,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        cv_param.cv.fit_on_valid = false;
        // on_validation = false
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let result_training = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng2,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        // Both should succeed and achieve results.
        assert!(!result_validation.importances.is_empty());
        assert!(!result_training.importances.is_empty());

        // The amounts may differ depending on the data used.
        // We are then just checking that the structures are consistent.
        let validation_collection_count = result_validation
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .count();

        let training_collection_count = result_training
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .count();

        // We can expect to have Collection importances in both cases.
        assert!(validation_collection_count > 0 || training_collection_count > 0);
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_scaled_importance_flag() {
        let cv = CV::test();
        let mut cv_param = Param::default();
        cv_param.cv.cv_best_models_ci_alpha = 0.01; // Include most models in FBM

        // scaled = true
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let result_scaled = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng1,
                &ImportanceAggregation::mean,
                true,
                false,
            )
            .unwrap();

        // scaled = false
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let result_unscaled = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng2,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        // Check that the is_scaled flag is correctly propagated
        for imp in &result_scaled.importances {
            if matches!(imp.scope, ImportanceScope::Collection) {
                assert!(
                    imp.is_scaled,
                    "Collection importance should be marked as scaled"
                );
            }
        }

        for imp in &result_unscaled.importances {
            if matches!(imp.scope, ImportanceScope::Collection) {
                assert!(
                    !imp.is_scaled,
                    "Collection importance should not be marked as scaled"
                );
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

        let result = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        assert!(
            result.importances.is_empty(),
            "Empty folds should produce no importances"
        );
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_feature_filtering_in_aggregation() {
        let cv = CV::test();
        let mut cv_param = Param::default();
        cv_param.cv.cv_best_models_ci_alpha = 0.01; // Include most models in FBM
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        // Check that no importance Collection has empty values
        let collection_importances: Vec<_> = result
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Collection))
            .collect();

        for imp in &collection_importances {
            // All Collection amounts must have been calculated
            // from non-empty values.
            assert!(
                !imp.importance.is_nan(),
                "Collection importance should not be NaN for feature {}",
                imp.feature_idx
            );
        }
    }

    #[test]
    fn test_compute_cv_oob_feature_importance_population_id_assignment() {
        let cv = CV::test();
        let mut cv_param = Param::default();
        cv_param.cv.cv_best_models_ci_alpha = 0.01; // Include most models in FBM
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = cv
            .compute_cv_oob_feature_importance(
                &cv_param,
                5,
                &mut rng,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

        // Check that Population amounts have correct IDs
        let population_importances: Vec<_> = result
            .importances
            .iter()
            .filter(|imp| matches!(imp.scope, ImportanceScope::Population { .. }))
            .collect();

        for imp in &population_importances {
            if let ImportanceScope::Population { id } = imp.scope {
                assert!(
                    id < cv.fold_collections.len(),
                    "Population ID {} should be less than number of folds {}",
                    id,
                    cv.fold_collections.len()
                );
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

        // Mock algorithm returing a test population compatible with Data::test() (only features 0 and 1)
        let mock_algo = |_train_data: &mut Data,
                         _param: &Param,
                         _running: Arc<AtomicBool>|
         -> Vec<Population> {
            let mut pop = Population::test_with_these_features(&[0, 1]);
            pop.individuals[0].fit = 0.9999;
            pop.individuals[1].fit = 0.9998;
            pop.compute_hash();
            vec![pop]
        };

        cv.pass(mock_algo, &param, Arc::clone(&running));
        assert!(cv.fold_collections.len() > 0);

        let importance_collection = cv
            .compute_cv_oob_feature_importance(
                &param,
                1,
                &mut rng,
                &ImportanceAggregation::mean,
                false,
                false,
            )
            .unwrap();

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

        assert_eq!(
            cv.validation_folds.len(),
            reconstructed_cv.validation_folds.len()
        );
        assert_eq!(cv.training_sets.len(), reconstructed_cv.training_sets.len());

        for i in 0..outer_folds {
            assert_eq!(
                cv.validation_folds[i].samples,
                reconstructed_cv.validation_folds[i].samples
            );
            assert_eq!(
                cv.training_sets[i].samples,
                reconstructed_cv.training_sets[i].samples
            );
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
                assert!(
                    !validation.contains(sample),
                    "Sample '{}' found in both training and validation for fold {}",
                    sample,
                    i
                );
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
        for (i, fold) in cv
            .validation_folds
            .iter()
            .chain(cv.training_sets.iter())
            .enumerate()
        {
            assert_eq!(
                &fold.features, reference_features,
                "Features mismatch in fold {}",
                i
            );
            assert_eq!(
                fold.feature_len, data.feature_len,
                "Feature length mismatch in fold {}",
                i
            );
        }
    }

    #[test]
    fn test_data_size_preservation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data: Data = Data::test();
        let folds = 4;
        let cv = CV::new(&data, folds, &mut rng);

        // Check that the sum of the sizes of the validation folds equals the original size
        let total_validation_size: usize =
            cv.validation_folds.iter().map(|fold| fold.sample_len).sum();
        assert_eq!(total_validation_size, data.sample_len);

        // Check that each drive set is the correct size
        for (i, training_set) in cv.training_sets.iter().enumerate() {
            assert_eq!(
                training_set.sample_len,
                data.sample_len - cv.validation_folds[i].sample_len
            );
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

            assert!(
                (fold_class0 as isize - expected_class0 as isize).abs() <= 1,
                "Class 0 distribution imbalanced in fold {}: expected ~{}, got {}",
                i,
                expected_class0,
                fold_class0
            );
            assert!(
                (fold_class1 as isize - expected_class1 as isize).abs() <= 1,
                "Class 1 distribution imbalanced in fold {}: expected ~{}, got {}",
                i,
                expected_class1,
                fold_class1
            );
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

    use crate::data::SampleAnnotations;
    fn create_sample_annotations(
        sample_len: usize,
        col_name: &str,
        values: Vec<String>,
    ) -> SampleAnnotations {
        let mut sample_tags = HashMap::new();
        for i in 0..sample_len {
            sample_tags.insert(i, vec![values[i].clone()]);
        }

        SampleAnnotations {
            tag_column_names: vec![col_name.to_string()],
            sample_tags,
        }
    }

    #[test]
    fn test_new_stratified_by_creates_correct_number_of_folds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        // Add sample annotations for stratification
        let annotations = create_sample_annotations(
            data.sample_len,
            "batch",
            vec![
                "A".to_string(),
                "B".to_string(),
                "A".to_string(),
                "B".to_string(),
                "A".to_string(),
                "B".to_string(),
            ],
        );
        data.sample_annotations = Some(annotations);

        let outer_folds = 3;
        let cv = CV::new_stratified_by(&data, outer_folds, &mut rng, "batch");

        assert_eq!(
            cv.validation_folds.len(),
            outer_folds,
            "Should create correct number of validation folds"
        );
        assert_eq!(
            cv.training_sets.len(),
            outer_folds,
            "Should create correct number of training sets"
        );
    }

    #[test]
    fn test_new_stratified_by_preserves_class_distribution() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::specific_test(30, 10);

        let class0_count = data.y.iter().filter(|&&y| y == 0).count();
        let _class1_count = data.y.iter().filter(|&&y| y == 1).count();

        let annotation_values: Vec<String> = (0..30)
            .map(|i| {
                if i % 2 == 0 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();

        let annotations = create_sample_annotations(data.sample_len, "batch", annotation_values);
        data.sample_annotations = Some(annotations);

        let outer_folds = 3; // 30 / 3 = 10 samples per fold
        let cv = CV::new_stratified_by(&data, outer_folds, &mut rng, "batch");

        let expected_size_class0 = class0_count / outer_folds;

        for (i, fold) in cv.validation_folds.iter().enumerate() {
            let fold_class0 = fold.y.iter().filter(|&&y| y == 0).count();

            // Tolerance of +1/-1 due to integer divisions
            assert!(
                (fold_class0 as isize - expected_size_class0 as isize).abs() <= 2,
                "Fold {}: Class 0 count {} is too far from expected {}",
                i,
                fold_class0,
                expected_size_class0
            );
        }
    }

    #[test]
    fn test_new_stratified_by_preserves_annotation_distribution() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // Create a dataset with controlled class distribution
        // Data::specific_test generates random classes, so we need to use a large enough dataset
        // to get a reasonable distribution, or create data manually
        let num_samples = 60;
        let num_features = 30;

        // Create data manually with exactly 30 of each class
        let mut X = HashMap::new();
        let mut data_rng = ChaCha8Rng::seed_from_u64(12345);
        for sample in 0..num_samples {
            for feature in 0..num_features {
                X.insert((sample, feature), data_rng.gen_range(0.0..1.0));
            }
        }

        // Create exactly 30 samples of each class
        let mut y: Vec<u8> = vec![0; 30];
        y.extend(vec![1; 30]);

        let mut data = Data {
            X,
            y,
            sample_len: num_samples,
            feature_class: HashMap::new(),
            feature_significance: HashMap::new(),
            features: (0..num_features)
                .map(|i| format!("feature_{}", i))
                .collect(),
            samples: (0..num_samples).map(|i| format!("sample_{}", i)).collect(),
            feature_selection: (0..num_features).collect(),
            feature_len: num_features,
            classes: vec!["class_0".to_string(), "class_1".to_string()],
            feature_annotations: None,
            sample_annotations: None,
        };

        // Assign batch annotations: even indices = A, odd indices = B
        let batch_values: Vec<String> = (0..num_samples)
            .map(|i| {
                if i % 2 == 0 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();

        let annotations = create_sample_annotations(data.sample_len, "batch", batch_values);
        data.sample_annotations = Some(annotations);

        let outer_folds = 3;
        let cv = CV::new_stratified_by(&data, outer_folds, &mut rng, "batch");

        // Count the distribution of annotations in the original data
        let annot = data.sample_annotations.as_ref().unwrap();
        let col_idx = annot
            .tag_column_names
            .iter()
            .position(|c| c == "batch")
            .unwrap();

        let mut batch_a_class0 = 0i32;
        let mut batch_b_class0 = 0;
        let mut batch_a_class1 = 0;
        let mut batch_b_class1 = 0;

        for i in 0..data.sample_len {
            let batch = &annot.sample_tags[&i][col_idx];
            if data.y[i] == 0 {
                if batch == "A" {
                    batch_a_class0 += 1;
                } else {
                    batch_b_class0 += 1;
                }
            } else {
                if batch == "A" {
                    batch_a_class1 += 1;
                } else {
                    batch_b_class1 += 1;
                }
            }
        }

        let mut folds_with_both_batches = 0;
        let mut total_batch_a_class0 = 0i32;
        let mut total_batch_b_class0 = 0;
        let mut total_batch_a_class1 = 0;
        let mut total_batch_b_class1 = 0;

        for fold in &cv.validation_folds {
            let fold_annot = fold.sample_annotations.as_ref().unwrap();
            let mut has_batch_a = false;
            let mut has_batch_b = false;

            let mut fold_batch_a_class0 = 0;
            let mut fold_batch_b_class0 = 0;
            let mut fold_batch_a_class1 = 0;
            let mut fold_batch_b_class1 = 0;

            for i in 0..fold.sample_len {
                let batch = &fold_annot.sample_tags[&i][col_idx];
                if batch == "A" {
                    has_batch_a = true;
                }
                if batch == "B" {
                    has_batch_b = true;
                }

                // Count by combination (class, batch) for this fold
                if fold.y[i] == 0 {
                    if batch == "A" {
                        fold_batch_a_class0 += 1;
                    } else {
                        fold_batch_b_class0 += 1;
                    }
                } else {
                    if batch == "A" {
                        fold_batch_a_class1 += 1;
                    } else {
                        fold_batch_b_class1 += 1;
                    }
                }
            }

            if has_batch_a && has_batch_b {
                folds_with_both_batches += 1;
            }

            // Accumulate totals for global verification
            total_batch_a_class0 += fold_batch_a_class0;
            total_batch_b_class0 += fold_batch_b_class0;
            total_batch_a_class1 += fold_batch_a_class1;
            total_batch_b_class1 += fold_batch_b_class1;
        }

        assert!(
            folds_with_both_batches > 0,
            "At least one fold should contain both batch A and B, but none do"
        );

        assert_eq!(
            total_batch_a_class0, batch_a_class0,
            "Total count of (class=0, batch=A) across folds should match original data"
        );
        assert_eq!(
            total_batch_b_class0, batch_b_class0,
            "Total count of (class=0, batch=B) across folds should match original data"
        );
        assert_eq!(
            total_batch_a_class1, batch_a_class1,
            "Total count of (class=1, batch=A) across folds should match original data"
        );
        assert_eq!(
            total_batch_b_class1, batch_b_class1,
            "Total count of (class=1, batch=B) across folds should match original data"
        );

        // Additional verification: total samples reconstructed
        let total_samples_in_folds = total_batch_a_class0
            + total_batch_b_class0
            + total_batch_a_class1
            + total_batch_b_class1;
        assert_eq!(
            total_samples_in_folds, data.sample_len as i32,
            "Total samples across all folds should equal original dataset size"
        );
    }

    #[test]
    #[should_panic(expected = "Sample annotations are required")]
    fn test_new_stratified_by_panics_without_annotations() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();

        // No annotations - should panic
        let _cv = CV::new_stratified_by(&data, 3, &mut rng, "batch");
    }

    #[test]
    #[should_panic(expected = "Stratification column 'nonexistent' not found")]
    fn test_new_stratified_by_panics_with_wrong_column() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        let annotations = create_sample_annotations(
            data.sample_len,
            "batch",
            vec!["A".to_string(); data.sample_len],
        );
        data.sample_annotations = Some(annotations);

        // Nonexistent column - should panic
        let _cv = CV::new_stratified_by(&data, 3, &mut rng, "nonexistent");
    }

    #[test]
    fn test_new_stratified_by_all_data_preserved() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        let annotations = create_sample_annotations(
            data.sample_len,
            "site",
            vec![
                "S1".to_string(),
                "S2".to_string(),
                "S1".to_string(),
                "S2".to_string(),
                "S1".to_string(),
                "S2".to_string(),
            ],
        );
        data.sample_annotations = Some(annotations);

        let outer_folds = 3;
        let cv = CV::new_stratified_by(&data, outer_folds, &mut rng, "site");

        // Verify that all data is preserved
        let mut collected_y = Vec::new();
        for fold in &cv.validation_folds {
            collected_y.extend(fold.y.iter().map(|&x| x as usize));
        }

        let mut real_y: Vec<usize> = data.y.iter().map(|&x| x as usize).collect();
        real_y.sort();
        collected_y.sort();

        assert_eq!(
            collected_y, real_y,
            "All y values should be preserved across validation folds"
        );
    }

    #[test]
    fn test_new_stratified_by_training_sets_correct() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        let annotations = create_sample_annotations(
            data.sample_len,
            "cohort",
            vec![
                "C1".to_string(),
                "C2".to_string(),
                "C1".to_string(),
                "C2".to_string(),
                "C1".to_string(),
                "C2".to_string(),
            ],
        );
        data.sample_annotations = Some(annotations);

        let outer_folds = 3;
        let cv = CV::new_stratified_by(&data, outer_folds, &mut rng, "cohort");

        // Verify that each training set = all data - corresponding validation fold
        for i in 0..outer_folds {
            let expected_size = data.sample_len - cv.validation_folds[i].sample_len;
            assert_eq!(
                cv.training_sets[i].sample_len, expected_size,
                "Training set {} should have correct size",
                i
            );
        }
    }

    #[test]
    fn test_new_vs_new_stratified_by_same_class_distribution() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();

        // Standard CV
        let cv_standard = CV::new(&data, 3, &mut rng1);

        // CV with uniform stratification (all samples have the same annotation)
        let mut data_with_annot = data.clone();
        let uniform_annotations = create_sample_annotations(
            data.sample_len,
            "uniform",
            vec!["same".to_string(); data.sample_len],
        );
        data_with_annot.sample_annotations = Some(uniform_annotations);

        let cv_stratified = CV::new_stratified_by(&data_with_annot, 3, &mut rng2, "uniform");

        // Verify that the class distribution is identical
        for i in 0..3 {
            let class0_standard = cv_standard.validation_folds[i]
                .y
                .iter()
                .filter(|&&y| y == 0)
                .count();
            let class1_standard = cv_standard.validation_folds[i]
                .y
                .iter()
                .filter(|&&y| y == 1)
                .count();

            let class0_stratified = cv_stratified.validation_folds[i]
                .y
                .iter()
                .filter(|&&y| y == 0)
                .count();
            let class1_stratified = cv_stratified.validation_folds[i]
                .y
                .iter()
                .filter(|&&y| y == 1)
                .count();

            assert_eq!(
                class0_standard, class0_stratified,
                "Fold {} should have same class 0 count",
                i
            );
            assert_eq!(
                class1_standard, class1_stratified,
                "Fold {} should have same class 1 count",
                i
            );
        }
    }

    #[test]
    fn test_new_vs_new_stratified_by_same_fold_sizes() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();

        let cv_standard = CV::new(&data, 3, &mut rng1);

        let mut data_with_annot = data.clone();
        let uniform_annotations = create_sample_annotations(
            data.sample_len,
            "uniform",
            vec!["same".to_string(); data.sample_len],
        );
        data_with_annot.sample_annotations = Some(uniform_annotations);

        let cv_stratified = CV::new_stratified_by(&data_with_annot, 3, &mut rng2, "uniform");

        // Verify that the fold sizes are identical
        for i in 0..3 {
            assert_eq!(
                cv_standard.validation_folds[i].sample_len,
                cv_stratified.validation_folds[i].sample_len,
                "Fold {} should have same size",
                i
            );

            assert_eq!(
                cv_standard.training_sets[i].sample_len, cv_stratified.training_sets[i].sample_len,
                "Training set {} should have same size",
                i
            );
        }
    }

    #[test]
    fn test_new_vs_new_stratified_by_reproducibility_with_uniform_annotation() {
        // Test that new_stratified_by with uniform annotation produces the same results as new()
        // for a given seed
        let seed = 12345_u64;
        let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);

        let data = Data::test();
        let cv_standard = CV::new(&data, 3, &mut rng1);

        let mut data_with_annot = data.clone();
        let uniform_annotations = create_sample_annotations(
            data.sample_len,
            "uniform",
            vec!["X".to_string(); data.sample_len],
        );
        data_with_annot.sample_annotations = Some(uniform_annotations);

        let cv_stratified = CV::new_stratified_by(&data_with_annot, 3, &mut rng2, "uniform");

        // Verify that the samples in each fold are identical
        for i in 0..3 {
            assert_eq!(
                cv_standard.validation_folds[i].samples, cv_stratified.validation_folds[i].samples,
                "Fold {} should have same samples",
                i
            );
        }
    }

    #[test]
    fn test_new_stratified_by_with_multiple_annotation_values() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        // Create annotations with multiple different values
        let annotations = create_sample_annotations(
            data.sample_len,
            "center",
            vec![
                "Paris".to_string(),
                "Lyon".to_string(),
                "Paris".to_string(),
                "Marseille".to_string(),
                "Lyon".to_string(),
                "Marseille".to_string(),
            ],
        );
        data.sample_annotations = Some(annotations);

        let outer_folds = 3;
        let cv = CV::new_stratified_by(&data, outer_folds, &mut rng, "center");

        assert_eq!(cv.validation_folds.len(), outer_folds);

        // Verify that the data is well distributed
        let total_samples: usize = cv.validation_folds.iter().map(|f| f.sample_len).sum();
        assert_eq!(
            total_samples, data.sample_len,
            "All samples should be distributed"
        );
    }

    #[test]
    fn test_new_stratified_by_with_two_folds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        let annotations = create_sample_annotations(
            data.sample_len,
            "treatment",
            vec![
                "Control".to_string(),
                "Treated".to_string(),
                "Control".to_string(),
                "Treated".to_string(),
                "Control".to_string(),
                "Treated".to_string(),
            ],
        );
        data.sample_annotations = Some(annotations);

        let cv = CV::new_stratified_by(&data, 2, &mut rng, "treatment");

        assert_eq!(cv.validation_folds.len(), 2);
        assert_eq!(cv.training_sets.len(), 2);

        // Each training set should contain approximately 50% of the data
        for i in 0..2 {
            let expected_train_size = data.sample_len - cv.validation_folds[i].sample_len;
            assert_eq!(cv.training_sets[i].sample_len, expected_train_size);
        }
    }

    #[test]
    fn test_new_stratified_by_no_overlap() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        let annotations = create_sample_annotations(
            data.sample_len,
            "group",
            vec![
                "G1".to_string(),
                "G2".to_string(),
                "G1".to_string(),
                "G2".to_string(),
                "G1".to_string(),
                "G2".to_string(),
            ],
        );
        data.sample_annotations = Some(annotations);

        let cv = CV::new_stratified_by(&data, 3, &mut rng, "group");

        // Verify that there is no overlap between the validation folds
        for i in 0..cv.validation_folds.len() {
            for j in (i + 1)..cv.validation_folds.len() {
                let samples_i = &cv.validation_folds[i].samples;
                let samples_j = &cv.validation_folds[j].samples;

                for sample in samples_i {
                    assert!(
                        !samples_j.contains(sample),
                        "Sample {} should not appear in both fold {} and fold {}",
                        sample,
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_new_stratified_by_feature_consistency() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        let annotations = create_sample_annotations(
            data.sample_len,
            "experiment",
            vec!["E1".to_string(); data.sample_len],
        );
        data.sample_annotations = Some(annotations);

        let cv = CV::new_stratified_by(&data, 3, &mut rng, "experiment");

        // Verify that all features are preserved in all folds
        for fold in cv.validation_folds.iter().chain(cv.training_sets.iter()) {
            assert_eq!(
                fold.features, data.features,
                "All folds should have same features as original data"
            );
            assert_eq!(
                fold.feature_len, data.feature_len,
                "All folds should have same feature_len as original data"
            );
        }
    }

    #[test]
    fn test_new_stratified_by_large_dataset() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::specific_test(100, 20); // 100 samples, 20 features

        // Create annotations with multiple groups
        let annotation_values: Vec<String> = (0..100).map(|i| format!("Batch{}", i % 5)).collect();

        let annotations = create_sample_annotations(100, "batch", annotation_values);
        data.sample_annotations = Some(annotations);

        let outer_folds = 5;
        let cv = CV::new_stratified_by(&data, outer_folds, &mut rng, "batch");

        assert_eq!(cv.validation_folds.len(), outer_folds);

        // Verify the distribution
        let total_samples: usize = cv.validation_folds.iter().map(|f| f.sample_len).sum();
        assert_eq!(total_samples, 100);

        // Each fold should have approximately 20 samples (100/5)
        for (i, fold) in cv.validation_folds.iter().enumerate() {
            assert!(
                fold.sample_len >= 15 && fold.sample_len <= 25,
                "Fold {} has {} samples, expected around 20",
                i,
                fold.sample_len
            );
        }
    }

    #[test]
    fn test_new_from_param_with_stratify_by() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();
        let annotations = create_sample_annotations(
            data.sample_len,
            "batch",
            vec![
                "A".to_string(),
                "B".to_string(),
                "A".to_string(),
                "B".to_string(),
                "A".to_string(),
                "B".to_string(),
            ],
        );
        data.sample_annotations = Some(annotations);

        let mut param = Param::default();
        param.cv.stratify_by = "batch".to_string();

        let cv = CV::new_from_param(&data, &param, &mut rng, 3);
        assert_eq!(cv.validation_folds.len(), 3);
    }

    #[test]
    fn test_new_from_param_without_stratify_by() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let param = Param::default(); // stratify_by

        let cv = CV::new_from_param(&data, &param, &mut rng, 3);
        assert_eq!(cv.validation_folds.len(), 3);
    }

    #[test]
    fn test_new_from_param_missing_annotations() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test(); // No annotation
        let mut param = Param::default();
        param.cv.stratify_by = "batch".to_string();

        // Should fallback to CV::new() without panicking
        let cv = CV::new_from_param(&data, &param, &mut rng, 3);
        assert_eq!(cv.validation_folds.len(), 3);
    }

    #[test]
    fn test_new_stratified_by_interaction_class_x_annotation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Create a perfectly balanced dataset:
        // 80 samples: 40 class 0 (20 batch A + 20 batch B), 40 class 1 (20 batch A + 20 batch B)
        let total_samples = 80;
        let samples_per_group = 20; // (class, batch) combinations

        let mut data = Data::specific_test(total_samples, 10); // 10 features

        // Build class labels: [0,0,...0 (40x), 1,1,...1 (40x)]
        data.y = (0..total_samples)
            .map(|i| if i < total_samples / 2 { 0 } else { 1 })
            .collect();

        // Build batch annotations: alternating A/B within each class
        // Class 0: A,B,A,B,... (20 A, 20 B)
        // Class 1: A,B,A,B,... (20 A, 20 B)
        let annotation_values: Vec<String> = (0..total_samples)
            .map(|i| {
                if i % 2 == 0 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();

        let annotations = create_sample_annotations(total_samples, "batch", annotation_values);
        data.sample_annotations = Some(annotations);

        let num_folds = 4; // 80 / 4 = 20 samples per fold
        let cv = CV::new_stratified_by(&data, num_folds, &mut rng, "batch");

        // Verify that each fold preserves the ratios within subgroups
        for (fold_idx, fold) in cv.validation_folds.iter().enumerate() {
            let fold_annot = fold.sample_annotations.as_ref().unwrap();
            let col_idx = fold_annot
                .tag_column_names
                .iter()
                .position(|c| c == "batch")
                .unwrap();

            // Count the 4 subgroups in this fold
            let mut count_0_a = 0;
            let mut count_0_b = 0;
            let mut count_1_a = 0;
            let mut count_1_b = 0;

            for sample_idx in 0..fold.sample_len {
                let class = fold.y[sample_idx];
                let batch = &fold_annot.sample_tags[&sample_idx][col_idx];

                match (class, batch.as_str()) {
                    (0, "A") => count_0_a += 1,
                    (0, "B") => count_0_b += 1,
                    (1, "A") => count_1_a += 1,
                    (1, "B") => count_1_b += 1,
                    _ => panic!("Unexpected class/batch combination"),
                }
            }

            // CRITICAL CHECK: The class 0/class 1 ratio must be preserved WITHIN each batch
            // In our balanced dataset, we expect to have as many class 0 as class 1
            // in batch A, and as many class 0 as class 1 in batch B

            // Tolerance of ±1 to handle rounding (20 samples per fold / 4 groups = 5 per group)
            let expected_per_group = samples_per_group / num_folds; // 20/4 = 5

            assert!(
                (count_0_a as isize - expected_per_group as isize).abs() <= 1,
                "Fold {}: count(class=0, batch=A) = {}, expected ~{} (±1)",
                fold_idx,
                count_0_a,
                expected_per_group
            );

            assert!(
                (count_0_b as isize - expected_per_group as isize).abs() <= 1,
                "Fold {}: count(class=0, batch=B) = {}, expected ~{} (±1)",
                fold_idx,
                count_0_b,
                expected_per_group
            );

            assert!(
                (count_1_a as isize - expected_per_group as isize).abs() <= 1,
                "Fold {}: count(class=1, batch=A) = {}, expected ~{} (±1)",
                fold_idx,
                count_1_a,
                expected_per_group
            );

            assert!(
                (count_1_b as isize - expected_per_group as isize).abs() <= 1,
                "Fold {}: count(class=1, batch=B) = {}, expected ~{} (±1)",
                fold_idx,
                count_1_b,
                expected_per_group
            );

            // Additional check: class 0/class 1 ratio must be balanced within each batch
            let total_batch_a = count_0_a + count_1_a;
            let total_batch_b = count_0_b + count_1_b;

            if total_batch_a > 0 {
                assert!(
                    (count_0_a as isize - count_1_a as isize).abs() <= 1,
                    "Fold {}: Batch A has unbalanced class distribution (class0={}, class1={})",
                    fold_idx,
                    count_0_a,
                    count_1_a
                );
            }

            if total_batch_b > 0 {
                assert!(
                    (count_0_b as isize - count_1_b as isize).abs() <= 1,
                    "Fold {}: Batch B has unbalanced class distribution (class0={}, class1={})",
                    fold_idx,
                    count_0_b,
                    count_1_b
                );
            }
        }

        // Global check: all data must be present exactly once
        let total_class_0_a: usize = cv
            .validation_folds
            .iter()
            .map(|fold| {
                let annot = fold.sample_annotations.as_ref().unwrap();
                let col_idx = annot
                    .tag_column_names
                    .iter()
                    .position(|c| c == "batch")
                    .unwrap();
                (0..fold.sample_len)
                    .filter(|&i| fold.y[i] == 0 && annot.sample_tags[&i][col_idx] == "A")
                    .count()
            })
            .sum();

        assert_eq!(
            total_class_0_a, samples_per_group,
            "Total count of (class=0, batch=A) across all folds should be {}",
            samples_per_group
        );
    }

    /// Extreme test case: very few samples per combination (class, annotation)
    /// compared to the number of requested folds. Checks the robustness of stratification
    /// when constraints are difficult to satisfy.
    #[test]
    fn test_new_stratified_by_very_small_samples_per_group() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Minimalist dataset: 6 samples, 2 classes, 2 batches
        // 4 possible combinations, with some underrepresented
        let mut data = Data::specific_test(6, 5); // 6 samples, 5 features

        // Configuration :
        // - Class 0: 4 samples (2 batch A, 2 batch B)
        // - Class 1: 2 samples (1 batch A, 1 batch B)
        data.y = vec![0, 0, 0, 0, 1, 1];

        let annotation_values = vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "B".to_string(),
        ];
        let annotations = create_sample_annotations(6, "batch", annotation_values);
        data.sample_annotations = Some(annotations);

        // Create 5 folds (more folds than samples in some subgroups!)
        let outer_folds = 5;
        let cv = CV::new_stratified_by(&data, outer_folds, &mut rng, "batch");

        assert_eq!(
            cv.validation_folds.len(),
            outer_folds,
            "Should create 5 folds"
        );
        assert_eq!(
            cv.training_sets.len(),
            outer_folds,
            "Should create 5 training sets"
        );

        let mut all_samples_in_folds = Vec::new();
        for fold in &cv.validation_folds {
            all_samples_in_folds.extend(fold.samples.clone());
        }
        all_samples_in_folds.sort();

        let mut original_samples = data.samples.clone();
        original_samples.sort();

        assert_eq!(
            all_samples_in_folds, original_samples,
            "All samples should be present exactly once across all folds"
        );

        let annot = data.sample_annotations.as_ref().unwrap();
        let col_idx = annot
            .tag_column_names
            .iter()
            .position(|c| c == "batch")
            .unwrap();

        // Count in the original dataset
        let mut original_counts = std::collections::HashMap::new();
        for i in 0..data.sample_len {
            let class = data.y[i];
            let batch = &annot.sample_tags[&i][col_idx];
            let key = format!("class{}_batch{}", class, batch);
            *original_counts.entry(key).or_insert(0) += 1;
        }

        // Count in the folds
        let mut fold_counts = std::collections::HashMap::new();
        for fold in &cv.validation_folds {
            let fold_annot = fold.sample_annotations.as_ref().unwrap();
            for i in 0..fold.sample_len {
                let class = fold.y[i];
                let batch = &fold_annot.sample_tags[&i][col_idx];
                let key = format!("class{}_batch{}", class, batch);
                *fold_counts.entry(key).or_insert(0) += 1;
            }
        }

        assert_eq!(
            original_counts, fold_counts,
            "Distribution of (class, batch) combinations should be preserved"
        );

        for i in 0..outer_folds {
            // No overlap
            let training_samples = &cv.training_sets[i].samples;
            let validation_samples = &cv.validation_folds[i].samples;
            for sample in validation_samples {
                assert!(
                    !training_samples.contains(sample),
                    "Sample '{}' found in both training and validation for fold {}",
                    sample,
                    i
                );
            }

            // Consistent size
            let expected_training_size = data.sample_len - cv.validation_folds[i].sample_len;
            assert_eq!(
                cv.training_sets[i].sample_len, expected_training_size,
                "Training set {} should have size = total - validation_fold_size",
                i
            );

            // Features preserved
            assert_eq!(
                cv.validation_folds[i].features, data.features,
                "Fold {} should preserve original features",
                i
            );
            assert_eq!(
                cv.training_sets[i].features, data.features,
                "Training set {} should preserve original features",
                i
            );
        }

        let non_empty_folds = cv
            .validation_folds
            .iter()
            .filter(|f| f.sample_len > 0)
            .count();
        assert!(
            non_empty_folds > 0,
            "At least some folds should be non-empty even with very small sample sizes"
        );

        // With 6 samples and 5 folds, some folds will have 1 sample, others 2, others 0
        // This is acceptable as long as all samples are present exactly once
        let total_in_folds: usize = cv.validation_folds.iter().map(|f| f.sample_len).sum();
        assert_eq!(
            total_in_folds, data.sample_len,
            "Total samples in folds should equal original dataset size"
        );
    }

    #[test]
    #[should_panic(expected = "Sample annotations are required for stratified CV")]
    fn test_cv_new_stratified_by_panic_on_missing_annotations() {
        let mut data = Data::test_with_these_features(&[0, 1, 2, 3]);
        data.sample_len = 5;
        data.y = vec![0, 1, 0, 1, 0];
        data.sample_annotations = None;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let _cv = CV::new_stratified_by(&data, 2, &mut rng, "batch");
    }

    #[test]
    #[should_panic(expected = "Sample index")]
    fn test_cv_new_stratified_by_panic_on_incomplete_annotation_line() {
        let mut data = Data::test_with_these_features(&[0, 1, 2, 3]);
        data.sample_len = 4;
        data.y = vec![0, 1, 0, 1];

        let mut sample_tags = HashMap::new();
        sample_tags.insert(0, vec!["ctrl".to_string()]);
        sample_tags.insert(1, vec!["treat".to_string(), "batchA".to_string()]);
        sample_tags.insert(2, vec!["ctrl".to_string()]);
        sample_tags.insert(3, vec!["treat".to_string()]);

        data.sample_annotations = Some(SampleAnnotations {
            tag_column_names: vec!["group".to_string(), "batch".to_string()],
            sample_tags,
        });

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let _cv = CV::new_stratified_by(&data, 2, &mut rng, "batch");
    }
}
