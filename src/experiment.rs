use crate::bayesian_mcmc::MCMCAnalysisTrace;
use crate::cinfo;
use crate::cv::CV;
use crate::data::Data;
use crate::param::Param;
use crate::population::Population;
use crate::utils::display_feature_importance_terminal;
use crate::voting::Jury;
use log::{debug, error, info, warn};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

//-----------------------------------------------------------------------------
// Importance structures and methods
//-----------------------------------------------------------------------------

/// Scope at which feature importance is computed
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ImportanceScope {
    /// Aggregated across all populations (e.g., inter-fold FBM results).
    Collection,
    /// Specific to a single population
    Population {
        /// Population ID, e.g., fold number in CV
        id: usize,
    },
    /// Specific to a single individual model
    Individual {
        /// Model hash
        model_hash: u64,
    },
}

/// Type of feature importance computed
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ImportanceType {
    /// Mean Decrease in Accuracy (permutation importance)
    MDA,
    /// Prevalence at the population level
    PrevalencePop,
    /// Prevalence at the cross-validation level
    PrevalenceCV,
    /// Coefficient values
    Coefficient,
    /// Posterior probability from Bayesian MCMC
    PosteriorProbability,
}

/// Method used to aggregate importance values
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(non_camel_case_types)]
pub enum ImportanceAggregation {
    /// Mean aggregation
    mean,
    /// Median aggregation
    median,
}

// To-do: re-implement these structs if needed later
// pub struct FeatureImportance {
//     // pub scope: ImportanceScope,
//     pub mda: Vec<Importance>, // cv/pop level
//     pub prevalence_pop_pct: Vec<f64>, // cv/pop level
//     pub coefficient_sign_pct: Vec<(f64, f64, f64)>, // cv/pop level
//     pub coefficient_abs: Vec<Importance>, // cv/pop level
//     pub posterior_prob: Vec<(f64, f64, f64)>, // cv?/pop level
//     pub prevalence_cv_pct: Vec<Importance>,  // cv level
// }

// pub struct FeatureImportanceAgg {
//     // pub scope: ImportanceScope,
//     pub mda: Vec<Importance>, // agg pop level
//     pub prevalence_pop_pct: Vec<f64>, // agg pop level
//     pub coefficient_sign_pct: Vec<(f64, f64, f64)>, // agg pop level
//     pub coefficient_abs: Vec<Importance>, // agg pop level
//     pub posterior_prob: Vec<(f64, f64, f64)>, // cv?/pop level

//     pub prevalence_cv_pct: Vec<Importance>,  // cv level
// }

/// Feature importance complete information
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Importance {
    /// Type of importance
    pub importance_type: ImportanceType,
    /// Index of the feature whose importance is computed
    pub feature_idx: usize,
    /// Scope of importance
    pub scope: ImportanceScope,
    /// Aggregation method used
    pub aggreg_method: Option<ImportanceAggregation>,
    /// Importance value
    pub importance: f64,
    /// Whether the importance is scaled
    pub is_scaled: bool,
    /// Dispersion of the importance value
    pub dispersion: f64,
    /// Percentage of the scope covered
    pub scope_pct: f64,
    /// If applicable, direction of the importance
    pub direction: Option<usize>,
}

/// Collection of feature importances
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ImportanceCollection {
    /// Vector of computed importances
    pub importances: Vec<Importance>,
}

impl ImportanceCollection {
    /// Creates a new empty ImportanceCollection
    pub fn new() -> ImportanceCollection {
        ImportanceCollection {
            importances: Vec::new(),
        }
    }

    /// Returns importances associated with a feature
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the feature
    ///
    /// # Returns
    ///
    /// ImportanceCollection containing importances for the specified feature
    pub fn feature(&self, idx: usize) -> ImportanceCollection {
        let importances = self
            .importances
            .iter()
            .filter(|imp| imp.feature_idx == idx)
            .cloned()
            .collect();

        ImportanceCollection { importances }
    }

    /// Filters importances based on scope and type
    ///
    /// # Arguments
    ///
    /// * `scope` - Optional scope to filter by
    /// * `imp_type` - Optional importance type to filter by
    ///
    /// # Returns
    ///
    /// ImportanceCollection containing filtered importances
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::experiment::{ImportanceCollection, Importance, ImportanceType, ImportanceScope, ImportanceAggregation};
    /// let importance_collection = ImportanceCollection::new();
    ///
    /// // Fill importance_collection with data
    /// // importance_collection.importances.push(Importance { feature_idx: 0, importance_type: ImportanceType::MDA, scope: ImportanceScope::Population { id: 0 }, aggreg_method: Some(ImportanceAggregation::mean), importance: 0.5, is_scaled: true, dispersion: 0.1, scope_pct: 1.0, direction: None });
    /// // importance_collection.importances.push(Importance { feature_idx: 1, importance_type: ImportanceType::Coefficient, scope: ImportanceScope::Collection, aggreg_method: Some(ImportanceAggregation::median), importance: 0.3, is_scaled: false, dispersion: 0.2, scope_pct: 0.8, direction: Some(1) });
    ///
    /// // Filter by scope and type
    /// let filtered = importance_collection.filter(Some(ImportanceScope::Population { id: 0 }), Some(ImportanceType::MDA));
    /// // assert_eq!(filtered.importances.len(), 1);
    /// // assert_eq!(filtered.importances[0].feature_idx, 0);
    /// // assert_eq!(filtered.importances[0].importance_type, ImportanceType::MDA);
    /// // assert!(matches!(filtered.importances[0].scope, ImportanceScope::Population { id: 0 }));
    /// ```
    ///
    pub fn filter(
        &self,
        scope: Option<ImportanceScope>,
        imp_type: Option<ImportanceType>,
    ) -> ImportanceCollection {
        let importances = self
            .importances
            .iter()
            .filter(|imp| {
                let scope_ok = match &scope {
                    None => true,
                    Some(ImportanceScope::Individual { .. }) => {
                        matches!(imp.scope, ImportanceScope::Individual { .. })
                    }
                    Some(ImportanceScope::Population { .. }) => {
                        matches!(imp.scope, ImportanceScope::Population { .. })
                    }
                    Some(ImportanceScope::Collection) => {
                        matches!(imp.scope, ImportanceScope::Collection)
                    }
                };

                let type_ok = imp_type
                    .as_ref()
                    .map_or(true, |t| imp.importance_type == *t);
                scope_ok && type_ok
            })
            .cloned()
            .collect();

        ImportanceCollection { importances }
    }

    /// Retrieves the top percentage of features based on importance.
    ///
    ///  # Arguments
    ///
    /// * `pct` - Percentage of top features to retrieve (between 0.0 and 1.0).
    ///
    /// # Returns
    ///
    /// ImportanceCollection containing the top percentage of features.
    ///
    /// # Panics
    ///
    /// Panics if `pct` is not between 0.0 and 1.0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::experiment::{ImportanceCollection, Importance, ImportanceType, ImportanceScope, ImportanceAggregation};
    /// let mut importance_collection = ImportanceCollection::new();
    /// // Fill importance_collection with data
    /// // importance_collection.importances.push(Importance { feature_idx: 0, importance_type: ImportanceType::MDA, scope: ImportanceScope::Population { id: 0 }, aggreg_method: Some(ImportanceAggregation::mean), importance: 0.5, is_scaled: true, dispersion: 0.1, scope_pct: 1.0, direction: None });
    /// // importance_collection.importances.push(Importance { feature_idx: 1, importance_type: ImportanceType::Coefficient, scope: ImportanceScope::Collection, aggreg_method: Some(ImportanceAggregation::median), importance: 0.3, is_scaled: false, dispersion: 0.2, scope_pct: 0.8, direction: Some(1) });
    /// // importance_collection.importances.push(Importance { feature_idx: 2, importance_type: ImportanceType::MDA, scope: ImportanceScope::Collection, aggreg_method: Some(ImportanceAggregation::mean), importance: 0.7, is_scaled: true, dispersion: 0.15, scope_pct: 0.9, direction: None });
    ///
    /// // Get top 50% features
    /// let top_features = importance_collection.get_top(0.5);
    /// // assert_eq!(top_features.importances.len(), 2);
    /// // assert_eq!(top_features.importances[0].feature_idx, 2); // importance 0.7
    /// // assert_eq!(top_features.importances[1].feature_idx, 0); // importance 0.5
    /// ```
    pub fn get_top(&self, pct: f64) -> ImportanceCollection {
        assert!((0.0..=1.0).contains(&pct));
        let mut subset = self.importances.clone();
        subset.sort_unstable_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        let keep = ((subset.len() as f64 * pct).ceil() as usize).max(1);
        subset.truncate(keep);
        ImportanceCollection {
            importances: subset,
        }
    }

    /// Displays feature importance in terminal format.
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the Data object.
    /// * `nb_features` - Number of features to display.
    ///
    /// # Returns
    ///
    /// String containing the formatted feature importance display.
    pub fn display_feature_importance_terminal(&self, data: &Data, nb_features: usize) -> String {
        let mut map: std::collections::HashMap<usize, (f64, f64)> =
            std::collections::HashMap::new();
        let mut agg = ImportanceAggregation::mean;

        let mut collection_importances = Vec::new();
        let mut population_importances = Vec::new();
        let mut individual_importances = Vec::new();

        for imp in &self.importances {
            match imp.scope {
                ImportanceScope::Collection => collection_importances.push(imp),
                ImportanceScope::Population { .. } => population_importances.push(imp),
                ImportanceScope::Individual { .. } => individual_importances.push(imp),
            }
        }

        let importances_to_use = if !collection_importances.is_empty() {
            collection_importances
        } else if !population_importances.is_empty() {
            population_importances
        } else {
            individual_importances
        };

        for imp in importances_to_use {
            map.insert(imp.feature_idx, (imp.importance, imp.dispersion));
            if let Some(a) = &imp.aggreg_method {
                agg = a.clone();
            }
        }

        //let consistent_directions = self.check_direction_consistency();

        display_feature_importance_terminal(data, &map, nb_features, &agg)
    }
}

//-----------------------------------------------------------------------------
// Experiment structures and methods
//-----------------------------------------------------------------------------

/// Experiment associated metadata
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ExperimentMetadata {
    /// Bayesian MCMC trace to analyze posterior distributions
    MCMC {
        /// Complete MCMC analysis trace
        trace: MCMCAnalysisTrace,
    },
    /// Voting Jury results
    Jury {
        /// Jury population
        jury: Jury,
    },
}

/// Complete experiment data and results
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Experiment {
    /// Experiment ID, i.e., timestamp and algorithm
    pub id: String,
    /// Timestamp of the experiment
    pub timestamp: String,
    /// Gpredomics version and git hash used
    pub gpredomics_version: String,
    /// Parameters used
    pub parameters: Param,

    /// Training data
    pub train_data: Data,
    /// If provided, test data
    pub test_data: Option<Data>,

    /// In CV-mode, Vec<Vec<Population>>.len() = outer_folds, in non-CV mode Vec<Vec<Population>>.len() = 1
    pub collections: Vec<Vec<Population>>,

    /// The best final population after training, i.e. the last generation for non-CV genetic algorithm and the best max_nb_of_models across k for non-CV beam.
    /// In CV-mode, the best models are extracted from above population (FBM) per fold and merged.
    pub final_population: Option<Population>,

    /// If requested, computed feature importance collection
    pub importance_collection: Option<ImportanceCollection>,

    /// In CV-mode, stores the train/validation IDs per fold to reconstruct CV object
    pub cv_folds_ids: Option<Vec<(Vec<String>, Vec<String>)>>,

    /// Execution time in seconds
    pub execution_time: f64,

    /// If available, other metadata associated with the experiment
    pub others: Option<ExperimentMetadata>,
}

impl Experiment {
    /// Computes feature importance for the experiment.
    ///
    /// Handles both cross-validation (CV) and non-CV modes.
    ///
    /// In non-CV mode, computes importance on the final population's Family of Best Models (FBM).
    /// In CV mode, reconstructs the CV object and computes intra-fold and inter-fold importances.
    ///
    /// # Panics
    ///
    /// Panics if CV reconstruction fails or if CV fold IDs are missing when expected.
    pub fn compute_importance(&mut self) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.parameters.general.seed);
        if matches!(self.cv_folds_ids, None) {
            info!("Computing importance on final population's FBM (non-CV mode)");
            self.importance_collection = Some(
                self.final_population
                    .as_ref()
                    .unwrap()
                    .select_best_population_with_method(
                        self.parameters.cv.cv_best_models_ci_alpha,
                        &self.parameters.cv.cv_fbm_ci_method,
                    )
                    .compute_pop_mda_feature_importance(
                        &self.train_data,
                        self.parameters.importance.n_permutations_mda,
                        &mut rng,
                        &self.parameters.importance.importance_aggregation,
                        self.parameters.importance.scaled_importance,
                        true,
                        None,
                    ),
            );
        } else {
            info!(
                "Computing CV importance with {} folds",
                self.cv_folds_ids.as_ref().unwrap().len()
            );
            if let Some(fold_ids) = &self.cv_folds_ids {
                debug!("Reconstructing CV object from Experiment...");
                let cv_result =
                    CV::reconstruct(&self.train_data, fold_ids.clone(), self.collections.clone());

                match cv_result {
                    Ok(cv) => {
                        debug!("Computing Intra-fold and Inter-fold importances...");
                        self.importance_collection = Some(
                            cv.compute_cv_mda_feature_importance(
                                &self.parameters,
                                self.parameters.importance.n_permutations_mda,
                                &mut rng,
                                &self.parameters.importance.importance_aggregation,
                                self.parameters.importance.scaled_importance,
                                true,
                            )
                            .expect("CV importance calculation failed."),
                        );
                    }
                    Err(e) => {
                        panic!("Failed to reconstruct CV structure: {}", e);
                    }
                }
            } else {
                error!("CV fold IDs are None but expected for CV importance calculation");
            }
        }
        info!("Importance computation complete");
    }

    /// Generates a formatted string displaying the experiment results.
    ///
    /// # Returns
    ///
    /// String containing the formatted experiment results.
    ///
    /// # Panics
    ///
    /// Panics if CV reconstruction fails or if no final population is available.
    pub fn display_results(&self) -> String {
        let mut text = String::new();
        text.push_str(&format!(
            "\n=============== Experiment {} ===============\n\n",
            self.id
        ));
        text.push_str(&format!(
            "Gpredomics version: v{}\n",
            self.gpredomics_version
        ));
        text.push_str(&format!("Timestamp: {}\n", self.timestamp));
        text.push_str(&format!("Algorithm: {}\n", self.parameters.general.algo));
        text.push_str(&format!("Execution time: {:.2}s\n", self.execution_time));
        text.push_str(&format!(
            "Parameters: \x1b[2;97m{:?}\x1b[0m\n",
            &self.parameters
        ));
        text.push_str(&format!("Experiment results:\n\n"));

        // Reconstruct CV to print each fold FBM as original display
        if let Some(cv_folds_ids) = self.cv_folds_ids.clone() {
            let cv = CV::reconstruct(&self.train_data, cv_folds_ids, self.collections.clone())
                .expect("CV reconstruction failed.");
            let mut fbm = cv.get_fbm(&self.parameters);

            fbm.fit(&self.train_data, &mut None, &None, &None, &self.parameters);
            fbm = fbm.sort();

            fbm.compute_hash();
            let mut final_pop = self.final_population.clone().unwrap();
            final_pop.compute_hash();

            let fbm_hashes: std::collections::HashSet<_> =
                fbm.individuals.iter().map(|i| i.hash).collect();
            let final_hashes: std::collections::HashSet<_> =
                final_pop.individuals.iter().map(|i| i.hash).collect();

            assert_eq!(fbm_hashes, final_hashes, "Something is wrong with the Experiment: reconstructed CV based FBM should be the same as final population");
            text.push_str(&format!(
                "{}\n",
                crate::utils::strip_ansi_if_needed(
                    "\x1b[1;93mDisplaying Family of best models across folds\x1b[0m",
                    self.parameters.general.display_colorful
                )
            ));
        }

        // Print final result
        if let Some(mut final_pop) = self.final_population.clone() {
            text.push_str(&format!(
                "{}\n",
                final_pop.display(&self.train_data, self.test_data.as_ref(), &self.parameters)
            ));
        } else if self.parameters.general.algo == "mcmc" {
            text.push_str(&format!(
                "{}\n",
                self.collections[0][0].clone().display(
                    &self.train_data,
                    self.test_data.as_ref(),
                    &self.parameters
                )
            ));
        } else {
            panic!("No final population available");
        }

        if let Some(ref importance_collection) = self.importance_collection {
            let top_features = if self.cv_folds_ids.is_some() {
                // CV-mode : priorize Collection importances (inter-fold)
                let collection_features =
                    importance_collection.filter(Some(ImportanceScope::Collection), None);
                if !collection_features.importances.is_empty() {
                    collection_features.get_top(0.1)
                } else {
                    importance_collection.get_top(0.1)
                }
            } else {
                // Non CV-mode : priorize Population importances
                let population_features =
                    importance_collection.filter(Some(ImportanceScope::Population { id: 0 }), None);
                if !population_features.importances.is_empty() {
                    population_features.get_top(0.1)
                } else {
                    importance_collection.get_top(0.1)
                }
            };

            text.push_str(&format!(
                "Top 10%:{}\n",
                top_features.display_feature_importance_terminal(
                    &self.train_data,
                    top_features.importances.len()
                )
            ));
        }

        if let Some(ref metadata) = self.others {
            match metadata {
                ExperimentMetadata::Jury { jury } => {
                    text.push_str(&format!(
                        "{}\n",
                        jury.display(&self.train_data, self.test_data.as_ref(), &self.parameters)
                    ));
                }
                _ => {}
            }
        }

        text
    }

    /// Evaluates the final population on a new dataset.
    ///
    /// # Arguments
    ///
    /// * `X_path` - Path to the feature data file.
    /// * `y_path` - Path to the label data file.
    ///
    /// # Panics
    ///
    /// Panics if loading the new data fails or if the new data is not compatible with the training data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::experiment::Experiment;
    /// let mut experiment = Experiment::load_auto("experiment.mp").unwrap();
    /// experiment.evaluate_on_new_dataset("new_X.csv", "new_y.csv");
    /// ```
    pub fn evaluate_on_new_dataset(&mut self, X_path: &str, y_path: &str) {
        let mut new_data = Data::new();

        let final_pop = self
            .final_population
            .as_mut()
            .expect("No final population available");

        if let Err(e) = new_data.load_data(X_path, y_path, self.parameters.data.features_in_rows) {
            panic!("Failed to load test data: {}", e);
        }

        debug!("{}", new_data);

        new_data.set_classes(self.parameters.data.classes.clone());

        if self.parameters.data.inverse_classes {
            new_data.inverse_classes();
        }

        if self.train_data.check_compatibility(&new_data) == false {
            panic!("New test data to evaluate is not compatible with train data");
        }

        let display_param = self.parameters.clone();

        crate::cinfo!(
            self.parameters.general.display_colorful,
            "{}",
            final_pop.display(&self.train_data, Some(&new_data), &display_param)
        );

        if let Some(ref mut metadata) = self.others {
            match metadata {
                ExperimentMetadata::Jury { jury } => {
                    cinfo!(
                        self.parameters.general.display_colorful,
                        "{}",
                        jury.display(&self.train_data, Some(&new_data), &self.parameters)
                    );
                }
                _ => {}
            }
        }
    }

    /// Saves the experiment in a suitable format based on file extension.
    pub fn save_auto<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        match ext.as_str() {
            "json" => self.save_json(path),
            "msgpack" | "mp" => self.save_messagepack(path),
            "bin" | "bincode" => self.save_bincode(path),
            _ => {
                warn!("Unknown format. Saving experiment in msgpack.");
                let mp_path = path.with_extension("mp");
                self.save_messagepack(mp_path)
            }
        }
    }

    /// Saves to JSON (human readable, but may have slight inaccuracies for decimal values)
    fn save_json<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        warn!("Due to JSON compression, a slight inaccuracy may occur for decimal values. Prefer the msgpack format if you want to read the experiments from another language.");
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Saves as MessagePack (language interoperable)
    fn save_messagepack<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rmp_serde::Serializer;

        let mut buf = Vec::new();
        self.serialize(&mut Serializer::new(&mut buf).with_struct_map())?;
        std::fs::write(path, buf)?;
        Ok(())
    }

    /// Saves as Bincode (compact binary, Rust-only)
    fn save_bincode<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// Loads the experiment from a file, automatically detecting the format based on file extension.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the experiment file.
    ///
    /// # Returns
    ///
    /// Result containing the loaded Experiment or an error.
    pub fn load_auto<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        match ext.as_str() {
            "json" => Self::load_json(path),
            "msgpack" | "mp" => Self::load_messagepack(path),
            "bin" | "bincode" => Self::load_bincode(path),
            _ => Self::load_with_fallback(path),
        }
    }

    /// Loads from JSON format
    fn load_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let experiment: Experiment = serde_json::from_str(&content)?;
        Ok(experiment)
    }

    /// Loads from MessagePack format
    fn load_messagepack<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let experiment: Experiment = rmp_serde::from_slice(&bytes)?;
        Ok(experiment)
    }

    /// Loads from Bincode format
    fn load_bincode<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let experiment: Experiment = bincode::deserialize(&bytes)?;
        Ok(experiment)
    }

    /// Attempts to load the experiment using multiple formats as a fallback mechanism.
    ///
    /// Tries MessagePack, then Bincode, and finally JSON.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the experiment file.
    ///
    /// # Returns
    ///
    /// Result containing the loaded Experiment or an error if all formats fail.
    fn load_with_fallback<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();

        if let Ok(experiment) = Self::load_messagepack(path) {
            return Ok(experiment);
        }

        if let Ok(experiment) = Self::load_bincode(path) {
            return Ok(experiment);
        }

        if let Ok(experiment) = Self::load_json(path) {
            return Ok(experiment);
        }

        Err("Unable to load the experience".into())
    }

    /// Computes voting results using the final population.
    ///
    /// If there is more than one expert in the final population, a Jury is created and evaluated.
    /// The results are stored in the `others` field of the Experiment.
    ///
    /// # Panics
    ///
    /// Panics if there is no final population available.
    pub fn compute_voting(&mut self) {
        let mut jury;
        let mut voting_pop = self.final_population.clone().unwrap();

        voting_pop.compute_all_metrics(&self.train_data, &self.parameters.general.fit);

        jury = Jury::new_from_param(&voting_pop, &self.train_data, &self.parameters);

        if jury.experts.individuals.len() > 1 {
            jury.evaluate(&self.train_data);
            self.others = Some(ExperimentMetadata::Jury { jury: jury })
        } else {
            warn!("An informative vote is requiring more than one expert!")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Data;
    use crate::population::Population;

    #[test]
    fn test_new_creates_empty_collection() {
        let collection = ImportanceCollection::new();
        assert!(collection.importances.is_empty());
    }

    #[test]
    fn test_feature_returns_existing_feature_importances() {
        let mut collection = ImportanceCollection::new();
        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 5,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.8,
            is_scaled: true,
            dispersion: 0.1,
            scope_pct: 1.0,
            direction: None,
        });
        collection.importances.push(Importance {
            importance_type: ImportanceType::Coefficient,
            feature_idx: 3,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.6,
            is_scaled: true,
            dispersion: 0.1,
            scope_pct: 1.0,
            direction: None,
        });

        let result = collection.feature(5);
        assert_eq!(result.importances.len(), 1);
        assert_eq!(result.importances[0].feature_idx, 5);
        assert_eq!(result.importances[0].importance, 0.8);
    }

    #[test]
    fn test_feature_returns_empty_for_nonexistent_feature() {
        let mut collection = ImportanceCollection::new();
        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 1,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.8,
            is_scaled: true,
            dispersion: 0.1,
            scope_pct: 1.0,
            direction: None,
        });

        let result = collection.feature(99);
        assert!(result.importances.is_empty());
    }

    #[test]
    fn test_filter_comprehensive_functionality() {
        let mut collection = ImportanceCollection::new();
        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 1,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.8,
            is_scaled: true,
            dispersion: 0.1,
            scope_pct: 1.0,
            direction: None,
        });
        collection.importances.push(Importance {
            importance_type: ImportanceType::Coefficient,
            feature_idx: 2,
            scope: ImportanceScope::Individual { model_hash: 123 },
            aggreg_method: Some(ImportanceAggregation::median),
            importance: 0.6,
            is_scaled: false,
            dispersion: 0.2,
            scope_pct: 0.5,
            direction: Some(1),
        });
        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 3,
            scope: ImportanceScope::Population { id: 0 },
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.7,
            is_scaled: true,
            dispersion: 0.15,
            scope_pct: 0.8,
            direction: None,
        });

        let result_none = collection.filter(None, None);
        assert_eq!(result_none.importances.len(), 3);

        let result_scope = collection.filter(Some(ImportanceScope::Collection), None);
        assert_eq!(result_scope.importances.len(), 1);
        assert!(matches!(
            result_scope.importances[0].scope,
            ImportanceScope::Collection
        ));

        let result_type = collection.filter(None, Some(ImportanceType::MDA));
        assert_eq!(result_type.importances.len(), 2);
        assert!(result_type
            .importances
            .iter()
            .all(|imp| imp.importance_type == ImportanceType::MDA));

        let result_both =
            collection.filter(Some(ImportanceScope::Collection), Some(ImportanceType::MDA));
        assert_eq!(result_both.importances.len(), 1);
        assert!(matches!(
            result_both.importances[0].scope,
            ImportanceScope::Collection
        ));
        assert_eq!(
            result_both.importances[0].importance_type,
            ImportanceType::MDA
        );
    }

    #[test]
    fn test_get_top_comprehensive_functionality() {
        let mut collection = ImportanceCollection::new();

        assert!(collection.get_top(0.5).importances.is_empty());

        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 1,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.3,
            is_scaled: true,
            dispersion: 0.1,
            scope_pct: 1.0,
            direction: None,
        });
        collection.importances.push(Importance {
            importance_type: ImportanceType::Coefficient,
            feature_idx: 2,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::median),
            importance: 0.9,
            is_scaled: false,
            dispersion: 0.2,
            scope_pct: 0.5,
            direction: Some(1),
        });
        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 3,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.6,
            is_scaled: true,
            dispersion: 0.15,
            scope_pct: 0.8,
            direction: None,
        });

        let result_zero = collection.get_top(0.0);
        assert_eq!(result_zero.importances.len(), 1);
        assert_eq!(result_zero.importances[0].importance, 0.9);

        let result_full = collection.get_top(1.0);
        assert_eq!(result_full.importances.len(), 3);

        assert!(result_full.importances[0].importance >= result_full.importances[1].importance);
        assert!(result_full.importances[1].importance >= result_full.importances[2].importance);
        assert_eq!(result_full.importances[0].importance, 0.9);
        assert_eq!(result_full.importances[1].importance, 0.6);
        assert_eq!(result_full.importances[2].importance, 0.3);

        let result_half = collection.get_top(0.5);
        assert_eq!(result_half.importances.len(), 2);
        assert_eq!(result_half.importances[0].importance, 0.9);
        assert_eq!(result_half.importances[1].importance, 0.6);
    }

    #[test]
    fn test_display_feature_importance_prioritizes_scopes_correctly() {
        let mut collection = ImportanceCollection::new();

        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 1,
            scope: ImportanceScope::Individual { model_hash: 123 },
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.8,
            is_scaled: true,
            dispersion: 0.1,
            scope_pct: 1.0,
            direction: None,
        });
        collection.importances.push(Importance {
            importance_type: ImportanceType::Coefficient,
            feature_idx: 1,
            scope: ImportanceScope::Population { id: 0 },
            aggreg_method: Some(ImportanceAggregation::median),
            importance: 0.6,
            is_scaled: false,
            dispersion: 0.2,
            scope_pct: 0.5,
            direction: Some(1),
        });
        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 1,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.4,
            is_scaled: true,
            dispersion: 0.15,
            scope_pct: 0.8,
            direction: None,
        });

        let data = Data::test_with_these_features(&[1, 2]);

        let result = collection.display_feature_importance_terminal(&data, 10);

        // Should prioritise Collection > Population > Individual
        // If Collection is present, should use Collection importance (0.4)
        assert!(!result.is_empty());
    }

    #[test]
    fn test_display_feature_importance_handles_empty_collection() {
        let collection = ImportanceCollection::new();
        let data = Data::test_with_these_features(&[1, 2]);

        let result = collection.display_feature_importance_terminal(&data, 10);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_display_feature_importance_handles_mixed_aggregation_methods() {
        let mut collection = ImportanceCollection::new();
        collection.importances.push(Importance {
            importance_type: ImportanceType::MDA,
            feature_idx: 1,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::mean),
            importance: 0.8,
            is_scaled: true,
            dispersion: 0.1,
            scope_pct: 1.0,
            direction: None,
        });
        collection.importances.push(Importance {
            importance_type: ImportanceType::Coefficient,
            feature_idx: 2,
            scope: ImportanceScope::Collection,
            aggreg_method: Some(ImportanceAggregation::median),
            importance: 0.6,
            is_scaled: false,
            dispersion: 0.2,
            scope_pct: 0.5,
            direction: Some(1),
        });

        let data = Data::test_with_these_features(&[1, 2]);

        let result = collection.display_feature_importance_terminal(&data, 10);
        assert!(!result.is_empty());
    }

    impl Experiment {
        /// Creates a test Experiment instance for testing purposes
        pub fn test() -> Experiment {
            Experiment {
                id: "test_exp_001".to_string(),
                timestamp: "2025-01-01T12:00:00Z".to_string(),
                gpredomics_version: "1.0.0".to_string(),
                parameters: Param::default(),
                train_data: Data::test(),
                cv_folds_ids: None,
                test_data: Some(Data::test2()),
                collections: vec![vec![
                    Population::test_with_n_overlapping_features(2, 20),
                    Population::test_with_n_overlapping_features(15, 12),
                ]],
                final_population: Some(Population::test()),
                importance_collection: None,
                execution_time: 42.5,
                others: None,
            }
        }
    }

    // #[test]
    // fn test_serialization_json_roundtrip() {
    //     let original_exp = Experiment::test();
    //     let file_path = "test_serialization_json_roundtrip";

    //     original_exp.save_json(&file_path).unwrap();

    //     let _ = Experiment::load_json(&file_path).unwrap();

    //     // Vérifications
    //     //assert_eq!(original_exp, loaded_exp);

    //     std::fs::remove_file("test_serialization_json_roundtrip").unwrap();
    // }

    #[test]
    fn test_serialization_messagepack_roundtrip() {
        let original_exp = Experiment::test();
        let file_path = "test_serialization_messagepack_roundtrip";

        original_exp.save_messagepack(&file_path).unwrap();
        let loaded_exp = Experiment::load_messagepack(&file_path).unwrap();

        assert_eq!(original_exp, loaded_exp);
        std::fs::remove_file("test_serialization_messagepack_roundtrip").unwrap();
    }

    #[test]
    fn test_serialization_bincode_roundtrip() {
        let original_exp = Experiment::test();
        let temp_file = "test_serialization_bincode_roundtrip";
        let file_path = temp_file;

        original_exp.save_bincode(&file_path).unwrap();
        let loaded_exp = Experiment::load_bincode(&file_path).unwrap();

        assert_eq!(original_exp, loaded_exp);
        std::fs::remove_file("test_serialization_bincode_roundtrip").unwrap();
    }

    #[test]
    fn test_auto_format_detection_by_extension() {
        let experiment = Experiment::test();

        // Test JSON
        let json_file = "test_auto_format_detection_by_extension.json";
        experiment.save_auto(json_file).unwrap();
        let _ = Experiment::load_auto(json_file).unwrap();
        //assert_eq!(experiment, loaded);

        // Test MessagePack
        let mp_file = "test_auto_format_detection_by_extension.msgpack";
        experiment.save_auto(mp_file).unwrap();
        let loaded = Experiment::load_auto(mp_file).unwrap();
        assert_eq!(experiment, loaded);

        // Test Bin
        let bin_file = "test_auto_format_detection_by_extension.bin";
        experiment.save_auto(bin_file).unwrap();
        let loaded = Experiment::load_auto(bin_file).unwrap();
        assert_eq!(experiment, loaded);

        std::fs::remove_file("test_auto_format_detection_by_extension.json").unwrap();
        std::fs::remove_file("test_auto_format_detection_by_extension.msgpack").unwrap();
        std::fs::remove_file("test_auto_format_detection_by_extension.bin").unwrap();
    }

    #[test]
    fn test_load_fallback_mechanism() {
        let experiment = Experiment::test();
        let temp_file = "test_load_fallback_mechanism";
        experiment.save_auto(temp_file).unwrap();

        let _ = std::fs::rename("test_load_fallback_mechanism.mp", temp_file);
        let loaded = Experiment::load_auto(temp_file).unwrap();
        assert_eq!(experiment, loaded);
        std::fs::remove_file(temp_file).unwrap();
    }

    use crate::voting::{VotingMethod, WeightingMethod};
    #[test]
    fn test_experiment_with_jury_metadata() {
        let mut experiment = Experiment::test();

        // Create minimal test data for Jury
        let jury = Jury::new(
            &Population::new(),
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        experiment.others = Some(ExperimentMetadata::Jury { jury });

        // Serialisation test with metadata Short
        let temp_file = "test_experiment_with_jury_metadata.json";
        experiment.save_auto(temp_file).unwrap();
        let loaded = Experiment::load_auto(temp_file).unwrap();

        match loaded.others {
            Some(ExperimentMetadata::Jury { .. }) => {}
            _ => panic!("Jury metadata not preserved during serialization"),
        }
        std::fs::remove_file("test_experiment_with_jury_metadata.json").unwrap();
    }

    #[test]
    fn test_file_extension_normalization() {
        let experiment = Experiment::test();

        // Test with extension in uppercase
        let temp_file = "test_file_extension_normalization.BIN";
        experiment.save_auto(temp_file).unwrap();
        let loaded = Experiment::load_auto(temp_file).unwrap();
        assert_eq!(experiment, loaded);
        std::fs::remove_file("test_file_extension_normalization.BIN").unwrap();
    }

    #[test]
    #[should_panic]
    fn test_load_nonexistent_file() {
        let _ = Experiment::load_auto("nonexistent_file.json").unwrap();
    }

    #[test]
    #[should_panic]
    fn test_evaluate_on_new_dataset_no_population() {
        let mut experiment = Experiment::test();
        experiment.final_population = None;

        experiment.evaluate_on_new_dataset("dummy_X.csv", "dummy_y.csv");
    }

    // Integration test for the complete flow
    #[test]
    fn test_complete_experiment_lifecycle() {
        let mut original = Experiment::test();
        original.execution_time = 123.45;

        original.final_population = Some(Population::new());

        let temp_file = "test_complete_experiment_lifecycle.json";
        original.save_auto(temp_file).unwrap();

        let reloaded = Experiment::load_auto(temp_file).unwrap();

        assert_eq!(original, reloaded);
        assert!(reloaded.final_population.is_some());

        cinfo!(
            reloaded.parameters.general.display_colorful,
            "{}",
            reloaded.display_results()
        );
        std::fs::remove_file("test_complete_experiment_lifecycle.json").unwrap();
    }

    #[test]
    fn test_compute_importance_non_cv_mode_uses_final_population() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = None;
        experiment.final_population = Some(Population::test());

        experiment.compute_importance();

        assert!(experiment.importance_collection.is_some());
        let importance = experiment.importance_collection.unwrap();
        assert!(!importance.importances.is_empty());
    }

    #[test]
    #[should_panic(expected = "unwrap")]
    fn test_compute_importance_non_cv_mode_panics_when_no_final_population() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = None;
        experiment.final_population = None;

        experiment.compute_importance();
    }

    #[test]
    fn test_compute_importance_cv_mode_reconstructs_cv_from_collection() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = Some(vec![
            (
                vec!["sample1".to_string(), "sample2".to_string()],
                vec!["sample3".to_string()],
            ),
            (
                vec!["sample3".to_string()],
                vec!["sample1".to_string(), "sample2".to_string()],
            ),
        ]);
        experiment.collections = vec![vec![Population::test()], vec![Population::test()]];

        experiment.compute_importance();

        assert!(experiment.importance_collection.is_some());
    }

    #[test]
    #[should_panic(expected = "Mismatch")]
    fn test_compute_importance_cv_mode_handles_no_collection() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = Some(vec![(
            vec!["sample1".to_string()],
            vec!["sample2".to_string()],
        )]);
        experiment.collections = vec![];
        experiment.compute_importance();

        assert!(experiment.importance_collection.is_none());
    }

    #[test]
    #[should_panic(expected = "Failed to reconstruct CV structure")]
    fn test_compute_importance_cv_mode_panics_when_cv_reconstruction_fails() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = Some(vec![(
            vec!["invalid_sample".to_string()],
            vec!["another_invalid".to_string()],
        )]);
        experiment.collections = vec![vec![Population::test()]];

        experiment.compute_importance();
    }

    #[test]
    fn test_compute_importance_uses_specified_thread_pool_settings() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = None;
        experiment.final_population = Some(Population::test());
        experiment.parameters.general.thread_number = 2;

        experiment.compute_importance();

        assert!(experiment.importance_collection.is_some());
    }

    #[test]
    #[should_panic(expected = "No final population available")]
    fn test_evaluate_on_new_dataset_panics_when_no_population() {
        let mut experiment = Experiment::test();

        experiment.final_population = None;

        experiment.evaluate_on_new_dataset("dummy_X.csv", "dummy_y.csv");
    }

    #[test]
    #[should_panic(expected = "Failed to load test data")]
    fn test_evaluate_on_new_dataset_panics_on_load_failure() {
        let mut experiment = Experiment::test();
        experiment.final_population = Some(Population::test());
        experiment.evaluate_on_new_dataset("nonexistent_X.csv", "nonexistent_y.csv");
    }

    #[test]
    fn test_evaluate_on_new_dataset_panics_on_incompatible_data() {
        use std::panic::AssertUnwindSafe;
        let mut experiment = Experiment::test();
        experiment.final_population = Some(Population::test());

        std::fs::write(
            "test_incompatible_X.csv",
            "feature1,feature2,feature3\n1,2,3\n",
        )
        .unwrap();
        std::fs::write("test_incompatible_y.csv", "class\n1\n").unwrap();

        // Use AssertUnwindSafe to wrap the closure
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            experiment
                .evaluate_on_new_dataset("test_incompatible_X.csv", "test_incompatible_y.csv");
        }));

        // Cleanup code
        let _ = std::fs::remove_file("test_incompatible_X.csv");
        let _ = std::fs::remove_file("test_incompatible_y.csv");

        // Assert that the code panicked as expected
        assert!(result.is_err(), "The test did not panic as expected");
    }
}
