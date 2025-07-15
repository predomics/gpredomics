use serde::{Deserialize, Serialize};
use crate::param::ImportanceAggregation;
use crate::data::Data;
use crate::{utils};
use crate::population::Population;
use crate::cv::CV;
use crate::param::Param;
use crate::bayesian_mcmc::MCMCAnalysisTrace;
use log::{debug, info, error};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use rayon::ThreadPoolBuilder;


#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ImportanceScope {
    // Importance can be computed on : an individual (oob), a population (FBM), a collection of population (between folds)
    Collection, // Collection of populations <=> Inter-folds FBM
    Population { id: usize }, // Intra-fold FBM (ID = Fold number)
    Individual { model_hash: u64 }, // Individual
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ImportanceType {
    OOB, 
    Coefficient, 
    PosteriorProbability 
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Importance {
    pub importance_type: ImportanceType,
    pub feature_idx: usize, 
    pub scope: ImportanceScope,
    pub aggreg_method: Option<ImportanceAggregation>,
    pub importance: f64, 
    pub is_scaled: bool,
    pub dispersion: f64,
    pub scope_pct: f64,
    pub direction: Option<usize>, // the associated class for MCMC & Coefficient
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ImportanceCollection {
    pub importances: Vec<Importance>
}

impl ImportanceCollection {
    pub fn new() -> ImportanceCollection {
        ImportanceCollection {
            importances: Vec::new()
        }
    }

    // Return importances associated with a feature
    pub fn feature(&self, idx: usize) -> ImportanceCollection {
        let importances = self
            .importances
            .iter()
            .filter(|imp| imp.feature_idx == idx)
            .cloned()
            .collect();

        ImportanceCollection { importances }
    }

    // Return importances associated with a scope and/or type
    pub fn filter(&self, scope: Option<ImportanceScope>, imp_type: Option<ImportanceType>,) -> ImportanceCollection {

        let importances = self.importances
            .iter()
            .filter(|imp| {

                let scope_ok = match &scope {
                    None => true,                                      
                    Some(ImportanceScope::Individual { .. })  =>
                        matches!(imp.scope, ImportanceScope::Individual { .. }),
                    Some(ImportanceScope::Population { .. })  =>
                        matches!(imp.scope, ImportanceScope::Population { .. }),
                    Some(ImportanceScope::Collection)        =>
                        matches!(imp.scope, ImportanceScope::Collection),
                };

                let type_ok  = imp_type.as_ref().map_or(true, |t| imp.importance_type == *t);
                scope_ok && type_ok
            })
            .cloned()
            .collect();

        ImportanceCollection { importances }
    }

    pub fn get_top(&self, pct: f64) -> ImportanceCollection {
        assert!((0.0..=1.0).contains(&pct));
        let mut subset = self.importances.clone();
        subset.sort_unstable_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        let keep = ((subset.len() as f64 * pct).ceil() as usize).max(1);
        subset.truncate(keep);
        ImportanceCollection { importances: subset }
    }

    pub fn display_feature_importance_terminal(&self, data: &Data, nb_features: usize) -> String {
        let mut map: std::collections::HashMap<usize, (f64, f64)> = std::collections::HashMap::new();
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
        
        utils::display_feature_importance_terminal(
            data, 
            &map, 
            nb_features, 
            &agg
        )
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ExperimentMetadata {
    MCMC { trace: MCMCAnalysisTrace }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Experiment {
    pub id: String,
    pub timestamp: String,
    pub gpredomics_version: String,
    pub algorithm: String,
    pub parameters: Param,

    pub train_data: Data,
    pub cv_folds_ids: Option<Vec<(Vec<String>, Vec<String>)>>,
    pub test_data: Option<Data>,

    // Results 
    pub collection: Option<Vec<Population>>, // onky if keep_trace==true
    pub final_population: Option<Population>,
    pub importance_collection: Option<ImportanceCollection>,

    // Metadata
    pub execution_time: f64,
    pub others: Option<ExperimentMetadata>
}

impl Experiment {
    pub fn compute_importance(&mut self) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.parameters.general.seed);

        let pool = ThreadPoolBuilder::new()
        .num_threads(self.parameters.general.thread_number)
        .build()
        .expect("Failed to build thread pool");

        pool.install(|| { 
            if matches!(self.cv_folds_ids, None)  {
                info!("Computing importance on final population's FBM (non-CV mode)");
                self.importance_collection = Some(self.final_population.as_ref().unwrap().select_best_population(self.parameters.cv.cv_best_models_ci_alpha).compute_pop_oob_feature_importance(&self.train_data, self.parameters.cv.n_permutations_oob, &mut rng, &self.parameters.cv.importance_aggregation, self.parameters.cv.scaled_importance, true, None));
                info!("{}", self.importance_collection.as_ref().unwrap().display_feature_importance_terminal(&self.train_data, 30));
            } else {
                info!("Computing CV importance with {} folds", self.cv_folds_ids.as_ref().unwrap().len());
                if let Some(fold_ids) = &self.cv_folds_ids {
                    debug!("Reconstructing CV object from Experiment...");
                    let cv_result = if let Some(collection) = self.collection.clone() {
                        CV::reconstruct(&self.train_data, fold_ids.clone(), collection)
                    } else {
                        error!("No population available for CV importance calculation");
                        return;
                    };
                    
                    match cv_result {
                        Ok(cv) => {
                            debug!("Computing Intra-fold and Inter-fold importances...");
                            self.importance_collection = Some(cv.compute_importance_from_cv(&self.parameters, self.parameters.cv.cv_best_models_ci_alpha, self.parameters.cv.n_permutations_oob,
                                &mut rng, &self.parameters.cv.importance_aggregation, self.parameters.cv.scaled_importance, true, true).expect("CV importance calculation failed."));
                            info!("{}", self.importance_collection.as_ref().unwrap().display_feature_importance_terminal(&self.train_data, 30));
                        }
                        Err(e) => {
                            error!("Failed to reconstruct CV structure: {}", e);
                        }
                    }
                } else {
                    error!("CV fold IDs are None but expected for CV importance calculation");
                }
            }
        })
    }
    

    pub fn save_auto<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
            let path = path.as_ref();
            let ext = path.extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();

            match ext.as_str() {
                "json" => self.save_json(path),
                "msgpack" | "mp" => self.save_messagepack(path),
                "bin" | "bincode" => self.save_bincode(path),
                _ => {
                    let json_path = path.with_extension("json");
                    self.save_json(json_path)
                }
            }
        }

        /// Save to JSON
        fn save_json<P: AsRef<std::path::Path>>(
            &self,
            path: P,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let json = serde_json::to_string_pretty(self)?;
            std::fs::write(path, json)?;
            Ok(())
        }

        /// Save to messagepack (R and Rust compatible)
        fn save_messagepack<P: AsRef<std::path::Path>>(
            &self,
            path: P,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let encoded = rmp_serde::to_vec(self)?;
            std::fs::write(path, encoded)?;
            Ok(())
        }

        /// Save as bincode (Rust compatible)
        fn save_bincode<P: AsRef<std::path::Path>>(
            &self,
            path: P,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let encoded = bincode::serialize(self)?;
            std::fs::write(path, encoded)?;
            Ok(())
        }

    pub fn load_auto<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let ext = path.extension()
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

    fn load_json<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let experiment: Experiment = serde_json::from_str(&content)?;
        Ok(experiment)
    }

    fn load_messagepack<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let experiment: Experiment = rmp_serde::from_slice(&bytes)?;
        Ok(experiment)
    }

    fn load_bincode<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let experiment: Experiment = bincode::deserialize(&bytes)?;
        Ok(experiment)
    }

    fn load_with_fallback<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        
        if let Ok(experiment) = Self::load_json(path) {
            return Ok(experiment);
        }
        
        if let Ok(experiment) = Self::load_messagepack(path) {
            return Ok(experiment);
        }
        
        if let Ok(experiment) = Self::load_bincode(path) {
            return Ok(experiment);
        }
        
        Err("Unable to load the experience".into())
    }

    pub fn display_results(&self) {
        println!("=== EXPERIMENT RESULTS ===");
        println!("GPREDOMICS version: v{}", self.gpredomics_version);
        println!("Experiment ID: {}", self.id);
        println!("Timestamp: {}", self.timestamp);
        println!("Algorithm: {}", self.algorithm);
        println!("Execution time: {:.2}s", self.execution_time);

        if let Some(fold_ids) = &self.cv_folds_ids {
            debug!("Reconstructing CV object from Experiment...");
            let cv_result = if let Some(collection) = self.collection.clone() {
                CV::reconstruct(&self.train_data, fold_ids.clone(), collection)
            } else {
                error!("No population available for displaying FBMs");
                return;
            };

            if let Ok(cv_data) = cv_result {
                let all_fold_fbms: Vec<(Population, Data)> = (0..cv_data.validation_folds.len())
                .map(|i| cv_data.extract_fold_fbm(i, self.parameters.cv.cv_best_models_ci_alpha)).collect();

                let mut merged_fbm = Population::new();
                for (fold_fbm, _) in &all_fold_fbms {
                    merged_fbm.individuals.extend(fold_fbm.individuals.clone());
                }

            for (i, (fold_fbm, valid_data)) in all_fold_fbms.iter().enumerate() {
                println!("\x1b[1;93mFold #{}\x1b[0m", i+1);
                // Validation fold = valid_data, train complet = &self.train_data
                println!("{}", fold_fbm.clone().display(valid_data, Some(&self.train_data), &self.parameters));
            }
            }

        } else {
            if let Some(mut final_pop) = self.final_population.clone() {
                println!("{}", final_pop.display(&self.train_data, self.test_data.as_ref(), &self.parameters));
            } else {
                println!("No final population available");
            }}
        
        if let Some(ref importance_collection) = self.importance_collection {
            let top_features = if self.cv_folds_ids.is_some() {
                // Mode CV : prioriser les importances Collection (inter-fold)
                let collection_features = importance_collection.filter(Some(ImportanceScope::Collection), None);
                if !collection_features.importances.is_empty() {
                    collection_features.get_top(0.1)
                } else {
                    // Fallback : toutes les importances sans filtre
                    importance_collection.get_top(0.1)
                }
            } else {
                // Mode non-CV : utiliser les importances Population
                let population_features = importance_collection.filter(
                    Some(ImportanceScope::Population { id: 0 }), 
                    None
                );
                if !population_features.importances.is_empty() {
                    population_features.get_top(0.1)
                } else {
                    // Fallback : toutes les importances sans filtre
                    importance_collection.get_top(0.1)
                }
            };
            
            println!("{}", top_features.display_feature_importance_terminal(&self.train_data, top_features.importances.len()));
        }
    }

      pub fn evaluate_on_new_dataset(&mut self, X_path: &str, y_path: &str) -> Result<String, String> {
        let mut new_data = Data::new();
        
        if let Err(e) = new_data.load_data(X_path, y_path) {
            return Err(format!("Failed to load test data: {}", e));
        }

        if self.train_data.check_compatibility(&new_data) == false {
            return Err(format!("New test data to evaluate isn't compatible with train data"));
        } 

        let final_pop = self.final_population.as_mut()
            .ok_or("No final population available in experiment")?;

        
        let results = final_pop.display(&self.train_data, Some(&new_data), &self.parameters);
        
        Ok(results)
    }
}

// pub struct Court {
//     judges: Population,
//     penalties: Option<Vec<f64>>
    
// }

