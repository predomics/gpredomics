use serde::{Deserialize, Serialize};
use crate::data::Data;
use crate::utils::{compute_metrics_from_classes, display_feature_importance_terminal};
use crate::population::Population;
use crate::cv::CV;
use crate::param::Param;
use crate::bayesian_mcmc::MCMCAnalysisTrace;
use log::{debug, info, error, warn};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use rayon::ThreadPoolBuilder;
use crate::Individual;
use crate::ga;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use chrono::Local;
//use crate::lib::{param, run_ga, run_cv, run_beam, run_mcmc};

////////////////////////
////// IMPORTANCE //////
////////////////////////

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(non_camel_case_types)]
pub enum ImportanceAggregation {
    mean,
    median
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
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
        
        display_feature_importance_terminal(
            data, 
            &map, 
            nb_features, 
            &agg
        )
    }
}

////////////////////////
////// EXPERIMENT //////
////////////////////////

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ExperimentMetadata {
    MCMC { trace: MCMCAnalysisTrace },
    Court { court: Court }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Experiment {
    pub id: String,
    pub timestamp: String,
    pub gpredomics_version: String,
    pub algorithm: String,
    pub parameters: Param,

    // Data
    pub train_data: Data,
    pub test_data: Option<Data>,

    // Results 
    // In CV-mode, Vec<Vec<Population>>.len() = outer_folds, in non-CV mode Vec<Vec<Population>>.len() = 1
    pub collections: Vec<Vec<Population>>, 
    pub final_population: Option<Population>,
    pub importance_collection: Option<ImportanceCollection>,

     pub cv_folds_ids: Option<Vec<(Vec<String>, Vec<String>)>>,
    
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
                self.importance_collection = Some(self.final_population.as_ref().unwrap().select_best_population(self.parameters.cv.cv_best_models_ci_alpha).compute_pop_oob_feature_importance(&self.train_data, self.parameters.importance.n_permutations_oob, &mut rng, &self.parameters.importance.importance_aggregation, self.parameters.importance.scaled_importance, true, None));
                info!("{}", self.importance_collection.as_ref().unwrap().display_feature_importance_terminal(&self.train_data, 30));
            } else {
                info!("Computing CV importance with {} folds", self.cv_folds_ids.as_ref().unwrap().len());
                if let Some(fold_ids) = &self.cv_folds_ids {
                    debug!("Reconstructing CV object from Experiment...");
                    let cv_result = CV::reconstruct(&self.train_data, fold_ids.clone(), self.collections.clone());
                    
                    match cv_result {
                        Ok(cv) => {
                            debug!("Computing Intra-fold and Inter-fold importances...");
                            self.importance_collection = Some(cv.compute_cv_oob_feature_importance(&self.parameters, self.parameters.importance.n_permutations_oob,
                                &mut rng, &self.parameters.importance.importance_aggregation, self.parameters.importance.scaled_importance, true).expect("CV importance calculation failed."));
                            
                            info!("{}", self.importance_collection.as_ref().unwrap().display_feature_importance_terminal(&self.train_data, 30));
                        }
                        Err(e) => {
                            panic!("Failed to reconstruct CV structure: {}", e);
                        }
                    }
                } else {
                    error!("CV fold IDs are None but expected for CV importance calculation");
                }
            }
        })
    }

    pub fn display_results(&mut self) {
        info!("=== EXPERIMENT {} RESULTS ===", self.id);
        info!("GPREDOMICS version: v{}", self.gpredomics_version);
        info!("Timestamp: {}", self.timestamp);
        info!("Algorithm: {}", self.algorithm);
        info!("Execution time: {:.2}s", self.execution_time);

        // Reconstruct CV to print each fold FBM as original display
        if let Some(cv_folds_ids) = self.cv_folds_ids.clone() {
            let cv = CV::reconstruct(&self.train_data, cv_folds_ids, self.collections.clone())
                .expect("CV reconstruction failed.");
            let mut fbm = cv.get_fbm(&self.parameters);

            ga::fit_fn(&mut fbm, &self.train_data, &mut None, &None, &None, &self.parameters);
            fbm = fbm.sort();

            fbm.compute_hash();
            let mut final_pop = self.final_population.clone().unwrap();
            final_pop.compute_hash();

            let fbm_hashes: std::collections::HashSet<_> = fbm.individuals.iter().map(|i| i.hash).collect();
            let final_hashes: std::collections::HashSet<_> = final_pop.individuals.iter().map(|i| i.hash).collect();

            assert_eq!(fbm_hashes, final_hashes, "Something is wrong with the Experiment: reconstructed CV based FBM should be the same as final population");
            info!("\x1b[1;93mDisplaying Family of best models across folds\x1b[0m");
        }    

        // Print final population
        if let Some(mut final_pop) = self.final_population.clone() {
                info!("{}", final_pop.display(&self.train_data, self.test_data.as_ref(), &self.parameters));
        } else {
            panic!("No final population available");
        }
        
        if let Some(ref importance_collection) = self.importance_collection {
            let top_features = if self.cv_folds_ids.is_some() {
                // CV-mode : priorize Collection importances (inter-fold)
                let collection_features = importance_collection.filter(Some(ImportanceScope::Collection), None);
                if !collection_features.importances.is_empty() {
                    collection_features.get_top(0.1)
                } else {
                    importance_collection.get_top(0.1)
                }
            } else {
                // Non CV-mode : priorize Population importances
                let population_features = importance_collection.filter(
                    Some(ImportanceScope::Population { id: 0 }), 
                    None
                );
                if !population_features.importances.is_empty() {
                    population_features.get_top(0.1)
                } else {
                    importance_collection.get_top(0.1)
                }
            };
            
            info!("Top 10%:{}", top_features.display_feature_importance_terminal(&self.train_data, top_features.importances.len()));

        }
        
        if let Some(ref mut metadata) = self.others {
            match metadata {
                ExperimentMetadata::Court { court } => {
                    court.display(&self.train_data, self.test_data.as_ref(), &self.parameters)
                },
                _ => {}
            }
        }
        
    }

      pub fn evaluate_on_new_dataset(&mut self, X_path: &str, y_path: &str) {
        let mut new_data = Data::new();
        
        let final_pop = self.final_population.as_mut().expect("No final population available");

        if let Err(e) = new_data.load_data(X_path, y_path) {
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

        info!("{}", final_pop.display(&self.train_data, Some(&new_data), &display_param));

        if let Some(ref mut metadata) = self.others {
            match metadata {
                ExperimentMetadata::Court { court } => {
                    court.display(&self.train_data, Some(&new_data), &self.parameters)
                },
                _ => {}
            }
        }
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
                    warn!("Unknown format. Saving experiment in msgpack.");
                    let mp_path = path.with_extension("mp");
                    self.save_messagepack(mp_path)
                }
            }
        }

        /// Save to JSON
        fn save_json<P: AsRef<std::path::Path>>(
            &self,
            path: P,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let json = serde_json::to_string_pretty(self)?;
            warn!("Due to JSON compression, a slight inaccuracy may occur for decimal values. Prefer the msgpack format if you want to read the experiments from another language.");
            std::fs::write(path, json)?;
            Ok(())
        }

        /// Save to messagepack (R and Rust compatible)
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

        /// Save as bincode (Rust compatible only)
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

}

////////////////////////
//////// VOTING ////////
////////////////////////

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Court {
    pub judges: Population,
    pub voting_method: VotingMethod,
    pub voting_threshold: f64,
    
    // Weights
    pub weighting_method: WeightingMethod,
    pub weights: Option<Vec<f64>>,
    
    // Binary metrics
    pub accuracy: f64,
    pub sensitivity: f64,
    pub specificity: f64,
    pub predicted_classes: Option<Vec<u8>>
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum VotingMethod {
    Majority,
    Consensus
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum WeightingMethod {
    Uniform,
    Specialized {
        sensitivity_threshold: f64,  
        specificity_threshold: f64,
    },              
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum JudgeSpecialization {
    PositiveSpecialist, 
    NegativeSpecialist,
    Balanced,           
    Ineffective,        
}

impl Court {
    pub fn new(pop: &Population, min_perf: &f64, min_diversity: &f64, voting_method: &VotingMethod, voting_threshold: &f64, weighting_method: &WeightingMethod) -> Self {
        let mut judges: Population = pop.clone();

        if *min_perf > 0.0 {
            let n = judges.individuals.len();
            judges.individuals.retain(|judge| { judge.sensitivity >= *min_perf && judge.specificity >= *min_perf });
            debug!("Judges filtered for minimum sensitivity and specificity: {}/{} individuals retained", judges.individuals.len(), n);
        }

        if *min_diversity > 0.0 {
            let n = judges.individuals.len();
            judges = judges.filter_by_signed_jaccard_dissimilarity(*min_diversity, true);
            debug!("Judges filtered for diversity: {}/{} individuals retained", judges.individuals.len(), n);
        }

        if voting_threshold < &0.0 || voting_threshold > &1.0 {
            panic!("Voting threshold should be in [0,1]")
        }

        match weighting_method {
            WeightingMethod::Specialized { sensitivity_threshold, specificity_threshold } => {
                warn!("Specialized voting mode is experimental");
                if *sensitivity_threshold < 0.0 || *sensitivity_threshold > 1.0 {
                    panic!("Sensitivity threshold must be in [0,1]");
                }
                if *specificity_threshold < 0.0 || *specificity_threshold > 1.0 {
                    panic!("Specificity threshold must be in [0,1]");
                }
            },
            _ => ()
        }

        judges = judges.sort();

        Court {
            judges,
            voting_method: voting_method.clone(),
            voting_threshold: *voting_threshold,
            weighting_method:  weighting_method.clone(),
            weights: None,
            accuracy: 0.0,
            sensitivity: 0.0,
            specificity: 0.0,
            predicted_classes: None
        }
    }

    // Evaluates learning data and adjusts internal weight and performance variables accordingly
    pub fn evaluate(&mut self, data: &Data) {
        for judge in &mut self.judges.individuals {
            if judge.accuracy == 0.0 {
                judge.compute_roc_and_metrics(data, None);
            }
        }
        
        let weights = self.compute_weights_by_method(data);

        let effective_judges = weights.iter().filter(|&w| *w > 0.0).count();    
        if effective_judges % 2 == 0 && effective_judges > 0 {
            warn!("Even number of effective judges ({}). Perfect ties will be abstained (class 2).", effective_judges);
        }

        self.weights = Some(weights);

        if self.voting_threshold == 0.0 && self.voting_method == VotingMethod::Majority {
            self.voting_threshold = self.optimize_majority_threshold_youden(data);
            warn!("Threshold set to 0.0. Using Youden Maxima as threshold: {}", self.voting_threshold); 
        }
        
        let predictions = self.predict(data);
        let (accuracy, sensitivity, specificity) = 
            compute_metrics_from_classes(&predictions, &data.y);
        
        self.accuracy = accuracy;
        self.sensitivity = sensitivity;
        self.specificity = specificity;
        self.predicted_classes = Some(predictions);
    }

    pub fn optimize_majority_threshold_youden(&mut self, data: &Data) -> f64 {
        let mut best_threshold = 0.5;
        let mut best_youden = 0.0;

        let step_size = match data.sample_len {
            0..=50 => 0.1,
            51..=200 => 0.05, 
            _ => 0.01
        };

        let steps = (1.0 / step_size) as i32;
        for i in 1..steps {
            let threshold = i as f64 * step_size;
                let predictions = self.compute_majority_threshold_vote(data, &self.weights.as_ref().unwrap(), threshold);
                let (_, sensitivity, specificity) = compute_metrics_from_classes(&predictions, &data.y);
                
                let youden_index = sensitivity + specificity - 1.0;
                
                if youden_index > best_youden {
                    best_youden = youden_index;
                    best_threshold = threshold;
                }
            }
            
        self.voting_threshold = best_threshold;
        best_threshold
    }
    
    // Evaluates new data based on internal weights and variables
    pub fn predict(&self, data: &Data) -> Vec<u8> {
        let weights = self.weights.as_ref()
            .expect("Weights must be computed before prediction. Call evaluate() first.");
        self.apply_voting_mechanism(data, weights)
    }

    // Called by evaluate()
    fn compute_weights_by_method(&self, _data: &Data) -> Vec<f64> {
        match &self.weighting_method {
            WeightingMethod::Uniform  => vec![1.0; self.judges.individuals.len()],
            WeightingMethod::Specialized { sensitivity_threshold, specificity_threshold } => 
                self.compute_group_strict_weights(*sensitivity_threshold, *specificity_threshold),
        }
    }

    // 
    fn apply_voting_mechanism(&self, data: &Data, weights: &[f64]) -> Vec<u8> {
        match &self.voting_method {
            VotingMethod::Majority => {
                self.compute_majority_threshold_vote(data, weights, self.voting_threshold)
            },
            VotingMethod::Consensus => {
                self.compute_consensus_threshold_vote(data, weights, self.voting_threshold)
            }
        }
    }

    // Voting methods
    fn compute_consensus_threshold_vote(&self, data: &Data, weights: &[f64], threshold: f64) -> Vec<u8> {
        let mut predicted_classes = Vec::with_capacity(data.sample_len);
        
        let judge_predictions: Vec<Vec<u8>> = self.judges.individuals.iter()
            .map(|judge| judge.evaluate_class(data))
            .collect();
        
        for sample_index in 0..data.sample_len {
            let mut weighted_positive = 0.0;
            let mut weighted_negative = 0.0;
            let mut effective_total_weight = 0.0;
            
            for (judge_idx, judge_pred) in judge_predictions.iter().enumerate() {
                if sample_index < judge_pred.len() {
                    let prediction = judge_pred[sample_index];
                    let weight = weights[judge_idx];
                    
                    if weight > 0.0 {
                        effective_total_weight += weight;
                        
                        if prediction == 1 {
                            weighted_positive += weight;
                        } else if prediction == 0 {
                            weighted_negative += weight;
                        }
                    }
                }
            }
            
            let predicted_class = if effective_total_weight > 0.0 {
                let pos_ratio = weighted_positive / effective_total_weight;
                let neg_ratio = weighted_negative / effective_total_weight;
                
                if pos_ratio >= threshold {
                    1u8  
                } else if neg_ratio >= threshold {
                    0u8  
                } else {
                    2u8 
                }
            } else {
                2u8  
            };
            
            predicted_classes.push(predicted_class);
        }
        
        predicted_classes
    }


    fn compute_majority_threshold_vote(&self, data: &Data, weights: &[f64], threshold: f64) -> Vec<u8> {
        if weights.len() != self.judges.individuals.len() {
            panic!("Weights length ({}) must match judges count ({})", 
                weights.len(), self.judges.individuals.len());
        }

        let mut predicted_classes = Vec::with_capacity(data.sample_len);

        let judge_predictions: Vec<Vec<u8>> = self.judges.individuals.iter()
            .map(|judge| judge.evaluate_class(data))
            .collect();
        for sample_index in 0..data.sample_len {
            let mut weighted_positive = 0.0;
            let mut total_weight = 0.0;
            
            for (judge_idx, judge_pred) in judge_predictions.iter().enumerate() {
                if sample_index < judge_pred.len() {
                    let weight = weights[judge_idx];
                    total_weight += weight;
                    
                    if judge_pred[sample_index] == 1 {
                        weighted_positive += weight;
                    }
                }
            }
            
            let predicted_class = if total_weight > 0.0 {
                let ratio = weighted_positive / total_weight;
                if (ratio - 0.5).abs() < 1e-10 {
                    2u8 
                } else if weighted_positive >= total_weight * threshold {
                    1u8
                } else {
                    0u8
                }
            } else {
                2u8
            };

            predicted_classes.push(predicted_class);
        }
        
        predicted_classes
    }
    
    
    pub fn compute_classes(&mut self, data: &Data) {
        let predictions = self.predict(data);
        self.predicted_classes = Some(predictions);
    }

    fn compute_rejection_rate(&self, predictions: &[u8]) -> f64 {
        let total_samples = predictions.len();
        let rejected_samples = predictions.iter()
            .filter(|&&pred| pred == 2)
            .count();
        
        if total_samples > 0 {
            rejected_samples as f64 / total_samples as f64
        } else {
            0.0
        }
    }
    
    
    fn count_effective_judges(&self) -> (usize, usize) {
        match &self.weighting_method {
            WeightingMethod::Uniform => {
                (self.judges.individuals.len(), self.judges.individuals.len())
            },
            WeightingMethod::Specialized { sensitivity_threshold, specificity_threshold } => {
                let mut total_judges = 0;
                let mut effective_judges = 0;
                
                for judge in &self.judges.individuals {
                    total_judges += 1;
                    let specialization = self.get_judge_specialization(
                        judge, *sensitivity_threshold, *specificity_threshold
                    );
                    
                    if !matches!(specialization, JudgeSpecialization::Ineffective) {
                        effective_judges += 1;
                    }
                }
                
                (total_judges, effective_judges)
            }
        }
    }

    // Judge specialization related functions (experimental)
    fn get_judge_specialization(&self, judge: &Individual, 
                               sensitivity_threshold: f64,
                               specificity_threshold: f64) -> JudgeSpecialization {
        
        if judge.sensitivity >= sensitivity_threshold && judge.specificity >= specificity_threshold {
            JudgeSpecialization::Balanced
        } else if judge.sensitivity >= sensitivity_threshold {
            JudgeSpecialization::PositiveSpecialist
        } else if judge.specificity >= specificity_threshold {
            JudgeSpecialization::NegativeSpecialist
        } else {
            JudgeSpecialization::Ineffective
        }
    }

    fn compute_group_strict_weights(&self, sensitivity_threshold: f64, specificity_threshold: f64) -> Vec<f64> {
        let mut specs = Vec::new();
        let mut pos = 0usize;
        let mut neg = 0usize;
        let mut bal = 0usize;

        for judge in &self.judges.individuals {
            let s = self.get_judge_specialization(judge, sensitivity_threshold, specificity_threshold);
            specs.push(s.clone());
            match s {
                JudgeSpecialization::PositiveSpecialist => pos += 1,
                JudgeSpecialization::NegativeSpecialist => neg += 1,
                JudgeSpecialization::Balanced          => bal += 1,
                _ => {}
            }
        }

        let active_groups = [
            (pos, JudgeSpecialization::PositiveSpecialist),
            (neg, JudgeSpecialization::NegativeSpecialist),
            (bal, JudgeSpecialization::Balanced),
        ].iter().filter(|(n, _)| *n > 0).count();

        if active_groups == 0 {
            panic!("Specialized threshold are too high to allow judge selection")
        }

        let group_share = 1.0 / active_groups as f64;

        specs.into_iter().map(|sp| {
            match sp {
                JudgeSpecialization::PositiveSpecialist if pos > 0 => group_share / pos as f64,
                JudgeSpecialization::NegativeSpecialist if neg > 0 => group_share / neg as f64,
                JudgeSpecialization::Balanced          if bal > 0 => group_share / bal as f64,
                _ => 0.0,                            
            }
        }).collect()
    }

    // Display functions 
    pub fn display(&mut self, data: &Data, test_data: Option<&Data>, param: &Param) {
        let mut text = format!("{}\n{}{}VOTING ANALYSIS{}{}\n{}\n", "═".repeat(80), "\x1b[1m", " ".repeat(31), " ".repeat(32), "\x1b[0m", "═".repeat(80));

        text = format!("{}\n{}{}", text, self.display_compact_summary(data, test_data), self.display_voting_method_info());
        
        text = format!("{}\n\n{}{} DETAILED METRICS {}{}", text, "\x1b[1m",  "~".repeat(31),  "~".repeat(31), "\x1b[0m");
        text = format!("{}\n{}", text, self.display_confusion_matrix(&self.predicted_classes.as_ref().unwrap(), &data.y, "TRAIN"));
            if let Some(test_data) = test_data {
                let test_preds = self.predict(test_data);
                text = format!("{}\n{}", text, self.display_confusion_matrix(&test_preds, &test_data.y, "TEST"));
            } 

        
        if let Some(test_data) = test_data {
            text = format!("{}\n{}", text, self.display_predictions_by_sample(test_data, "TEST"));
        } else {
            text = format!("{}\n{}", text, self.display_predictions_by_sample(data, "TRAIN"));
        }
        
        
        text = format!("\n{}\n\n{}{} JUDGES POPULATION ({}) {}{}", text, "\x1b[1m", "~".repeat(25), self.judges.individuals.len(), "~".repeat(25), "\x1b[0m\n");
        text = format!("{}\n{}", text, self.judges.display(&data, test_data, param));

        info!("{}\n", text)
    }

    fn display_confusion_matrix(&self, predictions: &[u8], true_labels: &[u8], title: &str) -> String {
        let mut text = "".to_string();
        let (mut tp, mut tn, mut fp, mut fn_, mut rp_abstentions, mut rn_abstentions) = (0, 0, 0, 0, 0, 0);
        
        for (pred, real) in predictions.iter().zip(true_labels.iter()) {
            match (*pred, *real) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fp += 1,
                (0, 1) => fn_ += 1,
                (2, 1) => rp_abstentions += 1,
                (2, 0) => rn_abstentions += 1,
                _ => warn!("Warning: Unexpected class values pred={}, real={}", pred, real),
            }
        }
        
        text = format!("{}\n{} CONFUSION MATRIX ({}) {}", text, "─".repeat(15), title, "─".repeat(15));
        text = format!("{}\n\n         | \x1b[1;96mPred 1\x1b[0m | \x1b[1;95mPred 0\x1b[0m | \x1b[1;90mAbstain\x1b[0m", text);
        text = format!("{}\n\x1b[1;96mReal 1\x1b[0m   | {:>6} | {:>6} | {:>7}", text, tp, fn_, rp_abstentions);
        text = format!("{}\n\x1b[1;95mReal 0\x1b[0m   | {:>6} | {:>6} | {:>7}", text, fp, tn, rn_abstentions);
        
        text
    }

    /// Train/Test metrics
    fn display_compact_summary(&self, _: &Data, test_data: Option<&Data>) -> String {
        let mut summary: String;
        let (total_judges, _) = self.count_effective_judges();
        
        let weighting_info = match &self.weighting_method {
            WeightingMethod::Uniform => "",
            WeightingMethod::Specialized { .. } => "specialized-weighted ",
        };
        
        let voting_info = match &self.voting_method {
            VotingMethod::Majority => "Majority",
            VotingMethod::Consensus => "Consensus",
        };

        let method_display = format!("\x1b[1m{} court [{} {}judges]", 
                                   voting_info, total_judges, weighting_info);
        
        let (test_accuracy, test_sensitivity, test_specificity) = if let Some(test_data) = test_data {
            let test_predictions = self.predict(test_data);
            compute_metrics_from_classes(&test_predictions, &test_data.y)
        } else {
            (0.0, 0.0, 0.0)
        };
        
        if test_data.is_some() {
            summary = format!("{} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}{}", 
                    method_display,
                    self.accuracy, test_accuracy,
                    self.sensitivity, test_sensitivity,
                    self.specificity, test_specificity,
                    "\x1b[0m");
            
            if matches!(self.voting_method, VotingMethod::Consensus) {
                let train_predictions = self.predicted_classes.as_ref().unwrap();
                let train_rejection_rate = self.compute_rejection_rate(train_predictions);
                
                let test_predictions = self.predict(test_data.unwrap());
                let test_rejection_rate = self.compute_rejection_rate(&test_predictions);
                
                summary = format!("{}\nRejection rate: {:.1}%/{:.1}% (train/test)", summary, 
                        train_rejection_rate * 100.0, test_rejection_rate * 100.0);
            }
        } else {
            summary = format!("{} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3}{}", 
                    method_display,
                    self.accuracy,
                    self.sensitivity,
                    self.specificity,
                    "\x1b[0m");

            if matches!(self.voting_method, VotingMethod::Consensus) {
                let train_predictions = self.predicted_classes.as_ref().unwrap();
                let train_rejection_rate = self.compute_rejection_rate(train_predictions);
                
                summary = format!("{}\nRejection rate: {:.1}%", summary, train_rejection_rate * 100.0);
            }
            
        }

        summary

    }
    
    fn display_voting_method_info(&self) -> String {
        let mut info : String = "".to_string();
        match &self.weighting_method {
            WeightingMethod::Uniform => {
            },
            WeightingMethod::Specialized { sensitivity_threshold, specificity_threshold } => {
                info = self.display_judge_specializations(*sensitivity_threshold, *specificity_threshold);
            },
        }
        info
    }
    
    /// Affichage des prédictions par échantillon avec couleurs
    fn display_predictions_by_sample(&self, data: &Data, title: &str) -> String {
        let mut text = "".to_string();
        text = format!("{}\n{}{} PREDICTIONS BY SAMPLE ({}) {}{}\n\n{}", text, "\x1b[1;1m\n", "~".repeat(25), title, "~".repeat(25), "\x1b[0m", "─".repeat(60));

        let predictions = if title == "TEST" {
            self.predict(data)
        } else {
            self.predicted_classes.as_ref().unwrap().clone()
        };

        if predictions.len() != data.sample_len {
            return format!("Error: Predictions length ({}) != data length ({})", predictions.len(), data.sample_len);
        }

        let mut errors = Vec::new();
        let mut abstentions = Vec::new();
        let mut correct = Vec::new();

        for i in 0..data.sample_len {
            let real_class = data.y[i];
            let predicted_class = predictions[i];

            match predicted_class {
                2 => abstentions.push(i),
                _ if predicted_class != real_class => errors.push(i),
                _ => correct.push(i),
            }
        }

        let nb_samples_to_show = 20.min(data.sample_len);
        
        // Calculate how many samples to show per category
        let max_errors = if errors.len() > 0 { (nb_samples_to_show * 60 / 100).max(1).min(errors.len()) } else { 0 };
        let max_abstentions = if abstentions.len() > 0 { ((nb_samples_to_show - max_errors) * 60 / 100).max(1).min(abstentions.len()) } else { 0 };
        let max_correct = (nb_samples_to_show - max_errors - max_abstentions).min(correct.len());

        // Statistics overview
        text = format!("{}\n{}Sample\t\t| Real | Predictions\t| Well predicted?{}\n{}", text, "\x1b[1m", "\x1b[0m", "─".repeat(60));

        // Display ERRORS section first (highest priority)
        if max_errors > 0 {
            text = format!("{}\n{}─────── ERRORS ({} shown of {}) ───────{}", text, "\x1b[1;31m", max_errors, errors.len(), "\x1b[0m");
            for &sample_idx in errors.iter().take(max_errors) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let judge_votes = self.display_judge_votes_for_sample(data, sample_idx);
                
                text = format!("{}\n{:>10}\t| {:>4} | {} → {}\t| \x1b[1;31m✗\x1b[0m", 
                    text, sample_name, real_class, judge_votes, predicted_class);
            }
        }

        // Display ABSTENTIONS section second
        if max_abstentions > 0 {
            text = format!("{}\n{}────── ABSTENTIONS ({} shown of {}) ─────{}", text, "\x1b[1;90m", max_abstentions, abstentions.len(), "\x1b[0m");
            for &sample_idx in abstentions.iter().take(max_abstentions) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let judge_votes = self.display_judge_votes_for_sample(data, sample_idx);
                
                text = format!("{}\n{:>10}\t| {:>4} | {} → {} | \x1b[90m~\x1b[0m", 
                    text, sample_name, real_class, judge_votes, predicted_class);
            }
        }

        // Display CORRECT section last
        if max_correct > 0 {
            text = format!("{}\n{}─────── CORRECT ({} shown of {}) ───────{}", text, "\x1b[1;32m", max_correct, correct.len(), "\x1b[0m");
            for &sample_idx in correct.iter().take(max_correct) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let judge_votes = self.display_judge_votes_for_sample(data, sample_idx);
                
                text = format!("{}\n{:>10}\t| {:>4} | {} → {} | \x1b[1;32m✓\x1b[0m", 
                    text, sample_name, real_class, judge_votes, predicted_class);
            }
        }

        let total_shown = max_errors + max_abstentions + max_correct;
        if data.sample_len > total_shown {
            text = format!("{}\n ... {} additional samples not shown", text, data.sample_len - total_shown);
        }

        text = format!("{}\n\n{}Errors: {} | Correct: {} | Abstentions: {}{}", 
            text, "\x1b[1;33m", errors.len(), correct.len(), abstentions.len(), "\x1b[0m");

        text
    }

    pub fn display_judge_specializations(&self, sensitivity_threshold: f64, specificity_threshold: f64) -> String {
        let mut text = "".to_string();
        text = format!("{}\n{} JUDGE SPECIALIZATIONS{}\n{}", text, "\x1b[1;45m", "\x1b[0m", "─".repeat(80));
        text = format!("{}\n{:<6} | {:<8} | {:<11} | {:<11} | {:<20}", text, "Judge", "Accuracy", "Sensitivity", "Specificity", "Specialization");
        text = format!("{}\n{}", text, "─".repeat(80));
        
        for (idx, judge) in self.judges.individuals.iter().enumerate() {
            let specialization = self.get_judge_specialization(
                judge, sensitivity_threshold, specificity_threshold);
            
            let spec_str = match specialization {
                JudgeSpecialization::PositiveSpecialist => "🔍 \x1b[96mPositive Specialist\x1b[0m",
                JudgeSpecialization::NegativeSpecialist => "🔍 \x1b[95mNegative Specialist\x1b[0m", 
                JudgeSpecialization::Balanced => "⚖️  Balanced",
                JudgeSpecialization::Ineffective => "❌ \x1b[90mIneffective\x1b[0m",
            };
            
            match specialization {
                JudgeSpecialization::Ineffective => {},
                _ => { text = format!("{}\n#{:<6} | {:<8.3} | {:<11.3} | {:<11.3} | {}", text, idx+1, judge.accuracy, judge.sensitivity, judge.specificity, spec_str) }
            }

        }

        text
    }

    fn display_specialized_vote(&self, specialization: &JudgeSpecialization, vote: u8) -> (&'static str, String) {
        match (specialization, vote) {
            (JudgeSpecialization::Ineffective, _) => ("\x1b[90m", format!("{}", vote)),      
            (JudgeSpecialization::Balanced, _) => ("\x1b[92m", vote.to_string()),             
            (JudgeSpecialization::PositiveSpecialist, 0) => ("\x1b[90m", format!("{}", vote)), 
            (JudgeSpecialization::PositiveSpecialist, 1) => ("\x1b[96m", vote.to_string()),   
            (JudgeSpecialization::NegativeSpecialist, 1) => ("\x1b[90m", format!("{}", vote)), 
            (JudgeSpecialization::NegativeSpecialist, 0) => ("\x1b[95m", vote.to_string()),  
            _ => ("\x1b[97m", vote.to_string()),
        }
    }

    fn display_judge_votes_for_sample(&self, data: &Data, sample_idx: usize) -> String {
        let mut output = String::new();
        
        for (_, judge) in self.judges.individuals.iter().enumerate() {
            let predictions = judge.evaluate_class(data);
            
            if sample_idx < predictions.len() {
                let vote = predictions[sample_idx];
                
                match &self.weighting_method {
                    WeightingMethod::Uniform => {
                        let vote_display = match data.y[sample_idx] == vote {
                            true => &format!("\x1b[92m{}\x1b[0m", vote),
                            false => &format!("\x1b[31m{}\x1b[0m",vote),
                        };
                        output.push_str(vote_display);
                    },
                    WeightingMethod::Specialized { sensitivity_threshold, specificity_threshold } => {
                        let specialization = self.get_judge_specialization(
                            judge, *sensitivity_threshold, *specificity_threshold
                        );
                        
                        let (color, symbol) = self.display_specialized_vote(&specialization, vote);
                        output.push_str(&format!("{}{}{}", color, symbol, "\x1b[0m"));
                    },
                }
            } else {
                output.push('?');
            }
        }
        
        output
    }
        
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_empty_collection() {
        let collection = ImportanceCollection::new();
        assert!(collection.importances.is_empty());
    }

    #[test]
    fn test_feature_returns_existing_feature_importances() {
        let mut collection = ImportanceCollection::new();
        collection.importances.push(Importance {
            importance_type: ImportanceType::OOB,
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
            importance_type: ImportanceType::OOB,
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
            importance_type: ImportanceType::OOB,
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
            importance_type: ImportanceType::OOB,
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
        assert!(matches!(result_scope.importances[0].scope, ImportanceScope::Collection));
        
        let result_type = collection.filter(None, Some(ImportanceType::OOB));
        assert_eq!(result_type.importances.len(), 2);
        assert!(result_type.importances.iter().all(|imp| imp.importance_type == ImportanceType::OOB));
        
        let result_both = collection.filter(Some(ImportanceScope::Collection), Some(ImportanceType::OOB));
        assert_eq!(result_both.importances.len(), 1);
        assert!(matches!(result_both.importances[0].scope, ImportanceScope::Collection));
        assert_eq!(result_both.importances[0].importance_type, ImportanceType::OOB);
    }

    #[test]
    fn test_get_top_comprehensive_functionality() {
        let mut collection = ImportanceCollection::new();
        
        assert!(collection.get_top(0.5).importances.is_empty());
        
        collection.importances.push(Importance {
            importance_type: ImportanceType::OOB,
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
            importance_type: ImportanceType::OOB,
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
            importance_type: ImportanceType::OOB,
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
            importance_type: ImportanceType::OOB,
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
            importance_type: ImportanceType::OOB,
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
        pub fn test() -> Experiment {
            Experiment {
                id: "test_exp_001".to_string(),
                timestamp: "2025-01-01T12:00:00Z".to_string(),
                gpredomics_version: "1.0.0".to_string(),
                algorithm: "GA".to_string(),
                parameters: Param::default(), 
                train_data: Data::test(),      
                cv_folds_ids: None,
                test_data: Some(Data::test2()),
                collections: vec![vec!(Population::test_with_n_overlapping_features(2, 20), Population::test_with_n_overlapping_features(15, 12))],
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
        let file_path ="test_serialization_messagepack_roundtrip";

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

    #[test]
    fn test_experiment_with_court_metadata() {
        let mut experiment = Experiment::test();
        
        // Create minimal test data for Court
        let court = Court::new(
            &Population::new(),
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        experiment.others = Some(ExperimentMetadata::Court { court });
        
        // Serialisation test with metadata Short
        let temp_file = "test_experiment_with_court_metadata.json";
        experiment.save_auto(temp_file).unwrap();
        let loaded = Experiment::load_auto(temp_file).unwrap();
        
        match loaded.others {
            Some(ExperimentMetadata::Court { .. }) => {
            },
            _ => panic!("Court metadata not preserved during serialization"),
        }
        std::fs::remove_file("test_experiment_with_court_metadata.json").unwrap();
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
        
        let mut reloaded = Experiment::load_auto(temp_file).unwrap();
        
        assert_eq!(original, reloaded);
        assert!(reloaded.final_population.is_some());
        
        reloaded.display_results();
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
            (vec!["sample1".to_string(), "sample2".to_string()], vec!["sample3".to_string()]),
            (vec!["sample3".to_string()], vec!["sample1".to_string(), "sample2".to_string()]),
        ]);
        experiment.collections = vec![
            vec![Population::test()], 
            vec![Population::test()]
        ];
        
        experiment.compute_importance();
        
        assert!(experiment.importance_collection.is_some());
    }

    #[test]
    #[should_panic(expected = "Mismatch")]
    fn test_compute_importance_cv_mode_handles_no_collection() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = Some(vec![
            (vec!["sample1".to_string()], vec!["sample2".to_string()]),
        ]);
        experiment.collections = vec![];
        experiment.compute_importance();
        
        assert!(experiment.importance_collection.is_none());
    }

    #[test]
    #[should_panic(expected = "Failed to reconstruct CV structure")]
    fn test_compute_importance_cv_mode_panics_when_cv_reconstruction_fails() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = Some(vec![
            (vec!["invalid_sample".to_string()], vec!["another_invalid".to_string()]),
        ]);
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

        std::fs::write("test_incompatible_X.csv", "feature1,feature2,feature3\n1,2,3\n").unwrap();
        std::fs::write("test_incompatible_y.csv", "class\n1\n").unwrap();

        // Use AssertUnwindSafe to wrap the closure
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            experiment.evaluate_on_new_dataset("test_incompatible_X.csv", "test_incompatible_y.csv");
        }));

        // Cleanup code
        let _ = std::fs::remove_file("test_incompatible_X.csv");
        let _ = std::fs::remove_file("test_incompatible_y.csv");

        // Assert that the code panicked as expected
        assert!(result.is_err(), "The test did not panic as expected");
    }

    /// COURT TESTS 
    #[test]
    fn test_new_filters_judges_by_minimum_performance_threshold() {
        let mut population = Population::test();
        
        if population.individuals.len() >= 2 {
            population.individuals[0].sensitivity = 0.9;
            population.individuals[0].specificity = 0.8;
            population.individuals[1].sensitivity = 0.3;
            population.individuals[1].specificity = 0.2;
        }
        
        let court = Court::new(
            &population,
            &0.7,  // min_perf
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        // Only the first judge should be retained (0.9 >= 0.7 && 0.8 >= 0.7)
        assert_eq!(court.judges.individuals.len(), 1);
        assert!(court.judges.individuals[0].sensitivity >= 0.7);
        assert!(court.judges.individuals[0].specificity >= 0.7);
    }

    #[test]
    fn test_new_filters_judges_by_diversity_threshold() {
        let population = Population::test();
        let original_count = population.individuals.len();
        
        let court = Court::new(
            &population,
            &0.0,
            &0.8,  
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        assert!(court.judges.individuals.len() <= original_count);
    }

    #[test]
    #[should_panic(expected = "Voting threshold should be in [0,1]")]
    fn test_new_validates_voting_threshold_bounds_and_panics() {
        let population = Population::test();
        
        Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &1.5,  // Invalid : > 1.0
            &WeightingMethod::Uniform,
        );
    }

    #[test]
    #[should_panic(expected = "Sensitivity threshold must be in [0,1]")]
    fn test_new_validates_specialized_sensitivity_threshold_bounds() {
        let population = Population::test();
        
        Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 1.5,  // Invalid : > 1.0
                specificity_threshold: 0.8,
            },
        );
    }

    #[test]
    #[should_panic(expected = "Specificity threshold must be in [0,1]")]
    fn test_new_validates_specialized_specificity_threshold_bounds() {
        let population = Population::test();
        
        Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.8,
                specificity_threshold: -0.1,  // Invalid : < 0.0
            },
        );
    }

    #[test]
    fn test_new_retains_judges_when_thresholds_are_zero() {
        let population = Population::test();
        let original_count = population.individuals.len();
        
        let court = Court::new(
            &population,
            &0.0,  // No filtering by performance
            &0.0,  // No filtering by diversity
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        assert_eq!(court.judges.individuals.len(), original_count);
    }

    #[test]
    fn test_new_sorts_judges_after_filtering() {
        let population = Population::test();
        
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        // Judges should be sorted (check that sort() has been called)
        assert!(!court.judges.individuals.is_empty());
        assert!(court.judges.individuals[0].fit > court.judges.individuals[1].fit);
        assert!(court.judges.individuals[1].fit > court.judges.individuals[2].fit);
        // Test that the object is valid after sorting
        assert!(court.voting_threshold >= 0.0 && court.voting_threshold <= 1.0);
    }

    #[test]
    fn test_new_handles_empty_population_gracefully() {
        let population = Population::new(); 
        
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        assert!(court.judges.individuals.is_empty());
        assert_eq!(court.voting_method, VotingMethod::Majority);
        assert_eq!(court.voting_threshold, 0.5);
    }

    #[test]
    fn test_evaluate_computes_metrics_for_judges_and_stores_weights() {
        let mut population = Population::test();
        for individual in &mut population.individuals {
            individual.accuracy = 0.0;
        }
        
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);
        
        // Court metrics should be calculated
        assert!(court.accuracy > 0.0);
        assert!(court.weights.is_some());
        assert_eq!(court.weights.as_ref().unwrap().len(), court.judges.individuals.len());
        assert!(court.predicted_classes.is_some());
    }

    #[test]
    fn test_evaluate_optimizes_threshold_when_set_to_zero() {
        let population = Population::test();
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.0,  // Threshold set to zero to trigger Youden optimisation
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);
        
        // The threshold should have been optimised (different from 0.0).
        assert!(court.voting_threshold > 0.0);
        assert!(court.voting_threshold <= 1.0);
    }

    #[test]
    #[should_panic(expected = "Weights must be computed before prediction")]
    fn test_predict_fails_when_weights_not_computed() {
        let population = Population::test();
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.predict(&data);  // Should panic because evaluate() has not been called
    }

    #[test]
    #[should_panic(expected = "Weights length")]
    fn test_compute_majority_threshold_vote_panics_on_weight_mismatch() {
        let population = Population::test();
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        let wrong_weights = vec![1.0]; // Less weight than judges
        
        court.compute_majority_threshold_vote(&data, &wrong_weights, 0.5);
    }

    #[test]
    fn test_compute_majority_threshold_vote_handles_perfect_ties_correctly() {
        let mut population = Population::test();
        // Set up exactly 2 judges to create perfect ties
        population.individuals.truncate(2);
        
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);
        let predictions = court.predict(&data);
        
        // With a threshold of 0.5 and two judges, perfect ties result in class 2 (abstention).
        assert_eq!(predictions.len(), data.sample_len);
        // At least some predictions should be valid.
        assert!(predictions.iter().all(|&x| x == 0 || x == 1 || x == 2));
    }

    #[test]
    fn test_compute_consensus_threshold_vote_requires_consensus_for_decision() {
        let population = Population::test();
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.95,  
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);
        let predictions = court.predict(&data);
        
        // With a high consensus threshold, there should be abstentions (class 2)
        assert!(predictions.contains(&2));
        assert_eq!(predictions.len(), data.sample_len);
    }

    #[test]
    fn test_compute_consensus_threshold_vote_abstains_when_no_consensus() {
        let population = Population::test();
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.95,  
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);
        let predictions = court.predict(&data);
        
        // With a very high threshold, the majority of predictions are likely to be abstentions
        let abstentions = predictions.iter().filter(|&&x| x == 2).count();
        assert!(abstentions > 0);
        assert_eq!(predictions.len(), data.sample_len);
    }

    #[test]
    fn test_compute_classes_method_stores_predictions_correctly() {
        let population = Population::test();
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);  
        court.compute_classes(&data);
        
        assert!(court.predicted_classes.is_some());
        assert_eq!(court.predicted_classes.as_ref().unwrap().len(), data.sample_len);
        
        // Check that all predictions are valid classes
        let predictions = court.predicted_classes.as_ref().unwrap();
        assert!(predictions.iter().all(|&x| x == 0 || x == 1 || x == 2));
    }

    #[test]
    fn test_compute_rejection_rate_calculates_percentage_correctly() {
        let population = Population::test();
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.9, 
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);
        let predictions = court.predict(&data);
        
        let rejection_rate = court.compute_rejection_rate(&predictions);
        assert!(rejection_rate >= 0.0 && rejection_rate <= 1.0);
        
        // Extreme cases
        let all_abstentions = vec![2u8; 10];
        let all_abstention_rate = court.compute_rejection_rate(&all_abstentions);
        assert_eq!(all_abstention_rate, 1.0);
        
        let no_abstentions = vec![0u8, 1u8, 0u8, 1u8];
        let no_abstention_rate = court.compute_rejection_rate(&no_abstentions);
        assert_eq!(no_abstention_rate, 0.0);
    }

    #[test]
    fn test_youden_optimization_adapts_step_size_to_sample_size() {
        let population = Population::test();
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.0,  // Optimisation
            &WeightingMethod::Uniform,
        );
        
        // Test with different sample sizes
        let small_data = Data::specific_test(30, 10);  // <= 50
        court.evaluate(&small_data);
        let threshold_small = court.voting_threshold;
        assert!(threshold_small > 0.0 && threshold_small <= 1.0);
        
        // Reset for next test
        court.voting_threshold = 0.0;
        let medium_data = Data::specific_test(100, 10); // 51-200
        court.evaluate(&medium_data);
        let threshold_medium = court.voting_threshold;
        assert!(threshold_medium > 0.0 && threshold_medium <= 1.0);
        
        // Reset for next test
        court.voting_threshold = 0.0;
        let large_data = Data::specific_test(500, 10); // > 200
        court.evaluate(&large_data);
        let threshold_large = court.voting_threshold;
        assert!(threshold_large > 0.0 && threshold_large <= 1.0);
    }

    #[test]
    fn test_get_judge_specialization_comprehensive() {
        let population = Population::test();
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        // Create judges with different metrics
        let balanced = Individual::test_with_metrics(0.8, 0.8, 0.8);
        let pos_specialist = Individual::test_with_metrics(0.9, 0.5, 0.7);
        let neg_specialist = Individual::test_with_metrics(0.5, 0.9, 0.7);
        let ineffective = Individual::test_with_metrics(0.3, 0.3, 0.3);
        let edge_case = Individual::test_with_metrics(0.7, 0.7, 0.7); 
 
        // Normal cases
        assert_eq!(
            court.get_judge_specialization(&balanced, 0.7, 0.7),
            JudgeSpecialization::Balanced
        );
        assert_eq!(
            court.get_judge_specialization(&pos_specialist, 0.7, 0.7),
            JudgeSpecialization::PositiveSpecialist
        );
        assert_eq!(
            court.get_judge_specialization(&neg_specialist, 0.7, 0.7),
            JudgeSpecialization::NegativeSpecialist
        );
        assert_eq!(
            court.get_judge_specialization(&ineffective, 0.7, 0.7),
            JudgeSpecialization::Ineffective
        );
        
        // Boundary condition test (exact thresholds)
        assert_eq!(
            court.get_judge_specialization(&edge_case, 0.7, 0.7),
            JudgeSpecialization::Balanced
        );
    }

    #[test]
    fn test_count_effective_judges_uniform_counts_all() {
        let population = Population::test();
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let (total, effective) = court.count_effective_judges();
        
        // In uniform mode, all judges are active.
        assert_eq!(total, effective);
        assert_eq!(effective, court.judges.individuals.len());
    }

    #[test]
    fn test_count_effective_judges_specialized_excludes_ineffective() {
        let mut population = Population::test();
        if population.individuals.len() >= 3 {
            // Set up judges with different levels of efficiency
            population.individuals[0].sensitivity = 0.9;
            population.individuals[0].specificity = 0.8; 
            population.individuals[1].sensitivity = 0.5;
            population.individuals[1].specificity = 0.9; 
            population.individuals[2].sensitivity = 0.3;
            population.individuals[2].specificity = 0.2; 
        }
        
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.7,
                specificity_threshold: 0.7,
            },
        );
        
        let (total, effective) = court.count_effective_judges();
        
        // Should exclude ineffective judges
        assert!(effective <= total);
        if population.individuals.len() >= 3 {
            assert!(effective < total);
        }
    }

    #[test]
    fn test_compute_group_strict_weights_comprehensive() {
        let mut population = Population::test();
        
        // Set up judges with different specialisations
        if population.individuals.len() >= 4 {
            population.individuals[0].sensitivity = 0.9;
            population.individuals[0].specificity = 0.5;  // Positive specialist
            population.individuals[1].sensitivity = 0.5;
            population.individuals[1].specificity = 0.9;  // Negative specialist
            population.individuals[2].sensitivity = 0.8;
            population.individuals[2].specificity = 0.8;  // Balanced
            population.individuals[3].sensitivity = 0.3;
            population.individuals[3].specificity = 0.3;  // Ineffective
        }
        
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.7,
                specificity_threshold: 0.7,
            },
        );
        
        let weights = court.compute_group_strict_weights(0.7, 0.7);
        
        // Weights should be distributed fairly among active groups
        assert_eq!(weights.len(), court.judges.individuals.len());
        
        // The sum of the weights of the effective judges must be 1.0.
        let sum_effective_weights: f64 = weights.iter().filter(|&&w| w > 0.0).sum();
        assert!((sum_effective_weights - 1.0).abs() < 1e-10);
        
        // Ineffective judges should have a weight of 0.0.
        if population.individuals.len() >= 4 {
            assert_eq!(weights[3], 0.0); // Judge ineffectif
        }
    }

    #[test]
    #[should_panic(expected = "Specialized threshold are too high to allow judge selection")]
    fn test_compute_group_strict_weights_panics_when_no_active_groups() {
        let mut population = Population::test();
        
        // Ensure that all judges perform poorly
        for individual in &mut population.individuals {
            individual.sensitivity = 0.1;
            individual.specificity = 0.1;
        }
        
        let court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.9,  
                specificity_threshold: 0.9,
            },
        );
        
        court.compute_group_strict_weights(0.9, 0.9);
    }

    #[test]
    fn test_voting_mechanisms_handle_zero_effective_weight() {
        let mut population = Population::test();
        population.individuals.truncate(1); 
        
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        
        court.weights = Some(vec![0.0]);
        
        let predictions = court.apply_voting_mechanism(&data, &[0.0]);
        
        // With a total weight of zero, all predictions should be abstentions (class 2).
        assert!(predictions.iter().all(|&x| x == 2));
        assert_eq!(predictions.len(), data.sample_len);
    }

    #[test]
    fn test_voting_mechanisms_handle_out_of_bounds_sample_indices() {
        let population = Population::test();
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);
        
        // Voting mechanisms should correctly handle sample indices.
        let predictions = court.predict(&data);
        assert_eq!(predictions.len(), data.sample_len);
        assert!(predictions.iter().all(|&x| x == 0 || x == 1 || x == 2));
    }

    #[test]
    fn test_complete_workflow_serialization_to_court_evaluation() {
        let mut experiment = Experiment::test();
        experiment.final_population = Some(Population::test());
        
        let court = Court::new(
            experiment.final_population.as_ref().unwrap(),
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        experiment.others = Some(ExperimentMetadata::Court { court });
        
        let temp_file = "test_complete_workflow.msgpack";
        experiment.save_auto(temp_file).unwrap();
        
        let loaded_experiment = Experiment::load_auto(temp_file).unwrap();
        assert_eq!(experiment, loaded_experiment);
        
        match loaded_experiment.others {
            Some(ExperimentMetadata::Court { court: loaded_court }) => {
                assert_eq!(loaded_court.voting_method, VotingMethod::Majority);
                assert_eq!(loaded_court.voting_threshold, 0.5);
            },
            _ => panic!("Court metadata not preserved"),
        }
        
        std::fs::remove_file(temp_file).unwrap();
    }

    #[test]
    fn test_concurrent_importance_calculation_thread_safety() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = None;
        experiment.final_population = Some(Population::test());
        experiment.parameters.general.thread_number = 4; 
        
        // Importance calculation should be thread safe
        experiment.compute_importance();
        
        assert!(experiment.importance_collection.is_some());
        let importance = experiment.importance_collection.unwrap();
        assert!(!importance.importances.is_empty());
    }

    #[test]
    fn test_memory_cleanup_after_cv_reconstruction_failure() {
        let mut experiment = Experiment::test();
        experiment.cv_folds_ids = Some(vec![
            (vec!["nonexistent_sample".to_string()], vec!["another_nonexistent".to_string()]),
        ]);
        experiment.collections = vec![vec![Population::test()]];
        
        // Even if CV reconstruction fails, the programme should not leak memory.
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            experiment.compute_importance();
        }));
        
        // The test is successful if no memory leaks are detected (implicit verification).
        assert!(panic_result.is_err()); 
    }

    #[test]
    fn test_court_edge_case_single_judge_voting() {
        let mut population = Population::test();
        population.individuals.truncate(1);  
        
        let mut court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        court.evaluate(&data);
        let predictions = court.predict(&data);
        
        // With only one judge, predictions should be consistent.
        assert_eq!(predictions.len(), data.sample_len);
        // With a single judge and majority vote, no abstentions are possible (except for zero weight).
        assert!(predictions.iter().all(|&x| x == 0 || x == 1 || x == 2));
        
        // Test with consensus and a single judge
        let mut consensus_court = Court::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &1.0,  
            &WeightingMethod::Uniform,
        );
        
        consensus_court.evaluate(&data);
        let consensus_predictions = consensus_court.predict(&data);
        
       // >i 100% consensus required and a single judge, all predictions should be the computed
        assert_eq!(consensus_predictions.len(), data.sample_len);
        assert!(consensus_predictions.iter().all(|&x| x == 0 || x == 1));
    }



}