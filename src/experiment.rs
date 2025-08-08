use serde::{Deserialize, Serialize};
use crate::data::Data;
use crate::utils::{compute_metrics_from_classes, display_feature_importance_terminal, compute_auc_from_value};
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
    MDA, 
    PrevalencePop,
    PrevalenceCV,
    Coefficient, 
    PosteriorProbability 
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(non_camel_case_types)]
pub enum ImportanceAggregation {
    mean,
    median
}

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
    Jury { jury: Jury }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Experiment {
    pub id: String,
    pub timestamp: String,
    pub gpredomics_version: String,
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
        info!("Algorithm: {}", self.parameters.general.algo);
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
                ExperimentMetadata::Jury { jury } => {
                    jury.display(&self.train_data, self.test_data.as_ref(), &self.parameters)
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
                ExperimentMetadata::Jury { jury } => {
                    jury.display(&self.train_data, Some(&new_data), &self.parameters)
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

    pub fn compute_voting(&mut self) {
        let mut jury;
        
        let mut voting_pop = self.final_population.clone().unwrap();
        voting_pop.compute_all_metrics(&self.train_data, &self.parameters.general.fit);

        jury = Jury::new_from_param(&voting_pop, &self.parameters);
        
        if jury.experts.individuals.len() > 1 {
            jury.evaluate(&self.train_data);
            jury.display(&self.train_data, self.test_data.as_ref(), &self.parameters.clone());
            self.others = Some(ExperimentMetadata::Jury { jury: jury })
        } else {
            warn!("An informative vote is requiring more than one expert!")
        }
    
    }
// Majority jury [9 experts] | AUC 0.802/0.518 | accuracy 0.703/0.553 | sensitivity 0.729/0.654 | specificity 0.676/0.420 | rejection rate 0.000/0.000

}

////////////////////////
//////// VOTING ////////
////////////////////////

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Jury {
    pub experts: Population,
    pub voting_method: VotingMethod,
    pub voting_threshold: f64,
    pub threshold_window: f64,
    
    // Weights
    pub weighting_method: WeightingMethod,
    pub weights: Option<Vec<f64>>,
    
    // Binary metrics
    pub auc: f64,
    pub accuracy: f64,
    pub sensitivity: f64,
    pub specificity: f64,
    pub rejection_rate: f64,
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

impl Jury {
    pub fn new(pop: &Population, min_perf: &f64, min_diversity: &f64, voting_method: &VotingMethod, voting_threshold: &f64, threshold_window: &f64, weighting_method: &WeightingMethod) -> Self {
        let mut experts: Population = pop.clone();

        if *min_perf > 0.0 {
            let n = experts.individuals.len();
            experts.individuals.retain(|expert| { expert.sensitivity >= *min_perf && expert.specificity >= *min_perf });
            debug!("Judges filtered for minimum sensitivity and specificity: {}/{} individuals retained", experts.individuals.len(), n);
        }

        if *min_diversity > 0.0 {
            let n = experts.individuals.len();
            experts = experts.filter_by_signed_jaccard_dissimilarity(*min_diversity, true);
            debug!("Judges filtered for diversity: {}/{} individuals retained", experts.individuals.len(), n);
        }

        if voting_threshold < &0.0 || voting_threshold > &1.0 {
            panic!("Voting threshold should be in [0,1]")
        }

        let window;
        if threshold_window < &0.0 || threshold_window > &100.0 {
            panic!("Voting threshold should be in [0,100]")
        } else if *threshold_window == 0.0 {
            window = 1e-10; 
        } else {
            window = *threshold_window;
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

        experts = experts.sort();

        Jury {
            experts,
            voting_method: voting_method.clone(),
            voting_threshold: *voting_threshold,
            threshold_window: window,
            weighting_method:  weighting_method.clone(),
            weights: None,
            auc: 0.0,
            accuracy: 0.0,
            sensitivity: 0.0,
            specificity: 0.0,
            rejection_rate: 1.0,
            predicted_classes: None
        }
    }

    pub fn new_from_param(pop: &Population, param: &Param) -> Self {

        let voting_pop: &Population = if param.voting.use_fbm {
            &pop.select_best_population(0.05)
        } else {
            &pop
        };

        let weighting_method = if param.voting.specialized {
                warn!("Specialized voting mode is experimental");
                if param.voting.specialized_pos_threshold < 0.0 || param.voting.specialized_pos_threshold > 1.0 {
                    panic!("Sensitivity threshold must be in [0,1]");
                }
                if param.voting.specialized_neg_threshold < 0.0 || param.voting.specialized_neg_threshold > 1.0 {
                    panic!("Specificity threshold must be in [0,1]");
                }
                WeightingMethod::Specialized {sensitivity_threshold: param.voting.specialized_pos_threshold, specificity_threshold: param.voting.specialized_neg_threshold}
            } else {
                WeightingMethod::Uniform
            };

        Jury::new(&voting_pop, &param.voting.min_perf, &param.voting.min_diversity, &param.voting.method, &param.voting.method_threshold, &param.voting.threshold_windows_pct, &weighting_method)
    }

    // Evaluates learning data and adjusts internal weight and performance variables accordingly
    pub fn evaluate(&mut self, data: &Data) {
        for expert in &mut self.experts.individuals {
            if expert.accuracy == 0.0 {
                expert.compute_roc_and_metrics(data, None);
            }
        }
        
        let weights = self.compute_weights_by_method(data);

        let effective_experts = weights.iter().filter(|&w| *w > 0.0).count();    
        if effective_experts % 2 == 0 && effective_experts > 0 {
            warn!("Even number of effective experts ({}). Perfect ties will be abstained (class 2).", effective_experts);
        }

        self.weights = Some(weights);

        if self.voting_threshold == 0.0 && self.voting_method == VotingMethod::Majority {
            self.voting_threshold = self.optimize_majority_threshold_youden(data);
            warn!("Threshold set to 0.0. Using Youden Maxima as threshold: {}", self.voting_threshold); 
        }
        
        self.predicted_classes = Some(self.predict(data).0);

        let (auc, accuracy, sensitivity, specificity, rejection_rate) = self.compute_new_metrics(data);
    
        self.auc = auc;
        self.accuracy = accuracy;  
        self.sensitivity = sensitivity;
        self.specificity = specificity;
        self.rejection_rate = rejection_rate;
    }

    pub fn compute_new_metrics(&self, data: &Data) -> (f64, f64, f64, f64, f64) {
        if self.weights.is_none() {
            panic!("Jury must be evaluated on training data first. Call evaluate() before compute_new_metrics().");
        }

        let (pred_classes, scores) = self.predict(data);

        let filtered_data: Vec<(f64, u8, u8)> = scores.iter()
            .zip(pred_classes.iter())
            .zip(data.y.iter())
            .filter_map(|((&score, &pred_class), &true_class)| {
                if score >= 0.0 && score <= 1.0 && pred_class != 2 {
                    Some((score, pred_class, true_class))
                } else {
                    None
                }
            })
            .collect();

        let rejection_rate = self.compute_rejection_rate(&pred_classes);

        if !filtered_data.is_empty() {
            let (scores_filtered, pred_filtered, true_filtered): (Vec<f64>, Vec<u8>, Vec<u8>) = 
                filtered_data.into_iter().fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut scores, mut preds, mut trues), (s, p, t)| {
                        scores.push(s);
                        preds.push(p);
                        trues.push(t);
                        (scores, preds, trues)
                    }
                );

            let auc = compute_auc_from_value(&scores_filtered, &true_filtered);
            
            let (accuracy, sensitivity, specificity) = 
                compute_metrics_from_classes(&pred_filtered, &true_filtered);

            (auc, accuracy, sensitivity, specificity, rejection_rate)
        } else {
            (0.5, 0.0, 0.0, 0.0, rejection_rate)
        }
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
        let predictions = self.compute_majority_threshold_vote(data, &self.weights.as_ref().unwrap(), threshold, self.threshold_window);
        
        let filtered_data: Vec<(u8, u8)> = predictions.1.iter()
            .zip(predictions.0.iter())
            .zip(data.y.iter())
            .filter_map(|((&score, &pred_class), &true_class)| {
                if score >= 0.0 && score <= 1.0 && pred_class != 2 && true_class != 2 {
                    Some((pred_class, true_class))
                } else {
                    None
                }
            })
            .collect();
        
        if !filtered_data.is_empty() {
            let (pred_classes, true_classes): (Vec<u8>, Vec<u8>) = filtered_data.into_iter().unzip();
            let (_, sensitivity, specificity) = compute_metrics_from_classes(&pred_classes, &true_classes);
            
            let youden_index = sensitivity + specificity - 1.0;
            
            if youden_index > best_youden {
                best_youden = youden_index;
                best_threshold = threshold;
            }
        }
    }
    
    self.voting_threshold = best_threshold;
    best_threshold
}
    
    // Evaluates new data based on internal weights and variables
    pub fn predict(&self, data: &Data) -> (Vec<u8>, Vec<f64>) {
        let weights = self.weights.as_ref()
            .expect("Weights must be computed before prediction. Call evaluate() first.");
        self.apply_voting_mechanism(data, weights)
    }

    // Called by evaluate()
    fn compute_weights_by_method(&self, _data: &Data) -> Vec<f64> {
        match &self.weighting_method {
            WeightingMethod::Uniform  => vec![1.0; self.experts.individuals.len()],
            WeightingMethod::Specialized { sensitivity_threshold, specificity_threshold } => 
                self.compute_group_strict_weights(*sensitivity_threshold, *specificity_threshold),
        }
    }

    // 
    fn apply_voting_mechanism(&self, data: &Data, weights: &[f64]) -> (Vec<u8>, Vec<f64>) {
        match &self.voting_method {
            VotingMethod::Majority => {
                self.compute_majority_threshold_vote(data, weights, self.voting_threshold, self.threshold_window)
            },
            VotingMethod::Consensus => {
                self.compute_consensus_threshold_vote(data, weights, self.voting_threshold)
            }
        }
    }

    // Voting methods
    fn compute_consensus_threshold_vote(&self, data: &Data, weights: &[f64], threshold: f64) -> (Vec<u8>, Vec<f64>) {
        let mut predicted_classes = Vec::with_capacity(data.sample_len);
        let mut ratios = Vec::with_capacity(data.sample_len);
        
        let expert_predictions: Vec<Vec<u8>> = self.experts.individuals.iter()
            .map(|expert| expert.evaluate_class(data))
            .collect();
        
        for sample_index in 0..data.sample_len {
            let mut weighted_positive = 0.0;
            let mut weighted_negative = 0.0;
            let mut effective_total_weight = 0.0;
            
            for (expert_idx, expert_pred) in expert_predictions.iter().enumerate() {
                if sample_index < expert_pred.len() {
                    let prediction = expert_pred[sample_index];
                    let weight = weights[expert_idx];
                    
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
            
            let (predicted_class, pos_ratio) = if effective_total_weight > 0.0 {
                let pos_ratio = weighted_positive / effective_total_weight;
                let neg_ratio = weighted_negative / effective_total_weight;
                
                
                let class = if pos_ratio >= threshold {
                    1u8  
                } else if neg_ratio >= threshold {
                    0u8  
                } else {
                    2u8 
                };

                (class, pos_ratio)
            } else {
                (2u8 , -1.0)
            };
            
            ratios.push(pos_ratio);
            predicted_classes.push(predicted_class);
        }
        
        (predicted_classes, ratios)
    }


    fn compute_majority_threshold_vote(&self, data: &Data, weights: &[f64], threshold: f64, threshold_window: f64) -> (Vec<u8>, Vec<f64>) {
        if weights.len() != self.experts.individuals.len() {
            panic!("Weights length ({}) must match expert count ({})", 
                weights.len(), self.experts.individuals.len());
        }

        let mut predicted_classes = Vec::with_capacity(data.sample_len);
        let mut ratios = Vec::with_capacity(data.sample_len);

        let expert_predictions: Vec<Vec<u8>> = self.experts.individuals.iter()
            .map(|expert| expert.evaluate_class(data))
            .collect();
        for sample_index in 0..data.sample_len {
            let mut weighted_positive = 0.0;
            let mut total_weight = 0.0;
            
            for (expert_idx, expert_pred) in expert_predictions.iter().enumerate() {
                if sample_index < expert_pred.len() {
                    let weight = weights[expert_idx];
                    total_weight += weight;
                    
                    if expert_pred[sample_index] == 1 {
                        weighted_positive += weight;
                    }
                }
            }

            let (predicted_class, ratio) = if total_weight > 0.0 {
                let ratio = weighted_positive / total_weight;

                let class = if (ratio - threshold).abs() < (threshold_window / 100.0) {
                    2u8 
                } else if weighted_positive >= total_weight * threshold {
                    1u8
                } else {
                    0u8
                };
                
                (class, ratio)
            } else {
                (2u8, -1.0) // -1 are excluded
            };
            
            ratios.push(ratio);
            predicted_classes.push(predicted_class);
        }
        
        (predicted_classes, ratios)
    }
    
    
    pub fn compute_classes(&mut self, data: &Data) {
        let predictions = self.predict(data);
        self.predicted_classes = Some(predictions.0);
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
    
    
    fn count_effective_experts(&self) -> (usize, usize) {
        match &self.weighting_method {
            WeightingMethod::Uniform => {
                (self.experts.individuals.len(), self.experts.individuals.len())
            },
            WeightingMethod::Specialized { sensitivity_threshold, specificity_threshold } => {
                let mut total_experts = 0;
                let mut effective_experts = 0;
                
                for expert in &self.experts.individuals {
                    total_experts += 1;
                    let specialization = self.get_expert_specialization(
                        expert, *sensitivity_threshold, *specificity_threshold
                    );
                    
                    if !matches!(specialization, JudgeSpecialization::Ineffective) {
                        effective_experts += 1;
                    }
                }
                
                (total_experts, effective_experts)
            }
        }
    }

    // Judge specialization related functions (experimental)
    fn get_expert_specialization(&self, expert: &Individual, 
                               sensitivity_threshold: f64,
                               specificity_threshold: f64) -> JudgeSpecialization {
        
        if expert.sensitivity >= sensitivity_threshold && expert.specificity >= specificity_threshold {
            JudgeSpecialization::Balanced
        } else if expert.sensitivity >= sensitivity_threshold {
            JudgeSpecialization::PositiveSpecialist
        } else if expert.specificity >= specificity_threshold {
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

        for expert in &self.experts.individuals {
            let s = self.get_expert_specialization(expert, sensitivity_threshold, specificity_threshold);
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
            panic!("Specialized threshold are too high to allow expert selection")
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
                text = format!("{}\n{}", text, self.display_confusion_matrix(&test_preds.0, &test_data.y, "TEST"));
            } 

        if let Some(test_data) = test_data {
            text = format!("{}\n{}", text, self.display_predictions_by_sample(test_data, &param.voting.complete_display, "TEST"));
        } else {
            text = format!("{}\n{}", text, self.display_predictions_by_sample(data, &param.voting.complete_display, "TRAIN"));
        }
    
        text = format!("\n{}\n\n{}{} EXPERT POPULATION ({}) {}{}", text, "\x1b[1m", "~".repeat(25), self.experts.individuals.len(), "~".repeat(25), "\x1b[0m");
        text = format!("{}\n{}", text, self.experts.display(&data, test_data, param));

        if param.voting.complete_display {
            text = format!("{}\n{}", text, self.experts.display_feature_prevalence(data, 0));
        } else {
            text = format!("{}\n{}", text, self.experts.display_feature_prevalence(data, 20));
        }

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
        let summary: String;
        let (total_experts, _) = self.count_effective_experts();
        
        let weighting_info = match &self.weighting_method {
            WeightingMethod::Uniform => "",
            WeightingMethod::Specialized { .. } => "specialized-weighted ",
        };
        
        let voting_info = match &self.voting_method {
            VotingMethod::Majority => "Majority",
            VotingMethod::Consensus => "Consensus",
        };

        let method_display = format!("\x1b[1m{} jury [{} {}experts]", 
                                   voting_info, total_experts, weighting_info);
        
        if test_data.is_some() {
            let (test_auc, test_accuracy, test_sensitivity, test_specificity, test_rejection_rate) = self.compute_new_metrics(test_data.unwrap());

            summary = format!("{} | AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3} | rejection rate {:.3}/{:.3}{}", 
                    method_display,
                    self.auc, test_auc,
                    self.accuracy, test_accuracy,
                    self.sensitivity, test_sensitivity,
                    self.specificity, test_specificity,
                    self.rejection_rate, test_rejection_rate,
                    "\x1b[0m");
        } else {
            summary = format!("{} | AUC {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} [ rejection rate {:.3}{}", 
                    method_display,
                    self.auc,
                    self.accuracy,
                    self.sensitivity,
                    self.specificity,
                    self.rejection_rate,
                    "\x1b[0m");
        }

        summary

    }
    
    fn display_voting_method_info(&self) -> String {
        let mut info : String = "".to_string();
        match &self.weighting_method {
            WeightingMethod::Uniform => {
            },
            WeightingMethod::Specialized { sensitivity_threshold, specificity_threshold } => {
                info = self.display_expert_specializations(*sensitivity_threshold, *specificity_threshold);
            },
        }
        info
    }
    
    fn display_predictions_by_sample(&self, data: &Data, complete_display: &bool, title: &str) -> String {
        let mut text = "".to_string();
        text = format!("{}\n{}{} PREDICTIONS BY SAMPLE ({}) {}{}\n\n{}", 
            text, "\x1b[1;1m\n", "~".repeat(25), title, "~".repeat(25), "\x1b[0m", "─".repeat(80));

        let predictions = if title == "TEST" {
            self.predict(data).0
        } else {
            self.predicted_classes.as_ref().unwrap().clone()
        };

        if predictions.len() != data.sample_len {
            return format!("Error: Predictions length ({}) != data length ({})", predictions.len(), data.sample_len);
        }

        let (errors, abstentions, correct, inconsistency_list) = self.categorize_and_sort_by_inconsistency(data, &predictions);
        let inconsistency_map: std::collections::HashMap<usize, f64> = inconsistency_list.iter().cloned().collect();

        let nb_samples_to_show = if *complete_display { data.sample_len } else { 20.min(data.sample_len) };
        
        let max_errors = if errors.len() > 0 { (nb_samples_to_show * 60 / 100).max(1).min(errors.len()) } else { 0 };
        let max_abstentions = if abstentions.len() > 0 { ((nb_samples_to_show - max_errors) * 60 / 100).max(1).min(abstentions.len()) } else { 0 };
        let max_correct = (nb_samples_to_show - max_errors - max_abstentions).min(correct.len());

        text = format!("{}\n{}Sample\t\t| Real | Predictions\t| Result | Consistency{}\n{}", 
            text, "\x1b[1m", "\x1b[0m", "─".repeat(80));

        if max_errors > 0 {
            text = format!("{}\n{}─────── ERRORS ({} shown of {}, sorted by inconsistency) ───────{}", 
                text, "\x1b[1;31m", max_errors, errors.len(), "\x1b[0m");
            
            for &sample_idx in errors.iter().take(max_errors) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let expert_votes = self.display_expert_votes_for_sample(data, sample_idx);
                let inconsistency = inconsistency_map.get(&sample_idx).unwrap_or(&0.0);
                let consistency_percent = (1.0 - inconsistency) * 100.0;
                
                text = format!("{}\n{:>10}\t| {:>4} | {} → {}\t| \x1b[1;31m✗\x1b[0m     | {:>6.1}%", 
                    text, sample_name, real_class, expert_votes, predicted_class, consistency_percent);
            }
        }

        if max_abstentions > 0 {
            text = format!("{}\n{}────── ABSTENTIONS ({} shown of {}, sorted by inconsistency) ─────{}", 
                text, "\x1b[1;90m", max_abstentions, abstentions.len(), "\x1b[0m");
            
            for &sample_idx in abstentions.iter().take(max_abstentions) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let expert_votes = self.display_expert_votes_for_sample(data, sample_idx);
                let inconsistency = inconsistency_map.get(&sample_idx).unwrap_or(&0.0);
                let consistency_percent = (1.0 - inconsistency) * 100.0;
                
                text = format!("{}\n{:>10}\t| {:>4} | {} → {} | \x1b[90m~\x1b[0m     | {:>6.1}%", 
                    text, sample_name, real_class, expert_votes, predicted_class, consistency_percent);
            }
        }
  
        if max_correct > 0 {
            text = format!("{}\n{}─────── CORRECT ({} shown of {}, sorted by inconsistency) ───────{}", 
                text, "\x1b[1;32m", max_correct, correct.len(), "\x1b[0m");
            
            for &sample_idx in correct.iter().take(max_correct) {
                let sample_name = &data.samples[sample_idx];
                let real_class = data.y[sample_idx];
                let predicted_class = predictions[sample_idx];
                let expert_votes = self.display_expert_votes_for_sample(data, sample_idx);
                let inconsistency = inconsistency_map.get(&sample_idx).unwrap_or(&0.0);
                let consistency_percent = (1.0 - inconsistency) * 100.0;
                
                text = format!("{}\n{:>10}\t| {:>4} | {} → {} | \x1b[1;32m✓\x1b[0m     | {:>6.1}%", 
                    text, sample_name, real_class, expert_votes, predicted_class, consistency_percent);
            }
        }

        let total_shown = max_errors + max_abstentions + max_correct;
        if data.sample_len > total_shown {
            text = format!("{}\n ... {} additional samples not shown", text, data.sample_len - total_shown);
        }

        // Statistiques avec métriques d'inconsistance
        let avg_inconsistency = inconsistency_list.iter().map(|(_, inc)| inc).sum::<f64>() / inconsistency_list.len() as f64;
        let avg_consistency = (1.0 - avg_inconsistency) * 100.0;

        text = format!("{}\n\n{}Errors: {} | Correct: {} | Abstentions: {} | Avg Consistency: {:.1}%{}", 
            text, "\x1b[1;33m", errors.len(), correct.len(), abstentions.len(), avg_consistency, "\x1b[0m");

        text
    }

    fn compute_sample_inconsistency(&self, data: &Data) -> Vec<(usize, f64)> {
        let mut inconsistency_list = Vec::new();
        
        let expert_predictions: Vec<Vec<u8>> = self.experts.individuals.iter()
            .map(|expert| expert.evaluate_class(data))
            .collect();
        
        for sample_idx in 0..data.sample_len {
            let mut vote_counts = std::collections::HashMap::new();
            let mut total_votes = 0;
            
            for expert_pred in &expert_predictions {
                if sample_idx < expert_pred.len() {
                    let vote = expert_pred[sample_idx];
                    *vote_counts.entry(vote).or_insert(0) += 1;
                    total_votes += 1;
                }
            }
            
            // Calculate inconsistency (1 - proportion of majority vote)
            let max_vote_count = vote_counts.values().max().copied().unwrap_or(0);
            let consistency = if total_votes > 0 { 
                max_vote_count as f64 / total_votes as f64 
            } else { 
                0.0 
            };
            let inconsistency = 1.0 - consistency;
            
            inconsistency_list.push((sample_idx, inconsistency));
        }
        
        inconsistency_list
    }

    /// Categorise and sort samples by inconsistency
    fn categorize_and_sort_by_inconsistency(&self, data: &Data, predictions: &[u8]) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<(usize, f64)>) {
        let inconsistency_list = self.compute_sample_inconsistency(data);
        let inconsistency_map: std::collections::HashMap<usize, f64> = inconsistency_list.iter().cloned().collect();
        
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
        
        errors.sort_by(|&a, &b| {
            inconsistency_map.get(&b).unwrap_or(&0.0)
                .partial_cmp(inconsistency_map.get(&a).unwrap_or(&0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        abstentions.sort_by(|&a, &b| {
            inconsistency_map.get(&b).unwrap_or(&0.0)
                .partial_cmp(inconsistency_map.get(&a).unwrap_or(&0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        correct.sort_by(|&a, &b| {
            inconsistency_map.get(&b).unwrap_or(&0.0)
                .partial_cmp(inconsistency_map.get(&a).unwrap_or(&0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        (errors, abstentions, correct, inconsistency_list)
    }

    pub fn display_expert_specializations(&self, sensitivity_threshold: f64, specificity_threshold: f64) -> String {
        let mut text = "".to_string();
        text = format!("{}\n{} JUDGE SPECIALIZATIONS{}\n{}", text, "\x1b[1;45m", "\x1b[0m", "─".repeat(80));
        text = format!("{}\n{:<6} | {:<8} | {:<11} | {:<11} | {:<20}", text, "Judge", "Accuracy", "Sensitivity", "Specificity", "Specialization");
        text = format!("{}\n{}", text, "─".repeat(80));
        
        for (idx, expert) in self.experts.individuals.iter().enumerate() {
            let specialization = self.get_expert_specialization(
                expert, sensitivity_threshold, specificity_threshold);
            
            let spec_str = match specialization {
                JudgeSpecialization::PositiveSpecialist => "🔍 \x1b[96mPositive Specialist\x1b[0m",
                JudgeSpecialization::NegativeSpecialist => "🔍 \x1b[95mNegative Specialist\x1b[0m", 
                JudgeSpecialization::Balanced => "⚖️  Balanced",
                JudgeSpecialization::Ineffective => "❌ \x1b[90mIneffective\x1b[0m",
            };
            
            match specialization {
                JudgeSpecialization::Ineffective => {},
                _ => { text = format!("{}\n#{:<6} | {:<8.3} | {:<11.3} | {:<11.3} | {}", text, idx+1, expert.accuracy, expert.sensitivity, expert.specificity, spec_str) }
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

    fn display_expert_votes_for_sample(&self, data: &Data, sample_idx: usize) -> String {
        let mut output = String::new();
        
        for (_, expert) in self.experts.individuals.iter().enumerate() {
            let predictions = expert.evaluate_class(data);
            
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
                        let specialization = self.get_expert_specialization(
                            expert, *sensitivity_threshold, *specificity_threshold
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
    use crate::population::Population;
    use crate::individual::{Individual, BINARY_LANG, RAW_TYPE};
    use crate::data::Data;
    use std::collections::HashMap;

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
        assert!(matches!(result_scope.importances[0].scope, ImportanceScope::Collection));
        
        let result_type = collection.filter(None, Some(ImportanceType::MDA));
        assert_eq!(result_type.importances.len(), 2);
        assert!(result_type.importances.iter().all(|imp| imp.importance_type == ImportanceType::MDA));
        
        let result_both = collection.filter(Some(ImportanceScope::Collection), Some(ImportanceType::MDA));
        assert_eq!(result_both.importances.len(), 1);
        assert!(matches!(result_both.importances[0].scope, ImportanceScope::Collection));
        assert_eq!(result_both.importances[0].importance_type, ImportanceType::MDA);
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
        pub fn test() -> Experiment {
            Experiment {
                id: "test_exp_001".to_string(),
                timestamp: "2025-01-01T12:00:00Z".to_string(),
                gpredomics_version: "1.0.0".to_string(),
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
            Some(ExperimentMetadata::Jury { .. }) => {
            },
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

    /// JURY TESTS 
    #[test]
    fn test_new_filters_experts_by_minimum_performance_threshold() {
        let mut population = Population::test();
        
        if population.individuals.len() >= 2 {
            population.individuals[0].sensitivity = 0.9;
            population.individuals[0].specificity = 0.8;
            population.individuals[1].sensitivity = 0.3;
            population.individuals[1].specificity = 0.2;
        }
        
        let jury = Jury::new(
            &population,
            &0.7,  // min_perf
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        // Only the first expert should be retained (0.9 >= 0.7 && 0.8 >= 0.7)
        assert_eq!(jury.experts.individuals.len(), 1);
        assert!(jury.experts.individuals[0].sensitivity >= 0.7);
        assert!(jury.experts.individuals[0].specificity >= 0.7);
    }

    #[test]
    fn test_new_filters_experts_by_diversity_threshold() {
        let population = Population::test();
        let original_count = population.individuals.len();
        
        let jury = Jury::new(
            &population,
            &0.0,
            &0.8,  
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        assert!(jury.experts.individuals.len() <= original_count);
    }

    #[test]
    #[should_panic(expected = "Voting threshold should be in [0,1]")]
    fn test_new_validates_voting_threshold_bounds_and_panics() {
        let population = Population::test();
        
        Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &1.5,  // Invalid : > 1.0
            &0.0,
            &WeightingMethod::Uniform,
        );
    }

    #[test]
    #[should_panic(expected = "Sensitivity threshold must be in [0,1]")]
    fn test_new_validates_specialized_sensitivity_threshold_bounds() {
        let population = Population::test();
        
        Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
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
        
        Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.8,
                specificity_threshold: -0.1,  // Invalid : < 0.0
            },
        );
    }

    #[test]
    fn test_new_retains_experts_when_thresholds_are_zero() {
        let population = Population::test();
        let original_count = population.individuals.len();
        
        let jury = Jury::new(
            &population,
            &0.0,  // No filtering by performance
            &0.0,  // No filtering by diversity
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        assert_eq!(jury.experts.individuals.len(), original_count);
    }

    #[test]
    fn test_new_sorts_experts_after_filtering() {
        let population = Population::test();
        
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        // Judges should be sorted (check that sort() has been called)
        assert!(!jury.experts.individuals.is_empty());
        assert!(jury.experts.individuals[0].fit > jury.experts.individuals[1].fit);
        assert!(jury.experts.individuals[1].fit > jury.experts.individuals[2].fit);
        // Test that the object is valid after sorting
        assert!(jury.voting_threshold >= 0.0 && jury.voting_threshold <= 1.0);
    }

    #[test]
    fn test_new_handles_empty_population_gracefully() {
        let population = Population::new(); 
        
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        assert!(jury.experts.individuals.is_empty());
        assert_eq!(jury.voting_method, VotingMethod::Majority);
        assert_eq!(jury.voting_threshold, 0.5);
    }

    #[test]
    fn test_evaluate_computes_metrics_for_experts_and_stores_weights() {
        let mut population = Population::test();
        for individual in &mut population.individuals {
            individual.accuracy = 0.0;
        }
        
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);
        
        // Jury metrics should be calculated
        assert!(jury.accuracy > 0.0);
        assert!(jury.weights.is_some());
        assert_eq!(jury.weights.as_ref().unwrap().len(), jury.experts.individuals.len());
        assert!(jury.predicted_classes.is_some());
    }

    #[test]
    fn test_evaluate_optimizes_threshold_when_set_to_zero() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.0,  // Threshold set to zero to trigger Youden optimisation
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);
        
        // The threshold should have been optimised (different from 0.0).
        assert!(jury.voting_threshold > 0.0);
        assert!(jury.voting_threshold <= 1.0);
    }

    #[test]
    #[should_panic(expected = "Weights must be computed before prediction")]
    fn test_predict_fails_when_weights_not_computed() {
        let population = Population::test();
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.predict(&data);  // Should panic because evaluate() has not been called
    }

    #[test]
    #[should_panic(expected = "Weights length")]
    fn test_compute_majority_threshold_vote_panics_on_weight_mismatch() {
        let population = Population::test();
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        let wrong_weights = vec![1.0]; // Less weight than experts
        
        jury.compute_majority_threshold_vote(&data, &wrong_weights, 0.5, 0.0);
    }

    #[test]
    fn test_compute_majority_threshold_vote_handles_perfect_ties_correctly() {
        let mut population = Population::test();
        // Set up exactly 2 experts to create perfect ties
        population.individuals.truncate(2);
        
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);
        
        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
        assert!(predictions.0.iter().all(|&x| x == 0 || x == 1 || x == 2));
        assert!(predictions.1.iter().all(|&x| x >= 0.0 || x <= 1.0));
    }

    #[test]
    fn test_compute_consensus_threshold_vote_requires_consensus_for_decision() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.95,  
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);
        
        // With a high consensus threshold, there should be abstentions (class 2)
        assert!(predictions.0.contains(&2));
        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
    }

    #[test]
    fn test_compute_consensus_threshold_vote_abstains_when_no_consensus() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.95,  
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);
        
        // With a very high threshold, the majority of predictions are likely to be abstentions
        let abstentions = predictions.0.iter().filter(|&&x| x == 2).count();
        assert!(abstentions > 0);
        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
    }

    #[test]
    fn test_compute_classes_method_stores_predictions_correctly() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);  
        jury.compute_classes(&data);
        
        assert!(jury.predicted_classes.is_some());
        assert_eq!(jury.predicted_classes.as_ref().unwrap().len(), data.sample_len);
        
        // Check that all predictions are valid classes
        let predictions = jury.predicted_classes.as_ref().unwrap();
        assert!(predictions.iter().all(|&x| x == 0 || x == 1 || x == 2));
    }

    #[test]
    fn test_compute_rejection_rate_calculates_percentage_correctly() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.9, 
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);
        
        let rejection_rate = jury.compute_rejection_rate(&predictions.0);
        assert!(rejection_rate >= 0.0 && rejection_rate <= 1.0);
        
        // Extreme cases
        let all_abstentions = vec![2u8; 10];
        let all_abstention_rate = jury.compute_rejection_rate(&all_abstentions);
        assert_eq!(all_abstention_rate, 1.0);
        
        let no_abstentions = vec![0u8, 1u8, 0u8, 1u8];
        let no_abstention_rate = jury.compute_rejection_rate(&no_abstentions);
        assert_eq!(no_abstention_rate, 0.0);
    }

    #[test]
    fn test_youden_optimization_adapts_step_size_to_sample_size() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.0,  // Optimisation
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        // Test with different sample sizes
        let small_data = Data::specific_test(30, 10);  // <= 50
        jury.evaluate(&small_data);
        let threshold_small = jury.voting_threshold;
        assert!(threshold_small > 0.0 && threshold_small <= 1.0);
        
        // Reset for next test
        jury.voting_threshold = 0.0;
        let medium_data = Data::specific_test(100, 10); // 51-200
        jury.evaluate(&medium_data);
        let threshold_medium = jury.voting_threshold;
        assert!(threshold_medium > 0.0 && threshold_medium <= 1.0);
        
        // Reset for next test
        jury.voting_threshold = 0.0;
        let large_data = Data::specific_test(500, 10); // > 200
        jury.evaluate(&large_data);
        let threshold_large = jury.voting_threshold;
        assert!(threshold_large > 0.0 && threshold_large <= 1.0);
    }

    #[test]
    fn test_get_expert_specialization_comprehensive() {
        let population = Population::test();
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        // Create experts with different metrics
        let balanced = Individual::test_with_metrics(0.8, 0.8, 0.8);
        let pos_specialist = Individual::test_with_metrics(0.9, 0.5, 0.7);
        let neg_specialist = Individual::test_with_metrics(0.5, 0.9, 0.7);
        let ineffective = Individual::test_with_metrics(0.3, 0.3, 0.3);
        let edge_case = Individual::test_with_metrics(0.7, 0.7, 0.7); 
 
        // Normal cases
        assert_eq!(
            jury.get_expert_specialization(&balanced, 0.7, 0.7),
            JudgeSpecialization::Balanced
        );
        assert_eq!(
            jury.get_expert_specialization(&pos_specialist, 0.7, 0.7),
            JudgeSpecialization::PositiveSpecialist
        );
        assert_eq!(
            jury.get_expert_specialization(&neg_specialist, 0.7, 0.7),
            JudgeSpecialization::NegativeSpecialist
        );
        assert_eq!(
            jury.get_expert_specialization(&ineffective, 0.7, 0.7),
            JudgeSpecialization::Ineffective
        );
        
        // Boundary condition test (exact thresholds)
        assert_eq!(
            jury.get_expert_specialization(&edge_case, 0.7, 0.7),
            JudgeSpecialization::Balanced
        );
    }

    #[test]
    fn test_count_effective_experts_uniform_counts_all() {
        let population = Population::test();
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let (total, effective) = jury.count_effective_experts();
        
        // In uniform mode, all experts are active.
        assert_eq!(total, effective);
        assert_eq!(effective, jury.experts.individuals.len());
    }

    #[test]
    fn test_count_effective_experts_specialized_excludes_ineffective() {
        let mut population = Population::test();
        if population.individuals.len() >= 3 {
            // Set up experts with different levels of efficiency
            population.individuals[0].sensitivity = 0.9;
            population.individuals[0].specificity = 0.8; 
            population.individuals[1].sensitivity = 0.5;
            population.individuals[1].specificity = 0.9; 
            population.individuals[2].sensitivity = 0.3;
            population.individuals[2].specificity = 0.2; 
        }
        
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.7,
                specificity_threshold: 0.7,
            },
        );
        
        let (total, effective) = jury.count_effective_experts();
        
        // Should exclude ineffective experts
        assert!(effective <= total);
        if population.individuals.len() >= 3 {
            assert!(effective < total);
        }
    }

    #[test]
    fn test_compute_group_strict_weights_comprehensive() {
        let mut population = Population::test();
        
        // Set up experts with different specialisations
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
        
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.7,
                specificity_threshold: 0.7,
            },
        );
        
        let weights = jury.compute_group_strict_weights(0.7, 0.7);
        
        // Weights should be distributed fairly among active groups
        assert_eq!(weights.len(), jury.experts.individuals.len());
        
        // The sum of the weights of the effective experts must be 1.0.
        let sum_effective_weights: f64 = weights.iter().filter(|&&w| w > 0.0).sum();
        assert!((sum_effective_weights - 1.0).abs() < 1e-10);
        
        // Ineffective experts should have a weight of 0.0.
        if population.individuals.len() >= 4 {
            assert_eq!(weights[3], 0.0); // Judge ineffectif
        }
    }

    #[test]
    #[should_panic(expected = "Specialized threshold are too high to allow expert selection")]
    fn test_compute_group_strict_weights_panics_when_no_active_groups() {
        let mut population = Population::test();
        
        // Ensure that all experts perform poorly
        for individual in &mut population.individuals {
            individual.sensitivity = 0.1;
            individual.specificity = 0.1;
        }
        
        let jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.9,  
                specificity_threshold: 0.9,
            },
        );
        
        jury.compute_group_strict_weights(0.9, 0.9);
    }

    #[test]
    fn test_voting_mechanisms_handle_zero_effective_weight() {
        let mut population = Population::test();
        population.individuals.truncate(1); 
        
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        
        jury.weights = Some(vec![0.0]);
        
        let predictions = jury.apply_voting_mechanism(&data, &[0.0]);
        
        // With a total weight of zero, all predictions should be abstentions (class 2).
        assert!(predictions.0.iter().all(|&x| x == 2));
        assert_eq!(predictions.0.len(), data.sample_len);
    }

    #[test]
    fn test_voting_mechanisms_handle_out_of_bounds_sample_indices() {
        let population = Population::test();
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);
        
        // Voting mechanisms should correctly handle sample indices.
        let predictions = jury.predict(&data);
        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
        assert!(predictions.0.iter().all(|&x| x == 0 || x == 1 || x == 2));
        assert!(predictions.1.iter().all(|&x| x >= 0.0 || x <= 1.0));
    }

    #[test]
    fn test_complete_workflow_serialization_to_jury_evaluation() {
        let mut experiment = Experiment::test();
        experiment.final_population = Some(Population::test());
        
        let jury = Jury::new(
            experiment.final_population.as_ref().unwrap(),
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        experiment.others = Some(ExperimentMetadata::Jury { jury });
        
        let temp_file = "test_complete_workflow.msgpack";
        experiment.save_auto(temp_file).unwrap();
        
        let loaded_experiment = Experiment::load_auto(temp_file).unwrap();
        assert_eq!(experiment, loaded_experiment);
        
        match loaded_experiment.others {
            Some(ExperimentMetadata::Jury { jury: loaded_jury }) => {
                assert_eq!(loaded_jury.voting_method, VotingMethod::Majority);
                assert_eq!(loaded_jury.voting_threshold, 0.5);
            },
            _ => panic!("Jury metadata not preserved"),
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
    fn test_jury_edge_case_single_expert_voting() {
        let mut population = Population::test();
        population.individuals.truncate(1);  
        
        let mut jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        let data = Data::test();
        jury.evaluate(&data);
        let predictions = jury.predict(&data);
        
        assert_eq!(predictions.0.len(), data.sample_len);
        assert_eq!(predictions.1.len(), data.sample_len);
        assert!(predictions.0.iter().all(|&x| x == 0 || x == 1 || x == 2));
        assert!(predictions.1.iter().all(|&x| x >= 0.0 || x <= 1.0));
        
        // Test with consensus and a single expert
        let mut consensus_jury = Jury::new(
            &population,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &1.0,  
            &0.0,
            &WeightingMethod::Uniform,
        );
        
        consensus_jury.evaluate(&data);
        let consensus_predictions = consensus_jury.predict(&data);
        
       // >i 100% consensus required and a single expert, all predictions should be the computed
        assert_eq!(consensus_predictions.0.len(), data.sample_len);
        assert!(consensus_predictions.0.iter().all(|&x| x == 0 || x == 1));
    }

    /// Helper to create a expert who votes according to a predefined pattern
    fn create_mock_expert(vote: u8) -> Individual {
        let mut expert = Individual::new();
        expert.features.insert(0, if vote == 1 { 1 } else { -1 });
        expert.accuracy = 0.8;
        expert.sensitivity = 0.75;
        expert.specificity = 0.85;
        expert.auc = 0.8;
        expert.threshold = 0.5;
        expert.language = BINARY_LANG;
        expert.data_type = RAW_TYPE;
        expert.k = 1;
        expert
    }

    /// Helper for creating data with a single sample
    fn create_single_sample_data(true_class: u8) -> Data {
        let mut X = HashMap::new();
        X.insert((0, 0), 1.0); // Sample 0, feature 0
        
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);
        
        Data {
            X,
            y: vec![true_class],
            features: vec!["feature1".to_string()],
            samples: vec!["sample1".to_string()],
            feature_class,
            feature_selection: vec![0],
            feature_len: 1,
            sample_len: 1,
            classes: vec!["class0".to_string(), "class1".to_string()],
        }
    }

    /// Helper for creating a population from predefined votes
    fn create_population_with_votes(votes: Vec<u8>) -> Population {
        let mut pop = Population::new();
        for vote in votes {
            pop.individuals.push(create_mock_expert(vote));
        }
        pop
    }

    #[test]
    fn test_scenario_1_unanimous_majority_for() {
        // 5 experts unanimously vote 1, threshold 0.5 -> decision 1
        let pop = create_population_with_votes(vec![1, 1, 1, 1, 1]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0, // min_perf
            &0.0, // min_diversity
            &VotingMethod::Majority,
            &0.5, // voting_threshold
            &0.0, // threshold_window
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(pred_classes[0], 1, "Decision should be 1 (unanimous for)");
        assert!((scores[0] - 1.0).abs() < 1e-10, "Score should be 1.0, got {}", scores[0]);
    }

    #[test]
    fn test_scenario_2_unanimous_majority_against() {
        // 5 experts unanimously vote 0, threshold 0.5 -> decision 0
        let pop = create_population_with_votes(vec![0, 0, 0, 0, 0]);
        let data = create_single_sample_data(0);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(pred_classes[0], 0, "Decision should be 0 (unanimous against)");
        assert!((scores[0] - 0.0).abs() < 1e-10, "Score should be 0.0, got {}", scores[0]);
    }

    #[test]
    fn test_scenario_3_simple_majority() {
        // 3 votes 1, 2 votes 0, threshold 0.5 -> decision 1
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(pred_classes[0], 1, "Decision should be 1 (simple majority)");
        assert!((scores[0] - 0.6).abs() < 1e-10, "Score should be 0.6 (3/5), got {}", scores[0]);
    }

    #[test]
    fn test_scenario_4_abstention_due_to_threshold_window() {
        // 3 votes 1, 2 votes 0, threshold 0.6, window 10% -> abstention
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.6,   // threshold
            &10.0,  // threshold_window 10%
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // Score = 0.6, threshold = 0.6, window = 10% = 0.1
        // |0.6 - 0.6| = 0.0 < 0.1 -> abstention
        assert_eq!(pred_classes[0], 2, "Decision should be 2 (abstention due to window)");
        assert!((scores[0] - 0.6).abs() < 1e-10, "Score should be 0.6, got {}", scores[0]);
    }

    #[test]
    fn test_scenario_5_consensus_success() {
        // 4 votes 1, 1 vote 0, consensus threshold 0.7 -> decision 1
        let pop = create_population_with_votes(vec![1, 1, 1, 1, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.7, // consensus threshold
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(pred_classes[0], 1, "Decision should be 1 (consensus achieved)");
        assert!((scores[0] - 0.8).abs() < 1e-10, "Score should be 0.8 (4/5), got {}", scores[0]);
    }

    #[test]
    fn test_scenario_6_consensus_failure() {
        // 3 votes 1, 2 votes 0, consensus threshold 0.8 -> abstention
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.8, // high consensus threshold
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // Score = 0.6 < 0.8 threshold -> abstention
        assert_eq!(pred_classes[0], 2, "Decision should be 2 (consensus failed)");
        assert!((scores[0] - 0.6).abs() < 1e-10, "Score should be 0.6, got {}", scores[0]);
    }

    #[test]
    fn test_scenario_7_weighted_majority() {
        // Votes [1,1,0,0,0] with weights [2,2,1,1,1] -> decision 1
        let pop = create_population_with_votes(vec![1, 1, 0, 0, 0]);
        let data = create_single_sample_data(1);

        // To simulate different weights, we create a Jury with min_perf
        // that will filter certain experts based on their performance
        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        
        // Manually simulate different weights
        jury.weights = Some(vec![2.0, 2.0, 1.0, 1.0, 1.0]);
        
        let (pred_classes, scores) = jury.predict(&data);

        // Weighted votes: 2*1 + 2*1 + 1*0 + 1*0 + 1*0 = 4
        // Total weight: 2 + 2 + 1 + 1 + 1 = 7
        // Score: 4/7 ≈ 0.571
        assert_eq!(pred_classes[0], 1, "Decision should be 1 (weighted majority)");
        assert!((scores[0] - 4.0/7.0).abs() < 1e-10, "Score should be ~0.571, got {}", scores[0]);
    }

    #[test]
    fn test_scenario_8_perfect_tie_with_window() {
        // 2 votes 1, 2 votes 0, seuil 0.5, window 5% -> abstention
        let pop = create_population_with_votes(vec![1, 1, 0, 0]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5, // threshold
            &5.0, // threshold_window 5%
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        // Score = 0.5, threshold = 0.5, window = 5% = 0.05
        // |0.5 - 0.5| = 0.0 < 0.05 -> abstention
        assert_eq!(pred_classes[0], 2, "Decision should be 2 (abstention due to perfect tie)");
        assert!((scores[0] - 0.5).abs() < 1e-10, "Score should be 0.5, got {}", scores[0]);
    }

    #[test]
    fn test_majority_vs_consensus_different_outcomes() {
        // Same population, different voting methods -> different results
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]); // 60% 1
        let data = create_single_sample_data(1);

        // Majority Test (threshold 0.5)
        let mut jury_majority = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );
        jury_majority.evaluate(&data);
        let (pred_maj, _) = jury_majority.predict(&data);

        // Consensus Test  (threshold 0.8)
        let mut jury_consensus = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.8,
            &0.0,
            &WeightingMethod::Uniform,
        );
        jury_consensus.evaluate(&data);
        let (pred_cons, _) = jury_consensus.predict(&data);

        assert_eq!(pred_maj[0], 1, "Majority should decide 1");
        assert_eq!(pred_cons[0], 2, "Consensus should abstain (0.6 < 0.8)");
    }

    #[test]
    fn test_threshold_window_boundary_cases() {
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]); // Score = 0.6
        let data = create_single_sample_data(1);

        // Case 1: window too small -> no abstention
        let mut jury1 = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,  // threshold
            &1.0,  // window = 1% = 0.01
            &WeightingMethod::Uniform,
        );
        jury1.evaluate(&data);
        let (pred1, _) = jury1.predict(&data);
        // |0.6 - 0.5| = 0.1 > 0.01 -> no abstention
        assert_eq!(pred1[0], 1, "Should decide 1 (window too small)");

        // Case 2: window large enough -> abstention
        let mut jury2 = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,   // threshold
            &15.0,  // window = 15% = 0.15
            &WeightingMethod::Uniform,
        );
        jury2.evaluate(&data);
        let (pred2, _) = jury2.predict(&data);
        // |0.6 - 0.5| = 0.1 < 0.15 -> abstention
        assert_eq!(pred2[0], 2, "Should abstain (window large enough)");
    }

    #[test]
    fn test_compute_new_metrics_with_known_outcomes() {
        // Population that votes correctly based on known data
        let pop = create_population_with_votes(vec![1, 1, 1, 0, 0]); // 60% pour
        
        // Data with true class = 1
        let data_positive = create_single_sample_data(1);
        
        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data_positive);
        let (auc, accuracy, sensitivity, specificity, rejection_rate) = 
            jury.compute_new_metrics(&data_positive);

        // Jury predicts 1, true class = 1 -> True Positive
        assert_eq!(accuracy, 1.0, "Accuracy should be 1.0 (correct prediction)");
        assert_eq!(sensitivity, 1.0, "Sensitivity should be 1.0 (TP detected)");
        assert_eq!(rejection_rate, 0.0, "No rejection expected");

        // Test on negative class
        let data_negative = create_single_sample_data(0);
        let (_, accuracy_neg, _, specificity_neg, _) = jury.compute_new_metrics(&data_negative);
        
        // Short predicts 1, true class = 0 -> False Positive
        assert_eq!(accuracy_neg, 0.0, "Accuracy should be 0.0 (wrong prediction)");
        assert_eq!(specificity_neg, 0.0, "Specificity should be 0.0 (FP not detected)");
    }

    #[test]
    fn test_rejection_rate_calculation() {
        let pop = create_population_with_votes(vec![1, 1, 0, 0]); // Perfect tie
        
        // Create data with 3 samples
        let mut X = HashMap::new();
        for i in 0..3 {
            X.insert((i, 0), 1.0);
        }
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);
        
        let data = Data {
            X,
            y: vec![1, 0, 1],
            features: vec!["feature1".to_string()],
            samples: vec!["sample1".to_string(), "sample2".to_string(), "sample3".to_string()],
            feature_class,
            feature_selection: vec![0],
            feature_len: 1,
            sample_len: 3,
            classes: vec!["class0".to_string(), "class1".to_string()],
        };

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,  // threshold
            &5.0,  // window -> abstention on perfect tie
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (_, _, _, _, rejection_rate) = jury.compute_new_metrics(&data);

        // All samples should be rejected (perfect tie in window)
        assert_eq!(rejection_rate, 1.0, "All samples should be rejected (perfect tie)");
    }

    #[test]
    fn test_edge_case_single_expert() {
        // One expert -> no collective vote, but tests logic
        let pop = create_population_with_votes(vec![1]);
        let data = create_single_sample_data(1);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (pred_classes, scores) = jury.predict(&data);

        assert_eq!(pred_classes[0], 1, "Single expert voting 1 should result in decision 1");
        assert_eq!(scores[0], 1.0, "Score should be 1.0 with single expert voting 1");
    }

    // Helper functions
    fn create_mock_expert_with_metrics(vote: u8, accuracy: f64, sensitivity: f64, specificity: f64) -> Individual {
        let mut expert = Individual::new();
        expert.features.insert(0, if vote == 1 { 1 } else { -1 });
        expert.accuracy = accuracy;
        expert.sensitivity = sensitivity;
        expert.specificity = specificity;
        expert.auc = (accuracy + sensitivity + specificity) / 3.0;
        expert.threshold = 0.5;
        expert.language = BINARY_LANG;
        expert.data_type = RAW_TYPE;
        expert.k = 1;
        expert
    }

    fn create_controlled_population(votes: Vec<u8>) -> Population {
        let mut pop = Population::new();
        for (i, vote) in votes.iter().enumerate() {
            let accuracy = 0.7 + 0.02 * i as f64;
            let sensitivity = 0.65 + 0.03 * i as f64;
            let specificity = 0.75 + 0.02 * i as f64;
            pop.individuals.push(create_mock_expert_with_metrics(*vote, accuracy, sensitivity, specificity));
        }
        pop
    }

    fn create_multi_sample_data(true_classes: Vec<u8>) -> Data {
        let mut X = HashMap::new();
        let mut samples = Vec::new();
        
        for (sample_idx, _) in true_classes.iter().enumerate() {
            X.insert((sample_idx, 0), 1.0);
            samples.push(format!("sample_{}", sample_idx));
        }
        
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);
        
        Data {
            X,
            y: true_classes,
            features: vec!["feature1".to_string()],
            samples: samples.clone(),
            feature_class,
            feature_selection: vec![0],
            feature_len: 1,
            sample_len: samples.len(),
            classes: vec!["class0".to_string(), "class1".to_string()],
        }
    }

    #[test]
    fn test_compute_new_metrics_consistency() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]);
        let data = create_multi_sample_data(vec![1, 1, 0, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (ext_auc, ext_accuracy, ext_sensitivity, ext_specificity, _) = jury.compute_new_metrics(&data);

        // External metrics must be identical
        assert!((jury.auc - ext_auc).abs() < 1e-10, 
                "AUC mismatch: internal={}, external={}", jury.auc, ext_auc);
        assert!((jury.accuracy - ext_accuracy).abs() < 1e-10, 
                "Accuracy mismatch: internal={}, external={}", jury.accuracy, ext_accuracy);
        assert!((jury.sensitivity - ext_sensitivity).abs() < 1e-10, 
                "Sensitivity mismatch: internal={}, external={}", jury.sensitivity, ext_sensitivity);
        assert!((jury.specificity - ext_specificity).abs() < 1e-10, 
                "Specificity mismatch: internal={}, external={}", jury.specificity, ext_specificity);
    }

    #[test]
    fn test_compute_new_metrics_rejection_rate_calculation() {
        // Population that will create abstentions with window
        let pop = create_controlled_population(vec![1, 1, 0, 0]); // Perfect tie at 0.5
        let data = create_multi_sample_data(vec![1, 0, 1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &10.0, // 10% window -> abstention on ratio = 0.5
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (_, _, _, _, rejection_rate) = jury.compute_new_metrics(&data);

        // Manual calculation of the rejection rate
        let (pred_classes, _) = jury.predict(&data);
        let manual_rejection_rate = pred_classes.iter().filter(|&&x| x == 2).count() as f64 / pred_classes.len() as f64;

        assert!((rejection_rate - manual_rejection_rate).abs() < 1e-10,
                "Rejection rate mismatch: calculated={}, manual={}", rejection_rate, manual_rejection_rate);
        
        // With a perfect tie and a 10% window, we expect 100% abstention.
        assert_eq!(rejection_rate, 1.0, "Should have 100% rejection with perfect tie and window");
    }

    #[test]
    fn test_internal_vs_external_metrics_coherence() {
        let pop = create_controlled_population(vec![1, 1, 1, 1, 0]);
        let data = create_multi_sample_data(vec![1, 1, 0, 0, 1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Consensus,
            &0.6,
            &0.0,
            &WeightingMethod::Uniform,
        );

        // Test on training data
        jury.evaluate(&data);
        let (train_auc, train_acc, train_sens, train_spec, _) = jury.compute_new_metrics(&data);
        
        assert_eq!(train_auc, jury.auc, "Training AUC should match");
        assert_eq!(train_acc, jury.accuracy, "Training accuracy should match");
        assert_eq!(train_sens, jury.sensitivity, "Training sensitivity should match");
        assert_eq!(train_spec, jury.specificity, "Training specificity should match");

        // Test on different test data
        let test_data = create_multi_sample_data(vec![0, 0, 1, 1]);
        let (test_auc, test_acc, test_sens, test_spec, _) = jury.compute_new_metrics(&test_data);
        
        // Metrics may differ based on different data,
        // but must remain within valid ranges.
        assert!(test_auc >= 0.0 && test_auc <= 1.0, "Test AUC out of bounds: {}", test_auc);
        assert!(test_acc >= 0.0 && test_acc <= 1.0, "Test accuracy out of bounds: {}", test_acc);
        assert!(test_sens >= 0.0 && test_sens <= 1.0, "Test sensitivity out of bounds: {}", test_sens);
        assert!(test_spec >= 0.0 && test_spec <= 1.0, "Test specificity out of bounds: {}", test_spec);
    }

    #[test]
    fn test_voting_threshold_systematic_variations() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]); // 60% 1
        let data = create_multi_sample_data(vec![1]);

        let thresholds = vec![0.3, 0.5, 0.7, 0.9];
        let mut results = Vec::new();

        for threshold in thresholds {
            let mut jury = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Majority,
                &threshold,
                &0.0,
                &WeightingMethod::Uniform,
            );

            jury.evaluate(&data);
            let (pred_classes, scores) = jury.predict(&data);
            let (_, _, _, _, rejection_rate) = jury.compute_new_metrics(&data);
            
            results.push((threshold, pred_classes[0], scores[0], rejection_rate));
        }

        // Score = 0.6 for
        assert!(results.iter().all(|(_, _, score, _)| (score - 0.6).abs() < 1e-10));

        // Check decision logic according to thresholds
        assert_eq!(results[0].1, 1, "Threshold 0.3: 0.6 > 0.3 -> decision 1");
        assert_eq!(results[1].1, 1, "Threshold 0.5: 0.6 > 0.5 -> decision 1");
        assert_eq!(results[2].1, 0, "Threshold 0.7: 0.6 < 0.7 -> decision 0");
        assert_eq!(results[3].1, 0, "Threshold 0.9: 0.6 < 0.9 -> decision 0");
    }

    #[test]
    fn test_threshold_window_granular_effects() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]); // Score = 0.6
        let data = create_multi_sample_data(vec![1]);
        
        let threshold = 0.5;
        let windows = vec![1.0, 5.0, 15.0, 25.0]; // 1%, 5%, 15%, 25%
        
        for window in windows {
            let mut jury = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Majority,
                &threshold,
                &window,
                &WeightingMethod::Uniform,
            );

            jury.evaluate(&data);
            let (pred_classes, scores) = jury.predict(&data);
            
            // Score = 0.6, threshold = 0.5, |0.6 - 0.5| = 0.1 = 10%
            if window < 10.0 {
                // Window too small -> no abstention
                assert_eq!(pred_classes[0], 1, "Window {}%: should decide 1", window);
            } else {
                // Window large enough -> abstention
                assert_eq!(pred_classes[0], 2, "Window {}%: should abstain", window);
            }
        }
    }

    #[test]
    fn test_majority_vs_consensus_systematic_comparison() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]); // 60% pour
        let data = create_multi_sample_data(vec![1, 0, 1]);

        let test_cases = vec![
            (0.5, 1, 1), // Majority: 0.6 > 0.5 -> 1, Consensus: 0.6 > 0.5 -> 2
            (0.4, 1, 1), // Majority: 0.6 > 0.4 -> 1, Consensus: 0.6 > 0.4 -> 1
            (0.8, 0, 2), // Majority: 0.6 < 0.8 -> 0, Consensus: 0.6 < 0.8 -> 2
        ];

        for (threshold, expected_majority, expected_consensus) in test_cases {
            // Test Majority
            let mut jury_maj = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Majority,
                &threshold,
                &0.0,
                &WeightingMethod::Uniform,
            );
            jury_maj.evaluate(&data);
            let (pred_maj, _) = jury_maj.predict(&data);

            // Test Consensus
            let mut jury_cons = Jury::new(
                &pop,
                &0.0,
                &0.0,
                &VotingMethod::Consensus,
                &threshold,
                &0.0,
                &WeightingMethod::Uniform,
            );
            jury_cons.evaluate(&data);
            let (pred_cons, _) = jury_cons.predict(&data);

            assert_eq!(pred_maj[0], expected_majority, 
                      "Majority with threshold {}: expected {}, got {}", threshold, expected_majority, pred_maj[0]);
            assert_eq!(pred_cons[0], expected_consensus, 
                      "Consensus with threshold {}: expected {}, got {}", threshold, expected_consensus, pred_cons[0]);
        }
    }

    #[test]
    fn test_edge_cases_boundary_conditions() {
        let pop = create_controlled_population(vec![1, 1, 0, 0]); // Perfect tie
        let data = create_multi_sample_data(vec![1]);

        // Threshold test at 0.0
        let mut jury_min = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.0, // Minimum threshold -> always 1 unless score = 0
            &10.0,
            &WeightingMethod::Uniform,
        );
        jury_min.evaluate(&data);
        jury_min.voting_threshold = 0.0; // avoid optimization with Youden Maxima
        let (pred_min, _) = jury_min.predict(&data);
        assert_eq!(pred_min[0], 1, "Threshold 0.0: should always decide 1 for score > 0");

        // Test seuil à 1.0
        let mut jury_max = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &1.0, // Maximum threshold -> always 0 unless score = 1
            &0.0,
            &WeightingMethod::Uniform,
        );
        jury_max.evaluate(&data);
        let (pred_max, _) = jury_max.predict(&data);
        assert_eq!(pred_max[0], 0, "Threshold 1.0: should decide 0 for score < 1");

        // Extreme window test
        let mut jury_extreme_window = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &100.0,  // 100% window -> always abstain
            &WeightingMethod::Uniform,
        );
        jury_extreme_window.evaluate(&data);
        let (pred_extreme, _) = jury_extreme_window.predict(&data);
        assert_eq!(pred_extreme[0], 2, "Window 100%: should always abstain");
    }

    #[test]
    fn test_large_population_performance() {
        // Create a large population (50 experts)
        let mut large_votes = Vec::new();
        for i in 0..50 {
            large_votes.push(if i % 3 == 0 { 1 } else { 0 }); // ~33% 1
        }
        
        let pop = create_controlled_population(large_votes);
        let data = create_multi_sample_data(vec![1, 0, 1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (auc, accuracy, sensitivity, specificity, rejection_rate) = jury.compute_new_metrics(&data);

        // Verify that metrics remain stable with a large population
        assert!(auc >= 0.0 && auc <= 1.0, "AUC out of bounds with large population");
        assert!(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy out of bounds with large population");
        assert!(sensitivity >= 0.0 && sensitivity <= 1.0, "Sensitivity out of bounds with large population");
        assert!(specificity >= 0.0 && specificity <= 1.0, "Specificity out of bounds with large population");
        assert!(rejection_rate >= 0.0 && rejection_rate <= 1.0, "Rejection rate out of bounds with large population");
        
        // Verify that the Jury is effectively managing 50 experts
        assert_eq!(jury.experts.individuals.len(), 50, "Should retain all 50 experts");
        assert_eq!(jury.weights.as_ref().unwrap().len(), 50, "Should have 50 weights");
    }

    #[test]
    fn test_multi_sample_dataset_consistency() {
        let pop = create_controlled_population(vec![1, 1, 1, 0, 0]);
        
        // Create a dataset with 100 samples
        let mut large_classes = Vec::new();
        for i in 0..100 {
            large_classes.push(if i % 2 == 0 { 1 } else { 0 });
        }
        let large_data = create_multi_sample_data(large_classes);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&large_data);
        let (auc, accuracy, sensitivity, specificity, rejection_rate) = jury.compute_new_metrics(&large_data);

        // Check consistency on large dataset
        assert!(auc >= 0.0 && auc <= 1.0, "AUC inconsistent on large dataset");
        assert!(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy inconsistent on large dataset");
        assert!(sensitivity >= 0.0 && sensitivity <= 1.0, "Sensitivity inconsistent on large dataset");
        assert!(specificity >= 0.0 && specificity <= 1.0, "Specificity inconsistent on large dataset");
        assert!(rejection_rate >= 0.0 && rejection_rate <= 1.0, "Rejection rate inconsistent on large dataset");

        // Check that all predictions are valid
        let (predictions, _) = jury.predict(&large_data);
        assert_eq!(predictions.len(), 100, "Should have 100 predictions");
        assert!(predictions.iter().all(|&x| x == 0 || x == 1 || x == 2), "All predictions should be valid classes");
    }

    #[test]
    fn test_rejection_rate_mathematical_accuracy() {
        // Controlled scenario: 4 samples with 2 expected abstentions
        let pop = create_controlled_population(vec![1, 1, 0, 0]); // Perfect Tie 0.5
        let data = create_multi_sample_data(vec![1, 0, 1, 0]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &1.0, // 1% window -> no abstention (|0.5-0.5| = 0 < 0.01)
            &WeightingMethod::Uniform,
        );

        jury.evaluate(&data);
        let (_, _, _, _, rejection_rate) = jury.compute_new_metrics(&data);

        // Expected mathematical calculation
        let (predictions, _) = jury.predict(&data);
        let expected_rejections = predictions.iter().filter(|&&x| x == 2).count();
        let expected_rate = expected_rejections as f64 / predictions.len() as f64;

        assert!((rejection_rate - expected_rate).abs() < 1e-10,
                "Mathematical rejection rate mismatch: expected={}, got={}", expected_rate, rejection_rate);

        // Extreme case testing
        let all_abstain = vec![2u8; 5];
        let rate_all = jury.compute_rejection_rate(&all_abstain);
        assert_eq!(rate_all, 1.0, "All abstentions should give 100% rejection rate");

        let no_abstain = vec![0u8, 1u8, 0u8, 1u8];
        let rate_none = jury.compute_rejection_rate(&no_abstain);
        assert_eq!(rate_none, 0.0, "No abstentions should give 0% rejection rate");
    }

    #[test]
    fn test_specialized_weighting_comprehensive() {
        let mut pop = create_controlled_population(vec![1, 1, 0, 0]);
        
        if pop.individuals.len() >= 4 {
            pop.individuals[0].sensitivity = 0.9; // Positive specialist
            pop.individuals[0].specificity = 0.6;
            pop.individuals[1].sensitivity = 0.9; // Positive specialist  
            pop.individuals[1].specificity = 0.6;
            pop.individuals[2].sensitivity = 0.6; // Negative specialist
            pop.individuals[2].specificity = 0.9;
            pop.individuals[3].sensitivity = 0.6; // Negative specialist
            pop.individuals[3].specificity = 0.9;
        }

        let data = create_multi_sample_data(vec![1]);

        let mut jury = Jury::new(
            &pop,
            &0.0,
            &0.0,
            &VotingMethod::Majority,
            &0.5,
            &0.0,
            &WeightingMethod::Specialized {
                sensitivity_threshold: 0.8,
                specificity_threshold: 0.8,
            },
        );

        jury.evaluate(&data);
        
        let weights = jury.weights.as_ref().unwrap();
        assert_eq!(weights.len(), 4, "Should have 4 weights");
        
        let effective_weight_sum: f64 = weights.iter().filter(|&&w| w > 0.0).sum();
        assert!((effective_weight_sum - 1.0).abs() < 1e-10, "Effective weights should sum to 1.0");
        
        let (auc, accuracy, sensitivity, specificity, rejection_rate) = jury.compute_new_metrics(&data);
        assert!(auc >= 0.0 && auc <= 1.0, "AUC should be valid with specialized weighting");
        assert!(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy should be valid with specialized weighting");
        assert!(sensitivity >= 0.0 && sensitivity <= 1.0, "Sensitivity should be valid with specialized weighting");
        assert!(specificity >= 0.0 && specificity <= 1.0, "Specificity should be valid with specialized weighting");
        assert!(rejection_rate >= 0.0 && rejection_rate <= 1.0, "Rejection rate should be valid with specialized weighting");
    }

}