use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Param {
    pub general: General,
    pub data: Data,                          // Nested struct for "data"
    pub ga: GA,
    pub beam: BEAM,
    pub mcmc: MCMC,
    pub cv: CV,
    pub gpu: GPU,
}

#[derive(Debug,Serialize,Deserialize,Clone)]
pub enum FitFunction {
    auc,
    specificity,
    sensitivity    
}


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum GpuMemoryPolicy {
    Strict,
    Adaptive,
    Performance,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum ImportanceAggregation {
    Mean,
    Median
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct General {
    pub seed: u64,
    #[serde(default = "algorithm_default")]  
    pub algo: String,
    #[serde(default = "language_default")]  
    pub language: String,
    #[serde(default = "data_type_default")]  
    pub data_type: String,    
    #[serde(default = "data_type_epsilon_default")]  
    pub data_type_epsilon: f64,
    #[serde(default = "thread_number_default")]  
    pub thread_number: usize,
    #[serde(default = "log_base_default")]  
    pub log_base: String,
    #[serde(default = "log_suffix_default")]  
    pub log_suffix: String,
    #[serde(default = "log_level_default")]  
    pub log_level: String,
    #[serde(default = "fit_default")]
    pub fit: FitFunction,
    #[serde(default = "penalty_default")] 
    pub k_penalty: f64,                       // A penalty of this per value of k is deleted from AUC in the fit function
    #[serde(default = "penalty_default")] 
    pub overfit_penalty: f64,
    #[serde(default = "penalty_default")] 
    pub fr_penalty: f64,
    #[serde(default = "nb_best_model_to_test_default")] 
    pub nb_best_model_to_test: u32,
    #[serde(default = "false_default")] 
    pub gpu: bool,
    #[serde(default = "false_default")] 
    pub cv: bool,
    #[serde(default = "display_level_default")] 
    pub display_level: usize,
    #[serde(default = "display_colorful_default")] 
    pub display_colorful: bool,
    #[serde(default = "feature_keep_trace_default")]   
    pub keep_trace: bool
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Data {
    pub X: String,                     
    pub y: String,
    #[serde(default = "empty_string")]                      
    pub Xtest: String,
    #[serde(default = "empty_string")]                     
    pub ytest: String,
    #[serde(default = "features_maximal_number_per_class_default")]                      
    pub features_maximal_number_per_class: usize,
    #[serde(default = "feature_selection_method_default")]                      
    pub feature_selection_method: String,
    #[serde(default = "feature_minimal_prevalence_pct_default")]                     
    pub feature_minimal_prevalence_pct: f64, // Minimum prevalence
    #[serde(default = "feature_maximal_pvalue_default")]                     
    pub feature_maximal_pvalue: f64, 
    #[serde(default = "feature_minimal_feature_value_default")]                      
    pub feature_minimal_feature_value: f64, 
    #[serde(default = "feature_minimal_log_abs_bayes_factor_default")]                      
    pub feature_minimal_log_abs_bayes_factor: f64, 
    #[serde(default = "class_names_default")]                      
    pub classes: Vec<String>, 
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GA {
    pub population_size: u32,                // Population size
    pub max_epochs: usize,                         // Number of epochs/generations
    #[serde(default = "min_epochs_default")]    
    pub min_epochs: usize,                     // Do not stop before reaching this
    #[serde(default = "max_age_best_model_default")]    
    pub max_age_best_model: usize,                 // Stop if the best model has not change for this long
    #[serde(default = "feature_kminkmax_default")]  
    pub kmin: usize,                           // Minimum value of k
    #[serde(default = "feature_kminkmax_default")]  
    pub kmax: usize,                           // Maximum value of k
    pub select_elite_pct: f64,               // Elite selection percentage
    #[serde(default = "feature_select_niche_pct_default")] 
    pub select_niche_pct: f64,              // Same as elite but split between competing date type/model
    pub select_random_pct: f64,              // Random selection percentage
    pub mutated_children_pct: f64,        // Mutated individuals percentage
    pub mutated_features_pct: f64,           // Mutated features percentage
    pub mutation_non_null_chance_pct: f64,    // Chance pct that a mutation gives an non null value
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BEAM {
    #[serde(default = "beam_method_default")]  
    pub method: String,
    #[serde(default = "feature_kminkmax_default")]  
    pub kmin: usize,                           // Minimum value of k
    #[serde(default = "feature_kminkmax_default")]  
    pub kmax: usize,                           // Maximum value of k
    #[serde(default = "best_models_ci_alpha_default")]
    pub best_models_ci_alpha: f64,                                 
    #[serde(default = "max_nb_of_models_default")]
    pub max_nb_of_models: usize,                                    
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MCMC {
    #[serde(default = "n_iter_default")]  
    pub n_iter: usize,
    #[serde(default = "n_burn_default")]
    pub n_burn: usize,
    #[serde(default = "lambda_default")]
    pub lambda: f64,
    #[serde(default = "nmin_default")]
    pub nmin: u32,
    #[serde(default = "empty_string")]                      
    pub save_trace_outdir: String,
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GPU {
    #[serde(default = "memory_policy_default")]  
    pub memory_policy: GpuMemoryPolicy,
    #[serde(default = "max_total_memory_mb_default")]  
    pub max_total_memory_mb: u64,                           
    #[serde(default = "max_buffer_size_mb_default")]  
    pub max_buffer_size_mb: u32,                           
    #[serde(default = "fallback_to_cpu_default")]
    pub fallback_to_cpu: bool,              
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CV {
    #[serde(default = "fold_number_default")]  
    pub fold_number: usize,
    #[serde(default = "cv_best_models_ci_alpha_default")]  
    pub cv_best_models_ci_alpha: f64,
    #[serde(default = "n_permutations_oob_default")]  
    pub n_permutations_oob: usize,
    #[serde(default = "scaled_importance_default")]  
    pub scaled_importance: bool,
    #[serde(default = "importance_aggregation_default")]  
    pub importance_aggregation: ImportanceAggregation
}

impl Default for General {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for Data {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for CV {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for GA {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for BEAM {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for MCMC {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for GPU {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap_or_else(|_| {
            GPU {
                    memory_policy: memory_policy_default(),
                    max_total_memory_mb: max_total_memory_mb_default(),
                    max_buffer_size_mb: max_buffer_size_mb_default(),
                    fallback_to_cpu: fallback_to_cpu_default(),
                }
            })
        }
}

impl Default for Param {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}


impl Param {
    pub fn new() -> Self {
        Self::default()
    }
}


pub fn get(param_file: String) -> Result<Param, Box<dyn Error>> {
    let param_file_reader = File::open(param_file)?;
    let param_reader = BufReader::new(param_file_reader);
    
    let config:Param = serde_yaml::from_reader(param_reader)?;

    Ok(config)
}



fn empty_string() -> String { "".to_string() }
fn min_epochs_default() -> usize { 10 }
fn max_age_best_model_default() -> usize { 10 }
fn algorithm_default() -> String { "ga".to_string() }
fn features_maximal_number_per_class_default() -> usize { 0 }
fn feature_selection_method_default() -> String { "wilcoxon".to_string() }
fn feature_minimal_prevalence_pct_default() -> f64 { 10.0 }
fn feature_maximal_pvalue_default() -> f64 { 0.5 }
fn feature_minimal_log_abs_bayes_factor_default() -> f64 { 2.0 }
fn feature_minimal_feature_value_default() -> f64 { 0.0 }
fn language_default() -> String { "binary".to_string() }
fn data_type_default() -> String { "raw".to_string() }
fn data_type_epsilon_default() -> f64 { 1e-5 }
fn thread_number_default() -> usize { 1 }
fn feature_kminkmax_default() -> usize { 0 }
fn feature_keep_trace_default() -> bool { true }
fn log_base_default() -> String { "".to_string() }
fn log_suffix_default() -> String { "log".to_string() }
fn log_level_default() -> String { "info".to_string() }
fn fold_number_default() -> usize { 5 }
fn cv_best_models_ci_alpha_default() -> f64 { 0.05 }
fn n_permutations_oob_default() -> usize { 100 }
fn scaled_importance_default() -> bool { false }
fn importance_aggregation_default() -> ImportanceAggregation { ImportanceAggregation::Mean }
fn penalty_default() -> f64 { 0.0 }
fn fit_default() -> FitFunction { FitFunction::auc }
fn nb_best_model_to_test_default() -> u32 { 10 }
fn feature_select_niche_pct_default() -> f64 { 0.0 }
fn false_default() -> bool { false }
fn display_level_default() -> usize { 2 }
fn display_colorful_default() -> bool { false }
fn beam_method_default() -> String { "combinatorial".to_string() }
fn best_models_ci_alpha_default() -> f64 { 0.05 }
fn max_nb_of_models_default() -> usize { 10000 }
fn class_names_default() -> Vec<String> { Vec::new() }
fn memory_policy_default() -> GpuMemoryPolicy { GpuMemoryPolicy::Adaptive }
fn max_total_memory_mb_default() -> u64 { 256 }
fn max_buffer_size_mb_default() -> u32 { 128 }
fn fallback_to_cpu_default() -> bool { true }
fn n_iter_default() -> usize { 10_000 }
fn n_burn_default() -> usize { 5_000 }
fn lambda_default() -> f64 { 0.001 }
fn nmin_default() -> u32 { 10 }