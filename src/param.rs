use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};
use std::error::Error;
use crate::experiment::VotingMethod;
use crate::experiment::ImportanceAggregation;
use log::{warn,error};

#[derive(Debug,Serialize,Deserialize,Clone, PartialEq)]
#[allow(non_camel_case_types)]
pub enum FitFunction {
    auc,
    specificity,
    sensitivity,  
    ExperimentalMcc
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum GpuMemoryPolicy {
    Strict,
    Adaptive,
    Performance,
}

// Field definitions and associated default values

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Param {
    #[serde(default)]
    pub general: General,
    #[serde(default)]
    pub voting: Voting,
    #[serde(default)]
    pub data: Data,
    #[serde(default)]
    pub ga: GA,
    #[serde(default)]
    pub beam: BEAM,
    #[serde(default)]
    pub mcmc: MCMC,
    #[serde(default)]
    pub cv: CV,
    #[serde(default)]
    pub importance: Importance,
    #[serde(default)]
    pub gpu: GPU,
    #[serde(default)]
    pub experimental: Experimental,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct General {
    #[serde(default = "seed_default")]
    pub seed: u64,
    #[serde(default = "algorithm_default")]  
    pub algo: String,
    #[serde(default = "language_default")]  
    pub language: String,
    #[serde(default = "data_type_default")]  
    pub data_type: String,    
    #[serde(default = "data_type_epsilon_default")]  
    pub data_type_epsilon: f64,
    #[serde(default = "one_default")]  
    pub thread_number: usize,
    #[serde(default = "log_base_default")]  
    pub log_base: String,
    #[serde(default = "log_suffix_default")]   
    pub log_suffix: String,
    #[serde(default = "log_level_default")]  
    pub log_level: String,
    #[serde(default = "fit_default")]
    pub fit: FitFunction,
    #[serde(default = "zero_default")] 
    pub k_penalty: f64,                       
    #[serde(default = "zero_default")] 
    pub fr_penalty: f64,
    #[serde(default = "n_model_to_display_default")] 
    pub n_model_to_display: u32,
    #[serde(default = "false_default")] 
    pub gpu: bool,
    #[serde(default = "false_default")] 
    pub cv: bool,
    #[serde(default = "display_level_default")] 
    pub display_level: usize,
    #[serde(default = "true_default")] 
    pub display_colorful: bool,
    #[serde(default = "feature_keep_trace_default")]   
    pub keep_trace: bool,
    #[serde(default = "save_experiment_default")] 
    pub save_exp: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Data {
    #[serde(default = "empty_string")]  
    pub X: String,                     
    #[serde(default = "empty_string")]  
    pub y: String,
    #[serde(default = "empty_string")]                      
    pub Xtest: String,
    #[serde(default = "empty_string")]                     
    pub ytest: String,
    #[serde(default = "uzero_default")]                      
    pub feature_maximal_number_per_class: usize,
    #[serde(default = "feature_selection_method_default")]                      
    pub feature_selection_method: String,
    #[serde(default = "feature_minimal_prevalence_pct_default")]                     
    pub feature_minimal_prevalence_pct: f64, 
    #[serde(default = "feature_maximal_pvalue_default")]                     
    pub feature_maximal_pvalue: f64, 
    #[serde(default = "zero_default")]                      
    pub feature_minimal_feature_value: f64, 
    #[serde(default = "feature_minimal_log_abs_bayes_factor_default")]                      
    pub feature_minimal_log_abs_bayes_factor: f64, 
    #[serde(default = "false_default")] 
    pub inverse_classes: bool,
    #[serde(default = "uzero_default")]
    pub n_validation_samples: usize,
    #[serde(default = "class_names_default")]                      
    pub classes: Vec<String>, 
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CV {
    #[serde(default = "folds_default")]  
    pub inner_folds: usize,
    #[serde(default = "zero_default")] 
    pub overfit_penalty: f64,
    #[serde(default = "folds_default")]  
    pub outer_folds: usize,
    #[serde(default = "false_default")] 
    pub fit_on_valid: bool,
    #[serde(default = "best_models_ci_alpha_default")]  
    pub cv_best_models_ci_alpha: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Voting {
  #[serde(default = "false_default")]
  pub vote: bool,
  #[serde(default = "true_default")]
  pub use_fbm: bool,
  #[serde(default = "half_default")] 
  pub min_perf: f64,
  #[serde(default = "diversity_voting_default")] 
  pub min_diversity: f64,       
  #[serde(default = "voting_default")]                      
  pub method: VotingMethod,   
  #[serde(default = "half_default")]                         
  pub method_threshold: f64,  
  #[serde(default = "zero_default")]        
  pub threshold_windows_pct: f64,
  #[serde(default = "false_default")]        
  pub complete_display: bool,
  #[serde(default = "false_default")]                        
  pub specialized: bool,                 
  #[serde(default = "specialized_default")]            
  pub specialized_pos_threshold: f64,
  #[serde(default = "specialized_default")]                      
  pub specialized_neg_threshold: f64                 
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct GA {
    #[serde(default = "pop_size_default")]    
    pub population_size: u32,        
    #[serde(default = "max_epochs_default")]          
    pub max_epochs: usize,                         
    #[serde(default = "min_epochs_default")]    
    pub min_epochs: usize,                     
    #[serde(default = "max_age_best_model_default")]    
    pub max_age_best_model: usize,                
    #[serde(default = "one_default")]  
    pub kmin: usize,                           
    #[serde(default = "kmax_default")]  
    pub kmax: usize,        
    #[serde(default = "ga_elite_pct_default")]                    
    pub select_elite_pct: f64,               
    #[serde(default = "zero_default")] 
    pub select_niche_pct: f64,        
    #[serde(default = "ga_random_pct_default")]   
    pub select_random_pct: f64,        
    #[serde(default = "ga_mut_children_pct_default")]         
    pub mutated_children_pct: f64,     
    #[serde(default = "ga_mut_features_pct_default")]          
    pub mutated_features_pct: f64,           
    #[serde(default = "ga_mut_non_null_pct_default")]   
    pub mutation_non_null_chance_pct: f64,    
    #[serde(default = "zero_default")] 
    pub forced_diversity_pct: f64,    
    #[serde(default = "uzero_default")] 
    pub forced_diversity_epochs: usize,    
    #[serde(default = "zero_default")] 
    pub random_sampling_pct: f64,         
    #[serde(default = "uzero_default")]                 
    pub random_sampling_epochs: usize,       
    #[serde(default = "uzero_default")]       
    pub n_epochs_before_global: usize,    
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BEAM {
    #[serde(default = "beam_method_default")]  
    pub method: String,
    #[serde(default = "one_default")]  
    pub kmin: usize,                           
    #[serde(default = "kmax_default")]  
    pub kmax: usize,                           
    #[serde(default = "best_models_ci_alpha_default")]
    pub best_models_ci_alpha: f64,                                 
    #[serde(default = "max_nb_of_models_default")]
    pub max_nb_of_models: usize,                                    
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Importance {
    #[serde(default = "false_default")] 
    pub compute_importance: bool,
    #[serde(default = "n_permutations_oob_default")]  
    pub n_permutations_oob: usize,
    #[serde(default = "false_default")]  
    pub scaled_importance: bool,
    #[serde(default = "importance_aggregation_default")]  
    pub importance_aggregation: ImportanceAggregation,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Experimental {
    #[serde(default = "false_default")] 
    pub threshold_ci: bool,
    #[serde(default = "zero_default")] 
    pub threshold_ci_penalty: f64,
    #[serde(default = "zero_default")] 
    pub threshold_ci_alpha: f64,
    #[serde(default = "uzero_default")] 
    pub threshold_ci_n_bootstrap: usize
}

// Default section definitions

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

impl Default for Importance {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for Voting {
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

impl Default for Experimental {
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

    if config.experimental.threshold_ci  {
        if !config.general.keep_trace {
            panic!("Experimental thresholdCI currently requires keep_trace==true")
        }

        if config.experimental.threshold_ci_alpha <= 0.0 || config.experimental.threshold_ci_alpha >= 1.0 {
            panic!("Configuration error: invalid experimental_threshold_ci_alpha value. Must be in range (0, 1).");
        }


        // Sanity check to prevent bad practices
        let B_min = 40.0/config.experimental.threshold_ci_alpha; //(Efron, 1987)
        let B_opt = 100.0/config.experimental.threshold_ci_alpha;
        if (config.experimental.threshold_ci_n_bootstrap as f64) < B_min {
            error!("Bootstrap sample size B={} is BELOW theoretical minimum B_min={}. Confidence interval quantiles may be undefined or severely biased.", config.experimental.threshold_ci_n_bootstrap, B_min);
            panic!("Bootstrap sample size B={} is BELOW theoretical minimum B_min={}. Confidence interval quantiles may be undefined or severely biased.", config.experimental.threshold_ci_n_bootstrap, B_min);
        } else if (config.experimental.threshold_ci_n_bootstrap as f64) < B_opt {
            warn!("Bootstrap sample size B={} is ABOVE minimum but BELOW optimal threshold B={}. Expect moderate instability and potential under-coverage.", config.experimental.threshold_ci_n_bootstrap, B_opt);
        }   
    }

    Ok(config)
}

// Default value definitions

fn seed_default() -> u64 { 4815162342 }
fn empty_string() -> String { "".to_string() }
fn min_epochs_default() -> usize { 10 }
fn max_epochs_default() -> usize { 200 }
fn max_age_best_model_default() -> usize { 10 }
fn algorithm_default() -> String { "ga".to_string() }
fn feature_selection_method_default() -> String { "wilcoxon".to_string() }
fn feature_minimal_prevalence_pct_default() -> f64 { 10.0 }
fn feature_maximal_pvalue_default() -> f64 { 0.5 }
fn feature_minimal_log_abs_bayes_factor_default() -> f64 { 2.0 }
fn language_default() -> String { "binary".to_string() }
fn data_type_default() -> String { "raw".to_string() }
fn data_type_epsilon_default() -> f64 { 1e-5 }
fn feature_keep_trace_default() -> bool { true }
fn save_experiment_default() -> String { "".to_string() }
fn log_base_default() -> String { "".to_string() }
fn log_suffix_default() -> String { "log".to_string() }
fn log_level_default() -> String { "info".to_string() }
fn folds_default() -> usize { 5 }
fn n_permutations_oob_default() -> usize { 100 }
fn importance_aggregation_default() -> ImportanceAggregation { ImportanceAggregation::mean }
fn fit_default() -> FitFunction { FitFunction::auc }
fn n_model_to_display_default() -> u32 { 10 }
fn false_default() -> bool { false }
fn true_default() -> bool { true }
fn display_level_default() -> usize { 2 }
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
fn zero_default() -> f64 { 0.0 }
fn uzero_default() -> usize { 0 }
fn half_default() -> f64 { 0.5 }
fn one_default() -> usize { 1 }
fn voting_default() -> VotingMethod { VotingMethod::Majority }
fn specialized_default() -> f64 { 0.6 }
fn diversity_voting_default() -> f64 { 5.0 }
fn kmax_default() -> usize { 200 }
fn pop_size_default() -> u32 { 5000 }
fn ga_elite_pct_default() -> f64 { 2.0 }
fn ga_random_pct_default() -> f64 { 2.0 }
fn ga_mut_children_pct_default() -> f64 { 80.0 }
fn ga_mut_features_pct_default() -> f64 { 20.0 }
fn ga_mut_non_null_pct_default() -> f64 { 20.0 }