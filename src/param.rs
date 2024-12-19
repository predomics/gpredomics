use std::fs::File;
use std::io::{BufRead, BufReader};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Serialize, Deserialize)]
pub struct Param {
    pub general: General,
    pub data: Data,                          // Nested struct for "data"
    pub ga: GA,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct General {
    pub seed: u64,
    #[serde(default = "algorithm_default")]  
    pub algo: String,
    #[serde(default = "thread_number_default")]  
    pub thread_number: usize,
    #[serde(default = "log_base_default")]  
    pub log_base: String,
    #[serde(default = "log_suffix_default")]  
    pub log_suffix: String,
    #[serde(default = "log_level_default")]  
    pub log_level: String
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Data {
    pub X: String,                      // Path to X data
    pub y: String,
    #[serde(default = "empty_string")]                      // Path to y data
    pub Xtest: String,
    #[serde(default = "empty_string")]                      // Path to y data
    pub ytest: String,
    #[serde(default = "pvalue_method_default")]                      // Path to y data
    pub pvalue_method: String,
    #[serde(default = "feature_minimal_prevalence_pct_default")]                      // Path to y data
    pub feature_minimal_prevalence_pct: f64, // Minimum prevalence
    #[serde(default = "feature_maximal_pvalue_default")]                      // Path to y data
    pub feature_maximal_pvalue: f64, // Minimum prevalence
    #[serde(default = "feature_minimal_feature_value_default")]                      // Path to y data
    pub feature_minimal_feature_value: f64, // Minimum prevalence
}

#[derive(Debug, Serialize, Deserialize)]
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
    pub kpenalty: f64,                       // A penalty of this per value of k is deleted from AUC in the fit function
    pub select_elite_pct: f64,               // Elite selection percentage
    pub select_random_pct: f64,              // Random selection percentage
    pub mutated_children_pct: f64,        // Mutated individuals percentage
    pub mutated_features_pct: f64,           // Mutated features percentage
    pub mutation_non_null_chance_pct: f64,    // Chance pct that a mutation gives an non null value
    #[serde(default = "feature_importance_permutations_default")]    
    pub feature_importance_permutations: usize,
    #[serde(default = "feature_keep_all_generations_default")]   
    pub keep_all_generations: bool
}


pub fn get(param_file: String) -> Result<Param, Box<dyn Error>> {
    // Open and read the X.tsv file
    let param_file_reader = File::open(param_file)?;
    let mut param_reader = BufReader::new(param_file_reader);
    
    let config:Param = serde_yaml::from_reader(param_reader)?;

    Ok(config)
}

fn empty_string() -> String { "".to_string() }
fn min_epochs_default() -> usize { 10 }
fn max_age_best_model_default() -> usize { 10 }
fn algorithm_default() -> String { "ga".to_string() }
fn pvalue_method_default() -> String { "studentt".to_string() }
fn feature_minimal_prevalence_pct_default() -> f64 { 10.0 }
fn feature_maximal_pvalue_default() -> f64 { 0.5 }
fn feature_importance_permutations_default() -> usize { 10 }
fn feature_minimal_feature_value_default() -> f64 { 0.0 }
fn thread_number_default() -> usize { 1 }
fn feature_kminkmax_default() -> usize { 0 }
fn feature_keep_all_generations_default() -> bool { true }
fn log_base_default() -> String { "".to_string() }
fn log_suffix_default() -> String { "log".to_string() }
fn log_level_default() -> String { "info".to_string() }
