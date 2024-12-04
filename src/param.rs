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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Data {
    pub X: String,                      // Path to X data
    pub y: String,
    #[serde(default = "empty_string")]                      // Path to y data
    pub Xtest: String,
    #[serde(default = "empty_string")]                      // Path to y data
    pub ytest: String,
    #[serde(default = "feature_minimal_prevalence_default")]                      // Path to y data
    pub feature_minimal_prevalence: u32, // Minimum prevalence
    #[serde(default = "feature_maximal_pvalue_default")]                      // Path to y data
    pub feature_maximal_pvalue: f64, // Minimum prevalence
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GA {
    pub population_size: u32,                // Population size
    pub epochs: u32,                         // Number of epochs/generations
    #[serde(default = "min_epochs_default")]    
    pub min_epochs: usize,                     // Do not stop before reaching this
    #[serde(default = "max_divergence_default")]    
    pub max_divergence: f64,                 // Stop only if fit function divergence is lower than this in average
    pub kmin: u32,                           // Minimum value of k
    pub kmax: u32,                           // Maximum value of k
    pub kpenalty: f64,                       // A penalty of this per value of k is deleted from AUC in the fit function
    pub select_elite_pct: f64,               // Elite selection percentage
    pub select_random_pct: f64,              // Random selection percentage
    pub mutated_children_pct: f64,        // Mutated individuals percentage
    pub mutated_features_pct: f64,           // Mutated features percentage
    pub mutation_non_null_chance_pct: f64    // Chance pct that a mutation gives an non null value
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
fn max_divergence_default() -> f64 { 0.01 }
fn algorithm_default() -> String { "ga".to_string() }
fn feature_minimal_prevalence_default() -> u32 { 10 }
fn feature_maximal_pvalue_default() -> f64 { 0.5 }