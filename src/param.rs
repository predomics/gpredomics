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
    pub seed: u64
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Data {
    pub X: String,                      // Path to X data
    pub y: String,                      // Path to y data
    pub feature_minimal_prevalence: u32, // Minimum prevalence
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GA {
    pub population_size: u32,                // Population size
    pub epochs: u32,                         // Number of epochs/generations
    pub kmin: u32,                           // Minimum value of k
    pub kmax: u32,                           // Maximum value of k
    pub select_elite_pct: f64,               // Elite selection percentage
    pub select_random_pct: f64,              // Random selection percentage
    pub mutated_individuals_pct: f64,        // Mutated individuals percentage
    pub mutated_features_pct: f64,           // Mutated features percentage
}


pub fn get(param_file: String) -> Result<Param, Box<dyn Error>> {
    // Open and read the X.tsv file
    let param_file_reader = File::open(param_file)?;
    let mut param_reader = BufReader::new(param_file_reader);
    
    let config:Param = serde_yaml::from_reader(param_reader)?;

    Ok(config)
}
