use rand::Rng;
use rand::prelude::SliceRandom;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::hash::Hash;
use std::hash::Hasher;
use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::{HashMap};
use statrs::function::logistic::logistic;
use statrs::function::erf::{erf, erf_inv};
use serde::{Serialize, Deserialize};
use crate::individual::{RAW_TYPE, PREVALENCE_TYPE, LOG_TYPE};
use argmin::{
    core::{CostFunction, Error as ArgminError, Executor},  
    solver::brent::BrentOpt,
};
use crate::experiment::{Importance, ImportanceCollection, ImportanceScope, ImportanceType};

use log::{debug, info, warn, error};
use crate::individual;
use crate::data::Data;
use crate::individual::{Individual, data_type, MCMC_GENERIC_LANG};
use crate::Population;
use crate::param::Param;

const BIG_NUMBER: f64 = 100.0;

//-----------------------------------------------------------------------------
// Utilities  

// Calculates the logarithm of the logistic function (log(1/(1+e^(-x)))) with protection against numerical underflow
// for extreme negative values. Returns a stable approximation when x is very negative.
fn log_logistic(x: f64) -> f64 {
    if x >= - BIG_NUMBER {
        logistic(x).ln()
    } else {
        x + logistic(-BIG_NUMBER).ln() + BIG_NUMBER
    }
}

// Generates a random number from a truncated normal distribution that must be positive.
// Uses the inverse error function method to correctly sample from the truncated distribution.
// Parameters:
// - mu: Mean of the underlying normal distribution
// - scale: Standard deviation of the underlying normal distribution
// - rng: Random number generator
fn truncnorm_pos(mu: f64, scale: f64, rng: &mut rand_chacha::ChaCha8Rng) -> f64 {
    // truncated normal positive random number 
    let erf0 = erf(- mu / (2_f64.sqrt() * scale));
    let u: f64 = rng.gen_range(0.0..1.0);
    mu + 2_f64.sqrt() * scale * erf_inv(u * (1.0 - erf0) + erf0)
}

// Generates a random number from a truncated normal distribution that must be negative.
// Uses the inverse error function method to correctly sample from the truncated distribution.
// Parameters:
// - mu: Mean of the underlying normal distribution
// - scale: Standard deviation of the underlying normal distribution
// - rng: Random number generator
fn truncnorm_neg(mu: f64, scale: f64, rng: &mut rand_chacha::ChaCha8Rng) -> f64 {
    // truncated normal negative random number
    let erf0 = erf(- mu / (2_f64.sqrt() * scale));
    let u: f64 = rng.gen_range(0.0..1.0);
    mu + 2_f64.sqrt() * scale * erf_inv(u * (1.0 + erf0) - 1.0)
}

// Generates a random number from a standard normal distribution using the inverse error function method.
// Parameters:
// - mu: Mean of the normal distribution
// - scale: Standard deviation of the normal distribution
fn random_normal(mu: f64, scale: f64, rng: &mut rand_chacha::ChaCha8Rng) -> f64 {
    // normal random number
    let u: f64 = rng.gen_range(0.0..1.0);
    mu + 2_f64.sqrt() * scale * erf_inv(2.0 * u - 1.0)
}


//-----------------------------------------------------------------------------
// Structure for holding data for MCMC

// Structure for making predictions using an ensemble of models from MCMC sampling.
// Contains multiple model configurations and their parameters for robust prediction.
// <=> Density
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Betas {
    pub a: f64,
    pub b: f64,
    pub c: f64
}

impl Betas {
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Betas { a, b, c }
    }

    pub fn get(&self) -> [f64;3] { [self.a, self.b, self.c] }

    pub fn set(&mut self, idx: usize, val: f64) {
        match idx {
            0 => self.a = val,
            1 => self.b = val,
            2 => self.c = val,
            _ => panic!("β index out of range (0..2)"),
        }
    }
}

impl Hash for Betas {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.a.to_bits());
        state.write_u64(self.b.to_bits());
        state.write_u64(self.c.to_bits());
    }
}

// Main structure for Bayesian prediction models that holds data and model parameters
// for MCMC sampling and posterior probability calculations.
pub struct BayesPred {
    data: Data,
    lambda: f64,
    data_type: u8,    
    epsilon: f64,     
}

impl BayesPred {
    pub fn new(data: &Data, lambda: f64, data_type: u8, epsilon: f64) -> BayesPred {
        BayesPred {
            data: data.clone(),
            lambda,
            data_type,
            epsilon,
        }
    }
    
    fn transform_value(&self, value: f64) -> f64 {
        match self.data_type {
            RAW_TYPE => value,
            PREVALENCE_TYPE => {
                if value > self.epsilon { 1.0 } else { 0.0 }
            },
            LOG_TYPE => {
                if value > 0.0 {
                    (value / self.epsilon).ln()
                } else {
                    0.0 
                }
            },
            _ => panic!("Unknown data-type {}", self.data_type)
        }
    }

    // Organizes features from an Individual into categorical groups:
    // - Positive features (coefficient = 1)
    // - Negative features (coefficient = -1)
    // - Intercept (always 1.0)
    // Returns a vector of vectors where each inner vector contains [pos_features, neg_features, intercept]
    // for each sample in the dataset.
    fn compute_feature_groups(&self, ind: &Individual) -> Vec<[f64; 3]> {
        let mut z = Vec::with_capacity(self.data.sample_len);
        for i_sample in 0..self.data.sample_len {
            let mut pos_features: f64 = 0.0;
            let mut neg_features: f64 = 0.0;
            for (&feature_idx, &coef) in ind.features.iter() {
                if let Some(&raw_value) = self.data.X.get(&(i_sample, feature_idx)) {
                    let value = self.transform_value(raw_value);
                    match coef {
                        1 => pos_features += value,
                        -1 => neg_features += value,
                        _ => ()
                    }
                }
            }
            z.push([pos_features, neg_features, 1.0])
        }
        z
    }


    // Efficiently updates feature groups when a feature coefficient changes without recomputing all groups.
    // Parameters:
    // - z: Current feature groups
    // - feature_idx: Index of the feature being updated
    // - old_coef: Previous coefficient value
    // - new_coef: New coefficient value
    // Returns a new vector of updated feature groups.
    pub fn update_feature_groups(&self, z: &[[f64; 3]], ind: &Individual, feature_idx: usize, new_coef: i8) -> Vec<[f64; 3]> {
       let old_coef = ind.get_coef(feature_idx);

        if old_coef == new_coef || !self.data.feature_selection.contains(&feature_idx) {
            return z.to_vec();
        }

        let mut z_new = z.to_vec();

        let delta_pos: f64 = match (old_coef, new_coef) {
            (0,  1) =>  1.0,  
            (1,  0) => -1.0,   
            (-1, 1) =>  1.0,   
            (1, -1) => -1.0,   
            _       =>  0.0,   
        };           

        let delta_neg: f64 = match (old_coef, new_coef) {
            (0, -1) =>  1.0,   
            (-1, 0) => -1.0, 
            (1, -1) =>  1.0,   
            (-1, 1) => -1.0,  
            _       =>  0.0,  
        };                 

        for i_sample in 0..self.data.sample_len {
            if let Some(&raw) = self.data.X.get(&(i_sample, feature_idx)) {
                let v = self.transform_value(raw);
                if delta_pos != 0.0 { z_new[i_sample][0] += delta_pos * v; }
                if delta_neg != 0.0 { z_new[i_sample][1] += delta_neg * v; }
            }
        }

    z_new
}



    // Calculates the log posterior probability for given model parameters and feature groups.
    // Combines the log-likelihood of the observed data with the log-prior of the parameters.
    // Parameters:
    // - beta: Model coefficients [beta_pos, beta_neg, beta_intercept]
    // - z: Feature groups [pos_features, neg_features, intercept]
    // Returns the log posterior probability.
    fn log_posterior(&self, ind: &Individual, z: &Vec<[f64;3]>) -> f64 {
        let [a,b,c] = ind.get_betas();
        let mut log_likelihood = 0.0;
        for (i_sample, z_sample) in z.iter().enumerate() {
            let y_sample = self.data.y[i_sample] as f64;
            let value = z_sample[0] * a + z_sample[1] * b + z_sample[2] * c;
            
            if y_sample == 1.0 {
                log_likelihood += log_logistic(value);
            } else {
                log_likelihood += log_logistic(-value);
            }
            
            // if log_logistic(value).is_infinite() || log_logistic(-value).is_infinite() {
            //     println!("v={}, logistic(x)={}, logistic(-x)={}", value, logistic(value), logistic(-value));
            //     process::exit(1);
            // }
        }
        
        let log_prior: f64 = -self.lambda * [a, b, c].iter().map(|v| v * v).sum::<f64>();
        log_likelihood + log_prior
    }


    // Computes the standard deviation for a specific model parameter, used to determine
    // appropriate proposal distribution width in MCMC sampling.
    // Parameters:
    // - beta: Current model coefficients
    // - z: Feature groups
    // - param_idx: Index of the parameter to compute sigma for (0=pos, 1=neg, 2=intercept)
    // Returns the standard deviation.
    fn compute_sigma_i(&self, ind: &Individual, z: &Vec<[f64;3]>, i: usize) -> f64 {
        let [a,b,c] = ind.get_betas();
        let mut cov_inv = 0.0;
        
        for (_i_sample, z_sample) in z.iter().enumerate() {
            let value = z_sample[0] * a + z_sample[1] * b + z_sample[2] * c;
            cov_inv += logistic(value) * logistic(-value) * z_sample[i].powf(2.0)
        }
        
        1.0 / (cov_inv + self.lambda).sqrt()
    }

}

// Helper structure for Brent optimization that implements the CostFunction trait.
// Used to minimize the negative log posterior probability for parameter optimization.
// Holds references to the Bayesian prediction model, parameter index, current beta values, and feature groups.
struct NegLogPostToMinimize<'a> {
    bp:   &'a BayesPred,
    i:    usize,
    ind:  &'a Individual,
    z:    &'a Vec<[f64;3]>,
}

// Implements the cost function for optimization.
// Returns the negative log posterior probability for the proposed parameter value.
impl CostFunction for NegLogPostToMinimize<'_> {
    type Param  = f64;
    type Output = f64;

    fn cost(&self, beta_i: &Self::Param) -> Result<Self::Output, ArgminError> {
        let mut clone = self.ind.clone();
        clone.set_beta(self.i, *beta_i);
        Ok(-self.bp.log_posterior(&clone, self.z))
    }

}


//-----------------------------------------------------------------------------
// Structure for MCMC tracing

// Structure for storing and processing results from MCMC sampling.
// Contains model statistics, feature probabilities, parameter estimates,
// and optional traces of the sampling process.
#[derive(Clone, Debug)]
pub struct MCMCAnalysisTrace {
    pub population: Population,
    pub feature_selection: Vec<usize>,
    pub feature_prob: HashMap<usize, (f64, f64, f64)>,  // idx -> (pos, neutre, neg)
    pub model_stats: HashMap<usize, (f64, f64)>,        // idx -> (moyenne, variance)
    pub beta_mean: [f64; 3],
    pub beta_var: [f64; 3],
    pub log_post_mean: f64,
    pub log_post_var: f64,
    pub log_post_trace: Vec<f64>,
    pub post_mean: f64,
    pub post_var: f64,
    pub param: Param
}

impl MCMCAnalysisTrace {
    // Creates a new MCMCAnalysisTrace instance to store MCMC sampling results.
    // Parameters:
    // - n_features: Total number of features in the model
    // - keep_trace: Whether to store detailed traces of sampling history
    // - feature_names: Optional vector of feature names for better interpretability
    pub fn new(feature_selection: &Vec<usize>, param:Param) -> Self {
        let n_features = feature_selection.len();
        let mut feature_prob = HashMap::with_capacity(n_features);
        let mut model_stats = HashMap::with_capacity(n_features);
        for idx in feature_selection {
                // Probabilities are divided with finalize()
                feature_prob.insert(*idx, (0.0, 0.0, 0.0));
                model_stats.insert(*idx, (0.0, 0.0));
        }
        
        MCMCAnalysisTrace {
            population: Population::new(), 
            feature_selection: feature_selection.clone(),
            feature_prob,
            model_stats,
            beta_mean: [0.0, 0.0, 0.0],
            beta_var: [0.0, 0.0, 0.0],
            log_post_mean: 0.0,
            log_post_var: 0.0,
            log_post_trace: Vec::with_capacity(n_features),
            post_mean: 0.0,
            post_var: 0.0,
            param: param
        }
    }

    // Updates the MCMC results with a new sample.
    // Accumulates statistics for parameters, feature probabilities, and model performance.
    // Parameters:
    // - beta: Current model coefficients
    // - individual: Current feature selection and coefficients
    // - log_post: Log posterior probability of the current sample
    pub fn update(&mut self, ind: &Individual, log_post: f64) {
        let [a,b,c] = ind.get_betas();
        self.beta_mean = [ self.beta_mean[0]+a, self.beta_mean[1]+b, self.beta_mean[2]+c ];
        self.beta_var  = [ self.beta_var[0]+a*a, self.beta_var[1]+b*b, self.beta_var[2]+c*c ];

        for idx in &self.feature_selection {
            let coef = ind.features.get(idx).copied().unwrap_or(0);
            let prob = self.feature_prob.get_mut(idx).unwrap();
            match coef { 1 => prob.0+=1.0, -1=>prob.2+=1.0, _=>prob.1+=1.0 }
            let stat = self.model_stats.get_mut(idx).unwrap();
            stat.0 += coef as f64;
            stat.1 += (coef as f64).powi(2);
        }

        self.log_post_mean += log_post;
        self.log_post_var  += log_post.powi(2);
        self.post_mean     += log_post.exp();
        self.post_var      += (2.0*log_post).exp();

        let mut clone = ind.clone();
        clone.epoch = self.population.individuals.len();
        clone.compute_hash();
        self.population.individuals.push(clone);
        self.log_post_trace.push(log_post);
    }

    
    // fn get_feature_name(&self, feature_idx: usize) -> String {
    //     match &self.feature_names {
    //         Some(names) if feature_idx < names.len() => names[feature_idx].clone(),
    //         _ => format!("feature_{}", feature_idx),
    //     }
    // }

    // Finalizes MCMC results by calculating means and variances from accumulated statistics.
    // Accounts for burn-in period and normalizes probabilities as needed.
    // Parameters:
    // - n_iter: Total number of MCMC iterations
    // - n_burn: Number of burn-in iterations to discard
    pub fn finalize(&mut self, n_iter: usize, n_burn: usize) {
        let n_mean = ((self.population.individuals[0].features.len() + 3) * (n_iter - n_burn)) as f64;
        
        // Beta
        self.beta_mean = self.beta_mean.iter()
            .map(|v| v / n_mean)
            .collect::<Vec<f64>>().try_into().unwrap();
        
        self.beta_var = self.beta_var.iter().zip(self.beta_mean.iter())
            .map(|(v, m)| (v - m.powf(2.0) * n_mean) / (n_mean - 1.0))
            .collect::<Vec<f64>>().try_into().unwrap();
        
        // Statistics
        for (_, stat) in &mut self.model_stats {
            stat.0 /= n_mean; 
            stat.1 = (stat.1 - stat.0.powf(2.0) * n_mean) / (n_mean - 1.0); 
        }
        
        // Probabilities
        for (_, prob) in &mut self.feature_prob {
            prob.0 /= n_mean; // pos
            prob.1 /= n_mean; // neutral
            prob.2 /= n_mean; // neg
            
            // Normalize
            let sum = prob.0 + prob.1 + prob.2;
            if (sum - 1.0).abs() > 1e-10 {
                let scale = 1.0 / sum;
                prob.0 *= scale;
                prob.1 *= scale;
                prob.2 *= scale;
            }
        }
        
        // Finalize
        self.log_post_mean /= n_mean;
        self.log_post_var = (self.log_post_var - self.log_post_mean.powf(2.0) * n_mean) / (n_mean - 1.0);
        self.post_mean /= n_mean;
        self.post_var = (self.post_var - self.post_mean.powf(2.0) * n_mean) / (n_mean - 1.0);
    }

    pub fn get_log_evidence(&self, n_classes:f64) -> f64 {
        let n_features = self.population.individuals[0].features.len() as f64;
        self.post_mean.log10() - n_features * n_classes.log10()
    }

    // Returns final MCMC results with informative logging of key statistics.
    // Identifies and reports the most important features based on sampling.
    // Parameters:
    // - bp: Reference to the BayesPred model
    // Returns a clone of the results for further processing
    // pub fn get_results(&self, bp: &BayesPred) -> Self {
    //     log::info!("MCMC Analysis Results:");
    //     log::info!("----------------------");
    //     log::info!("Beta coefficients: a={:.4}, b={:.4}, c={:.4}", 
    //                self.beta_mean[0], self.beta_mean[1], self.beta_mean[2]);
    //     log::info!("Log Posterior mean: {:.4}, variance: {:.4}", 
    //                self.log_post_mean, self.log_post_var);
        
    //     let mut top_features: Vec<(&String, f64)> = self.feature_prob.iter()
    //         .map(|(name, (pos, _, _))| (name, *pos))
    //         .collect();
        
    //     top_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
    //     log::info!("Top 5 positive features:");
    //     for (i, (name, prob)) in top_features.iter().take(5).enumerate() {
    //         log::info!("  {}. {} - Probability: {:.4}", i+1, name, prob);
    //     }
        
    //     self.clone()
    // }

    pub fn export_to_files(&self, data: &Data, outdir: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};
        use std::fs;
        
        fs::create_dir_all(outdir)?;
        
        // Export p_mean
        let p_mean_path = format!("{}/P_mean.tsv", outdir);
        let mut file = BufWriter::new(File::create(&p_mean_path)?);
        writeln!(file, "FEATURE\tPOS\tNUL\tNEG")?;
        let mut feature_idx: Vec<_> = self.feature_prob.keys().collect();
        feature_idx.sort();
        for idx in &feature_idx {
            let (pos, neutral, neg) = self.feature_prob.get(idx).unwrap();
            writeln!(file, "{}\t{}\t{}\t{}", data.features[**idx], pos, neutral, neg)?;
        }
        
        // Export betas coefficients
        let beta_trace_path = format!("{}/betas.tsv", outdir);
        let mut file = BufWriter::new(File::create(&beta_trace_path)?);
        writeln!(file, "a\tb\tc")?;
        for ind in &self.population.individuals {
            let beta = ind.betas.as_ref().expect("MCMC individual without betas");
            writeln!(file, "\t{}\t{}\t{}", beta.a, beta.b, beta.c)?;
        }

        // Export features
        let model_trace_path = format!("{}/features.tsv", outdir);
        let mut file = BufWriter::new(File::create(&model_trace_path)?);
        let names: Vec<String> = feature_idx.iter()
            .map(|&idx| data.features[*idx].clone())
            .collect();
        writeln!(file, "{}", names.join("\t"))?;
        for ind in &self.population.individuals {
            let row: Vec<String> = feature_idx
                .iter()
                .map(|orig| ind.features.get(orig).copied().unwrap_or(0).to_string())
                .collect();
            writeln!(file, "\t{}", row.join("\t"))?;
        }
            
        // Export statistics
        let means_vars_path = format!("{}/means_vars.tsv", outdir);
        let mut file = BufWriter::new(File::create(&means_vars_path)?);
        
        writeln!(file, "parameter\tmean\tvariance")?;
        writeln!(file, "a\t{}\t{}", self.beta_mean[0], self.beta_var[0])?;
        writeln!(file, "b\t{}\t{}", self.beta_mean[1], self.beta_var[1])?;
        writeln!(file, "c\t{}\t{}", self.beta_mean[2], self.beta_var[2])?;
        
        for (idx, (mean, variance)) in self.model_stats.iter() {
                writeln!(file, "{}\t{}\t{}", data.features[*idx], mean, variance)?;
        }
        
        writeln!(file, "logPost\t{}\t{}", self.log_post_mean, self.log_post_var)?;
        writeln!(file, "Post\t{}\t{}", self.post_mean, self.post_var)?;
        
        Ok(())
    }

    // Return the feature_prob in a ImportanceCollectiob
    pub fn get_importance(&self) -> ImportanceCollection {
        let mut importances = Vec::with_capacity(self.feature_prob.len() * 2);

        for (&idx, &(p_pos, _p_neut, p_neg)) in &self.feature_prob {
            // POS ------------------------------------------------------------
            if p_pos > 0.0 {
                importances.push(Importance {
                    importance_type : ImportanceType::PosteriorProbability,
                    feature_idx     : idx,
                    scope           : ImportanceScope::Population { id: 0 }, 
                    aggreg_method   : None,
                    importance      : p_pos,                      
                    dispersion      : (p_pos * (1.0 - p_pos) / self.population.individuals.len() as f64).sqrt(), // Bernoulli(P) repeated n_iter times
                    is_scaled       : false,
                    scope_pct       : 1.0,
                    direction       : Some(1)
                });
            }
            // NEG ------------------------------------------------------------
            if p_neg > 0.0 {
                importances.push(Importance {
                    importance_type : ImportanceType::PosteriorProbability,
                    feature_idx     : idx,
                    scope           : ImportanceScope::Population { id: 0 }, 
                    aggreg_method   : None,
                    importance      : p_neg,                      
                    dispersion      : (p_neg * (1.0 - p_neg) / self.population.individuals.len() as f64).sqrt(),
                    is_scaled       : false,
                    scope_pct       : 1.0,
                    direction       : Some(0)
                });
            }
        }
        ImportanceCollection { importances }
    }


//     pub fn import_from_files(outdir: &str) -> std::io::Result<MCMCAnalysisTrace> {
//         use std::fs::File;
//         use polars::prelude::*;

//         let p_mean_path = format!("{}/P_mean.tsv", outdir);
//         let df_p_mean = CsvReader::from_path(&p_mean_path)?
//             .with_separator(b'\t')
//             .has_header(true)
//             .finish()?;
        
//         let beta_path = format!("{}/betas.tsv", outdir);
//         let df_betas = CsvReader::from_path(&beta_path)?
//             .with_separator(b'\t')
//             .has_header(true)
//             .finish()?;
        
//         let models_path = format!("{}/models.tsv", outdir);
//         let df_models = CsvReader::from_path(&models_path)?
//             .with_separator(b'\t')
//             .has_header(true)
//             .finish()?;

//         let feature_names: Vec<String> = df_models.get_column_names()
//             .into_iter()
//             .map(|s| s.to_string())
//             .collect();
        

//         let mut beta_trace: Vec<[f64; 3]> = Vec::new();
//         let a_col = df_betas.column("a")?.f64()?.to_vec();
//         let b_col = df_betas.column("b")?.f64()?.to_vec();
//         let c_col = df_betas.column("c")?.f64()?.to_vec();
        
//         for i in 0..a_col.len() {
//             beta_trace.push([
//                 a_col[i].unwrap_or(0.0),
//                 b_col[i].unwrap_or(0.0),
//                 c_col[i].unwrap_or(0.0)
//             ]);
//         }

//         let mut population = Population::new();
//         for row_idx in 0..df_models.height() {
//             let mut individual = Individual::new();
            
//             for (col_idx, col_name) in feature_names.iter().enumerate() {
//                 let value = df_models.get(row_idx, col_idx)?;
//                 if let AnyValue::Int32(coef) = value {
//                     if coef != 0 {
//                         individual.features.insert(col_idx, coef as i8);
//                     }
//                 }
//             }
            
//             population.individuals.push(individual);
//         }
        
//         let mut trace = MCMCAnalysisTrace::new(
//             feature_names.len(),
//             true,
//             Some(feature_names)
//         );
        
//         trace.beta_trace = Some(beta_trace);
//         trace.population_trace = Some(population);
        
//         // To add : df_means_vars
//         // if let Ok(df_means_vars) = CsvReader::from_path(format!("{}/means_vars.tsv", outdir))
//         //     .with_separator(b'\t')
//         //     .has_header(true)
//         //     .finish() {
            
//         //     
//         //   
//         // }
        
//         Ok(trace)
//     }
}

//-----------------------------------------------------------------------------
// Algorithms

// Implements Sequential Backward Selection using MCMC to progressively eliminate 
// features while monitoring model performance.
// Starts with all features and removes the least important ones sequentially.
// Parameters:
// - data: Dataset to model
// - param: Configuration parameters for the MCMC process
// Returns a vector of results for each model size with performance metrics.
pub fn run_mcmc_sbs(data: &Data, param: &Param, rng: &mut ChaCha8Rng, running: Arc<AtomicBool>) -> Vec<(u32, f64, f64, f64, usize, ChaCha8Rng)> {
    let time = Instant::now();
    let mut data_train = data.clone();
    let nmax = data_train.feature_selection.len() as u32;
    let mut post_mean = Vec::new();
    let mut feature_to_drop = Vec::new();
    let mut rng_trace = Vec::new();

    let data_types: Vec<&str> = param.general.data_type.split(",").collect();
    let data_type = data_types[0];
    if data_types.len() > 1 { warn!("MCMC allows only one datatype per launch currently. Keeping: {}", data_type)}

    for n in (param.mcmc.nmin..=nmax).rev() {
        debug!("n = {}, (#Features, #Samples) = ({}, {})", n, &data_train.feature_selection.len(), data_train.sample_len);
        let bp = BayesPred::new(&data_train, param.mcmc.lambda, individual::data_type(data_type), param.general.data_type_epsilon);

        rng_trace.push(rng.clone());
        let res = compute_mcmc(&bp, param, rng);

        post_mean.push(res.post_mean);
        
        // Drop neutral feature but keep it
        let mut feature_vec: Vec<_> = res.feature_prob.iter().collect();
        feature_vec.sort_by(|a, b| a.0.cmp(b.0));

        let (idx, (_, neutral_prob, _)) = feature_vec.iter()
            .max_by(|(_, (_, neutral1, _)), (_, (_, neutral2, _))| {
                neutral1.total_cmp(neutral2)
            })
            .map(|(idx, &prob)| (*idx, prob))
            .unwrap_or_else(|| panic!("Couldn't find maximum neutral probability"));

        let feature_name = &data_train.features[*idx];
        debug!("Feature {} has maximum neutral probability: {}", feature_name, neutral_prob);
        
        data_train.feature_selection.retain(|keep_idx| keep_idx != idx);
        
        info!("[{}/{}] | Dropping feature: {}", data.feature_selection.len()-(n as usize), data.feature_selection.len()-(param.mcmc.nmin) as usize, feature_name);
        feature_to_drop.push(idx.clone());

        // Stop SBS loop if user want to safe quit from this step
        if !running.load(Ordering::Relaxed) {
            info!("Signal received");
            break
        }

    }

    let elapsed = time.elapsed();
    info!("SBS computed {:?} steps in {:.2?}", post_mean.len(), elapsed);

    // Calculate metrics
    let nn: Vec<u32> = (param.mcmc.nmin..=nmax).rev().collect();
    let log_post_mean: Vec<f64> = post_mean.iter().map(|v| v.log10()).collect();
    let n_classes = 3.0 as f64; 
    let log_evidence: Vec<f64> = nn.iter()
        .zip(log_post_mean.iter())
        .map(|(n, lpm)| lpm - (*n as f64) * n_classes.log10())
        .collect();

    nn.into_iter()
        .zip(post_mean)
        .zip(log_post_mean)
        .zip(log_evidence)
        .zip(feature_to_drop)
        .zip(rng_trace) 
        .map(|(((((n, pm), lpm), le), idx), rng)| (n, pm, lpm, le, idx, rng))
        .collect()
}

// Core MCMC implementation for Bayesian model sampling.
// Uses Metropolis-Hastings algorithm with Brent optimization for continuous parameters
// and discrete proposal updates for feature coefficients.
// Parameters:
// - bp: Bayesian prediction model
// - param: MCMC configuration parameters
// - rng: Random number generator with specified seed
// Returns the MCMC sampling results after burn-in and convergence.
pub fn compute_mcmc(bp: &BayesPred, param: &Param, rng: &mut ChaCha8Rng) -> MCMCAnalysisTrace {
    if param.mcmc.n_burn>=param.mcmc.n_iter {
        error!("n_iter should be greater than n_burn!");
        panic!("n_iter should be greater than n_burn!");
    }

    //let _nvals = 3;
    
    // Initialize coefficients 
    let mut ind = Individual::new();
    ind.language = MCMC_GENERIC_LANG;
    ind.data_type = data_type(&param.general.data_type);
    ind.epsilon  = param.general.data_type_epsilon;
    ind.betas = Some(Betas::new(1.0, -1.0, 1.0));

    for idx in &bp.data.feature_selection {
        if let &coef @ (-1|0|1) = [1,0,-1].choose(rng).unwrap() {
            if coef != 0 { ind.features.insert(*idx, coef); }
        }
    }
    ind.k = ind.features.len();

    
    let nbeta: usize = 3;
    let mut z = bp.compute_feature_groups(&ind);
    let mut log_post: f64 = 0.0;

    let mut res_mcmc = MCMCAnalysisTrace::new(&bp.data.feature_selection, param.clone());

    debug!("Computing MCMC...");
    
    let solver = BrentOpt::new(-10_f64.powf(4_f64), 10_f64.powf(4_f64));
    for n in 0..param.mcmc.n_iter {
        if param.mcmc.n_iter>1000 && ((n as f64 == param.mcmc.n_iter as f64 * 0.25) || (n as f64 == param.mcmc.n_iter as f64 * 0.50) ||  (n as f64 == param.mcmc.n_iter as f64 * 0.75)) {
            debug!("MCMC : {}% iterations finished: a={:.4}, b={:.4}, c={:.4}", (n as f64 / param.mcmc.n_iter as f64) as f64 *100.0, ind.get_betas()[0], ind.get_betas()[1], ind.get_betas()[2]);
        }
        for i in 0..nbeta {
            let cost = NegLogPostToMinimize {
                bp: bp,
                i: i,
                ind: &ind,
                z: &z,
            };
        
            let res = Executor::new(cost, solver.clone())
                .configure(|state| state.max_iters(100))
                .run()
                .unwrap();

            ind.set_beta(i, res.state.param.unwrap());

            let scale_i = bp.compute_sigma_i(&ind, &z, i);
            let mut b_i = ind.get_betas()[i];
            b_i = if (b_i/scale_i).abs() > 10.0 {
                    random_normal(b_i, scale_i, rng)
                } else {
                    match i {
                        0 => truncnorm_pos(b_i, scale_i, rng),
                        1 => truncnorm_neg(b_i, scale_i, rng),
                        _ => random_normal   (b_i, scale_i, rng)
                    }
                };
            ind.set_beta(i, b_i);

            log_post = bp.log_posterior(&ind, &z);

            if n > param.mcmc.n_burn {
                res_mcmc.update(&ind, log_post);
            }
        }
        

        for &feature_idx in &bp.data.feature_selection {
            let current_coef = ind.get_coef(feature_idx);
            let new_coef = (current_coef + [2, 3].choose(rng).unwrap()) % 3 - 1;
            let z_new = bp.update_feature_groups(&z, &mut ind, feature_idx, new_coef);

            let log_post_new = bp.log_posterior(&ind, &z_new);
            let diff_log_post = log_post_new - log_post;
            let u: f64 = rng.gen();

            if diff_log_post > 0.0 || u < diff_log_post.exp() {
                ind.set_coef(feature_idx, new_coef);
                log_post = log_post_new;
                z        = z_new;
            }
            
            if n > param.mcmc.n_burn {
                res_mcmc.update(&ind, log_post);   
            }

        }
    }

    debug!("Finalizing MCMC...");

    res_mcmc.finalize(param.mcmc.n_iter, param.mcmc.n_burn);

    res_mcmc
}


// Identifies and returns the best model from sequential backward selection results.
// Selects based on log evidence and filters out unnecessary features.
// Parameters:
// - data: Original dataset
// - results: Results from sequential backward selection
// - param: Configuration parameters
// - rng: Random number generator
// Returns the MCMC results for the best model configuration.
pub fn get_best_mcmc_sbs(data: &Data, results: &[(u32, f64, f64, f64, usize, ChaCha8Rng)], param: &Param) -> MCMCAnalysisTrace {
    let best_models = results.iter()
        .max_by(|(_, _, _, le1, _, _), (_, _, _, le2, _, _)| le1.partial_cmp(le2).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or_else(|| panic!("Couldn't find best models"));
    
    let n_best = best_models.0;
    
    let features_to_drop: Vec<usize> = results.iter()
        .filter(|(n, _, _, _, _, _)| *n > n_best)
        .map(|(_, _, _, _, idx, _)| idx.clone())
        .collect();
    
    let mut data_filtered = data.clone();
    for feature_idx in &features_to_drop {
        data_filtered.feature_selection.retain(|keep_idx| keep_idx != feature_idx);
    }
    
    info!("\nBest Models: n = {}, data dimensions = ({}, {})", n_best, data_filtered.feature_selection.len(), data_filtered.sample_len);

    let data_types: Vec<&str> = param.general.data_type.split(",").collect();
    let data_type = data_types[0];
    if data_types.len() > 1 { warn!("MCMC allows only one datatype per launch currently. Keeping: {}", data_type)}

    let now = Instant::now();

    // Use the saved seed to reproduce the exact MCMC 
    let bp = BayesPred::new(&data_filtered, param.mcmc.lambda,individual::data_type(data_type), param.general.data_type_epsilon);
    let rng = &mut best_models.5.clone();
    let res: MCMCAnalysisTrace = compute_mcmc(&bp, param, rng);

    if param.mcmc.save_trace_outdir.len() > 0 {
        if let Err(e) = res.clone().export_to_files(&data, &param.mcmc.save_trace_outdir.to_string()) {
            error!("Failed to export MCMC results: {}", e);
        } else {
            info!("Best MCMC trace saved to {}", &param.mcmc.save_trace_outdir.to_string())
        }
    }
    
    let elapsed = now.elapsed();
    info!("Elapsed: {:.2?}", elapsed);
    
    res
    
}

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_log_logistic_threshold() {
        // Below threshold
        assert!(log_logistic(-BIG_NUMBER - 1.0).is_finite());
        // Threshold
        assert_eq!(log_logistic(-BIG_NUMBER), logistic(-BIG_NUMBER).ln());
        // Above threshold
        assert!((log_logistic(0.0) - logistic(0.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_logistic_threshold_positive_negative() {
        // Positive
        let test_values = [1.0, 5.0, 10.0];
        for x in test_values.iter() {
            let result = log_logistic(*x);
            let expected = logistic(*x).ln();
            assert!((result - expected).abs() < 1e-10);
        }
        // Negative
        let x = -10.0; 
        assert!((log_logistic(x) - logistic(x).ln()).abs() < 1e-10);
        // Negative and tiny
        assert!(log_logistic(-1000.0).is_finite() || log_logistic(-1000.0) == f64::NEG_INFINITY);
    }

    #[test]
    fn test_truncnorm_pos_positive() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        for _ in 0..1_000 {
            let x = truncnorm_pos(0.0, 1.0, &mut rng);
            assert!(x > 0.0, "sample = {x} is not strictly positive");
        }
    }

    #[test]
    fn test_truncnorm_neg_negative() {
        let mut rng = ChaCha8Rng::seed_from_u64(456);
        for _ in 0..1_000 {
            let x = truncnorm_neg(0.0, 1.0, &mut rng);
            assert!(x < 0.0, "sample = {x} is not strictly negative");
        }
    }

    #[test]
    fn test_random_normal_basic_stats() {
        let mut rng = ChaCha8Rng::seed_from_u64(789);
        const MU: f64 = 2.0;
        const SIGMA: f64 = 3.0;
        const N: usize = 20_000;

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..N {
            let x = random_normal(MU, SIGMA, &mut rng);
            sum += x;
            sum_sq += x * x;
        }

        let sample_mean = sum / N as f64;
        let sample_var = sum_sq / N as f64 - sample_mean.powi(2);

        assert!((sample_mean - MU).abs() < 0.2, "mean {sample_mean}");
        assert!((sample_var - SIGMA * SIGMA).abs() < 0.5, "var {sample_var}");
    }

}

  
// pub fn save_results_to_tsv(results: &[(u32, f64, f64, f64, String)], path: &str) -> std::io::Result<()> {
//     let file = File::create(path)?;
//     let mut writer = BufWriter::new(file);
    
//     
//     writeln!(writer, "nfeat\tPosterior mean\tLog Posterior mean\tLog Evidence\tFeature to drop")?;
    
//     
//     for (nfeat, post_mean, log_post, log_evidence, feature_name) in results {
//         writeln!(writer, "{}\t{}\t{}\t{}\t{}", nfeat, post_mean, log_post, log_evidence, feature_name)?;
//     }
    
//     Ok(())
// }

