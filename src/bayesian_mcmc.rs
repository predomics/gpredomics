use rand::Rng;
use rand::prelude::SliceRandom;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::process;
use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::{HashMap};
use rayon::prelude::*;
use statrs::function::logistic::logistic;
use statrs::function::erf::{erf, erf_inv};
use serde::{Serialize, Deserialize};
use argmin::{
    core::{CostFunction, Error as argmin_Error, Executor},
    solver::brent::BrentOpt,
};

use log::{debug, info, error};

use crate::data::Data;
use crate::individual::{Individual, MCMC_GENERIC_LANG};
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
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>
}

impl Betas {
    pub fn new(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> Self {
        Betas { a, b, c }
    }

    pub fn iter(&self) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
        self.a.iter().zip(self.b.iter()).zip(self.c.iter())
            .map(|((&a, &b), &c)| (a, b, c))
    }
}



#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MCMC {
    pub features_collection: Vec<HashMap<usize,i8>>,
    pub betas_collection: Betas
}

impl MCMC {
    // Creates a new prediction ensemble from a population of models.
    // Parameters:
    // - population: Reference to the population containing model configurations
    // - betas: Optional vector of beta coefficients for each model
    pub fn new(features_collection: Vec<HashMap<usize,i8>>, betas_collection: Betas) -> Self {
        MCMC {
            features_collection,
            betas_collection
        }
    }

    // Predicts class probabilities for each sample in the provided dataset.
    // Combines predictions from all models in the ensemble for robust estimation.
    // Parameters:
    // - data: Dataset to make predictions on
    // Returns a vector of probability pairs [prob_class0, prob_class1] for each sample.
    pub fn predict_proba(&self, data: &Data) -> Vec<[f64; 2]> {
        let probabilities;
        
        probabilities = (0..data.sample_len).into_par_iter().map(|i_sample| {
            let mut probs = [0.0, 0.0];
            for (features, betas) in self.features_collection.iter().zip(self.betas_collection.iter()) {
                let mut pos_features = 0.0;
                let mut neg_features = 0.0;
                
                for (&feat_idx, &coef) in features {
                    if let Some(&value) = data.X.get(&(i_sample, feat_idx)) {
                        match coef {
                            1 => pos_features += value,
                            -1 => neg_features += value,
                            _ => (),
                        }
                    }
                }
                
                let z = pos_features * betas.0 + neg_features * betas.1 + betas.2;
                probs[0] += logistic(-z);
                probs[1] += logistic(z);
            }
            
            probs[0] /= self.features_collection.len() as f64;
            probs[1] /= self.features_collection.len() as f64;

            assert!(probs[0] >= 0.0 && probs[0] <= 1.0, "p should be between 0 and 1");
            assert!(probs[1] >= 0.0 && probs[1] <= 1.0, "p should be between 0 and 1");

            probs
            
        }).collect::<Vec<[f64; 2]>>();
        
        probabilities
    }
    
    // Predicts discrete class labels by thresholding the probability predictions.
    // Uses the ensemble of models to determine the most likely class for each sample.
    // Parameters:
    // - data: Dataset to make predictions on
    // Returns a vector of predicted class labels.
    pub fn predict_class(&self, data: &Data) -> Vec<u8> {
        let probs = self.predict_proba(data);
        probs.iter()
            .map(|p| if p[1] > p[0] { 1 } else { 0 })
            .collect()
    }
}


// Main structure for Bayesian prediction models that holds data and model parameters
// for MCMC sampling and posterior probability calculations.
pub struct BayesPred {
    data: Data,
    lambda: f64,
}

impl BayesPred {
    // Creates a new BayesPred instance with the provided data and regularization parameter.
    // Parameters:
    // - data: Reference to the dataset to be modeled
    // - lambda: Regularization strength parameter (lambda) for model complexity penalization
    pub fn new(data: &Data, lambda: f64) -> BayesPred {
        BayesPred {
            data: data.clone(),
            lambda: lambda,
        }
    }

    // Organizes features from an Individual into categorical groups:
    // - Positive features (coefficient = 1)
    // - Negative features (coefficient = -1)
    // - Intercept (always 1.0)
    // Returns a vector of vectors where each inner vector contains [pos_features, neg_features, intercept]
    // for each sample in the dataset.
    fn compute_feature_groups(&self, features: &HashMap<usize,i8>) -> Vec<[f64; 3]> {
        let mut z = Vec::with_capacity(self.data.sample_len);
        for i_sample in 0..self.data.sample_len {
            let mut pos_features: f64 = 0.0;
            let mut neg_features: f64 = 0.0;
            
            for (&feature_idx, &coef) in features {
                if let Some(&value) = self.data.X.get(&(i_sample, feature_idx)) {
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
    fn update_feature_groups(&self, z: &Vec<[f64; 3]>, feature_idx: usize, old_coef: i8, new_coef: i8) -> Vec<[f64; 3]> {
        if old_coef == new_coef || !self.data.feature_selection.contains(&feature_idx){
            return z.clone();
        }
        
        let mut z_new = z.clone();
        
        let delta_pos = match (old_coef, new_coef) {
            (0, 1) => 1.0,
            (1, 0) => -1.0,
            (-1, 1) => 1.0,
            (1, -1) => -1.0,
            _ => 0.0,
        };
        
        let delta_neg = match (old_coef, new_coef) {
            (0, -1) => 1.0,
            (-1, 0) => -1.0,
            (1, -1) => 1.0,
            (-1, 1) => -1.0,
            _ => 0.0,
        };
        
        for i_sample in 0..self.data.sample_len {
            if let Some(&value) = self.data.X.get(&(i_sample, feature_idx)) {
                if delta_pos != 0.0 {
                    z_new[i_sample][0] += delta_pos * value;
                }
                if delta_neg != 0.0 {
                    z_new[i_sample][1] += delta_neg * value;
                }
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
    fn log_posterior(&self, beta: &[f64; 3], z: &Vec<[f64; 3]>) -> f64 {
        let mut log_likelihood = 0.0;
        for (i_sample, z_sample) in z.iter().enumerate() {
            let y_sample = self.data.y[i_sample] as f64;
            let value = z_sample[0] * beta[0] + z_sample[1] * beta[1] + z_sample[2] * beta[2];
            
            if y_sample == 1.0 {
                log_likelihood += log_logistic(value);
            } else {
                log_likelihood += log_logistic(-value);
            }
            
            if log_logistic(value).is_infinite() || log_logistic(-value).is_infinite() {
                println!("v={}, logistic(x)={}, logistic(-x)={}", value, logistic(value), logistic(-value));
                process::exit(1);
            }
        }
        
        let log_prior: f64 = -self.lambda * beta.iter().map(|v| v*v).sum::<f64>();
        log_likelihood + log_prior
    }


    // Computes the standard deviation for a specific model parameter, used to determine
    // appropriate proposal distribution width in MCMC sampling.
    // Parameters:
    // - beta: Current model coefficients
    // - z: Feature groups
    // - param_idx: Index of the parameter to compute sigma for (0=pos, 1=neg, 2=intercept)
    // Returns the standard deviation.
    fn compute_sigma_i(&self, beta: &[f64; 3], z: &Vec<[f64; 3]>, param_idx: usize) -> f64 {
        let mut cov_inv = 0.0;
        
        for (_i_sample, z_sample) in z.iter().enumerate() {
            let value = z_sample[0] * beta[0] + z_sample[1] * beta[1] + z_sample[2] * beta[2];
            cov_inv += logistic(value) * logistic(-value) * z_sample[param_idx].powf(2.0)
        }
        
        1.0 / (cov_inv + self.lambda).sqrt()
    }

}

// Helper structure for Brent optimization that implements the CostFunction trait.
// Used to minimize the negative log posterior probability for parameter optimization.
// Holds references to the Bayesian prediction model, parameter index, current beta values, and feature groups.
struct NegLogPostToMinimize<'a> {
    bp: &'a BayesPred,
    i: usize,
    beta: &'a [f64; 3],
    z: &'a Vec<[f64; 3]>,
}

// Implements the cost function for optimization.
// Returns the negative log posterior probability for the proposed parameter value.
impl CostFunction for NegLogPostToMinimize<'_> {
    type Param = f64;
    type Output = f64;

    // Implements the cost function for optimization.
    // Returns the negative log posterior probability for the proposed parameter value.
    // Parameters:
    // - beta_i: Proposed value for the parameter being optimized
    // Returns the negative log posterior (to be minimized).
    fn cost(&self, beta_i: &Self::Param) -> Result<Self::Output, argmin_Error> {
        let mut new_beta = self.beta.clone();
        new_beta[self.i] = *beta_i;
        Ok(- self.bp.log_posterior(&new_beta, &self.z))
    }
}


//-----------------------------------------------------------------------------
// Structure for MCMC tracing

// Structure for storing and processing results from MCMC sampling.
// Contains model statistics, feature probabilities, parameter estimates,
// and optional traces of the sampling process.
#[derive(Clone, Debug)]
pub struct MCMCAnalysisTrace {
    pub coefs: MCMC,
    pub feature_prob: HashMap<usize, (f64, f64, f64)>,  // idx -> (pos, neutre, neg)
    pub model_stats: HashMap<usize, (f64, f64)>,        // idx -> (moyenne, variance)
    pub beta_mean: [f64; 3],
    pub beta_var: [f64; 3],
    pub log_post_mean: f64,
    pub log_post_var: f64,
    pub log_post_trace: Vec<f64>,
    pub post_mean: f64,
    pub post_var: f64,
}

impl MCMCAnalysisTrace {
    // Creates a new MCMCAnalysisTrace instance to store MCMC sampling results.
    // Parameters:
    // - n_features: Total number of features in the model
    // - keep_trace: Whether to store detailed traces of sampling history
    // - feature_names: Optional vector of feature names for better interpretability
    pub fn new(feature_selection: &Vec<usize>) -> Self {
        let n_features = feature_selection.len();
        let mut feature_prob = HashMap::with_capacity(n_features);
        let mut model_stats = HashMap::with_capacity(n_features);
        for idx in feature_selection {
                // Probabilities are divided with finalize()
                feature_prob.insert(*idx, (0.0, 0.0, 0.0));
                model_stats.insert(*idx, (0.0, 0.0));
        }
        
        MCMCAnalysisTrace {
            coefs: MCMC::new(Vec::with_capacity(n_features),
             Betas::new(Vec::with_capacity(n_features), Vec::with_capacity(n_features), Vec::with_capacity(n_features))), 
            feature_prob,
            model_stats,
            beta_mean: [0.0, 0.0, 0.0],
            beta_var: [0.0, 0.0, 0.0],
            log_post_mean: 0.0,
            log_post_var: 0.0,
            log_post_trace: Vec::with_capacity(n_features),
            post_mean: 0.0,
            post_var: 0.0,
        }
    }

    // Updates the MCMC results with a new sample.
    // Accumulates statistics for parameters, feature probabilities, and model performance.
    // Parameters:
    // - beta: Current model coefficients
    // - individual: Current feature selection and coefficients
    // - log_post: Log posterior probability of the current sample
    pub fn update(&mut self, beta: &[f64; 3], features: &HashMap<usize,i8>, log_post: f64) {
        self.beta_mean = [
            self.beta_mean[0] + beta[0], 
            self.beta_mean[1] + beta[1], 
            self.beta_mean[2] + beta[2]
        ];
        
        self.beta_var = [
            self.beta_var[0] + beta[0].powf(2.0),
            self.beta_var[1] + beta[1].powf(2.0),
            self.beta_var[2] + beta[2].powf(2.0)
        ];
        
        for (feature_idx, coef) in features {
            let model_stat = self.model_stats.entry(*feature_idx).or_insert((0.0, 0.0));
            model_stat.0 += *coef as f64;
            model_stat.1 += (*coef as f64).powf(2.0);

            let feature_prob = self.feature_prob.entry(*feature_idx).or_insert((0.0, 0.0, 0.0));
            match *coef {
                1 => feature_prob.0 += 1.0, 
                -1 => feature_prob.2 += 1.0, 
                _ => feature_prob.1 += 1.0, 
            }
        }
        
        self.log_post_mean += log_post;
        self.log_post_var += log_post.powf(2.0);
        self.post_mean += log_post.exp();
        self.post_var += (2.0 * log_post).exp();

        self.coefs.betas_collection.a.push(beta[0]);
        self.coefs.betas_collection.b.push(beta[1]);
        self.coefs.betas_collection.c.push(beta[2]);
        self.coefs.features_collection.push(features.clone());

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
        let n_mean = ((self.coefs.features_collection[0].len() + 3) * (n_iter - n_burn)) as f64;
        
        // Beta
        self.beta_mean = self.beta_mean.iter()
            .map(|v| v / n_mean)
            .collect::<Vec<f64>>().try_into().unwrap();
        
        self.beta_var = self.beta_var.iter().zip(self.beta_mean.iter())
            .map(|(v, m)| (v - m.powf(2.0) * n_mean) / (n_mean - 1.0))
            .collect::<Vec<f64>>().try_into().unwrap();
        
        // Statistics
        for (_, stat) in &mut self.model_stats {
            stat.0 /= n_mean; // Normalisation de la moyenne
            stat.1 = (stat.1 - stat.0.powf(2.0) * n_mean) / (n_mean - 1.0); // Calcul de la variance
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
        let n_features = self.coefs.features_collection[0].len() as f64;
        self.post_mean.log10() - n_features * n_classes.log10()
    }

    pub fn get_ind(&self, param: &Param) -> Individual {

        Individual {
                features: HashMap::new(),
                auc: 0.0,
                specificity: 0.0,
                sensitivity: 0.0,
                accuracy: 0.0,
                threshold: 0.5,
                fit: self.get_log_evidence(3.0),
                k: self.coefs.features_collection[0].len(),
                epoch: 0,   
                language: MCMC_GENERIC_LANG,
                data_type: 0,
                hash: 0,
                epsilon: param.general.data_type_epsilon,
                parents: None,
                mcmc: Some(self.coefs.clone())
            }
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

    // pub fn export_to_files(&self, outdir: &str) -> std::io::Result<()> {
    //     use std::fs::File;
    //     use std::io::{BufWriter, Write};
    //     use std::fs;
        
    //     fs::create_dir_all(outdir)?;
        
    //     let p_mean_path = format!("{}/P_mean.tsv", outdir);
    //     let mut file = BufWriter::new(File::create(&p_mean_path)?);
        
    //     writeln!(file, "FEATURE\tPOS\tNUL\tNEG")?;
        
    //     let mut feature_names: Vec<_> = self.feature_prob.keys().collect();
    //     feature_names.sort();
        
    //     for name in feature_names {
    //         let (pos, neutral, neg) = self.feature_prob.get(name).unwrap();
    //         writeln!(file, "{}\t{}\t{}\t{}", name, pos, neutral, neg)?;
    //     }
        
    //     if let Some(beta_trace) = &self.beta_trace {
    //         let beta_trace_path = format!("{}/betas.tsv", outdir);
    //         let mut file = BufWriter::new(File::create(&beta_trace_path)?);
            
    //         writeln!(file, "a\tb\tc")?;
    //         for beta in beta_trace {
    //             writeln!(file, "{}\t{}\t{}", beta[0], beta[1], beta[2])?;
    //         }
    //     }
        
    //     if let Some(pop_trace) = &self.population_trace {
    //         let model_trace_path = format!("{}/models.tsv", outdir);
    //         let mut file = BufWriter::new(File::create(&model_trace_path)?);
            
    //         if let Some(names) = &self.feature_names {
    //             writeln!(file, "{}", names.join("\t"))?;
    //             for individual in &pop_trace.individuals {
    //                 let row: Vec<String> = names.iter()
    //                     .enumerate()
    //                     .map(|(idx, _)| {
    //                         individual.features.get(&idx)
    //                             .copied()
    //                             .unwrap_or(0)
    //                             .to_string()
    //                     })
    //                     .collect();
                    
    //                 writeln!(file, "{}", row.join("\t"))?;
    //             }
    //         }
    //     }
        
    //     let means_vars_path = format!("{}/means_vars.tsv", outdir);
    //     let mut file = BufWriter::new(File::create(&means_vars_path)?);
        
    //     writeln!(file, "parameter\tmean\tvariance")?;
        
    //     writeln!(file, "a\t{}\t{}", self.beta_mean[0], self.beta_var[0])?;
    //     writeln!(file, "b\t{}\t{}", self.beta_mean[1], self.beta_var[1])?;
    //     writeln!(file, "c\t{}\t{}", self.beta_mean[2], self.beta_var[2])?;
        
    //     if let Some(names) = &self.feature_names {
    //         for (name, (mean, variance)) in self.model_stats.iter() {
    //             writeln!(file, "{}\t{}\t{}", name, mean, variance)?;
    //         }
    //     }
        
    //     writeln!(file, "logPost\t{}\t{}", self.log_post_mean, self.log_post_var)?;
    //     writeln!(file, "Post\t{}\t{}", self.post_mean, self.post_var)?;
        
    //     Ok(())
    // }

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
pub fn run_mcmc_sbs(data: &Data, param: &Param, rng: &mut ChaCha8Rng, running: Arc<AtomicBool>) -> Vec<(u32, f64, f64, f64, usize)> {
    let time = Instant::now();
    let mut data_train = data.clone();
    let nmax = data_train.feature_selection.len() as u32;
    let mut post_mean = Vec::new();
    let mut feature_to_drop = Vec::new();

    for n in (param.mcmc.nmin..=nmax).rev() {
        debug!("n = {}, (#Features, #Samples) = ({}, {})", n, &data_train.feature_selection.len(), data_train.sample_len);
        let bp = BayesPred::new(&data_train, param.mcmc.lambda);

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
        .map(|((((n, pm), lpm), le), idx)| (n, pm, lpm, le, idx))
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

    let nvals = 3;
    
    // Initialize coefficients 
    let mut features : HashMap<usize,i8> = HashMap::with_capacity(bp.data.feature_selection.len());
    for feature_idx in &bp.data.feature_selection {
        let random_val = [1, 0, -1].choose(rng).unwrap();
        features.insert(*feature_idx, *random_val);
    }

    let mut beta = [1_f64, -1_f64, 1_f64];
    let nbeta: usize = 3;
    let mut z = bp.compute_feature_groups(&features);
    let mut log_post: f64 = 0.0;

    let mut res_mcmc = MCMCAnalysisTrace::new(&bp.data.feature_selection);

    debug!("Computing MCMC...");
    
    let solver = BrentOpt::new(-10_f64.powf(4_f64), 10_f64.powf(4_f64));
    for n in 0..param.mcmc.n_iter {
        if param.mcmc.n_iter>1000 && ((n as f64 == param.mcmc.n_iter as f64 * 0.25) || (n as f64 == param.mcmc.n_iter as f64 * 0.50) ||  (n as f64 == param.mcmc.n_iter as f64 * 0.75)) {
            debug!("MCMC : {}% iterations finished: a={:.4}, b={:.4}, c={:.4}", (n as f64 / param.mcmc.n_iter as f64) as f64 *100.0, beta[0], beta[1], beta[2]);
        }
        for i in 0..nbeta {
            let cost = NegLogPostToMinimize {
                bp: bp,
                i: i,
                beta: &beta,
                z: &z,
            };
        
            let res = Executor::new(cost, solver.clone())
                .configure(|state| state.max_iters(100))
                .run()
                .unwrap();

            beta[i] = res.state.param.unwrap();
            let scale_i = bp.compute_sigma_i(&beta, &z, i);

            if beta[i].abs() / scale_i > 10_f64 {
                beta[i] = random_normal(beta[i], scale_i, rng)
            } else {
                match i {
                    0_usize => beta[i] = truncnorm_pos(beta[i], scale_i, rng),
                    1_usize => beta[i] = truncnorm_neg(beta[i], scale_i, rng),
                    2_usize => beta[i] = random_normal(beta[i], scale_i, rng),
                    _ => beta[i] = 0.0,
                }
            }

            log_post = bp.log_posterior(&beta, &z);
            if n > param.mcmc.n_burn {
                res_mcmc.update(&beta, &features, log_post);
            }
        }

        for &feature_idx in &bp.data.feature_selection {
            let current_coef = &features.get(&feature_idx).copied().unwrap_or(0);
            let new_coef = (current_coef + [2, 3].choose(rng).unwrap()) % nvals - 1;
            let z_new = bp.update_feature_groups(&z, feature_idx, *current_coef, new_coef);
            let log_post_new = bp.log_posterior(&beta, &z_new);
            let diff_log_post = log_post_new - log_post;
            let u: f64 = rng.gen_range(0.0..1.0);
            
            if diff_log_post > 0.0 || u < diff_log_post.exp() {
                features.insert(feature_idx, new_coef);
                log_post = log_post_new;
                z = z_new;
            }
            
            if n > param.mcmc.n_burn {
                res_mcmc.update(&beta, &features, log_post);
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
pub fn get_best_mcmc_sbs(data: &Data, results: &[(u32, f64, f64, f64, usize)], param: &Param, mut rng: ChaCha8Rng) -> MCMCAnalysisTrace {
    let best_models = results.iter()
        .max_by(|(_, _, _, le1, _), (_, _, _, le2, _)| le1.partial_cmp(le2).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or_else(|| panic!("Couldn't find best models"));
    
    let n_best = best_models.0;
    
    let features_to_drop: Vec<usize> = results.iter()
        .filter(|(n, _, _, _, _)| *n > n_best)
        .map(|(_, _, _, _, idx)| idx.clone())
        .collect();
    
    let mut data_filtered = data.clone();
    for feature_idx in &features_to_drop {
        data_filtered.feature_selection.retain(|keep_idx| keep_idx != feature_idx);
    }
    
    info!("\nBest Models: n = {}, data dimensions = ({}, {})", 
        n_best, data_filtered.feature_selection.len(), data_filtered.sample_len);
    
    let now = Instant::now();
    let bp = BayesPred::new(&data_filtered, param.mcmc.lambda);
    let res: MCMCAnalysisTrace = compute_mcmc(&bp, param, &mut rng);

    // if param.mcmc.save_trace_outdir.len() > 0 {
    //     if let Err(e) = res.clone().export_to_files(&param.mcmc.save_trace_outdir.to_string()) {
    //         error!("Failed to export MCMC results: {}", e);
    //     } else {
    //         debug!("Best MCMC trace saved to {}", &param.mcmc.save_trace_outdir.to_string())
    //     }
    // }
    
    let elapsed = now.elapsed();
    info!("Elapsed: {:.2?}", elapsed);
    
    res
    
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

