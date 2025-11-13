#![allow(non_snake_case)]

pub mod bayesian_mcmc;
pub mod beam;
pub mod data;
pub mod utils;
pub mod individual;
pub mod param;
pub mod population;
pub mod ga;
mod cv;
pub mod gpu;
pub mod experiment;
pub mod voting;

use crate::experiment::{Experiment, ExperimentMetadata};
use data::Data;
use individual::Individual;
use population::Population;
use cv::CV;
use rand_chacha::ChaCha8Rng;
use chrono::Local;
use rand::prelude::*;
use param::Param;
use crate::ga::ga;
use crate::beam::{beam, keep_n_best_model_within_collection};
use crate::bayesian_mcmc::mcmc;

use log::{debug};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::process::Command;

pub fn run(param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let start = std::time::Instant::now();
    let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();

    // Load train data
    let mut data = Data::new();
    let _ = data.load_data(&param.data.X.to_string(), &param.data.y.to_string(), param.data.features_in_rows);
    data.set_classes(param.data.classes.clone());
    if param.data.inverse_classes { data.inverse_classes(); }
    cinfo!(param.general.display_colorful, "\x1b[2;97m{:?}\x1b[0m", data);  

    // Launch training
    let (collections, final_population, cv_folds_ids, meta) = if param.general.cv {
        run_cv_training(&data, &param, running)
    } else {
        let (collection, final_population, meta) = run_training(&mut data, &param, running);
        (vec![collection], final_population, None, meta)
    };

    // Loading test data
    let test_data = if !param.data.Xtest.is_empty() {
        debug!("Loading test data...");
        let mut td = Data::new();
        let _ = td.load_data(&param.data.Xtest, &param.data.ytest, param.data.features_in_rows);
        td.set_classes(param.data.classes.clone());
        if param.data.inverse_classes { td.inverse_classes(); }
        if param.general.algo == "mcmc" { td = td.remove_class(2); }
        td
    } else {
        Data::new()
    };

    // Build experiment
    let output = Command::new("git").args(&["rev-parse", "HEAD"]).output().unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap().chars().take(7).collect::<String>();
    let gpredomics_version = format!("{}#{}", env!("CARGO_PKG_VERSION"), git_hash);
    let exec_time = start.elapsed().as_secs_f64();

    let mut exp = Experiment {
        id: format!("{}_{}_{}", param.general.save_exp.split('.').next().unwrap(), param.general.algo, timestamp).to_string(),
        gpredomics_version: gpredomics_version,
        timestamp: timestamp.clone(),

        train_data: data,
        test_data: Some(test_data),

        final_population: Some(final_population),
        collections: collections,
        
        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids: cv_folds_ids,
        others: meta
    };

    if param.importance.compute_importance {
        cinfo!(param.general.display_colorful, "Computing feature importance...");
        let start_importance = std::time::Instant::now();
        
        exp.compute_importance();
        
        let importance_time = start_importance.elapsed().as_secs_f64();
        cinfo!(param.general.display_colorful, "Importance calculation completed in {:.2}s", importance_time);
    } else {
        cinfo!(param.general.display_colorful, "Skipping importance calculation (disabled in parameters)");
    }

    // Voting
    if param.voting.vote {
        exp.compute_voting();
    } else {
        cinfo!(param.general.display_colorful, "Voting stage ignored (disabled in parameters)");
    }

    exp
}

pub fn run_on_data(data: &mut Data, test_data: Option<&Data>, param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let start = std::time::Instant::now();
    let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();

    cinfo!(param.general.display_colorful, "\x1b[2;97m{:?}\x1b[0m", data);  

    // Launch training
    let (collections, final_population, cv_folds_ids, meta) = if param.general.cv {
        run_cv_training(&data, &param, running)
    } else {
        let (collection, final_population, meta) = run_training(data, &param, running);
        (vec![collection], final_population, None, meta)
    };

    // Build experiment
    let git_hash = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.chars().take(7).collect::<String>())
        .unwrap_or_else(|| "unknown".to_string());
    let gpredomics_version = format!("{}#{}", env!("CARGO_PKG_VERSION"), git_hash);
    let exec_time = start.elapsed().as_secs_f64();

    let mut exp = Experiment {
        id: format!("{}_{}_{}", param.general.save_exp.split('.').next().unwrap(), param.general.algo, timestamp).to_string(),
        gpredomics_version: gpredomics_version,
        timestamp: timestamp.clone(),

        train_data: data.clone(),
        test_data: test_data.cloned(),

        final_population: Some(final_population),
        collections: collections,
        
        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids: cv_folds_ids,
        others: meta
    };

    if param.importance.compute_importance && exp.parameters.general.algo != "mcmc"  {
        cinfo!(param.general.display_colorful, "Computing feature importance...");
        
        let start_importance = std::time::Instant::now();
        exp.compute_importance();
        let importance_time = start_importance.elapsed().as_secs_f64();
        
        cinfo!(param.general.display_colorful, "Importance calculation completed in {:.2}s", importance_time);
    } else {
        cinfo!(param.general.display_colorful, "Skipping importance calculation (disabled in parameters)");
    }

    // Voting
    if param.voting.vote && exp.parameters.general.algo != "mcmc" {
        exp.compute_voting();
    } else {
        cinfo!(param.general.display_colorful, "Voting stage ignored (disabled in parameters)");
    }

    exp
}

pub fn run_training(data: &mut Data, param: &Param, running: Arc<AtomicBool>) -> (Vec<Population>, Population, Option<ExperimentMetadata>) {
    let meta;
    let collection; 
    let final_population;
    match param.general.algo.as_str() {
        "ga" => {
            cinfo!(param.general.display_colorful, "Training using Genetic Algorithm\n-----------------------------------------------------");
            (collection, meta) = (ga(data, &mut None, &param, running), None);
            final_population = collection[collection.len()-1].clone()
        },
        "beam" => {
            cinfo!(param.general.display_colorful, "Training using Beam Search\n-----------------------------------------------------");
            (collection, meta) = (beam(data, &mut None, &param, running), None);
            final_population = keep_n_best_model_within_collection(&collection, param.beam.max_nb_of_models as usize);
        },
        "mcmc" => {
            cinfo!(param.general.display_colorful, "Training using MCMC\n-----------------------------------------------------");
            (collection, meta) = mcmc(data, param, running);
            final_population = Population::new();
        },
        _ => {
            panic!("Unknown algorithm: {}", param.general.algo);
        }
    };
    (collection, final_population, meta)
}

pub fn run_cv_training(data: &Data, param: &Param, running: Arc<AtomicBool>) -> (Vec<Vec<Population>>, Population, Option<Vec<(Vec<String>, Vec<String>)>>, Option<ExperimentMetadata>) {
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let mut folds = CV::new(&data, param.cv.outer_folds, &mut rng);
    let cv_folds_ids = Some(folds.get_ids());

    let collections; 
    let mut final_population; 

    let mut run_param = param.clone();
    run_param.general.gpu = false;

    match param.general.algo.as_str() {
        "ga" => folds.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| { 
            ga(d, &mut None, p, r)
            }, &run_param, running),
        "beam" => folds.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| { 
            beam(d, &mut None, p, r)
        }, &run_param, running),
        _ => {
            panic!("CV mode is only available for GA and Beam Search currently.");
        }
    };

    final_population = folds.get_fbm(&run_param);
    final_population.fit(&data, &mut None, &None, &None, &run_param);
    final_population = final_population.sort();

    collections = folds.fold_collections;
    (collections, final_population, cv_folds_ids, None)
}
