//! Gpredomics: Interpretable machine learning for omics data
//!
//! # Overview
//!
//! Gpredomics is a Rust library designed for building interpretable predictive models using genetic algorithms (GA) and beam search.
//! It provides tools for data handling, model training, feature importance computation, and cross-validation.
//! This library is a reimplementation and extension of the original R Predomics package, optimized for performance and scalability.
//!
//! # Modules
//!
//! ## Core Modules
//! * `utils`: Provides utility functions and macros.
//! * `gpu`: Contains GPU acceleration utilities.
//!
//! ## Data and Parameter Management
//! * `experiment`- Defines the Experiment structure to encapsulate training results and metadata.
//! * `param`- Manages parameter configurations.
//! * `data`- Handles data loading, preprocessing, and management.
//! * `cv`- Provides cross-validation utilities.
//!
//! ## Model Components
//! * `individual`- Defines the Individual structure representing candidate solutions.
//! * `population`- Defines the Population structure representing collections of individuals.
//! * `voting`- Implements voting mechanisms for ensemble predictions.
//!
//! ## Algorithms
//! * `ga` - Implements genetic algorithm functionalities (equivalent to terga2).
//! * `beam`- Contains functions for beam search algorithms (equivalent to terbeam).
//! * `bayesian_mcmc`- Implements Bayesian MCMC algorithms for model training.
//!
//! # Notes
//!
//! Gpredomics is also accessible via the GpredomicsR package, which provides R bindings for this Rust library.
//! This package also provides additional functionalities for easier integration with R workflows and graphical outputs.
//!
//! # References
//! Prifti E, Chevaleyre Y, Hanczar B, Belda E, Danchin A, Clément K, Zucker J (2020).
//! Interpretable and accurate prediction models for metagenomics data, GigaScience, 2020.
//! doi:10.1093/gigascience/giaa010
//!
#![allow(non_snake_case)]
#![deny(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

/// Provides GPU acceleration utilities.
pub mod gpu;
/// Module declarations
/// Provides utility functions and macros.
pub mod utils;

/// Provides cross-validation utilities.
pub mod cv;
/// Handles data loading, preprocessing, and management.
pub mod data;
/// Defines the Experiment structure to encapsulate training results and metadata.
pub mod experiment;
/// Manages parameter configurations.
pub mod param;

/// Defines the Individual structure representing candidate solutions.
pub mod individual;
/// Defines the Population structure representing collections of individuals.
pub mod population;
/// Implements voting mechanisms for ensemble predictions.
pub mod voting;

/// Implements Bayesian MCMC algorithms for model training.
pub mod bayesian_mcmc;
/// Contains functions for beam search algorithms.
pub mod beam;
/// Implements genetic algorithm functionalities.
pub mod ga;

use crate::bayesian_mcmc::mcmc;
use crate::beam::{beam, keep_n_best_model_within_collection};
use crate::experiment::{Experiment, ExperimentMetadata};
use crate::ga::ga;
use chrono::Local;
use cv::CV;
use data::Data;
use individual::Individual;
use param::Param;
use population::Population;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use log::{debug, error, warn};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Executes a complete training experiment from parameter configuration.
///
/// # Arguments
///
/// * `param` - Reference to parameter configuration
/// * `running` - Atomic flag to control execution state
///
/// # Returns
///
/// Complete `Experiment` with results and metadata
///
/// # Description
///
/// * Loads training data from file paths specified in parameters
/// * Optionally splits data into train/test sets
/// * Performs training using specified algorithm (GA, Beam, or MCMC)
/// * Computes feature importance if enabled
/// * Executes voting procedures if configured
pub fn run(param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let start = std::time::Instant::now();
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();

    // Load train data
    let mut data = Data::new();
    let _ = data.load_data(
        &param.data.X.to_string(),
        &param.data.y.to_string(),
        param.data.features_in_rows,
    );
    data.set_classes(param.data.classes.clone());
    if param.data.inverse_classes {
        data.inverse_classes();
    }
    cinfo!(
        param.general.display_colorful,
        "\x1b[2;97m{:?}\x1b[0m",
        data
    );

    if !param.data.feature_annotations.is_empty() {
        match data.load_feature_annotation(&param.data.feature_annotations) {
            Ok(fa) => data.feature_annotations = Some(fa),
            Err(e) => warn!(
                "Could not load feature annotations '{}': {}",
                param.data.feature_annotations, e
            ),
        }
    }

    if !param.data.sample_annotations.is_empty() {
        match data.load_sample_annotation(&param.data.sample_annotations) {
            Ok(sa) => data.sample_annotations = Some(sa),
            Err(e) => warn!(
                "Could not load sample annotations '{}': {}",
                param.data.sample_annotations, e
            ),
        }
    }

    let mut test_data: Option<Data> = None;
    if (param.data.Xtest.is_empty() && param.data.ytest.is_empty())
        && param.data.holdout_ratio > 0.0
    {
        cinfo!(
            param.general.display_colorful,
            "Performing train/test split with holdout ratio of {}...",
            param.data.holdout_ratio
        );
        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(param.general.seed);
        let stratify_by: Option<&str> = if data.sample_annotations.is_some() {
            Some(param.cv.stratify_by.as_str())
        } else {
            None
        };
        let mut _holdout: Data = Data::new();
        (data, _holdout) = data.train_test_split(param.data.holdout_ratio, &mut rng, stratify_by);
        test_data = Some(_holdout);
        cinfo!(
            false,
            "Train/test split: {} train samples, {} test samples",
            data.sample_len,
            test_data.as_ref().unwrap().sample_len
        );
    } else if (param.data.Xtest.is_empty() && param.data.ytest.is_empty())
        && param.data.holdout_ratio == 0.0
    {
        warn!("No test data (Xtest/ytest) provided and holdout_ratio is set to 0.");
    }

    // Launch training
    let (collections, final_population, cv_folds_ids, meta) = if param.general.cv {
        run_cv_training(&data, &param, running)
    } else {
        let (collection, final_population, meta) =
            run_training(&mut data, &mut None, &param, running);
        (vec![collection], final_population, None, meta)
    };

    // Loading test data
    if test_data.is_none() && (!param.data.Xtest.is_empty() && !param.data.ytest.is_empty()) {
        debug!("Loading test data...");
        let mut td = Data::new();
        let _ = td.load_data(
            &param.data.Xtest,
            &param.data.ytest,
            param.data.features_in_rows,
        );
        td.set_classes(param.data.classes.clone());
        if param.data.inverse_classes {
            td.inverse_classes();
        }
        if param.general.algo == "mcmc" {
            td = td.remove_class(2);
        }
        if data.check_compatibility(&td) {
            test_data = Some(td);
        } else {
            warn!("Test data is not compatible with training data: classes or features differ. Ignoring test data.");
        }
    }

    // Build experiment
    let git_hash = option_env!("GPREDOMICS_GIT_SHA").unwrap_or("unknown");
    let gpredomics_version = format!("{}#{}", env!("CARGO_PKG_VERSION"), git_hash);
    let exec_time = start.elapsed().as_secs_f64();

    let mut exp = Experiment {
        id: format!(
            "{}_{}_{}",
            param.general.save_exp.split('.').next().unwrap(),
            param.general.algo,
            timestamp
        )
        .to_string(),
        gpredomics_version: gpredomics_version,
        timestamp: timestamp.clone(),

        train_data: data,
        test_data: test_data,

        final_population: Some(final_population),
        collections: collections,

        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids: cv_folds_ids,
        others: meta,
    };

    if param.importance.compute_importance {
        cinfo!(
            param.general.display_colorful,
            "Computing feature importance..."
        );
        let start_importance = std::time::Instant::now();

        exp.compute_importance();

        let importance_time = start_importance.elapsed().as_secs_f64();
        cinfo!(
            param.general.display_colorful,
            "Importance calculation completed in {:.2}s",
            importance_time
        );
    } else {
        cinfo!(
            param.general.display_colorful,
            "Skipping importance calculation (disabled in parameters)"
        );
    }

    // Voting
    if param.voting.vote {
        exp.compute_voting();
    } else {
        cinfo!(
            param.general.display_colorful,
            "Voting stage ignored (disabled in parameters)"
        );
    }

    exp
}

/// Executes training on pre-loaded data structures.
///
/// # Arguments
///
/// * `data` - Training dataset
/// * `test_data` - Optional test dataset
/// * `param` - Parameter configuration
/// * `running` - Execution control flag
///
/// # Returns
///
/// `Experiment` with training results
///
/// # Notes
///
/// This function is used when called from GpredomicsR package.
pub fn run_on_data(
    mut data: Data,
    mut test_data: Option<Data>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> Experiment {
    let start = std::time::Instant::now();
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();

    cinfo!(
        param.general.display_colorful,
        "\x1b[2;97m{:?}\x1b[0m",
        data
    );

    if test_data.is_none() && param.data.holdout_ratio > 0.0 {
        cinfo!(
            param.general.display_colorful,
            "Performing train/test split with holdout ratio of {}...",
            param.data.holdout_ratio
        );
        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(param.general.seed);

        let stratify_by: Option<&str> = if data.sample_annotations.is_some() {
            Some(param.cv.stratify_by.as_str())
        } else {
            None
        };

        let (train, holdout) =
            data.train_test_split(param.data.holdout_ratio, &mut rng, stratify_by);
        data = train;
        test_data = Some(holdout);

        cinfo!(
            false,
            "Train/test split: {} train samples, {} test samples",
            data.sample_len,
            test_data.as_ref().unwrap().sample_len
        );
    } else if param.data.Xtest.is_empty()
        && param.data.ytest.is_empty()
        && param.data.holdout_ratio == 0.0
    {
        warn!("No test data (Xtest/ytest) provided and holdout_ratio is set to 0.");
    }

    // Launch training
    let (collections, final_population, cv_folds_ids, meta) = if param.general.cv {
        run_cv_training(&data, &param, running)
    } else {
        let (collection, final_population, meta) =
            run_training(&mut data, &mut None, &param, running);
        (vec![collection], final_population, None, meta)
    };

    // Control test data
    if test_data.is_some() && !data.check_compatibility(test_data.as_ref().unwrap()) {
        warn!("Test data is not compatible with training data: classes or features differ. Ignoring test data.");
        test_data = None;
    }

    // Build experiment
    let git_hash = option_env!("GPREDOMICS_GIT_SHA").unwrap_or("unknown");
    let gpredomics_version = format!("{}#{}", env!("CARGO_PKG_VERSION"), git_hash);
    let exec_time = start.elapsed().as_secs_f64();

    let mut exp = Experiment {
        id: format!(
            "{}_{}_{}",
            param.general.save_exp.split('.').next().unwrap(),
            param.general.algo,
            timestamp
        )
        .to_string(),
        gpredomics_version: gpredomics_version,
        timestamp: timestamp.clone(),

        train_data: data,
        test_data: test_data,

        final_population: Some(final_population),
        collections: collections,

        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids: cv_folds_ids,
        others: meta,
    };

    if param.importance.compute_importance && exp.parameters.general.algo != "mcmc" {
        cinfo!(
            param.general.display_colorful,
            "Computing feature importance..."
        );

        let start_importance = std::time::Instant::now();
        exp.compute_importance();
        let importance_time = start_importance.elapsed().as_secs_f64();

        cinfo!(
            param.general.display_colorful,
            "Importance calculation completed in {:.2}s",
            importance_time
        );
    } else {
        cinfo!(
            param.general.display_colorful,
            "Skipping importance calculation (disabled in parameters)"
        );
    }

    // Voting
    if param.voting.vote && exp.parameters.general.algo != "mcmc" {
        exp.compute_voting();
    } else {
        cinfo!(
            param.general.display_colorful,
            "Voting stage ignored (disabled in parameters)"
        );
    }

    exp
}

/// Continues training from an existing population on pre-loaded data structures (designed for R integration).
///
/// # Arguments
/// * `initial_pop` - Optional initial population to continue from
/// * `data` - Training data
/// * `test_data` - Optional test data
/// * `param` - Configuration parameters
/// * `running` - Control flag for execution
///
/// # Returns
/// `Experiment` with results
///
/// # Restrictions
/// Currently only supports genetic algorithm (GA) mode.
///
/// # Notes
/// This function is primarily used when called from GpredomicsR package.
pub fn run_pop_and_data(
    initial_pop: &mut Option<Population>,
    mut data: Data,
    mut test_data: Option<Data>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> Experiment {
    let start = std::time::Instant::now();
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();

    if param.general.algo != "ga" {
        error!("Training from an initial population is currently only supported for the genetic algorithm (GA).");
        panic!("Training from an initial population is currently only supported for the genetic algorithm (GA).");
    }

    cinfo!(
        param.general.display_colorful,
        "\x1b[2;97m{:?}\x1b[0m",
        data
    );

    if test_data.is_none() && param.data.holdout_ratio > 0.0 {
        cinfo!(
            param.general.display_colorful,
            "Performing train/test split with holdout ratio of {}...",
            param.data.holdout_ratio
        );
        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(param.general.seed);

        let stratify_by: Option<&str> = if data.sample_annotations.is_some() {
            Some(param.cv.stratify_by.as_str())
        } else {
            None
        };

        let (train, holdout) =
            data.train_test_split(param.data.holdout_ratio, &mut rng, stratify_by);
        data = train;
        test_data = Some(holdout);

        cinfo!(
            false,
            "Train/test split: {} train samples, {} test samples",
            data.sample_len,
            test_data.as_ref().unwrap().sample_len
        );
    } else if param.data.Xtest.is_empty()
        && param.data.ytest.is_empty()
        && param.data.holdout_ratio == 0.0
    {
        warn!("No test data (Xtest/ytest) provided and holdout_ratio is set to 0.");
    }

    // Launch training
    let (collections, final_population, cv_folds_ids, meta) = if param.general.cv {
        panic!("Cross-validation with initial population is not currently supported.");
    } else {
        let (collection, final_population, meta) =
            run_training(&mut data, initial_pop, &param, running);
        (vec![collection], final_population, None, meta)
    };

    // Control test data
    if test_data.is_some() && !data.check_compatibility(test_data.as_ref().unwrap()) {
        warn!("Test data is not compatible with training data: classes or features differ. Ignoring test data.");
        test_data = None;
    }

    // Build experiment
    let git_hash = option_env!("GPREDOMICS_GIT_SHA").unwrap_or("unknown");
    let gpredomics_version = format!("{}#{}", env!("CARGO_PKG_VERSION"), git_hash);
    let exec_time = start.elapsed().as_secs_f64();

    let mut exp = Experiment {
        id: format!(
            "{}_{}_{}",
            param.general.save_exp.split('.').next().unwrap(),
            param.general.algo,
            timestamp
        )
        .to_string(),
        gpredomics_version: gpredomics_version,
        timestamp: timestamp.clone(),

        train_data: data,
        test_data: test_data,

        final_population: Some(final_population),
        collections: collections,

        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids: cv_folds_ids,
        others: meta,
    };

    if param.importance.compute_importance && exp.parameters.general.algo != "mcmc" {
        cinfo!(
            param.general.display_colorful,
            "Computing feature importance..."
        );

        let start_importance = std::time::Instant::now();
        exp.compute_importance();
        let importance_time = start_importance.elapsed().as_secs_f64();

        cinfo!(
            param.general.display_colorful,
            "Importance calculation completed in {:.2}s",
            importance_time
        );
    } else {
        cinfo!(
            param.general.display_colorful,
            "Skipping importance calculation (disabled in parameters)"
        );
    }

    // Voting
    if param.voting.vote && exp.parameters.general.algo != "mcmc" {
        exp.compute_voting();
    } else {
        cinfo!(
            param.general.display_colorful,
            "Voting stage ignored (disabled in parameters)"
        );
    }

    exp
}

/// Core training function that dispatches to appropriate algorithm.
///
/// # Arguments
///
/// * `data` - Mutable reference to training data
/// * `initial_pop` - Optional starting population
/// * `param` - Algorithm parameters
/// * `running` - Execution control
///
/// # Returns
///
/// * Collection of populations across iterations
/// * Final population
/// * Optional experiment metadata
pub fn run_training(
    data: &mut Data,
    initial_pop: &mut Option<Population>,
    param: &Param,
    running: Arc<AtomicBool>,
) -> (Vec<Population>, Population, Option<ExperimentMetadata>) {
    let meta;
    let collection;
    let final_population;
    match param.general.algo.as_str() {
        "ga" => {
            cinfo!(param.general.display_colorful, "Training using Genetic Algorithm\n-----------------------------------------------------");
            (collection, meta) = (ga(data, &mut None, initial_pop, &param, running), None);
            final_population = collection[collection.len() - 1].clone()
        }
        "beam" => {
            cinfo!(
                param.general.display_colorful,
                "Training using Beam Search\n-----------------------------------------------------"
            );
            (collection, meta) = (beam(data, &mut None, initial_pop, &param, running), None);
            final_population = keep_n_best_model_within_collection(
                &collection,
                param.beam.max_nb_of_models as usize,
            );
        }
        "mcmc" => {
            cinfo!(
                param.general.display_colorful,
                "Training using MCMC\n-----------------------------------------------------"
            );
            (collection, meta) = mcmc(data, param, running);
            final_population = Population::new();
        }
        _ => {
            panic!("Unknown algorithm: {}", param.general.algo);
        }
    };
    (collection, final_population, meta)
}

/// Performs cross-validated training.
///
/// # Arguments
/// * `data` - Training dataset
/// * `param` - Configuration with CV settings
/// * `running` - Control flag
///
/// # Returns
/// A tuple with:
/// * Collections from all CV folds
/// * Aggregated final population
/// * Fold sample IDs
/// * Optional metadata
///
/// **Note:** CV mode currently supports GA and Beam Search only.
pub fn run_cv_training(
    data: &Data,
    param: &Param,
    running: Arc<AtomicBool>,
) -> (
    Vec<Vec<Population>>,
    Population,
    Option<Vec<(Vec<String>, Vec<String>)>>,
    Option<ExperimentMetadata>,
) {
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    let mut folds = CV::new_from_param(&data, param, &mut rng, param.cv.outer_folds);
    let cv_folds_ids = Some(folds.get_ids());

    let collections;
    let mut final_population;

    let mut run_param = param.clone();
    run_param.general.gpu = false;

    match param.general.algo.as_str() {
        "ga" => folds.pass(
            |d: &mut Data, p: &Param, r: Arc<AtomicBool>| ga(d, &mut None, &mut None, p, r),
            &run_param,
            running,
        ),
        "beam" => folds.pass(
            |d: &mut Data, p: &Param, r: Arc<AtomicBool>| beam(d, &mut None, &mut None, p, r),
            &run_param,
            running,
        ),
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
