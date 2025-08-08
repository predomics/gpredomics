#![allow(non_snake_case)]

pub mod bayesian_mcmc;
pub mod beam;
pub mod data;
mod utils;
pub mod individual;
pub mod param;
pub mod population;
mod ga;
mod cv;
pub mod gpu;
pub mod experiment;

use crate::experiment::{Experiment, ExperimentMetadata};
use data::Data;
use individual::Individual;
use population::Population;
use cv::CV;
use rand_chacha::ChaCha8Rng;
use chrono::Local;
use rand::prelude::*;
use param::Param;

use log::{debug, info, warn, error};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// a very basic use
// pub fn basic_test(param: &Param) {
//     info!("                          BASIC TEST\n-----------------------------------------------------");
//     // define some data
//     let mut data = Data::new();
//     data.X.insert((0,0), 0.1);
//     data.X.insert((0,1), 0.2);
//     data.X.insert((0,2), 0.3);
//     data.X.insert((2,0), 0.9);
//     data.X.insert((2,1), 0.8);
//     data.X.insert((2,2), 0.7);    
//     data.feature_len = 3;
//     data.sample_len = 3;    data.samples = string_vec! ["a","b","c"];
//     data.features = string_vec! ["msp1","msp2","msp3"];
//     data.y = vec! [0,1,1];
//     data.feature_len = 3;
//     data.sample_len = 3;
//     info!("{:?}", data);

//     // create a model
//     let mut my_individual = Individual::new();
//     my_individual.features.insert(0, 1);
//     my_individual.features.insert(2, -1);
//     my_individual.compute_hash();
//     info!("my individual: {:?}",my_individual.features);
//     info!("my individual hash: {}",my_individual.hash);
//     info!("my individual evaluation: {:?}",my_individual.evaluate(&data));
//     // shoud display 1.0 (the AUC is 1.0)
//     info!("my individual AUC: {:?}",my_individual.compute_auc(&data));
    
//     let mut my_individual2 = Individual::new();
//     my_individual2.features.insert(0, 1);
//     my_individual2.features.insert(1, -1);
//     my_individual2.compute_hash();
//     info!("my individual2 {:?}",my_individual2.features);
//     info!("my individual2 hash: {}",my_individual2.hash);
//     info!("my individual2 evaluation: {:?}",my_individual2.evaluate(&data));
//     // shoud display 1.0 (the AUC is 1.0)
//     info!("my individual2 AUC: {:?}",my_individual2.compute_auc(&data));


//     let mut data2=Data::new();
//     let _ = data2.load_data(param.data.X.as_str(),param.data.y.as_str());
//     let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
//     let parent1 = Individual::random(&data2, &mut rng);
//     let parent2 = Individual::random(&data2, &mut rng);
//     let mut parents=population::Population::new();
//     parents.individuals.push(parent1);
//     parents.individuals.push(parent2);

//     let children_number = param.ga.population_size as usize-parents.individuals.len();
//     let mut children = ga::cross_over(&parents, data2.feature_len, children_number, &mut rng);
//     for (i,individual) in parents.individuals.iter().enumerate() { info!("Parent #{}: {:?}",i,individual); }
//     for (i,individual) in children.individuals.iter().enumerate() { info!("Child #{}: {:?}",i,individual); }
//     let feature_selection:Vec<usize> = (0..data2.feature_len).collect();
//     ga::mutate(&mut children, param, &feature_selection, &mut rng);
//     children.compute_hash();
//     let clone_number = children.remove_clone();
//     if clone_number>0 { warn!("There were {} clone(s)",clone_number); }
//     for (i,individual) in children.individuals.iter().enumerate() { info!("Mutated Child #{}: {:?}",i,individual); }    

// }

// /// a more elaborate use with random models
// pub fn random_run(param: &Param) {
//     info!("                          RANDOM TEST\n-----------------------------------------------------");
//     // use some data
//     let mut data = Data::new();
//     let _ = data.load_data(param.data.X.as_str(),param.data.y.as_str());
//     let mut rng = ChaCha8Rng::seed_from_u64(42);

//     let mut auc_max = 0.0;
//     let mut best_individual: Individual = Individual::new();
//     for _ in 0..1000 {
//         let mut my_individual = Individual::random(&data, &mut rng);


//         let auc = my_individual.compute_auc(&data);
//         if auc>auc_max {auc_max=auc;best_individual=my_individual;}
//     }
//     warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
// }

// /// a more elaborate use with random models
// pub fn gpu_random_run(param: &Param) {
//     info!("                          GPU RANDOM TEST\n-----------------------------------------------------");
//     // use some data
//     let mut data = Data::new();
//     let _ = data.load_data(param.data.X.as_str(),param.data.y.as_str());
//     info!("Selecting features...");
//     data.select_features(param);
//     let mut rng = ChaCha8Rng::seed_from_u64(42);

//     //let mut auc_max = 0.0;
//     //let mut best_individual: Individual = Individual::new();
//     let nb_individuals: usize = 1000;


//     let assay = gpu::GpuAssay::new(&data.X, &data.feature_selection, data.sample_len, nb_individuals as usize, &param.gpu);

//     let mut individuals:Vec<Individual> = (0..nb_individuals).map(|_i| {Individual::random(&data, &mut rng)}).collect();
//     individuals = individuals.into_iter()
//         .map(|i| {
//             // we filter random features for selected features, not efficient but we do not care
//             let mut new = Individual::new();
//             new.features = i.features.into_iter()
//                     .filter(|(i,_f)| {data.feature_selection.contains(i)})
//                     .collect();
//             new.data_type = *vec![individual::RAW_TYPE, individual::LOG_TYPE, individual::PREVALENCE_TYPE].choose(&mut rng).unwrap();
//             new.language = *vec![individual::TERNARY_LANG, individual::RATIO_LANG].choose(&mut rng).unwrap();
//             new
//         }).collect();
        

//     for _ in 0..1000 {
        
//         //println!("First individual {:?}", individuals[0]);

//         let scores = assay.compute_scores(&individuals, param.general.data_type_epsilon as f32);

//         println!("First scores {:?}", &scores[0..4]);
//         //println!("Last scores {:?}", &scores[scores.len()-4..scores.len()]);

//     }

//     //let auc = my_individual.compute_auc(&data);
//     //warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
// }

pub fn run_ga(param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let start = std::time::Instant::now();

    info!("Genetic algorithm\n-----------------------------------------------------");

    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let mut run_param = param.clone();
    
    // Load train data
    let mut data = Data::new();
    let _ = data.load_data(&run_param.data.X.to_string(), &run_param.data.y.to_string());
    data.set_classes(run_param.data.classes.clone());
    if run_param.data.inverse_classes { data.inverse_classes(); }
    info!("\x1b[2;97m{:?}\x1b[0m", data);  

    let collections: Vec<Vec<Population>> ;
    let mut final_population: Population;
    let mut cv_folds_ids: Option<Vec<(Vec<std::string::String>, Vec<std::string::String>)>> = None;

    // Computing Experiment
    if param.general.cv {
        let mut folds = CV::new(&data, run_param.cv.outer_folds, &mut rng);
        cv_folds_ids = Some(folds.get_ids());

        run_param.general.gpu = false;
        if run_param.general.thread_number >= param.cv.outer_folds {
            run_param.general.thread_number = param.general.thread_number/param.cv.outer_folds; 
            info!("\x1b[1;93mCross-validation parallelization using {} threads per fold\x1b[0m", run_param.general.thread_number);
        } else {
            run_param.general.thread_number = 1; 
        }

        folds.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| {
            ga::ga(d, &mut None, p, r)  
        }, &run_param, run_param.general.thread_number, running);

        final_population = folds.clone().get_fbm(&param);
        ga::fit_fn(&mut final_population, &data, &mut None, &None, &None, &run_param);
        final_population = final_population.sort();

        collections = folds.fold_collections;

        info!("\x1b[1;93mDisplaying Family of best models across folds\x1b[0m");
    } else {
        collections = vec![ga::ga(&mut data, &mut None, &param, running)];
        final_population = collections[0][collections[0].len()-1].clone();
    }

    // Loading test data
    let test_data = if !param.data.Xtest.is_empty() {
        debug!("Loading test data...");
        let mut td = Data::new();
        let _ = td.load_data(&param.data.Xtest, &param.data.ytest);
        td.set_classes(param.data.classes.clone());
        if param.data.inverse_classes { td.inverse_classes(); }
        td
    } else {
        Data::new()
    };

    info!("{}", final_population.display(&data, Some(&test_data), param));

    let exec_time = start.elapsed().as_secs_f64();
    Experiment {
        id: format!("{}_ga_{}", param.general.save_exp.split('.').next().unwrap(), timestamp).to_string(),
        gpredomics_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: timestamp,

        train_data: data,
        test_data: Some(test_data),

        final_population: Some(final_population),
        collections: collections,
        
        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids: cv_folds_ids,
        others: None
    }

}

// pub fn run_mixed(param: &Param, running: Arc<AtomicBool>) -> Experiment {
//     let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
//     let start = std::time::Instant::now();

//     info!("Genetic algorithm\n-----------------------------------------------------");

//     let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(param.general.seed);
//     let mut run_param = param.clone();
    
//     // Load train data
//     let mut data = Data::new();
//     let _ = data.load_data(&run_param.data.X.to_string(), &run_param.data.y.to_string());
//     data.set_classes(run_param.data.classes.clone());
//     if run_param.data.inverse_classes { data.inverse_classes(); }
//     info!("\x1b[2;97m{:?}\x1b[0m", data);  

//     let collections: Vec<Vec<Population>> ;
//     let mut final_population: Population;
//     let mut cv_folds_ids: Option<Vec<(Vec<std::string::String>, Vec<std::string::String>)>> = None;

//     // Computing Experiment
//     if param.general.cv {
//         let mut folds = CV::new(&data, run_param.cv.outer_folds, &mut rng);
//         cv_folds_ids = Some(folds.get_ids());

//         run_param.general.gpu = false;
//         if run_param.general.thread_number >= param.cv.outer_folds {
//             run_param.general.thread_number = param.general.thread_number/param.cv.outer_folds; 
//             info!("\x1b[1;93mCross-validation parallelization using {} threads per fold\x1b[0m", run_param.general.thread_number);
//         } else {
//             run_param.general.thread_number = 1; 
//         }

//         folds.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| {
//             ga::ga(d, &mut None, p, r)  
//         }, &run_param, run_param.general.thread_number, running);

//         final_population = folds.clone().get_fbm(&param);
//         ga::fit_fn(&mut final_population, &data, &mut None, &None, &None, &run_param);
//         final_population = final_population.sort();

//         collections = folds.fold_collections;

//         info!("\x1b[1;93mDisplaying Family of best models across folds\x1b[0m");
//     } else {
//         paramTinyModels = param.clone()
//         paramHugeModels = param.clone()
//         paramTinyModels.general.k_penalty = 0.005
//         paramHugeModels.general.k_penalty = 0.0001

//         tinyPop = ga::ga(&mut data, &mut None, &paramTinyModels, running)
//         pop = tinyPop.individuals.extend(ga::ga(&mut data, &mut None, &paramHugeModels, running).individuals)
//         ga::fit_fn(&mut pop, &data, &mut None, &None, &None, &run_param);

//         collections = vec![];
//         final_population = collections[0][collections.len()-1].clone();
//     }

//     // Loading test data
//     let test_data = if !param.data.Xtest.is_empty() {
//         debug!("Loading test data...");
//         let mut td = Data::new();
//         let _ = td.load_data(&param.data.Xtest, &param.data.ytest);
//         td.set_classes(param.data.classes.clone());
//         if param.data.inverse_classes { td.inverse_classes(); }
//         td
//     } else {
//         Data::new()
//     };

//     info!("{}", final_population.display(&data, Some(&test_data), param));

//     let exec_time = start.elapsed().as_secs_f64();
//     Experiment {
//         id: format!("{}_ga_{}", param.general.save_exp.split('.').next().unwrap(), timestamp).to_string(),
//         gpredomics_version: env!("CARGO_PKG_VERSION").to_string(),
//         timestamp: timestamp,

//         train_data: data,
//         test_data: Some(test_data),

//         final_population: Some(final_population),
//         collections: collections,
        
//         importance_collection: None,
//         execution_time: exec_time,
//         parameters: param.clone(),

//         cv_folds_ids: cv_folds_ids,
//         others: None
//     }

//}

pub fn run_beam(param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let start = std::time::Instant::now();

    info!("Beam algorithm\n-----------------------------------------------------");

    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let mut run_param = param.clone();
    
    // Load train data
    let mut data = Data::new();
    let _ = data.load_data(&run_param.data.X.to_string(), &run_param.data.y.to_string());
    data.set_classes(run_param.data.classes.clone());
    if run_param.data.inverse_classes { data.inverse_classes(); }
    info!("\x1b[2;97m{:?}\x1b[0m", data);  

    let collections: Vec<Vec<Population>> ;
    let mut final_population: Population;
    let mut cv_folds_ids: Option<Vec<(Vec<std::string::String>, Vec<std::string::String>)>> = None;
    
    // Computing Experiment
    if param.general.cv {
        let mut folds = CV::new(&data, run_param.cv.outer_folds, &mut rng);
        cv_folds_ids = Some(folds.get_ids());

        run_param.general.gpu = false;
        if run_param.general.thread_number >= param.cv.outer_folds {
            run_param.general.thread_number = param.general.thread_number/param.cv.outer_folds; 
            info!("\x1b[1;93mCross-validation parallelization using {} threads per fold\x1b[0m", run_param.general.thread_number);
        } else {
            run_param.general.thread_number = 1; 
        }

        folds.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| {
            beam::beam(d, &mut None, p, r)  
        }, &run_param, run_param.general.thread_number, running);

        final_population = folds.clone().get_fbm(&param);
        ga::fit_fn(&mut final_population, &data, &mut None, &None, &None, &run_param);
        final_population = final_population.sort();

        collections = folds.fold_collections;

        info!("\x1b[1;93mDisplaying Family of best models across folds\x1b[0m");
    } else {
        collections = vec![beam::beam(&mut data, &mut None, &param, running)];
        final_population = beam::keep_n_best_model_within_collection(&collections[0], run_param.beam.max_nb_of_models as usize);

    }
    
    // Loading test data
    let test_data = if !param.data.Xtest.is_empty() {
        debug!("Loading test data...");
        let mut td = Data::new();
        let _ = td.load_data(&param.data.Xtest, &param.data.ytest);
        td.set_classes(param.data.classes.clone());
        if param.data.inverse_classes { td.inverse_classes(); }
        td
    } else {
        Data::new()
    };

    info!("\x1b[1;93mTop model rankings for [{}, {}] interval\x1b[0m", param.beam.kmin, final_population.individuals[0].features.len());
    info!("{}", final_population.display(&data, Some(&test_data), param));

    let exec_time = start.elapsed().as_secs_f64();

    Experiment {
        id: format!("{}_beam_{}", param.general.save_exp.split('.').next().unwrap(), timestamp).to_string(),
        gpredomics_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: timestamp,

        train_data: data,
        test_data: Some(test_data),

        final_population: Some(final_population),
        collections: collections,
        
        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids: cv_folds_ids,
        others:None
    }
}

pub fn run_mcmc(param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let start = std::time::Instant::now();
    warn!("MCMC algorithm is still in beta!");
    warn!(" - results cannot be guaranteed,");
    warn!(" - isn't GPU-compatible,");
    warn!(" - isn't multi-threaded,");
    warn!(" - contains only one 'GENERIC' language.");

    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);

    if param.general.data_type.split(',').count() > 1 {
        error!("MCMC currently only allows one data type per launch");
        panic!("MCMC currently only allows one data type per launch");
    }
    // Load Data using Gpredomics Data structure
    let mut data = Data::new();
    let _ = data.load_data(param.data.X.as_str(), param.data.y.as_str());
    data.set_classes(param.data.classes.clone());
    if param.data.inverse_classes { data.inverse_classes(); }

    if param.general.cv {
        warn!("Cross-validation is not compatible yet with MCMC.")
    }

    // Each Gpredomics function currently handles class 2 as unknown
    // As MCMC does not, remove unknown sample before analysis
    data = data.remove_class(2);

    // Selecting features
    data.select_features(param);

    if data.feature_selection.len() < param.mcmc.nmin as usize {
        warn!("SBS can not be launched according to required parameters: {} pre-selected features < {} minimum features to keep after SBS",
                data.feature_selection.len(), param.mcmc.nmin as usize)
    }
        
    let mut mcmc_result;
    if param.mcmc.nmin != 0 && data.feature_selection.len() > param.mcmc.nmin as usize {
        // Executing SBS
        info!("Launching MCMC with SBS (λ={}, {}<->{} features...)", 
            param.mcmc.lambda, data.feature_selection.len(), param.mcmc.nmin);
        let results = bayesian_mcmc::run_mcmc_sbs(&data, param, &mut rng, running);
        
        // Displaying summary of SBS traces
        for (nfeat, post_mean, _, log_evidence, feature_idx, _) in &results {
            info!("Features: {}, Posterior: {:.4e}, Log Evid: {:.4}, Removed: {}", nfeat, post_mean, log_evidence, data.features[*feature_idx]);
        }
    
        // Extract best MCMC trace
        info!("Computing full posterior for optimal feature subset...");
        mcmc_result = bayesian_mcmc::get_best_mcmc_sbs(&data, &results, &param);
        
        //println!("{:?}",mcmc_result.MCMC);
        
    } else {
        info!("Launching MCMC without SBS (λ={}, using all {} features...)", 
            param.mcmc.lambda, data.feature_selection.len());

        let data_types: Vec<&str> = param.general.data_type.split(",").collect();
        let data_type = data_types[0];
        if data_types.len() > 1 { warn!("MCMC allows only one datatype per launch currently. Keeping: {}", data_type)}

        let bp: bayesian_mcmc::BayesPred = bayesian_mcmc::BayesPred::new(&data, param.mcmc.lambda, individual::data_type(data_type), param.general.data_type_epsilon);
        mcmc_result = bayesian_mcmc::compute_mcmc(&bp, param, &mut rng);
    }

    // Building complete posterior distribution
    let mut pop: Population = mcmc_result.population;
    mcmc_result.population = Population::new();

    // Using MCMC models to compute prediction on test data
    let mut test_data = Data::new();
    if !param.data.Xtest.is_empty() {
        let _ = test_data.load_data(&param.data.Xtest, &param.data.ytest);
        test_data.set_classes(param.data.classes.clone());
        if param.data.inverse_classes { test_data.inverse_classes(); }
        test_data = test_data.remove_class(2);
    }

    info!("{}", pop.display(&data, Some(&test_data), &param));
    
    let exec_time = start.elapsed().as_secs_f64();
    let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    Experiment {
        id: format!("{}_mcmc_{}_{}", param.general.save_exp.split('.').next().unwrap(), param.general.algo, timestamp).to_string(),
        gpredomics_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: timestamp,

        train_data: data,
        test_data: Some(test_data),

        // Currently keep MCMC last pop in collections to be compatible with GpredomicsR
        final_population: None,
        collections: vec![vec![pop]],
        
        importance_collection: None,
        execution_time: exec_time,

        cv_folds_ids: None,
        parameters: param.clone(),
        others: Some(ExperimentMetadata::MCMC { trace: mcmc_result })
    }
}