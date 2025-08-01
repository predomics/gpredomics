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
use rand_chacha::ChaCha8Rng;
use chrono::Local;
use rand::prelude::*;
use param::Param;
use std::collections::{HashSet, HashMap};

use log::{debug, info, warn, error};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// a very basic use
pub fn basic_test(param: &Param) {
    info!("                          BASIC TEST\n-----------------------------------------------------");
    // define some data
    let mut data = Data::new();
    data.X.insert((0,0), 0.1);
    data.X.insert((0,1), 0.2);
    data.X.insert((0,2), 0.3);
    data.X.insert((2,0), 0.9);
    data.X.insert((2,1), 0.8);
    data.X.insert((2,2), 0.7);    
    data.feature_len = 3;
    data.sample_len = 3;    data.samples = string_vec! ["a","b","c"];
    data.features = string_vec! ["msp1","msp2","msp3"];
    data.y = vec! [0,1,1];
    data.feature_len = 3;
    data.sample_len = 3;
    info!("{:?}", data);

    // create a model
    let mut my_individual = Individual::new();
    my_individual.features.insert(0, 1);
    my_individual.features.insert(2, -1);
    my_individual.compute_hash();
    info!("my individual: {:?}",my_individual.features);
    info!("my individual hash: {}",my_individual.hash);
    info!("my individual evaluation: {:?}",my_individual.evaluate(&data));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual AUC: {:?}",my_individual.compute_auc(&data));
    
    let mut my_individual2 = Individual::new();
    my_individual2.features.insert(0, 1);
    my_individual2.features.insert(1, -1);
    my_individual2.compute_hash();
    info!("my individual2 {:?}",my_individual2.features);
    info!("my individual2 hash: {}",my_individual2.hash);
    info!("my individual2 evaluation: {:?}",my_individual2.evaluate(&data));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual2 AUC: {:?}",my_individual2.compute_auc(&data));


    let mut data2=Data::new();
    let _ = data2.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let parent1 = Individual::random(&data2, &mut rng);
    let parent2 = Individual::random(&data2, &mut rng);
    let mut parents=population::Population::new();
    parents.individuals.push(parent1);
    parents.individuals.push(parent2);

    let children_number = param.ga.population_size as usize-parents.individuals.len();
    let mut children = ga::cross_over(&parents, data2.feature_len, children_number, &mut rng);
    for (i,individual) in parents.individuals.iter().enumerate() { info!("Parent #{}: {:?}",i,individual); }
    for (i,individual) in children.individuals.iter().enumerate() { info!("Child #{}: {:?}",i,individual); }
    let feature_selection:Vec<usize> = (0..data2.feature_len).collect();
    ga::mutate(&mut children, param, &feature_selection, &mut rng);
    children.compute_hash();
    let clone_number = children.remove_clone();
    if clone_number>0 { warn!("There were {} clone(s)",clone_number); }
    for (i,individual) in children.individuals.iter().enumerate() { info!("Mutated Child #{}: {:?}",i,individual); }    

}

/// a more elaborate use with random models
pub fn random_run(param: &Param) {
    info!("                          RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut data = Data::new();
    let _ = data.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut auc_max = 0.0;
    let mut best_individual: Individual = Individual::new();
    for _ in 0..1000 {
        let mut my_individual = Individual::random(&data, &mut rng);


        let auc = my_individual.compute_auc(&data);
        if auc>auc_max {auc_max=auc;best_individual=my_individual;}
    }
    warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}


/// a more elaborate use with random models
pub fn gpu_random_run(param: &Param) {
    info!("                          GPU RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut data = Data::new();
    let _ = data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("Selecting features...");
    data.select_features(param);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    //let mut auc_max = 0.0;
    //let mut best_individual: Individual = Individual::new();
    let nb_individuals: usize = 1000;


    let assay = gpu::GpuAssay::new(&data.X, &data.feature_selection, data.sample_len, nb_individuals as usize, &param.gpu);

    let mut individuals:Vec<Individual> = (0..nb_individuals).map(|_i| {Individual::random(&data, &mut rng)}).collect();
    individuals = individuals.into_iter()
        .map(|i| {
            // we filter random features for selected features, not efficient but we do not care
            let mut new = Individual::new();
            new.features = i.features.into_iter()
                    .filter(|(i,_f)| {data.feature_selection.contains(i)})
                    .collect();
            new.data_type = *vec![individual::RAW_TYPE, individual::LOG_TYPE, individual::PREVALENCE_TYPE].choose(&mut rng).unwrap();
            new.language = *vec![individual::TERNARY_LANG, individual::RATIO_LANG].choose(&mut rng).unwrap();
            new
        }).collect();
        

    for _ in 0..1000 {
        
        //println!("First individual {:?}", individuals[0]);

        let scores = assay.compute_scores(&individuals, param.general.data_type_epsilon as f32);

        println!("First scores {:?}", &scores[0..4]);
        //println!("Last scores {:?}", &scores[scores.len()-4..scores.len()]);

    }

    //let auc = my_individual.compute_auc(&data);
    //warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}

/// the Genetic Algorithm test
pub fn run_ga(param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let start = std::time::Instant::now();
    info!("Genetic algorithm\n-----------------------------------------------------");

    let mut data = Data::new();
    let _ = data.load_data(param.data.X.as_str(),param.data.y.as_str());
    data.set_classes(param.data.classes.clone());
    if param.data.inverse_classes { data.inverse_classes(); }
    info!("\x1b[2;97m{:?}\x1b[0m", data);  

    let (mut run_test_data, run_data): (Option<Data>,Option<Data>) = if param.cv.overfit_penalty>0.0 {
        let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
        let mut cv=cv::CV::new(&data, param.cv.outer_folds, &mut rng);
        (Some(cv.validation_folds.remove(0)),Some(cv.training_sets.remove(0)))
    } else { (None,None) };

    let populations = if let Some(mut this_data)= run_data {
        ga::ga(&mut this_data, &mut run_test_data, &param, running)
    } else {
        ga::ga(&mut data, &mut run_test_data, &param, running)
    };
    let generations = populations.len();
    let mut population = populations[generations-1].clone();

    debug!("Length of population {}",population.individuals.len());
    let nb_model_to_test = if param.general.nb_best_model_to_test>0 {param.general.nb_best_model_to_test as usize} else {population.individuals.len()};
    debug!("Testing {} models",nb_model_to_test);

    let mut test_data=Data::new();
    if param.data.Xtest.len()>0 && param.data.ytest.len()>0  {
        let _ = test_data.load_data(&param.data.Xtest, &param.data.ytest);
        test_data.set_classes(param.data.classes.clone());
        if param.data.inverse_classes { test_data.inverse_classes(); }
        info!("{}", population.display(&data, Some(&test_data), param));
    }
    else {
        info!("{}", population.display(&data, None, param));
    }

    let exec_time = start.elapsed().as_secs_f64();
    let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    Experiment {
        id: format!("{}_ga_{}", param.general.save_exp.split('.').next().unwrap(), timestamp).to_string(),
        gpredomics_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: timestamp,
        algorithm: param.general.algo.clone(),

        train_data: data,
        test_data: Some(test_data),

        final_population: Some(population),
        collection: Some(populations),
        
        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids: None,
        others: None
    }

}

pub fn run_beam(param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let start = std::time::Instant::now();
    let mut data = Data::new();
    let _ = data.load_data(&param.data.X.to_string(), &param.data.y.to_string());
    data.set_classes(param.data.classes.clone());
    if param.data.inverse_classes { data.inverse_classes(); }
    
    info!("\x1b[2;97m{:?}\x1b[0m", data);  

    let mut collection = beam::beam(&mut data, &mut None, param, running);
    
    let mut last_pop = collection.last_mut().unwrap().clone();
    let mut final_pop = beam::keep_n_best_model_within_collection(&collection, param.beam.max_nb_of_models as usize);
    
    let mut data_test = Data::new();
    if param.data.Xtest.len()>0 && param.data.ytest.len()>0 {
        let _ = data_test.load_data(&param.data.Xtest.to_string(), &param.data.ytest.to_string());
        data_test.set_classes(param.data.classes.clone());
        if param.data.inverse_classes { data_test.inverse_classes(); }
        info!("\x1b[1;93mTop model rankings for k={:?}\x1b[0m", final_pop.individuals[0].features.len());
        info!("{}", last_pop.display(&data, Some(&data_test), param));
        info!("\x1b[1;93mTop model rankings for [{}, {}] interval\x1b[0m", param.beam.kmin, final_pop.individuals[0].features.len());
        info!("{}", final_pop.display(&data, Some(&data_test), param));
    } else {
        info!("\x1b[1;93mTop model rankings for k={:?}\x1b[0m", final_pop.individuals[0].features.len());
        info!("{}", last_pop.display(&data, None, param));
        info!("\x1b[1;93mTop model rankings for [{}, {}] interval\x1b[0m", param.beam.kmin, final_pop.individuals[0].features.len());
        info!("{}", final_pop.display(&data, None, param));
    }

    let exec_time = start.elapsed().as_secs_f64();
    let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    Experiment {
        id: format!("{}_beam_{}", param.general.save_exp.split('.').next().unwrap(), timestamp).to_string(),
        gpredomics_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: timestamp,
        algorithm: param.general.algo.clone(),

        train_data: data,
        test_data: Some(data_test),

        final_population: Some(final_pop),
        collection: Some(collection),
        
        importance_collection: None,
        execution_time: exec_time,
        parameters: param.clone(),

        cv_folds_ids:None,
        others:None
    }
}

pub fn run_cv(param: &Param, running: Arc<AtomicBool>) -> Experiment {
    let start = std::time::Instant::now();
    
    info!("\x1b[1;93mCross-Validation\x1b[0m\n-----------------------------------------------------");
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    
    // Loading data
    let mut data = Data::new();
    let _ = data.load_data(param.data.X.as_str(),param.data.y.as_str());
    data.set_classes(param.data.classes.clone());
    if param.data.inverse_classes { data.inverse_classes(); }
    info!("\x1b[2;97m{:?}\x1b[0m", data); 

    // Split into k outer folds 
    let mut cv_data = cv::CV::new(&data, param.cv.outer_folds, &mut rng);
    let mut cv_param = param.clone();
    cv_param.general.thread_number = param.general.thread_number/param.cv.outer_folds; 
    cv_param.general.gpu = false;

    if cv_param.general.thread_number >= param.cv.outer_folds {
        info!("\x1b[1;93mCross-validation parallelization using {} threads per fold\x1b[0m", cv_param.general.thread_number);
    }

    // Launch algorithm k times 
    cv_data.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| {
        match param.general.algo.as_str() {
            "ga" => ga::ga(d, &mut None, p, r),
            "beam" => beam::beam(d, &mut None, p, r),
            _ => panic!("Such algorithm is not available with cross-validation."),
        }
    }, &cv_param, cv_param.general.thread_number, running);

    let cv_folds_train_test_ids: Vec<(Vec<String>, Vec<String>)> = cv_data.get_ids();

    // Extract each Family of Best Models and merge them
    let all_fold_fbms: Vec<(Population, Data)> = (0..cv_data.validation_folds.len())
        .map(|i| cv_data.extract_fold_fbm(i, param))
        .collect();

    let mut merged_fbm = Population::new();
    for (fold_fbm, _) in &all_fold_fbms {
        merged_fbm.individuals.extend(fold_fbm.individuals.clone());
    }

    // Display Family of Best Models performance on validation fold VS 
    for (i, (fold_fbm, valid_data)) in all_fold_fbms.iter().enumerate() {
        info!("\x1b[1;93mFold #{}\x1b[0m", i+1);
        // Validation fold = valid_data, train complet = &self.train_data
        info!("{}", fold_fbm.clone().display(valid_data, Some(&data), &param));
    }

    // Load data to test 
    let test_data = if !param.data.Xtest.is_empty() {
        let mut td = Data::new();
        let _ = td.load_data(&param.data.Xtest, &param.data.ytest);
        td.set_classes(param.data.classes.clone());
        if param.data.inverse_classes { td.inverse_classes(); }
        td
    } else {
        Data::new()
    };

    // Fit the population on all train data
    ga::fit_fn(&mut merged_fbm, &data, &mut Some(test_data.clone()), &None, &None, param);
    info!("\x1b[1;93mFamily of best models across folds\x1b[0m");
    info!("{}", merged_fbm.display(&data, Some(&test_data), &param));

    if data.feature_len > 10000 {
        warn!("Large dataset. Removing non-selected features from memory to reduce experiment...");
        
        // Collecter toutes les features présentes dans les training_sets
        let mut union_features = HashSet::new();
        for training_set in &cv_data.training_sets {
            for &feature in &training_set.feature_selection {
                union_features.insert(feature);
            }
        }
        
        let mut new_x = HashMap::new();
        for (&(sample, feature), &value) in &data.X {
            if union_features.contains(&feature) {
                new_x.insert((sample, feature), value);
            }
        }
        
        data.X = new_x;
        
        info!("Non-kept features removed. Dataset compacted from {} to {} features.", 
            data.feature_len, union_features.len());
    }

    let exec_time = start.elapsed().as_secs_f64();
    let timestamp =  Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    Experiment {
        id: format!("{}_cv_{}_{}", param.general.save_exp.split('.').next().unwrap(), param.general.algo, timestamp).to_string(),
        gpredomics_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: timestamp,
        algorithm: param.general.algo.clone(),

        train_data: data,
        test_data: Some(test_data),

        final_population: Some(merged_fbm),
        collection: cv_data.fold_populations,
        
        importance_collection: None,
        execution_time: exec_time,

        cv_folds_ids: Some(cv_folds_train_test_ids),
        parameters: param.clone(),
        others: None
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
        algorithm: param.general.algo.clone(),

        train_data: data,
        test_data: Some(test_data),

        final_population: Some(pop),
        collection: None,
        
        importance_collection: None,
        execution_time: exec_time,

        cv_folds_ids: None,
        parameters: param.clone(),
        others: Some(ExperimentMetadata::MCMC { trace: mcmc_result })
    }
}
