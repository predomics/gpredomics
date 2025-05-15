#![allow(non_snake_case)]

pub mod beam;
pub mod data;
mod utils;
pub mod individual;
pub mod param;
pub mod population;
mod ga;
mod cv;
pub mod gpu;

use data::Data;
use individual::Individual;
use population::Population;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use rand::prelude::*;
use param::Param;

use log::{debug, info, warn};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// a very basic use
pub fn basic_test(param: &Param) {
    info!("                          BASIC TEST\n-----------------------------------------------------");
    // define some data
    let mut my_data = Data::new();
    my_data.X.insert((0,0), 0.1);
    my_data.X.insert((0,1), 0.2);
    my_data.X.insert((0,2), 0.3);
    my_data.X.insert((2,0), 0.9);
    my_data.X.insert((2,1), 0.8);
    my_data.X.insert((2,2), 0.7);    
    my_data.feature_len = 3;
    my_data.sample_len = 3;    my_data.samples = string_vec! ["a","b","c"];
    my_data.features = string_vec! ["msp1","msp2","msp3"];
    my_data.y = vec! [0,1,1];
    my_data.feature_len = 3;
    my_data.sample_len = 3;
    info!("{:?}", my_data);

    // create a model
    let mut my_individual = Individual::new();
    my_individual.features.insert(0, 1);
    my_individual.features.insert(2, -1);
    my_individual.compute_hash();
    info!("my individual: {:?}",my_individual.features);
    info!("my individual hash: {}",my_individual.hash);
    info!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual AUC: {:?}",my_individual.compute_auc(&my_data));
    
    let mut my_individual2 = Individual::new();
    my_individual2.features.insert(0, 1);
    my_individual2.features.insert(1, -1);
    my_individual2.compute_hash();
    info!("my individual2 {:?}",my_individual2.features);
    info!("my individual2 hash: {}",my_individual2.hash);
    info!("my individual2 evaluation: {:?}",my_individual2.evaluate(&my_data));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual2 AUC: {:?}",my_individual2.compute_auc(&my_data));


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
    let mut my_data = Data::new();
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut auc_max = 0.0;
    let mut best_individual: Individual = Individual::new();
    for _ in 0..1000 {
        let mut my_individual = Individual::random(&my_data, &mut rng);


        let auc = my_individual.compute_auc(&my_data);
        if auc>auc_max {auc_max=auc;best_individual=my_individual;}
    }
    warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}


/// a more elaborate use with random models
pub fn gpu_random_run(param: &Param) {
    info!("                          GPU RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut my_data = Data::new();
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("Selecting features...");
    my_data.select_features(param);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    //let mut auc_max = 0.0;
    //let mut best_individual: Individual = Individual::new();
    let nb_individuals: usize = 1000;


    let assay = gpu::GpuAssay::new(&my_data.X, &my_data.feature_selection, my_data.sample_len, nb_individuals as usize, &param.gpu);

    let mut individuals:Vec<Individual> = (0..nb_individuals).map(|_i| {Individual::random(&my_data, &mut rng)}).collect();
    individuals = individuals.into_iter()
        .map(|i| {
            // we filter random features for selected features, not efficient but we do not care
            let mut new = Individual::new();
            new.features = i.features.into_iter()
                    .filter(|(i,_f)| {my_data.feature_selection.contains(i)})
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

    //let auc = my_individual.compute_auc(&my_data);
    //warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}

/// the Genetic Algorithm test
pub fn run_ga(param: &Param, running: Arc<AtomicBool>) -> (Vec<Population>,Data,Data) {
    info!("Genetic algorithm\n-----------------------------------------------------");
    let mut my_data = Data::new();
    
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    my_data.set_classes(param.data.classes.clone());
    info!("\x1b[2;97m{:?}\x1b[0m", my_data);  
    //let has_auc = true;

    let (mut run_test_data, run_data): (Option<Data>,Option<Data>) = if param.general.overfit_penalty>0.0 {
        let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
        let mut cv=cv::CV::new(&my_data, param.cv.fold_number, &mut rng);
        (Some(cv.validation_folds.remove(0)),Some(cv.training_sets.remove(0)))
    } else { (None,None) };

    let mut populations = if let Some(mut this_data)=run_data {
        ga::ga(&mut this_data, &mut run_test_data, &param, running)
    } else {
        ga::ga(&mut my_data, &mut run_test_data, &param, running)
    };
    
    //if param.general.overfit_penalty == 0.0 
    //    {   match param.general.fit.to_lowercase().as_str() {
    //            "auc" => {
    //                info!("Fitting by AUC with k penalty {}",param.general.k_penalty);
    //                ga::ga(&mut my_data,&param,running, 
    //                    |p: &mut Population,d: &Data| { 
    //                    p.auc_fit(d, param.general.k_penalty, param.general.thread_number); 
    //                } )
    //            },
    //            "specificity"|"sensitivity" => {
    //                info!("Fitting by objective {} with k penalty {}",param.general.fit,param.general.k_penalty);
    //                let fpr_penalty = if param.general.fit.to_lowercase().as_str()=="specificity" {1.0} else {param.general.fr_penalty}; 
    //                let fnr_penalty = if param.general.fit.to_lowercase().as_str()=="sensitivity" {1.0} else {param.general.fr_penalty}; 
    //                info!("FPR penalty {}  |  FNR penalty {}",fpr_penalty,fnr_penalty);
    //                has_auc = false;
    //                ga::ga(&mut my_data,&param,running, 
    //                    |p: &mut Population,d: &Data| { 
    //                        p.objective_fit(d, fpr_penalty,fnr_penalty,param.general.k_penalty,
    //                            param.general.thread_number); 
    //                    } )
    //            },
    //            other => { error!("Unrecognised fit {}",other); panic!("Unrecognised fit {}",other)}
    //        }
    //    } else {
    //        let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    //        let cv=cv::CV::new(&my_data, param.cv.fold_number, &mut rng);
//
    //        match param.general.fit.to_lowercase().as_str() {
    //            "auc" => {
    //                info!("Fitting by AUC with k penalty {} and overfit penalty {}",param.general.k_penalty, param.general.overfit_penalty);
    //                ga::ga(&mut cv.datasets[0].clone(),&param,running, 
    //                |p: &mut Population,d: &Data| { 
    //                    p.auc_nooverfit_fit(d,
    //                        param.general.k_penalty, &cv.folds[0].clone(), param.general.overfit_penalty,
    //                        param.general.thread_number); } )
    //            },
    //            "specificity"|"sensitivity" => {
    //                info!("Fitting by objective {} with k penalty {} and overfit penalty {}",param.general.fit,param.general.k_penalty, param.general.overfit_penalty);
    //                let fpr_penalty = if param.general.fit.to_lowercase().as_str()=="specificity" {1.0} else {param.general.fr_penalty}; 
    //                let fnr_penalty = if param.general.fit.to_lowercase().as_str()=="sensitivity" {1.0} else {param.general.fr_penalty}; 
    //                info!("FPR penalty {}  |  FNR penalty {}",fpr_penalty,fnr_penalty);
    //                has_auc = false;
    //                ga::ga(&mut cv.datasets[0].clone(),&param,running, 
    //                    |p: &mut Population,d: &Data| { 
    //                        p.objective_nooverfit_fit(d, fpr_penalty,fnr_penalty,param.general.k_penalty,
    //                            &cv.folds[0].clone(), param.general.overfit_penalty, param.general.thread_number); 
    //                    } )
    //            },
    //            other => { error!("Unrecognised fit {}",other); panic!("Unrecognised fit {}",other)}
    //        }
    //    };
    let generations = populations.len();
    let population= &mut populations[generations-1];

        debug!("Length of population {}",population.individuals.len());
        let nb_model_to_test = if param.general.nb_best_model_to_test>0 {param.general.nb_best_model_to_test as usize} else {population.individuals.len()};
        debug!("Testing {} models",nb_model_to_test);

    let mut test_data=Data::new();
    if param.data.Xtest.len()>0 {
        let _ = test_data.load_data(&param.data.Xtest, &param.data.ytest);
        test_data.set_classes(param.data.classes.clone());
        
        info!("{}", population.display(&my_data, Some(&test_data), param));

        // // Prepare the evaluation pool
        // let pool = ThreadPoolBuilder::new()
        //     .num_threads(param.general.thread_number)
        //     .build()
        //     .expect("Failed to build thread pool");

        // // Compute the final metrics
        // pool.install(|| {
        //     let results: Vec<String> = population.individuals[..nb_model_to_test]
        //         .par_iter_mut()
        //         .enumerate()
        //         .map(|(i, individual)| {
        //             (individual.threshold, individual.accuracy, individual.sensitivity, individual.specificity) = individual.compute_threshold_and_metrics(&my_data);
        //             individual.display(&my_data, Some(&test_data), &param.general.algo, 2, true)
        //         })
        //         .collect();

        //     // Output results in order
        //     for result in results {
        //         info!("{}", result);
        //     }
        // });
    }
    else {
        info!("{}", population.display(&my_data, None, param));
    }

    (populations,my_data,test_data) 

}

pub fn run_beam(param: &Param, running: Arc<AtomicBool>) -> (Vec<Population>,Data,Data) {
    let mut data = Data::new();
    let _ = data.load_data(&param.data.X.to_string(), &param.data.y.to_string());
    data.set_classes(param.data.classes.clone());
    
    info!("\x1b[2;97m{:?}\x1b[0m", data);  

    let mut collection = beam::beam(&mut data, &mut None, param, running);
    
    let mut final_pop = Population::new();
    final_pop.individuals = collection.last_mut().unwrap().individuals.clone();
    let mut top_ten_pop = beam::keep_n_best_model_within_collection(&collection, param.general.nb_best_model_to_test as usize);
    
    let mut data_test = Data::new();
    if param.data.Xtest.len()>0 {
        let _ = data_test.load_data(&param.data.Xtest.to_string(), &param.data.ytest.to_string());
        data_test.set_classes(param.data.classes.clone());
        info!("\x1b[1;93mTop model rankings for k={:?}\x1b[0m", final_pop.individuals[0].features.len());
        info!("{}", final_pop.display(&data, Some(&data_test), param));
        info!("\x1b[1;93mTop model rankings for [{}, {}] interval\x1b[0m", param.beam.kmin, final_pop.individuals[0].features.len());
        info!("{}", top_ten_pop.display(&data, Some(&data_test), param));
    } else {
        info!("\x1b[1;93mTop model rankings for k={:?}\x1b[0m", final_pop.individuals[0].features.len());
        info!("{}", final_pop.display(&data, None, param));
        info!("\x1b[1;93mTop model rankings for [{}, {}] interval\x1b[0m", param.beam.kmin, final_pop.individuals[0].features.len());
        info!("{}", top_ten_pop.display(&data, None, param));
    }


    (collection, data, data_test)
}

pub fn run_cv_ga(param: &Param, running: Arc<AtomicBool>) -> (cv::CV,Data,Data) {
    info!("Genetic algorithm with \x1b[1;93mCross-Validation\x1b[0m\n-----------------------------------------------------");
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let mut data = Data::new();
    let _ = data.load_data(param.data.X.as_str(),param.data.y.as_str());
    data.set_classes(param.data.classes.clone());
    info!("\x1b[2;97m{:?}\x1b[0m", data); 

    // CV
    let mut crossval = cv::CV::new(&data, param.cv.fold_number, &mut rng);
    let mut cv_param = param.clone();
    cv_param.general.thread_number = param.general.thread_number/param.cv.fold_number; 
    cv_param.general.gpu = false;

    if cv_param.general.thread_number>= param.cv.fold_number {
        info!("\x1b[1;93mCross-validation parallelization using {} threads per fold\x1b[0m", cv_param.general.thread_number);
    }

    let results=crossval.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| 
        { ga::ga(d,&mut None,p,r) }, &cv_param, cv_param.general.thread_number, running);
    
    let mut cv_fmb_pop = Population::new();

    // Computing OOB with validation folds based on FBM
    let mut importances: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut features_significant_observations: HashMap<usize, usize> = Default::default();
    let mut features_classes: HashMap<usize, i8> = Default::default();
    for (i,(mut fold_last_population, train, mut test)) in results.into_iter().enumerate() {
        
        // Fit last population on validation folds instead of training folds in order to select the models most likely to generalize.
        ga::fit_fn(&mut fold_last_population, &mut test, &mut None, &None, &None, &cv_param);

        let mut fold_last_fbm = fold_last_population.select_best_population(param.cv.cv_best_models_ci_alpha);
        fold_last_fbm = fold_last_fbm.sort();
        
        let fold_importances = fold_last_fbm.compute_pop_oob_feature_importance(&crossval.validation_folds[i], param.cv.n_permutations_oob, &mut rng, &param.cv.importance_aggregation, param.cv.scaled_importance);
        for (key, value) in fold_importances.iter() {
            importances.entry(*key).or_insert_with(Vec::new).push(*value);
            if train.feature_class.contains_key(key) {
                *features_significant_observations.entry(*key).or_insert(0) += 1;
                let associated_class = train.feature_class[key];
                if associated_class == 0 {
                    *features_classes.entry(*key).or_insert(0) += -1;
                } else if associated_class == 1 {
                    *features_classes.entry(*key).or_insert(0) += 1;
                }
            }
        }

        cv_fmb_pop.individuals.extend(fold_last_fbm.individuals);
    }

    // Sort best models according to their fit on validation folds
    // Display them and their metrics on data and potential data_test
    cv_fmb_pop = cv_fmb_pop.sort();
    
    info!("\x1b[1;93mDisplaying best models according to their generalization on associated validation fold (fit). Metrics are calculated on the global training set (including training and validation folds) and, if supplied, on the additional test set.\x1b[0m");
    cv_fmb_pop.compute_all_metrics(&data);
    cv_fmb_pop.remove_clone();

    let mut data_test = Data::new();
    if param.data.Xtest.len()>0 {
        let _ = data_test.load_data(&param.data.Xtest.to_string(), &param.data.ytest.to_string());
        data_test.set_classes(param.data.classes.clone());
        info!("{}", cv_fmb_pop.display(&data, Some(&data_test), param));
    } else {
        info!("{}", cv_fmb_pop.display(&data, None, param));
    }

    let mut final_importances: HashMap<usize, f64> = HashMap::new();

    for (key, values) in &importances {
        let agg = match param.cv.importance_aggregation.as_str() {
           "median" => {
                let mut v = values.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if v.len() % 2 == 1 {
                    v[v.len() / 2]
                } else {
                    (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
                }
            },
            _ => {
                let sum: f64 = values.iter().sum();
                sum / values.len() as f64
            }
        };
        final_importances.insert(*key, agg);
    }

    let mut importance_vec: Vec<(&usize, &f64)> = final_importances.iter().collect();
    importance_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    let n = 150;
    info!("\x1b[1;93mRank\t\tFeature\t\t{}\x1b[0m",  match param.cv.importance_aggregation.as_str() {"median" => "Importance (Median)", _ => "Importance (Mean)"}
);
    // Colouring if the feature is associated with the same class in all FBM models (or unassociated)
    for (rank, (feature_idx, importance)) in importance_vec.iter().take(n).enumerate() {
        if features_classes[*feature_idx] == -(features_significant_observations[*feature_idx] as i8)  {
             info!("\x1b[1;93m#{}\x1b[0m\t\t\x1b[95m{}\x1b[0m\t\t{:.4e}", rank + 1, data.features[**feature_idx], importance);
        } else if features_classes[*feature_idx] == features_significant_observations[*feature_idx] as i8 {
            info!("\x1b[1;93m#{}\x1b[0m\t\t\x1b[96m{}\x1b[0m\t\t{:.4e}", rank + 1, data.features[**feature_idx], importance);
        } else {
            info!("\x1b[1;93m#{}\x1b[0m\t\t{}\t\t{:.4e}", rank + 1, data.features[**feature_idx], importance);
        }
    }

    // if param.data.Xtest.len()>0 {
    //     let _ = test_data.load_data(&param.data.Xtest, &param.data.ytest);
    //     test_data.set_classes(param.data.classes.clone());
        
    //     for (i,(mut best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
    //         let holdout_auc=best_model.compute_auc(&test_data);
    //         let (threshold, accuracy, sensitivity, specificity) = 
    //             best_model.compute_threshold_and_metrics(&test_data);
    //         info!("Model #{} [gen:{}] [k={}]: train AUC {:.3}  | test AUC {:.3} | holdout AUC {:.3} | threshold {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} | {:?}",
    //                     i+1,best_model.epoch,best_model.k,train_auc,test_auc,holdout_auc,threshold,accuracy,sensitivity,specificity,best_model);
    //         info!("Features importance on train+test... ");
    //         info!("{}",
    //             best_model.compute_oob_feature_importance(&my_data, param.ga.feature_importance_permutations,&mut rng)
    //                 .into_iter()
    //                 .map(|feature_importance| { format!("[{:.4}]",feature_importance) })
    //                 .collect::<Vec<String>>()
    //                 .join(" ")
    //         );
    //         info!("Features importance on holdout... ");
    //         info!("{}",
    //             best_model.compute_oob_feature_importance(&test_data, param.ga.feature_importance_permutations,&mut rng)
    //                 .into_iter()
    //                 .map(|feature_importance| { format!("[{:.4}]",feature_importance) })
    //                 .collect::<Vec<String>>()
    //                 .join(" ")
    //         );
    //     }    
    // }
    // else {
    //     for (i,(best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
    //         warn!("Model #{} [gen:{}] [k={}]: train AUC {:.3} | test AUC {:.3} | {:?}",i+1,best_model.epoch,best_model.k,train_auc,test_auc,best_model);
    //     }    
    // }

    (crossval,data,data_test) 

}

pub fn run_cv_beam(param: &Param, running: Arc<AtomicBool>) -> (cv::CV,Data,Data) {
    info!("Beam algorithm with \x1b[1;93mCross-Validation\x1b[0m\n-----------------------------------------------------");
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let mut data = Data::new();
    let _ = data.load_data(param.data.X.as_str(),param.data.y.as_str());
    data.set_classes(param.data.classes.clone());
    info!("\x1b[2;97m{:?}\x1b[0m", data); 

    let mut data_test = Data::new();
    let _ = data_test.load_data(&param.data.Xtest.to_string(), &param.data.ytest.to_string());
    data_test.set_classes(param.data.classes.clone());

    // CV
    let mut crossval = cv::CV::new(&data, param.cv.fold_number, &mut rng);
    let mut cv_param = param.clone();
    cv_param.general.thread_number = param.general.thread_number/param.cv.fold_number; 
    cv_param.general.gpu = false;
    
    if cv_param.general.thread_number >= param.cv.fold_number {
        info!("\x1b[1;93mCross-validation parallelization using {} thread per fold\x1b[0m", cv_param.general.thread_number);
    }

    let results=crossval.pass(|d: &mut Data, p: &Param, r: Arc<AtomicBool>| 
        { beam::beam(d,&mut None,p,r) }, &cv_param, cv_param.general.thread_number, running);
    
    let mut cv_fmb_pop = Population::new();

    // Computing OOB with validation folds based on FBM
    let mut importances: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut features_significant_observations: HashMap<usize, usize> = Default::default();
    let mut features_classes: HashMap<usize, i8> = Default::default();
    for (i,(mut fold_last_population, train, mut test)) in results.into_iter().enumerate() {
        
        // Fit last population on validation folds instead of training folds in order to select the models most likely to generalize.
        ga::fit_fn(&mut fold_last_population, &mut test, &mut None, &None, &None, &cv_param);

        let mut fold_last_fbm = fold_last_population.select_best_population(param.cv.cv_best_models_ci_alpha);
        fold_last_fbm = fold_last_fbm.sort();
        
        let fold_importances = fold_last_fbm.compute_pop_oob_feature_importance(&crossval.validation_folds[i], param.cv.n_permutations_oob, &mut rng, &param.cv.importance_aggregation, param.cv.scaled_importance);
        for (key, value) in fold_importances.iter() {
            importances.entry(*key).or_insert_with(Vec::new).push(*value);
            if train.feature_class.contains_key(key) {
                *features_significant_observations.entry(*key).or_insert(0) += 1;
                let associated_class = train.feature_class[key];
                if associated_class == 0 {
                    *features_classes.entry(*key).or_insert(0) += -1;
                } else if associated_class == 1 {
                    *features_classes.entry(*key).or_insert(0) += 1;
                }
            }
        }

        cv_fmb_pop.individuals.extend(fold_last_fbm.individuals);
    }

    // Sort best models according to their fit on validation folds
    // Display them and their metrics on data and potential data_test
    cv_fmb_pop = cv_fmb_pop.sort();
    
    info!("\x1b[1;93mDisplaying best models according to their generalization on associated validation fold (fit). Metrics are calculated on the global training set (including training and validation folds) and, if supplied, on the additional test set.\x1b[0m");
    cv_fmb_pop.compute_all_metrics(&data);
    cv_fmb_pop.remove_clone();
    
    let mut data_test = Data::new();
    if param.data.Xtest.len()>0 {
        let _ = data_test.load_data(&param.data.Xtest.to_string(), &param.data.ytest.to_string());
        data_test.set_classes(param.data.classes.clone());
        info!("{}", cv_fmb_pop.display(&data, Some(&data_test), param));
    } else {
        info!("{}", cv_fmb_pop.display(&data, None, param));
    }

    let mut final_importances: HashMap<usize, f64> = HashMap::new();

    for (key, values) in &importances {
        let agg = match param.cv.importance_aggregation.as_str() {
           "median" => {
                let mut v = values.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if v.len() % 2 == 1 {
                    v[v.len() / 2]
                } else {
                    (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
                }
            },
            _ => {
                let sum: f64 = values.iter().sum();
                sum / values.len() as f64
            }
        };
        final_importances.insert(*key, agg);
    }

    let mut importance_vec: Vec<(&usize, &f64)> = final_importances.iter().collect();
    importance_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    let n = 150;
    info!("\x1b[1;93mRank\t\tFeature\t\t{}\x1b[0m",  match param.cv.importance_aggregation.as_str() {"median" => "Importance (Median)", _ => "Importance (Mean)"}
);
    // Colouring if the feature is associated with the same class in all FBM models (or unassociated)
    for (rank, (feature_idx, importance)) in importance_vec.iter().take(n).enumerate() {
        if features_classes[*feature_idx] == -(features_significant_observations[*feature_idx] as i8)  {
             info!("\x1b[1;93m#{}\x1b[0m\t\t\x1b[95m{}\x1b[0m\t\t{:.4e}", rank + 1, data.features[**feature_idx], importance);
        } else if features_classes[*feature_idx] == features_significant_observations[*feature_idx] as i8 {
            info!("\x1b[1;93m#{}\x1b[0m\t\t\x1b[96m{}\x1b[0m\t\t{:.4e}", rank + 1, data.features[**feature_idx], importance);
        } else {
            info!("\x1b[1;93m#{}\x1b[0m\t\t{}\t\t{:.4e}", rank + 1, data.features[**feature_idx], importance);
        }
    }

    (crossval,data,data_test) 

}