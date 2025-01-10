#![allow(non_snake_case)]

pub mod data;
mod utils;
mod individual;
pub mod param;
pub mod population;
mod ga;
mod cv;

use data::Data;
use individual::Individual;
use population::Population;
use rand_chacha::ChaCha8Rng;
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
    info!("my individual evaluation: {:?}",my_individual.evaluate(&my_data, 0.0));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual AUC: {:?}",my_individual.compute_auc(&my_data, 0.0));

    let mut my_individual2 = Individual::new();
    my_individual2.features.insert(0, 1);
    my_individual2.features.insert(1, -1);
    my_individual2.compute_hash();
    info!("my individual2 {:?}",my_individual2.features);
    info!("my individual2 hash: {}",my_individual2.hash);
    info!("my individual2 evaluation: {:?}",my_individual2.evaluate(&my_data, 0.0));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual2 AUC: {:?}",my_individual2.compute_auc(&my_data, 0.0));


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

        let auc = my_individual.compute_auc(&my_data, 0.0);
        if auc>auc_max {auc_max=auc;best_individual=my_individual;}
    }
    warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}


/// the Genetic Algorithm test
pub fn ga_run(param: &Param, running: Arc<AtomicBool>) -> (Vec<Population>,Data,Data) {
    info!("                          GA TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("{:?}", my_data); 
    let mut has_auc = true;

    let mut populations = if param.general.overfit_penalty == 0.0 
        {
            if param.general.fit.to_lowercase().as_str()=="auc" {
                info!("Fitting by AUC with k penalty {}",param.general.k_penalty);
                ga::ga(&mut my_data,&param,running, 
                |p: &mut Population,d: &Data| { 
                    p.auc_fit(d, param.general.data_type_epsilon, param.general.k_penalty); 
                } )
            } else {
                info!("Fitting by objective {} with k penalty {}",param.general.fit,param.general.k_penalty);
                let fpr_penalty = if param.general.fit.to_lowercase().as_str()=="specificity" && param.general.fpr_penalty==0.0 {1.0} else {param.general.fpr_penalty}; 
                let fnr_penalty = if param.general.fit.to_lowercase().as_str()=="sensitivity" && param.general.fnr_penalty==0.0 {1.0} else {param.general.fnr_penalty}; 
                info!("FPR penalty {}  |  FNR penalty {}",fpr_penalty,fnr_penalty);
                has_auc = false;
                ga::ga(&mut my_data,&param,running, 
                    |p: &mut Population,d: &Data| { 
                        p.objective_fit(d, param.general.data_type_epsilon, fpr_penalty,fnr_penalty,param.general.k_penalty); 
                    } )
            }
        } else {
            let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
            let cv=cv::CV::new(&my_data, param.cv.fold_number, &mut rng);

            if param.general.fit.to_lowercase().as_str()=="auc" {
                info!("Fitting by AUC with k penalty {} and overfit penalty {}",param.general.k_penalty, param.general.overfit_penalty);
                ga::ga(&mut cv.datasets[0].clone(),&param,running, 
                |p: &mut Population,d: &Data| { 
                    p.auc_nooverfit_fit(d, param.general.data_type_epsilon,
                         param.general.k_penalty, &cv.folds[0].clone(), param.general.overfit_penalty); } )
            } else {
                info!("Fitting by objective {} with k penalty {} and overfit penalty {}",param.general.fit,param.general.k_penalty, param.general.overfit_penalty);
                let fpr_penalty = if param.general.fit.to_lowercase().as_str()=="specificity" && param.general.fpr_penalty==0.0 {1.0} else {param.general.fpr_penalty}; 
                let fnr_penalty = if param.general.fit.to_lowercase().as_str()=="sensitivity" && param.general.fnr_penalty==0.0 {1.0} else {param.general.fnr_penalty}; 
                info!("FPR penalty {}  |  FNR penalty {}",fpr_penalty,fnr_penalty);
                has_auc = false;
                ga::ga(&mut cv.datasets[0].clone(),&param,running, 
                    |p: &mut Population,d: &Data| { 
                        p.objective_nooverfit_fit(d, param.general.data_type_epsilon, fpr_penalty,fnr_penalty,param.general.k_penalty,
                            &cv.folds[0].clone(), param.general.overfit_penalty); 
                    } )
            }
        };
    let generations = populations.len();
    let population= &mut populations[generations-1];

    let mut test_data=Data::new();
    if param.data.Xtest.len()>0 {
        let _ = test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        debug!("Length of population {}",population.individuals.len());
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            let mut auc=individual.auc;
            let test_auc=individual.compute_auc(&test_data, param.general.data_type_epsilon);
            if has_auc {
                individual.auc = auc;
                (individual.threshold, individual.accuracy, individual.sensitivity, individual.specificity) = 
                    individual.compute_threshold_and_metrics(&my_data, param.general.data_type_epsilon);
            } else {
                auc = individual.compute_auc(&my_data, param.general.data_type_epsilon);
            }
            let (tp, fp, tn, fn_count) = individual.calculate_confusion_matrix(&test_data, param.general.data_type_epsilon);
            info!("Model #{} [k={}] [gen:{}] threshold {:.3} : AUC {:.3}/{:.3} |  accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3} \n   < {:?} >",
                        i+1,individual.k,individual.epoch,individual.threshold,
                        test_auc,auc,
                        (tp+tn) as f64/(fp+tp+fn_count+tn) as f64,individual.accuracy,
                        if tp+fn_count>0 {tp as f64/(tp+fn_count) as f64} else {0.0},individual.sensitivity,
                        if tn+fp>0 {tn as f64/(tn+fp) as f64} else {0.0},individual.specificity,
                        individual);
        }    
    }
    else {
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            if has_auc {
                (individual.threshold, individual.accuracy, individual.sensitivity, individual.specificity) = 
                    individual.compute_threshold_and_metrics(&test_data, param.general.data_type_epsilon);
            } else {
                individual.compute_auc(&my_data, param.general.data_type_epsilon);
            }
            info!("Model #{} [k={}] [gen:{}]: train AUC {:.3}",i+1,individual.k,individual.epoch,individual.auc);
        }    
    }

    (populations,my_data,test_data) 

}


/// the Genetic Algorithm test with Crossval (not useful but test CV)
pub fn gacv_run(param: &Param, running: Arc<AtomicBool>) -> (cv::CV,Data,Data) {
    info!("                          GA CV TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("{:?}", my_data); 

    let mut crossval = cv::CV::new(&my_data, 10, &mut rng);

    let results=crossval.pass(|d: &mut Data,p: &Param,r: Arc<AtomicBool>| 
        { ga::ga(d,p,r,|p: &mut Population,d: &Data| { 
            p.auc_fit(d, param.general.data_type_epsilon, param.general.k_penalty); 
        }) }, &param, param.general.thread_number, running);
    
    let mut test_data=Data::new();
    if param.data.Xtest.len()>0 {
        let _ = test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        for (i,(mut best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            let holdout_auc=best_model.compute_auc(&test_data, param.general.data_type_epsilon);
            let (threshold, accuracy, sensitivity, specificity) = 
                best_model.compute_threshold_and_metrics(&test_data, param.general.data_type_epsilon);
            info!("Model #{} [gen:{}] [k={}]: train AUC {:.3}  | test AUC {:.3} | holdout AUC {:.3} | threshold {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} | {:?}",
                        i+1,best_model.epoch,best_model.k,train_auc,test_auc,holdout_auc,threshold,accuracy,sensitivity,specificity,best_model);
            info!("Features importance on train+test... ");
            info!("{}",
                best_model.compute_oob_feature_importance(&my_data, param.ga.feature_importance_permutations,param.general.data_type_epsilon,&mut rng)
                    .into_iter()
                    .map(|feature_importance| { format!("[{:.4}]",feature_importance) })
                    .collect::<Vec<String>>()
                    .join(" ")
            );
            info!("Features importance on holdout... ");
            info!("{}",
                best_model.compute_oob_feature_importance(&test_data, param.ga.feature_importance_permutations,param.general.data_type_epsilon,&mut rng)
                    .into_iter()
                    .map(|feature_importance| { format!("[{:.4}]",feature_importance) })
                    .collect::<Vec<String>>()
                    .join(" ")
            );
        }    
    }
    else {
        for (i,(best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            warn!("Model #{} [gen:{}] [k={}]: train AUC {:.3} | test AUC {:.3} | {:?}",i+1,best_model.epoch,best_model.k,train_auc,test_auc,best_model);
        }    
    }

    (crossval,my_data,test_data) 

}


