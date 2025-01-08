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

use log::{debug, info, warn, trace};


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
    info!("my individual: {:?}",my_individual.features);
    info!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual AUC: {:?}",my_individual.compute_auc(&my_data));


    let mut data2=Data::new();
    data2.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let parent1 = Individual::random(&data2, &mut rng);
    let parent2 = Individual::random(&data2, &mut rng);
    let mut parents=population::Population::new();
    parents.individuals.push(parent1);
    parents.individuals.push(parent2);
    let mut children = ga::cross_over(&parents, &param, data2.feature_len, &mut rng);
    for (i,individual) in parents.individuals.iter().enumerate() { info!("Parent #{}: {:?}",i,individual); }
    for (i,individual) in children.individuals.iter().enumerate() { info!("Child #{}: {:?}",i,individual); }
    let feature_selection:Vec<usize> = (0..data2.feature_len).collect();
    ga::mutate(&mut children, param, &feature_selection, &mut rng);
    for (i,individual) in children.individuals.iter().enumerate() { info!("Mutated Child #{}: {:?}",i,individual); }    

}

/// a more elaborate use with random models
pub fn random_run(param: &Param) {
    info!("                          RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut my_data = Data::new();
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut auc_max = 0.0;
    let mut best_individual: Individual = Individual::new();
    for i in 0..10000 {
        let mut my_individual = Individual::random(&my_data, &mut rng);

        let auc = my_individual.compute_auc(&my_data);
        if auc>auc_max {auc_max=auc;best_individual=my_individual;}
    }
    warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}


/// the Genetic Algorithm test
pub fn ga_run(param: &Param, running: Arc<AtomicBool>) -> (Vec<Population>,Data,Data) {
    info!("                          GA TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("{:?}", my_data); 

    let mutate = if param.general.algo.contains("ga2") { ga::mutate2 } else { ga::mutate };

    let mut populations = if param.general.algo.contains("no_overfit") 
        {
            ga::ga(&mut my_data,&param,running, mutate, 
                |p: &mut Population,d: &Data| { p.evaluate_with_k_penalty(d, param.ga.kpenalty); } )
        } else {
            let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
            let mut cv=cv::CV::new(&my_data, param.cv.fold_number, &mut rng);

            ga::ga(&mut cv.datasets[0].clone(),&param,running, mutate, 
                |p: &mut Population,d: &Data| { p.evaluate_with_kno_penalty(d, param.ga.kpenalty, &cv.folds[0].clone(), param.cv.overfit_penalty); } )
        };
    let generations = populations.len();
    let mut population= &mut populations[generations-1];

    let mut test_data=Data::new();
    if param.data.Xtest.len()>0 {
        test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        debug!("Length of population {}",population.individuals.len());
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            let auc=individual.auc;
            let test_auc=individual.compute_auc(&test_data);
            individual.auc = auc;
            debug!("test aux ok");
            let (threshold, accuracy, sensitivity, specificity) = individual.compute_threshold_and_metrics(&test_data);
            debug!("compute threshold");
            info!("Model #{} [k={}] [gen:{}]: train AUC {}  | test AUC {} | threshold {} | accuracy {} | sensitivity {} | specificity {} | {:?}",
                        i+1,individual.k,individual.n,auc,test_auc,threshold,accuracy,sensitivity,specificity,individual);
        }    
    }
    else {
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            let auc=individual.auc;
            info!("Model #{} [k={}] [gen:{}]: train AUC {}",i+1,individual.k,individual.n,auc);
        }    
    }

    (populations,my_data,test_data) 

}


/// the Genetic Algorithm test with Crossval (not useful but test CV)
pub fn gacv_run(param: &Param, running: Arc<AtomicBool>) -> (cv::CV,Data,Data) {
    info!("                          GA CV TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("{:?}", my_data); 

    let mut crossval = cv::CV::new(&my_data, 10, &mut rng);

    let results=crossval.pass(|d: &mut Data,p: &Param,r: Arc<AtomicBool>| 
        { ga::ga(d,p,r,
            ga::mutate,
            |p: &mut Population,d: &Data| { p.evaluate_with_k_penalty(d, param.ga.kpenalty.clone()); }) }, &param, param.general.thread_number, running);
    
    let mut test_data=Data::new();
    if param.data.Xtest.len()>0 {
        test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        for (i,(mut best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            let holdout_auc=best_model.compute_auc(&test_data);
            let (threshold, accuracy, sensitivity, specificity) = best_model.compute_threshold_and_metrics(&test_data);
            info!("Model #{} [gen:{}] [k={}]: train AUC {:.3}  | test AUC {:.3} | holdout AUC {:.3} | threshold {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} | {:?}",
                        i+1,best_model.n,best_model.k,train_auc,test_auc,holdout_auc,threshold,accuracy,sensitivity,specificity,best_model);
            info!("Features importance on train+test... ");
            info!("{}",
                best_model.compute_oob_feature_importance(&my_data, param.ga.feature_importance_permutations,&mut rng)
                    .into_iter()
                    .map(|feature_importance| { format!("[{:.4}]",feature_importance) })
                    .collect::<Vec<String>>()
                    .join(" ")
            );
            info!("Features importance on holdout... ");
            info!("{}",
                best_model.compute_oob_feature_importance(&test_data, param.ga.feature_importance_permutations,&mut rng)
                    .into_iter()
                    .map(|feature_importance| { format!("[{:.4}]",feature_importance) })
                    .collect::<Vec<String>>()
                    .join(" ")
            );
        }    
    }
    else {
        for (i,(best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            warn!("Model #{} [gen:{}] [k={}]: train AUC {:.3} | test AUC {:.3} | {:?}",i+1,best_model.n,best_model.k,train_auc,test_auc,best_model);
        }    
    }

    (crossval,my_data,test_data) 

}


