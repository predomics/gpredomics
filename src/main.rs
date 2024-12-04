mod data;
mod utils;
mod individual;
mod param;
mod population;
mod ga;
mod cv;
mod hyper;

use data::Data;
use individual::Individual;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;
use param::Param;
use std::process;

/// a very basic use
fn basic_test() {
    println!("                          BASIC TEST\n-----------------------------------------------------");
    // define some data
    let mut my_data = Data::new();
    my_data.X = vec! [ vec! [0.1,0.2,0.3], vec! [0.0, 0.0, 0.0], vec! [0.9,0.8,0.7] ];
    my_data.samples = string_vec! ["a","b","c"];
    my_data.features = string_vec! ["msp1","msp2","msp3"];
    my_data.y = vec! [0,1,1];
    println!("{:?}", my_data);

    // create a model
    let mut my_individual = Individual::new();
    my_individual.features = vec! [1,0,-1];
    println!("my individual: {:?}",my_individual.features);
    println!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));
    // shoud display 1.0 (the AUC is 1.0)
    println!("my individual AUC: {:?}",my_individual.compute_auc(&my_data));
}

/// a more elaborate use with random models
fn random_run(param: &Param) {
    println!("                          RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut my_data = Data::new();
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut auc_max = 0.0;
    let mut best_individual: Individual = Individual::new();
    for i in 0..10000 {
        let mut my_individual = Individual::random(&my_data, &mut rng);

        let auc = my_individual.compute_auc(&my_data);
        if (auc>auc_max) {auc_max=auc;best_individual=my_individual;}
    }
    println!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}


/// the Genetic Algorithm test
fn ga_run(param: &Param) {
    println!("                          GA TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    println!("{:?}", my_data); 
    
    let mut populations = ga::ga(&mut my_data,&param);

    let mut population=populations.pop().unwrap();

    if param.data.Xtest.len()>0 {
        let mut test_data=Data::new();
        test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            let auc=individual.auc;
            let test_auc=individual.compute_auc(&test_data);
            let (threshold, accuracy, sensitivity, specificity) = individual.compute_threshold_and_metrics(&test_data);
            println!("Model #{} [k={}]: train AUC {}  | test AUC {} | threshold {} | accuracy {} | sensitivity {} | specificity {} | {:?}",
                        i+1,individual.k,auc,test_auc,threshold,accuracy,sensitivity,specificity,individual);
        }    
    }
    else {
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            let auc=individual.auc;
            println!("Model #{} [k={}]: train AUC {}",i+1,individual.k,auc);
        }    
    }


}


/// the Genetic Algorithm test with Crossval (not useful but test CV)
fn gacv_run(param: &Param) {
    println!("                          GA CV TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    println!("{:?}", my_data); 

    let mut crossval = cv::CV::new(&my_data, 10, &mut rng);
    let results=crossval.pass(ga::ga, &param);
    
    if param.data.Xtest.len()>0 {
        let mut test_data=Data::new();
        test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        for (i,(mut best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            let holdout_auc=best_model.compute_auc(&test_data);
            let (threshold, accuracy, sensitivity, specificity) = best_model.compute_threshold_and_metrics(&test_data);
            //println!("Model #{} [k={}]: train AUC {:.3} | test AUC {:.3} | holdout AUC {:.3} | {:?}",i+1,best_model.k,train_auc,test_auc,holdout_auc,best_model);
            println!("Model #{} [k={}]: train AUC {:.3}  | test AUC {:.3} | holdout AUC {:.3} | threshold {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} | {:?}",
                        i+1,best_model.k,train_auc,test_auc,holdout_auc,threshold,accuracy,sensitivity,specificity,best_model);
            print!("Features importance on train+test: ");
            for feature_importance in best_model.compute_oob_feature_importance(&my_data, param.ga.feature_importance_permutations,&mut rng) {
                print!("[{:.4}] ",feature_importance);
            }
            println!();
            print!("Features importance on holdout: ");
            for feature_importance in best_model.compute_oob_feature_importance(&test_data, param.ga.feature_importance_permutations,&mut rng) {
                print!("[{:.4}] ",feature_importance);
            }
            println!();
        }    
    }
    else {
        for (i,(best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            println!("Model #{} [k={}]: train AUC {:.3} | test AUC {:.3} | {:?}",i+1,best_model.k,train_auc,test_auc,best_model);
        }    
    }


}

fn main() {
    let param= param::get("param.yaml".to_string()).unwrap();
    match param.general.algo.as_str() {
        "random" => random_run(&param),
        "ga" => ga_run(&param),
        "ga+cv" => gacv_run(&param),
        other => { println!("ERROR! No such algorithm {}", other);  process::exit(1); }
    }
    //basic_test();
}

