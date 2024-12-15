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
fn basic_test(param: &Param) {
    println!("                          BASIC TEST\n-----------------------------------------------------");
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
    my_data.y = vec! [0.0,1.0,1.0];
    my_data.feature_len = 3;
    my_data.sample_len = 3;
    println!("{:?}", my_data);

    // create a model
    let mut my_individual = Individual::new();
    my_individual.features.insert(0, 1);
    my_individual.features.insert(2, -1);
    println!("my individual: {:?}",my_individual.features);
    println!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));
    // shoud display 1.0 (the AUC is 1.0)
    println!("my individual AUC: {:?}",my_individual.compute_auc(&my_data));


    let mut data2=Data::new();
    data2.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let parent1 = Individual::random(&data2, &mut rng);
    let parent2 = Individual::random(&data2, &mut rng);
    let mut parents=population::Population::new();
    parents.individuals.push(parent1);
    parents.individuals.push(parent2);
    let mut children = ga::cross_over(&parents, &param, data2.feature_len, &mut rng);
    for (i,individual) in parents.individuals.iter().enumerate() { println!("Parent #{}: {:?}",i,individual); }
    for (i,individual) in children.individuals.iter().enumerate() { println!("Child #{}: {:?}",i,individual); }
    let feature_selection:Vec<usize> = (0..data2.feature_len).collect();
    ga::mutate(&mut children, param, &feature_selection, &mut rng);
    for (i,individual) in children.individuals.iter().enumerate() { println!("Mutated Child #{}: {:?}",i,individual); }    

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
    let results=crossval.pass(ga::ga, &param, param.general.thread_number);
    
    if param.data.Xtest.len()>0 {
        let mut test_data=Data::new();
        test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        for (i,(mut best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            let holdout_auc=best_model.compute_auc(&test_data);
            let (threshold, accuracy, sensitivity, specificity) = best_model.compute_threshold_and_metrics(&test_data);
            //println!("Model #{} [k={}]: train AUC {:.3} | test AUC {:.3} | holdout AUC {:.3} | {:?}",i+1,best_model.k,train_auc,test_auc,holdout_auc,best_model);
            println!("Model #{} [gen:{}] [k={}]: train AUC {:.3}  | test AUC {:.3} | holdout AUC {:.3} | threshold {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} | {:?}",
                        i+1,best_model.n,best_model.k,train_auc,test_auc,holdout_auc,threshold,accuracy,sensitivity,specificity,best_model);
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
            println!("Model #{} [gen:{}] [k={}]: train AUC {:.3} | test AUC {:.3} | {:?}",i+1,best_model.n,best_model.k,train_auc,test_auc,best_model);
        }    
    }


}

fn main() {
    let param= param::get("param.yaml".to_string()).unwrap();
    match param.general.algo.as_str() {
        "basic" => basic_test(&param),
        "random" => random_run(&param),
        "ga" => ga_run(&param),
        "ga+cv" => gacv_run(&param),
        other => { println!("ERROR! No such algorithm {}", other);  process::exit(1); }
    }
    //basic_test();
}

