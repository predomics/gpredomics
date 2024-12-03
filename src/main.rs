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
fn random_test() {
    println!("                          RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut my_data = Data::new();
    my_data.load_data("sample/Derosa2022/X.tsv","sample/Derosa2022/y.tsv");
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
fn ga_test() {
    println!("                          GA TEST\n-----------------------------------------------------");
    let param= param::get("param.yaml".to_string()).unwrap();
    let mut my_data = Data::new();
    
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    println!("{:?}", my_data); 
    
    let mut populations = ga::ga(&mut my_data,&param);

    println!("Best model: {:?}",populations.pop().unwrap().individuals[0]);

}



/// the Genetic Algorithm test
fn ga_test_qin2014() {
    println!("                          GA TEST\n-----------------------------------------------------");
    let param= param::get("param.yaml".to_string()).unwrap();
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
            println!("Model #{} [k={}]: train AUC {}  / test AUC {} : {:?}",i+1,individual.k,auc,test_auc,individual);
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
fn gacv_test_qin2014() {
    println!("                          GA CV TEST\n-----------------------------------------------------");
    let param= param::get("param.yaml".to_string()).unwrap();
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
            println!("Model #{} [k={}]: train AUC {:.3} | test AUC {:.3} | holdout AUC {:.3} | {:?}",i+1,best_model.k,train_auc,test_auc,holdout_auc,best_model);
        }    
    }
    else {
        for (i,(best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            println!("Model #{} [k={}]: train AUC {:.3} | test AUC {:.3} | {:?}",i+1,best_model.k,train_auc,test_auc,best_model);
        }    
    }


}

fn main() {
    gacv_test_qin2014();
}

