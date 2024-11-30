mod data;
mod utils;
mod individual;
mod param;
mod population;
mod ga;

use data::Data;
use individual::Individual;

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
    my_data.load_data("sample/X.tsv","sample/y.tsv");

    let mut auc_max = 0.0;
    let mut best_individual: Individual = Individual::new();
    for i in 0..10000 {
        let my_individual = Individual::random(&my_data);

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

fn main() {
    ga_test();
}

