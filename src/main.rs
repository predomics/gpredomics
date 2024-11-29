mod data;
mod utils;
mod individual;
mod param;
mod population;
mod ga;

use data::Data;
use individual::Individual;

fn main() {
    let mut my_data = Data::new();
    //my_data.X = vec! [ vec! [0.1,0.2,0.3], vec! [0.0, 0.0, 0.0], vec! [0.9,0.8,0.7] ];
    //my_data.samples = string_vec! ["a","b","c"];
    //my_data.features = string_vec! ["msp1","msp2","msp3"];
    //my_data.y = vec! [0,1,1];
    my_data.load_data("samples/X.tsv","samples/y.tsv");
    println!("{:?}", my_data); // Example usage
    
    //let (averages,prevalences)=my_data.compute_feature_stats();
    //println!("averages {:?}",averages);
    //println!("prevalences {:?}",prevalences);
    

    let mut auc_max = 0.0;
    let mut best_individual: Individual = Individual::new();
    for i in 0..100000 {
    let my_individual = Individual::random(&my_data);
    //let mut my_individual = Individual::new();
    //my_individual.features = vec! [1,0,-1];

    //println!("my individual: {:?}",my_individual.features);
    //println!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));
    let auc = my_individual.compute_auc(&my_data);
    if (auc>auc_max) {auc_max=auc;best_individual=my_individual;}
    }
    println!("auc max: {} model: {:?}",auc_max, best_individual.features);
}

