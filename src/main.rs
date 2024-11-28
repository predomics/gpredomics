mod data;
mod utils;
mod individuals;

use data::Data;
use individuals::Individual;

fn main() {
    let mut my_data = Data::new();
    my_data.X = vec! [ vec! [0.1,0.2,0.3], vec! [0.0, 0.0, 0.0], vec! [0.9,0.8,0.7] ];
    my_data.samples = string_vec! ["a","b","c"];
    my_data.features = string_vec! ["msp1","msp2","msp3"];
    my_data.y = vec! [0,1,1];
    //my_data.load_data("samples/X.tsv","samples/y.tsv");
    println!("{:?}", my_data); // Example usage
    
    //let my_individual = Individual::random(&my_data);
    let mut my_individual = Individual::new();
    my_individual.features = vec! [1,0,-1];

    println!("my individual: {:?}",my_individual.features);
    println!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));
    println!("my individual auc: {:?}",my_individual.compute_auc(&my_data));
    

}

