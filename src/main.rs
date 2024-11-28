mod data;
mod utils;
mod individuals;

use data::Data;
use individuals::Individual;

fn main() {
    let mut my_data = Data::new();
    my_data.load_data("samples/X.tsv","samples/y.tsv");
    println!("{:?}", my_data); // Example usage
    
    let my_individual = individuals::Individual::random(&my_data);
    println!("my individual: {:?}",my_individual.features);
    println!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));

}

