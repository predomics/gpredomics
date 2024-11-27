mod data; // Declare the data module

use data::Data; // Bring the Data struct into scope

fn main() {
    let mut my_data = data::Data::new();
    my_data.load_data("samples/X.tsv","samples/y.tsv");
    println!("{:?}", my_data.X); // Example usage
}

