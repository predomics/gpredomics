use std::collections::HashMap;

use rand::rngs::ThreadRng;
use rand::Rng;
use statrs::statistics::Statistics;
use statrs::distribution::{ContinuousCDF, StudentsT};
use rand_chacha::ChaCha8Rng;
use rand::seq::SliceRandom; 


/// a macro to declare simple Vec<String>
#[macro_export]
macro_rules! string_vec {
    ($($x:expr),*) => {
        vec![$($x.into()),*]
    };
}

pub fn generate_random_vector(reference_size: usize, rng: &mut ChaCha8Rng) -> Vec<i8> {
    // chose k variables amount feature_selection
    // set a random coeficient for these k variables


    // Generate a vector of random values: 1, 0, or -1
    (0..reference_size).map(|i| { rng.gen_range(-1..1) }).collect()
}




/// a function used essentially in CV that split randomly a Vec<T> into p Vec<T> of approximatively the same size
pub fn split_into_balanced_random_chunks<T: std::clone::Clone>(vec: Vec<T>, p: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<T>> {
    // Step 1: Shuffle the original vector
    let mut shuffled = vec;
    shuffled.shuffle(rng);

    // Step 2: Determine sizes for balanced chunks
    let n = shuffled.len();
    let base_size = n / p; // Minimum size for each chunk
    let extra_elements = n % p; // Remaining elements to distribute

    // Step 3: Create chunks with balanced sizes
    let mut chunks = Vec::new();
    let mut start = 0;

    for i in 0..p {
        let chunk_size = base_size + if i < extra_elements { 1 } else { 0 }; // Add one extra element to the first `extra_elements` chunks
        let end = start + chunk_size;
        chunks.push(shuffled[start..end].to_vec());
        start = end;
    }

    chunks
}

/// shuffle a feature
pub fn shuffle_row(X: &mut HashMap<(usize, usize), f64>, sample_len: usize, feature: usize, rng: &mut ChaCha8Rng) {
    // Extract all the column indices and values for the given row
    let mut feature_values: Vec<(usize, f64)> = (0..sample_len)
        .filter_map(|i| X.get(&(i, feature)).map(|&val| (i, val)))
        .collect();

    // Shuffle the column indices
    feature_values.shuffle(rng);

    // Update the matrix with shuffled values
    for (_, &(i, value)) in feature_values.iter().enumerate() {
        X.insert((i, feature), value);
    }
}
