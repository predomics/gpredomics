use std::collections::HashMap;
use rand::SeedableRng;
use rand::Rng;
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
    (0..reference_size).map(|_| { rng.gen_range(-1..2) }).collect()
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
    let feature_values: Vec<f64> = (0..sample_len)
        .filter_map(|i| X.remove(&(i, feature)))
        .collect();

    // Shuffle the column indices
    //feature_values.shuffle(rng);
    let new_samples: Vec<usize> = (0..sample_len).collect::<Vec<usize>>()
                        .choose_multiple(rng, feature_values.len()).copied().collect();


    // Update the matrix with shuffled values
    for (value,new_sample) in feature_values.iter().zip(new_samples.iter()) {
        X.insert((*new_sample, feature), *value);
    }
}


// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    // tests for generate_random_vector
    #[test]
    fn test_generate_random_vector_vector_size() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 10;
        let vector = generate_random_vector(size, &mut rng);
        assert_eq!(vector.len(), size, "the generated vector does not match the input size");
    }

    #[test]
    fn test_generate_random_vector_random_values() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 100;
        let vector = generate_random_vector(size, &mut rng);

        for &value in &vector {
            assert!(value == -1 || value == 0 || value == 1, "the generated vector contains value.s outside [-1 ; 1]");
        }
    }

    #[test]
    fn test_generate_random_vector_empty_vector() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 0;
        let vector = generate_random_vector(size, &mut rng);
        assert!(vector.is_empty(), "the generated vector should be empty for an input size of 0");
    }

    #[test]
    fn test_generate_random_vector_deterministic_output_and_reproductibility() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let size = 10;

        let vector1 = generate_random_vector(size, &mut rng1);
        let vector2 = generate_random_vector(size, &mut rng2);

        assert_eq!(vector1, vector2, "the same seed generated two different vectors");
        assert_eq!(vector1, vec![-1, 1, -1, 1, 1, 0, 0, 0, 1, -1], "the generated vector isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }

    // tests for split_into_balanced_random_chunks
    #[test]
    fn test_split_into_balanced_random_chunks_split_remainder_division() {
        let chunks = split_into_balanced_random_chunks(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(chunks.len(), 3, "the count of chunks does not match the input");
        assert_eq!(chunks[0].len(), 4, "the first chunk must have one more value when this is a remainder division");
        assert_eq!(chunks[1].len(), 3, "the count of value per chunck is not respected");
        assert_eq!(chunks[2].len(), 3, "the count of value per chunck is not respected");
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_remainderless_division() {
        let chunks = split_into_balanced_random_chunks(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(chunks.len(), 3, "the count of chunks does not match the input");
        assert_eq!(chunks[0].len(), 3, "the first chunk must have the same number of value when this is a remainderless division");
        assert_eq!(chunks[1].len(), 3, "the count of value per chunck is not respected");
        assert_eq!(chunks[2].len(), 3, "the count of value per chunck is not respected");
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_into_single_chunk() {
        let chunks = split_into_balanced_random_chunks(vec![1, 2, 3, 4, 5], 1, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(chunks.len(), 1, "the count of chunks does not match the input");
        assert_eq!(chunks[0].len(), 5, "when splitted in one part, the chunk must be equal to the input vector");
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_empty_vectors() {
        let vec: Vec<i32> = vec![];
        let chunks = split_into_balanced_random_chunks(vec, 3, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(chunks.len(), 3, "the count of chunks does not match the input");
        for chunk in chunks {
            assert!(chunk.is_empty(), "empty vector should to the formation of empty chunks");
        }
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_more_chunks_than_elements() {
        let vec = vec![1, 2, 3];
        let chunks = split_into_balanced_random_chunks(vec.clone(), 5, &mut ChaCha8Rng::seed_from_u64(42));

        // vecs 1, 2, 3 should contain one value, vecs 4 and 5 should be empty
        assert_eq!(chunks.len(), 5, "the count of chunks does not match the input");
        assert_eq!(chunks.iter().filter(|chunk| !chunk.is_empty()).count(), vec.len(), "the chunks exceeding the values count must be empty");
    }

    #[test]
    fn test_split_into_balanced_random_chunks_deterministic_split_and_reproductibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = 4;

        let chunks1 = split_into_balanced_random_chunks(vec.clone(), p, &mut rng);
        let chunks2 = split_into_balanced_random_chunks(vec.clone(), p, &mut rng2);

        assert_eq!(chunks1, chunks2, "the same seed generated two different chunks");
        assert_eq!(chunks1, vec![vec![1, 4, 6], vec![8, 10, 5], vec![9, 2], vec![7, 3]], "the generated chunks isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }

    // tests for shuffle_row
    #[test]
    fn test_shuffle_row_preserves_values_and_reorder() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();

        X.insert((0, 0), 1.0);
        X.insert((1, 0), 2.0);
        X.insert((2, 0), 3.0);
        X.insert((3, 0), 4.0);

        let sample_len = 4;
        let feature = 0;

        shuffle_row(&mut X, sample_len, feature, &mut rng);

        let values: Vec<f64> = (0..sample_len).filter_map(|i| X.get(&(i, feature)).copied()).collect();
        assert_eq!(values.len(), 4, "HashMap must contain the same number of values after shuffle");
        assert!(values.contains(&1.0), "the shuffle must conserve HashMap values");
        assert!(values.contains(&2.0), "the shuffle must conserve HashMap values");
        assert!(values.contains(&3.0), "the shuffle must conserve HashMap values");
        assert!(values.contains(&4.0), "the shuffle must conserve HashMap values");
        assert_ne!(values, vec![1.0, 2.0, 3.0, 4.0], "the shuffle must conserve HashMap values");
    }

    #[test]
    fn test_shuffle_row_empty_column() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();

        let sample_len = 4;
        let feature = 0;

        shuffle_row(&mut X, sample_len, feature, &mut rng);

        for i in 0..sample_len {
            assert!(!X.contains_key(&(i, feature)), "the shuffle of an empty HashMap should also be empty");
        }
    }

    #[test]
    fn test_shuffle_row_deterministic_shuffle_and_reproductibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X1: HashMap<(usize, usize), f64> = HashMap::new();
        
        X1.insert((0, 0), 1.0);
        X1.insert((1, 0), 2.0);
        X1.insert((2, 0), 3.0);
        X1.insert((3, 0), 4.0);

        let mut X2 = X1.clone();
        let sample_len = 4;
        let feature = 0;

        shuffle_row(&mut X1, sample_len, feature, &mut ChaCha8Rng::seed_from_u64(42));
        shuffle_row(&mut X2, sample_len, feature, &mut rng);
        
        assert_eq!(X1, X2, "the same seed generated two different chunks");
        assert_eq!(X1.get(&(0, 0)), Some(&2.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(2, 0)), Some(&1.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(1, 0)), Some(&3.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(3, 0)), Some(&4.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }   
}
