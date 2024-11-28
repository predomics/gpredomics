use rand::Rng;

/// a macro to declare simple Vec<String>
#[macro_export]
macro_rules! string_vec {
    ($($x:expr),*) => {
        vec![$($x.into()),*]
    };
}

pub fn generate_random_vector(reference_size: usize) -> Vec<i8> {
    let mut rng = rand::thread_rng();

    // Generate a vector of random values: 1, 0, or -1
    (0..reference_size).map(|_| {
        match rng.gen_range(0..3) {
            0 => 1,    // 0 maps to 1
            1 => 0,    // 1 maps to 0
            2 => -1,   // 2 maps to -1
            _ => unreachable!(),
        }
    }).collect()
}

