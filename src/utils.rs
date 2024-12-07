use rand::rngs::ThreadRng;
use rand::Rng;
use statrs::statistics::Statistics;
use statrs::distribution::{ContinuousCDF, StudentsT};
use rand_chacha::ChaCha8Rng;
use rand::seq::SliceRandom; 
use statrs::distribution::Normal;// For random shuffling

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



pub fn compare_classes_studentt(values: &Vec<f64>, targets: &Vec<u8>, max_p_value: f64, min_prevalence: f64, min_mean_value: f64) -> u8 {
    // Separate values into two classes
    let class_0: Vec<f64> = values.iter().zip(targets.iter())
        .filter(|(_, &class)| class == 0)
        .map(|(&value, _)| value)
        .collect();

    let class_1: Vec<f64> = values.iter().zip(targets.iter())
        .filter(|(_, &class)| class == 1)
        .map(|(&value, _)| value)
        .collect();

    // Calculate means
    let mean_0 = class_0.iter().copied().sum::<f64>() / class_0.len() as f64;
    let mean_1 = class_1.iter().copied().sum::<f64>() / class_1.len() as f64;

    if mean_0<min_mean_value && mean_1<min_mean_value { return 2 }

    // Calculate t-statistic (simple, equal variance assumption)
    let n0 = class_0.len() as f64;
    let n1 = class_1.len() as f64;
    let var0 = class_0.iter().map(|x| (x - mean_0).powi(2)).sum::<f64>() / (n0 - 1.0);
    let var1 = class_1.iter().map(|x| (x - mean_1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let prev0 = class_0.iter().filter(|&&x| x != 0.0).count() as f64 / n0;
    let prev1 = class_1.iter().filter(|&&x| x != 0.0).count() as f64 / n1;
    let pooled_std = ((var0 / n0) + (var1 / n1)).sqrt();
    if pooled_std > 0.0 {
        let t_stat = (mean_0 - mean_1) / pooled_std;

        // Compute p-value
        let degrees_of_freedom = (n0 + n1 - 2.0).round();
        let t_dist = StudentsT::new(0.0, 1.0, degrees_of_freedom).unwrap();
        //println!("t_stat {} n0 {} n1 {} var0 {} var1 {} prev0 {} prev1 {}",t_stat,n0,n1,var0,var1,prev0,prev1);
        let cumulative = t_dist.cdf(t_stat.abs()); // CDF up to |t_stat|
        let upper_tail = 1.0 - cumulative;         // Upper-tail area
        let p_value = 2.0 * upper_tail;            // Two-tailed test

        // Interpretation
        if (p_value < max_p_value) && (prev0 > min_prevalence || prev1 > min_prevalence) {
            if mean_0 > mean_1 {
                0
            } else {
                1
            }
        } else {
            2
        }
    }
    else {2}
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

pub fn compare_classes_wilcoxon(values: &Vec<f64>, targets: &Vec<u8>, max_p_value: f64, min_prevalence: f64, min_mean_value: f64) -> u8 {
    // Separate values into two classes
    let mut class_0: Vec<f64> = Vec::new();
    let mut class_1: Vec<f64> = Vec::new();

    for (&value, &class) in values.iter().zip(targets.iter()) {
        if class == 0 {
            class_0.push(value);
        } else if class == 1 {
            class_1.push(value);
        }
    }

    // Check if both classes have enough data points for statistical testing
    if class_0.is_empty() || class_1.is_empty() {
        return 2; // Unable to compare due to insufficient data
    }

    // Calculate means
    let mean_0 = class_0.iter().copied().sum::<f64>() / class_0.len() as f64;
    let mean_1 = class_1.iter().copied().sum::<f64>() / class_1.len() as f64;

    //println!("Means: {} vs {}",mean_0,mean_1);
    if mean_0<min_mean_value && mean_1<min_mean_value { return 2 }

    // Compute prevalence for each class
    let n0 = class_0.len() as f64;
    let n1 = class_1.len() as f64;
    let prev0 = n0 / (n0 + n1);
    let prev1 = n1 / (n0 + n1);

    // Skip comparison if prevalence is below the minimum threshold
    if prev0 < min_prevalence && prev1 < min_prevalence {
        return 2;
    }

    // Combine both classes with their labels
    let mut combined: Vec<(f64, u8)> = class_0
        .iter()
        .map(|&value| (value, 0))
        .chain(class_1.iter().map(|&value| (value, 1)))
        .collect();

    // Sort combined values by value, breaking ties arbitrarily
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign ranks
    let mut ranks = vec![0.0; combined.len()];
    let mut i = 0;
    while i < combined.len() {
        let start = i;
        while i + 1 < combined.len() && combined[i].0 == combined[i + 1].0 {
            i += 1;
        }
        let rank = (start + i + 1) as f64 / 2.0;
        for j in start..=i {
            ranks[j] = rank;
        }
        i += 1;
    }

    // Compute rank sums
    let rank_sum_0: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|((_, class), _)| *class == 0)
        .map(|(_, &rank)| rank)
        .sum();

    // Compute U statistic
    let u_stat = rank_sum_0 - (n0 * (n0 + 1.0)) / 2.0;

    // Compute p-value using normal approximation
    let mean_u = n0 * n1 / 2.0;
    let std_u = ((n0 * n1 * (n0 + n1 + 1.0)) / 12.0).sqrt();
    let z = (u_stat - mean_u) / std_u;

    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * (1.0 - normal_dist.cdf(z.abs())); // Two-tailed p-value

    // Interpretation
    if p_value < max_p_value {
        if rank_sum_0 > (n0 * n1 / 2.0) {
            0
        } else {
            1
        }
    } else {
        2
    }
}
