use rand::Rng;
use statrs::statistics::Statistics;
use statrs::distribution::{ContinuousCDF, StudentsT};

/// a macro to declare simple Vec<String>
#[macro_export]
macro_rules! string_vec {
    ($($x:expr),*) => {
        vec![$($x.into()),*]
    };
}

pub fn generate_random_vector(reference_size: usize) -> Vec<i8> {
    // chose k variables amount feature_selection
    // set a random coeficient for these k variables


    let mut rng = rand::thread_rng();

    // Generate a vector of random values: 1, 0, or -1
    (0..reference_size).map(|i| { rng.gen_range(-1..1) }).collect()
}



pub fn compare_classes(values: &Vec<f64>, targets: &Vec<u8>, max_p_value: f64, min_prevalence: f64) -> u8 {
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

