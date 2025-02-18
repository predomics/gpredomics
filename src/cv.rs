use rayon::prelude::*;
use crate::data::Data;
use crate::population::Population;
use crate::param::Param;
use crate::individual::Individual;
use crate::utils;
use std::sync::{Arc, Mutex};
use rand_chacha::ChaCha8Rng;
use log::info;

use std::sync::atomic::AtomicBool;

/// This class implement Cross Validation dataset, e.g. split the Data in N folds and create N subset of Data each with its test subset.
pub struct CV {
    pub folds: Vec<Data>,
    pub datasets: Vec<Data>
}

impl CV {
    pub fn new(data: &Data, fold_number: usize, rng: &mut ChaCha8Rng) -> CV {

        let mut indices_class0:Vec<usize> = Vec::new();
        let mut indices_class1:Vec<usize> = Vec::new();
        for (i,f) in data.y.iter().enumerate() {
            if *f==0 { indices_class0.push(i) } else if *f==1 { indices_class1.push(i) }
        }

        let indices_class0_folds = utils::split_into_balanced_random_chunks(indices_class0, fold_number, rng);
        let indices_class1_folds = utils::split_into_balanced_random_chunks(indices_class1, fold_number, rng);

        let folds: Vec<Data>  = indices_class0_folds.into_iter().zip(indices_class1_folds.into_iter())
                    .map( |(i1,i2)| {i1.into_iter().chain(i2).collect::<Vec<usize>>()} )
                    .map( |i| { data.subset(i) } )
                    .collect();

        let mut datasets: Vec<Data> = Vec::new();
        for i in 0..fold_number {
            let mut dataset: Data = if i==0 { folds[1].clone() } else { folds[0].clone() };

            for j in 1..fold_number {
                if j==i { continue }
                else {
                    dataset.add(&folds[j]);
                }
            }

            datasets.push(dataset);
        }

        CV {
            folds: folds,
            datasets: datasets
        }
    }

    pub fn pass<F>(
        &mut self,
        algo: F,
        param: &Param,
        thread_number: usize,
        running: Arc<AtomicBool>
    ) -> Vec<(Individual, f64, f64)>
    where F: Fn(&mut Data, &Param, Arc<AtomicBool>) -> Vec<Population> + std::marker::Send + std::marker::Sync, {
        // Configure the thread pool with the specified thread number
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_number)
            .build()
            .unwrap();

        // Arc-Mutex to collect results from threads safely
        let results_per_fold = Arc::new(Mutex::new(Vec::new()));

        thread_pool.install(|| {
            self.datasets
                .par_iter_mut()
                .zip(self.folds.par_iter_mut())
                .enumerate()
                .for_each(|(i, (train, test))| {
                    // Train and evaluate model
                    info!("Completing fold...");

                    let mut best_model: Individual =
                        algo(train, param, Arc::clone(&running)).pop().unwrap().individuals.into_iter().take(1).next().unwrap();
                    let train_auc = best_model.auc;
                    let test_auc = best_model.compute_auc(test);

                    info!(
                        "Fold #{}  |  Train AUC: {:.3}  |  Test AUC: {:.3}",
                        i+1, train_auc, test_auc
                    );

                    // Store the results
                    results_per_fold
                        .lock()
                        .unwrap()
                        .push((best_model, train_auc, test_auc));
                });
        });

        // Extract results from the Arc-Mutex
        Arc::try_unwrap(results_per_fold)
            .unwrap()
            .into_inner()
            .unwrap()
    }
}

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    fn create_test_data() -> Data {
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        let mut feature_class: HashMap<usize, u8> = HashMap::new();

        // Simulate data
        X.insert((0, 0), 0.1);
        X.insert((0, 1), 0.4);
        X.insert((1, 0), 0.2);
        X.insert((1, 1), 0.5);
        X.insert((2, 0), 0.3);
        X.insert((2, 1), 0.6);
        X.insert((3, 0), 0.7);
        X.insert((3, 1), 0.8);
        X.insert((4, 0), 0.1);
        X.insert((4, 1), 0.2);
        X.insert((5, 0), 0.9);
        X.insert((5, 1), 0.8);
        feature_class.insert(0, 0);
        feature_class.insert(1, 1);

        Data {
            X,
            y: vec![0, 1, 0, 1, 1, 1],
            features: vec!["feature1".to_string(), "feature2".to_string()],
            samples: vec!["sample1".to_string(), "sample2".to_string(), "sample3".to_string(), "sample4".to_string(), "sample5".to_string(), "sample6".to_string()],
            feature_class,
            feature_selection: vec![0, 1],
            feature_len: 2,
            sample_len: 6,
        }
    }

    #[test]
    fn test_cv_new_creates_correct_number_of_folds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = create_test_data();
        let fold_number = 3;
        let cv = CV::new(&data, fold_number, &mut rng);
        assert_eq!(cv.folds.len(), fold_number);
        assert_eq!(cv.datasets.len(), fold_number);
    }

    #[test]
    fn test_cv_new_distributes_y_correctly_and_preserve_them() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = create_test_data();
        let fold_number = 3;
        let cv = CV::new(&data, fold_number, &mut rng);

        // Check that y are correctly splitted
        let expected_size = (data.y.len() + fold_number - 1) / fold_number;
        for fold in &cv.folds {
            let fold_size = fold.y.len();
            assert!((fold_size as isize - expected_size as isize).abs() <= 1);
        }

        // Check that all data is preserved across all folds
        let mut real_y: Vec<usize> = data.y.iter().map(|&x| x as usize).collect();
        let mut collected_y = Vec::new();
        for fold in &cv.folds {
            collected_y.extend(fold.y.iter().map(|&x| x as usize));
        }
        real_y.sort();
        collected_y.sort();
 
         assert_eq!(collected_y, real_y);
    }

    // add a unit test to check if y are correctly distribued ? 

    #[test]
    fn test_cv_new_reproductibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = create_test_data();
        let fold_number = 3;
        let cv = CV::new(&data, fold_number, &mut rng);
        
        let mut X1: HashMap<(usize, usize), f64> = HashMap::new();
        let mut X2: HashMap<(usize, usize), f64> = HashMap::new();
        let mut X3: HashMap<(usize, usize), f64> = HashMap::new();

        X1.insert((0, 0), 0.3);
        X1.insert((0, 1), 0.6);
        X1.insert((1, 0), 0.2);
        X1.insert((1, 1), 0.5);
        X1.insert((2, 0), 0.1);
        X1.insert((2, 1), 0.2);
        X2.insert((0, 0), 0.1);
        X2.insert((0, 1), 0.4);
        X2.insert((1, 0), 0.7);
        X2.insert((1, 1), 0.8);
        X3.insert((0, 0), 0.9);
        X3.insert((0, 1), 0.8);

        assert_eq!(cv.folds[0].X, X1);
        assert_eq!(cv.folds[1].X, X2);
        assert_eq!(cv.folds[2].X, X3);
        assert_eq!(cv.folds[0].y, [0, 1, 1]);
        assert_eq!(cv.folds[1].y, [0, 1]);
        assert_eq!(cv.folds[2].y, [1]);

        assert_eq!(cv.folds[0].samples, ["sample3", "sample2", "sample5"]);
        assert_eq!(cv.folds[1].samples, ["sample1", "sample4"]);
        assert_eq!(cv.folds[2].samples, ["sample6"]);
        assert_eq!(cv.folds[0].sample_len, 3);
        assert_eq!(cv.folds[1].sample_len, 2);
        assert_eq!(cv.folds[2].sample_len, 1);

        for fold in &cv.folds {
            assert_eq!(fold.features, ["feature1", "feature2"]);
            assert_eq!(fold.feature_len, 2);
        }
    }

    // tests for pass -> need to check with Raynald

}