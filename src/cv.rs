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
                    let test_auc = best_model.compute_auc(test, param.general.data_type_epsilon);

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
